"""Auditable fresh collection, independent training and latest-weight recording."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import multiprocessing
from pathlib import Path
import threading
import time


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False,
                                    default=str), encoding='utf-8')
    for attempt in range(8):
        try:
            temporary.replace(path)
            break
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(.01*2**attempt)


def task_success(task, summary, seconds):
    duration = summary.get('time_seconds', -1.)
    if (not isinstance(duration, (int, float)) or not math.isfinite(duration)
            or abs(duration-seconds) > 1e-6 or summary.get('terminated', False)
            or any(summary.get(k, False) for k in
                   ('uav_collision', 'ship_collision', 'out_of_bounds'))):
        return False
    key = 'coverage_ratio' if task == 'coverage' else 'visible_fraction'
    value = summary.get(key, 0.)
    return (isinstance(value, (int, float)) and math.isfinite(value)
            and value >= 1.-1e-9
            and (task != 'tracking' or summary.get('first_loss_time') is None))


def freeze_latest(task, model_dir, frozen_dir):
    import torch
    source = Path(model_dir)/'last.pt'
    payload = source.read_bytes()
    state = torch.load(io.BytesIO(payload), map_location='cpu', weights_only=False)
    if state.get('task') != task:
        raise ValueError('Latest checkpoint task mismatch')
    sha = hashlib.sha256(payload).hexdigest()
    target = Path(frozen_dir)/f'{task}-last-{sha}.pt'
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.read_bytes() != payload:
        raise ValueError('Frozen checkpoint bytes changed')
    if not target.exists():
        temporary=target.with_suffix('.pt.tmp')
        temporary.write_bytes(payload)
        temporary.replace(target)
    return dict(path=str(target.resolve()), source=str(source.resolve()), sha256=sha,
                epoch=state['epoch'])


def formal_protocol(args, config, tasks):
    root = Path(__file__).resolve().parents[1]
    source_files = sorted([root/'main.py', root/'run.sh', *root.joinpath('src').glob('*.py'),
                           *root.joinpath('visual').glob('*.js'), root/'visual/index.html',
                           root/'visual/styles.css'])
    source = hashlib.sha256()
    for path in source_files:
        source.update(path.relative_to(root).as_posix().encode())
        source.update(path.read_bytes())
    for task in tasks:
        offset = 10000 if task == 'tracking' else 0
        collection_start = args.collection_seed+offset
        test_start = args.formal_test_seed+offset
        if set(range(collection_start, collection_start+args.episodes*args.collection_attempt_multiplier)).intersection(
                range(test_start, test_start+args.formal_test_episodes)):
            raise ValueError('Final-test seeds overlap the expert collection attempt range')
        if args.expert_seed+(task == 'tracking') in range(test_start,test_start+args.formal_test_episodes):
            raise ValueError('Final-test seeds overlap expert preflight')
    protocol=dict(schema='maritime-formal-protocol-v1', episodes_per_task=args.episodes,
        source_sha256=source.hexdigest(),
        validation_fraction=args.validation_fraction, latest_checkpoint_name='last.pt',
        test_outcomes_used_for_weight_selection=False,
        split_unit='scene_group_id', simulation=asdict(config),
        training={key: getattr(args, key) for key in ('epochs','batch_size','sequence_length',
            'learning_rate','weight_decay','grad_clip','training_seed','hidden_size',
            'correction_weight_max','field_steering_loss_weight','normalize_vectors',
            'normalization_min_std','field_cloud_geometry_encoder','field_frame',
            'field_mask_previous_command','coverage_pan_reference','tracking_pan_reference',
            'coverage_direction_reference','tracking_direction_reference')},
        collection_seed=args.collection_seed, expert_seed=args.expert_seed,
        recording={key:getattr(args,key) for key in ('browser_recording','recording_fps',
            'recording_width','recording_height','recording_duration_seconds','recording_bitrate')},
        tasks={task: dict(modes=(['none', 'forecast'] if task == 'coverage'
                and args.formal_coverage_safety_layer == 'forecast' else ['none']),
            primary_mode=args.formal_coverage_safety_layer if task == 'coverage' else 'none',
            fresh_seeds=list(range(args.formal_test_seed + (10000 if task == 'tracking' else 0),
                                  args.formal_test_seed + (10000 if task == 'tracking' else 0)
                                  + args.formal_test_episodes)),
            duration_seconds=(args.collection_coverage_seconds if task == 'coverage'
                              else args.collection_tracking_seconds)) for task in tasks})
    # Config tuples and their persisted JSON lists have identical semantics.
    return json.loads(json.dumps(protocol,allow_nan=False))


class FormalStatus:
    def __init__(self, output_dir, run_name, tasks, server=None):
        self.path = Path(output_dir)/'formal-status.json'
        self.server = server
        self.lock = threading.RLock()
        self.value = dict(run_name=run_name, phase='preparing', task=None, status='running',
                          tasks={task: {} for task in tasks})
        self.update()

    def update(self, **values):
        with self.lock:
            self.value.update(values)
            self.value['updated_at'] = datetime.now(timezone.utc).isoformat()
            atomic_json(self.path, self.value)
            if self.server:
                self.server.publish(dict(type='run_status', **self.value))

    def task_update(self, task, key, values):
        with self.lock:
            self.value['tasks'][task].setdefault(key, {}).update(values)
            self.update()


class ProgressMonitor:
    def __init__(self, status, output_dir, tasks, seconds):
        self.status, self.out, self.tasks, self.seconds = status, Path(output_dir), tasks, seconds
        self.stop = threading.Event()
        self.poll_lock = threading.Lock()
        self.thread = threading.Thread(target=self.loop, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def poll(self):
        with self.poll_lock:
            self._poll()

    def _poll(self):
        for task in self.tasks:
            try:
                manifest = self.out/'datasets'/task/'accepted-manifest.json'
                progress = self.out/'models'/task/'training-status.json'
                if manifest.is_file():
                    value = json.loads(manifest.read_text(encoding='utf-8'))
                    self.status.task_update(task, 'collection', dict(
                        accepted=len(value.get('accepted', [])), target=value.get('target_episodes'),
                        rejected=len(value.get('rejected', [])), metrics=value.get('metrics', {}),
                        complete=value.get('complete', False)))
                if progress.is_file():
                    value = json.loads(progress.read_text(encoding='utf-8'))
                    value['train_loss'] = (value.get('train_metrics') or {}).get('loss')
                    value['validation_loss'] = (value.get('validation_metrics') or {}).get('loss')
                    self.status.task_update(task, 'training', value)
            except (OSError, ValueError) as error:
                print(f'[monitor] {task}: {error}', flush=True)
        compact = {task: {key: {k: v for k, v in data.items() if k in (
            'accepted', 'target', 'rejected', 'epoch', 'epochs', 'phase', 'operation',
            'processed_steps', 'train_loss', 'validation_loss', 'gradient_norm')}
            for key, data in self.status.value['tasks'][task].items()
            if key in ('collection', 'training')} for task in self.tasks}
        print('[formal-monitor] '+json.dumps(compact, ensure_ascii=False), flush=True)

    def loop(self):
        while not self.stop.wait(self.seconds):
            self.poll()

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join(timeout=5)
        self.poll()


def model_options(args, config, task):
    if not config.potential_enabled:
        raise ValueError('Formal protocol requires the agreed local field-v2 inputs')
    options = dict(potential_field=True, mask_previous_command=args.field_mask_previous_command,
        field_frame=args.field_frame, pan_residual=args.field_pan_residual,
        cloud_geometry_encoder=('shared_pool_v1' if task == 'coverage' else 'slots')
            if args.field_cloud_geometry_encoder == 'auto' else args.field_cloud_geometry_encoder,
        pan_reference=args.coverage_pan_reference if task == 'coverage' else args.tracking_pan_reference,
        direction_reference=args.coverage_direction_reference if task == 'coverage' else args.tracking_direction_reference)
    options.update(reference_orbit_radius=config.coverage_repair_orbit_radius if task == 'coverage'
                   else config.tracking_desired_distance,
                   reference_radial_gain=config.coverage_repair_radial_gain if task == 'coverage'
                   else config.tracking_radial_gain)
    return options


def _rollout(job, server=None):
    from config import Config
    from env_coverage import CoverageEnv
    from env_tracking import TrackingEnv
    from expert_field import FieldCoverageExpert, FieldTrackingExpert
    from rollouts import run_episode, write_json, write_replay_index
    config = Config.from_dict(job['config'])
    task = job['task']
    env = (CoverageEnv if task == 'coverage' else TrackingEnv)(config, render_mode='json')
    if job['actor'] == 'expert':
        actor = (FieldCoverageExpert if task == 'coverage' else FieldTrackingExpert)(config)
    else:
        import torch
        from .policy_tracking import NeuralPolicy
        torch.set_num_threads(job.get('inference_num_threads',1))
        actor = NeuralPolicy.from_checkpoint(job['checkpoint']['path'], device=job.get('inference_device','cpu'))
        if actor.checkpoint_metadata.get('sha256') != job['checkpoint']['sha256']:
            raise ValueError('Loaded latest checkpoint SHA256 differs from frozen protocol')
        if job['mode'] == 'forecast':
            from .policy_constraints import ForecastConstrainedPolicy
            actor = ForecastConstrainedPolicy(actor, config, task)
    try:
        record, path = run_episode(env, actor, seed=job['seed'], actor_name=job['actor'],
            output_dir=job['output_dir'], name=job['name'], server=server,
            render_every=job['replay_every'], replay_every=job['replay_every'],
            progress_every=job.get('progress_every', 300), scene=job.get('scene'))
        if record['metadata']['scene_group_id'] in job.get('excluded_scene_groups', []):
            raise ValueError('Final-test layout overlaps collected train/validation scenes')
        if job['actor'] == 'policy':
            record['metadata'].update(model_sha256=job['checkpoint']['sha256'],
                model_checkpoint=job['checkpoint']['source'],
                control_mode=record['metadata'].get('control_mode', 'pure_network'))
            write_json(path, record)
            write_replay_index(path, record)
        return dict(task=task, seed=job['seed'], mode=job.get('mode', 'expert'),
            path=str(path.resolve()), summary=record['summary'],
            success=task_success(task, record['summary'], config.to_dict()[f'{task}_episode_seconds']))
    finally:
        env.close()


def _test_rollout(job):
    """Keep failed independent cases without erasing other task outcomes."""
    try:
        return _rollout(job)
    except Exception as error:
        path=Path(job['output_dir'])/(job['name']+'.failure.json')
        result=dict(task=job['task'],seed=job['seed'],mode=job['mode'],path=str(path.resolve()),
                    success=False,summary=dict(terminated=True,termination_reason='test_runtime_failure',
                    error=dict(type=type(error).__name__,message=str(error))))
        atomic_json(path,result)
        return result


def run_formal(args, config, tasks, server, command):
    from .collection import collect_expert_episodes
    from .train import train_policy
    out = Path(args.output_dir).resolve()
    if not config.potential_enabled:
        raise ValueError('Formal run requires field-v2; use --potential-field')
    if args.browser_recording and server is None:
        raise ValueError('Browser recording requires --visualize')
    if args.browser_recording:
        from .browser_recording import _playwright_runtime
        _playwright_runtime()  # Fail early rather than after hours of fresh training.
    if not 0 < args.validation_fraction < 1:
        raise ValueError('Formal run requires disjoint training and validation subsets')
    if not args.collection_require_task_success:
        raise ValueError('Formal expert collection requires complete task success')
    protocol = formal_protocol(args, config, tasks)
    protocol_path = out/'formal-protocol.json'
    if protocol_path.exists():
        previous = json.loads(protocol_path.read_text(encoding='utf-8'))
        if not args.formal_resume or previous != protocol:
            raise ValueError('Formal protocol already exists or changed; use a fresh output directory')
    atomic_json(protocol_path, protocol)
    status = FormalStatus(out, args.run_name, tasks, server)
    report = dict(protocol=protocol, tasks={}, complete=False)
    report_path = out/'formal-report.json'
    configs = {task: replace(config, **{f'{task}_episode_seconds':
        protocol['tasks'][task]['duration_seconds']}) for task in tasks}
    try:
        with ProgressMonitor(status, out, tasks, args.formal_monitor_seconds) as monitor:
            for task in tasks:
                status.update(phase='expert_preflight', task=task)
                job = dict(task=task, config=asdict(configs[task]), actor='expert',
                    seed=args.expert_seed+(task == 'tracking'), output_dir=str(out/'preflight'),
                    name=f'{task}-expert', replay_every=args.collection_replay_every,
                    progress_every=args.progress_every)
                preflight = _rollout(job, server)
                report['tasks'][task] = dict(expert=preflight)
                atomic_json(report_path, report)
                if not preflight['success']:
                    raise RuntimeError(f'{task} expert preflight failed: {preflight["path"]}')
            for task in tasks:
                status.update(phase='collecting', task=task)
                collection = collect_expert_episodes(task, configs[task], out/'datasets'/task,
                    episodes=args.episodes, seed=args.collection_seed+(10000 if task == 'tracking' else 0),
                    workers=args.collection_workers, attempt_multiplier=args.collection_attempt_multiplier,
                    replay_every=args.collection_replay_every, progress_every=args.progress_every,
                    server=server, resume=args.collection_resume,
                    require_task_success=args.collection_require_task_success)
                report['tasks'][task]['collection'] = {key: collection[key] for key in
                    ('manifest_path', 'complete', 'metrics', 'identity_hash')}
                atomic_json(report_path, report)
                monitor.poll()
            for task in tasks:
                status.update(phase='training', task=task)
                model_dir = out/'models'/task
                latest = model_dir/'last.pt'
                training = train_policy(task, out/'datasets'/task/'accepted-manifest.json',
                    model_dir, Path(args.tf_log_dir)/task/args.run_name,
                    epochs=args.epochs, batch_size=args.batch_size, sequence_length=args.sequence_length,
                    learning_rate=args.learning_rate, weight_decay=args.weight_decay, grad_clip=args.grad_clip,
                    validation_fraction=args.validation_fraction, seed=args.training_seed,
                    hidden_size=args.hidden_size, device=args.device, num_threads=args.num_threads,
                    normalize_vectors=args.normalize_vectors, normalization_min_std=args.normalization_min_std,
                    correction_weight_max=args.correction_weight_max, correction_threshold=args.correction_threshold,
                    field_steering_loss_weight=args.field_steering_loss_weight,
                    dagger_recovery_min_steps=args.dagger_recovery_min_steps,
                    model_config=model_options(args, config, task),
                    metadata={'simulation_config': asdict(configs[task]), 'command': command,
                              'formal_source_sha256':protocol['source_sha256']},
                    progress_every=args.training_progress_every, progress_seconds=args.training_progress_seconds,
                    load_workers=args.training_load_workers,
                    resume_checkpoint=latest if args.formal_resume and latest.exists() else None)
                report['tasks'][task]['training'] = training
                report['tasks'][task]['latest'] = freeze_latest(task, model_dir, out/'frozen-latest')
                atomic_json(report_path, report)
            jobs = []
            for task in tasks:
                manifest=json.loads((out/'datasets'/task/'accepted-manifest.json').read_text(encoding='utf-8'))
                from .datasets import read_episode_metadata
                excluded_groups=list({read_episode_metadata(path).get('scene_group_id') for path in manifest['paths']})
                for mode in protocol['tasks'][task]['modes']:
                    for seed in protocol['tasks'][task]['fresh_seeds']:
                        jobs.append(dict(task=task, config=asdict(configs[task]), actor='policy',
                            seed=seed, mode=mode, checkpoint=report['tasks'][task]['latest'],
                            inference_device=args.inference_device,inference_num_threads=args.inference_num_threads,
                            output_dir=str(out/'closed-loop-tests'/task/mode), name=f'policy-{seed}',
                            excluded_scene_groups=excluded_groups,
                            replay_every=args.formal_test_replay_every, progress_every=args.progress_every))
            status.update(phase='closed_loop_tests', task=None)
            with ProcessPoolExecutor(max_workers=args.formal_test_workers,
                    mp_context=multiprocessing.get_context('spawn')) as executor:
                for result in executor.map(_test_rollout, jobs):
                    report['tasks'][result['task']].setdefault('tests', []).append(result)
                    atomic_json(report_path, report)
                    print('[formal-test] '+json.dumps(result, ensure_ascii=False), flush=True)
            for task in tasks:
                mode = args.formal_coverage_safety_layer if task == 'coverage' else 'none'
                status.update(phase='latest_policy_demo', task=task)
                expert = json.loads(Path(report['tasks'][task]['expert']['path']).read_text(encoding='utf-8'))
                demo = _rollout(dict(task=task, config=asdict(configs[task]), actor='policy',
                    seed=expert['seed'], scene=expert['scene'], mode=mode,
                    checkpoint=report['tasks'][task]['latest'], output_dir=str(out/'replays'),
                    inference_device=args.inference_device,inference_num_threads=args.inference_num_threads,
                    name=f'{task}-latest-policy', replay_every=1,
                    progress_every=args.progress_every), server)
                report['tasks'][task]['demo'] = demo
                atomic_json(report_path, report)
                if args.browser_recording:
                    if server is None:
                        raise ValueError('Browser recording requires --visualize')
                    status.update(phase='recording', task=task)
                    from .browser_recording import record_policy_replay
                    report['tasks'][task]['recording'] = record_policy_replay(server.url,
                        demo['path'], out, fps=args.recording_fps, width=args.recording_width,
                        height=args.recording_height, duration_seconds=args.recording_duration_seconds,
                        headless=args.recording_headless)
                    atomic_json(report_path, report)
            report['complete'] = True
            for task,item in report['tasks'].items():
                item['primary_mode']=protocol['tasks'][task]['primary_mode']
                item['mode_results']={mode:dict(passed=sum(row['success'] for row in item['tests']
                    if row['mode']==mode),total=sum(row['mode']==mode for row in item['tests']),
                    all_passed=all(row['success'] for row in item['tests'] if row['mode']==mode))
                    for mode in protocol['tasks'][task]['modes']}
            report['task_goals_passed'] = all(item['demo']['success'] and all(
                row['success'] for row in item['tests'] if row['mode'] ==
                (args.formal_coverage_safety_layer if task == 'coverage' else 'none'))
                for task, item in report['tasks'].items())
            atomic_json(report_path, report)
            status.update(phase='complete', status='complete' if report['task_goals_passed'] else 'goals_failed')
    except BaseException as error:
        status.update(phase='failed', status='failed', error=dict(type=type(error).__name__, message=str(error)))
        atomic_json(report_path, report)
        raise
    print('[formal-complete] '+json.dumps(dict(task_goals_passed=report['task_goals_passed'],
                                              report=str(report_path)), ensure_ascii=False), flush=True)
    return report
