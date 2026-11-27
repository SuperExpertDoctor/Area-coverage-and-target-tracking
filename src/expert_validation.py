"""Reproducible closed-loop expert stress tests, independent of training.

Run from the project root with ``python src/expert_validation.py``. The case
manifest is written before any rollout; failed algorithms are never filtered.
Successes retain summaries and sampled trajectories, failures full replays.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback

from dynamics import FixedWingState


DEFAULT_COVERAGE_SEEDS = (2000, *range(3000, 3029))
DEFAULT_BOUNDARY_SCENE_SEEDS = (4000, 4001, 4002)
DEFAULT_TRACKING_SEEDS = tuple(range(5000, 5010))
_PROCESS_SOURCE_ID = None


def _worker_init():
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[name] = '1'


def _write_json(path, value, *, compact=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False,
                                    indent=None if compact else 2,
                                    separators=(',', ':') if compact else None), encoding='utf-8')
    temporary.replace(path)


def _source_fingerprint():
    root = Path(__file__).parent
    names = ('config.py', 'dynamics.py', 'geometry.py', 'navigation.py', 'scenario.py',
             'observations.py', 'env_base.py', 'env_coverage.py', 'env_tracking.py',
             'expert_coverage.py', 'expert_tracking.py', 'expert_validation.py',
             'potential_field.py', 'expert_field.py')
    hashes = {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names}
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return digest, hashes


def _physical_config_hash(config_dict):
    physical = {name:value for name,value in config_dict.items() if not name.startswith('visual_')}
    return hashlib.sha256(json.dumps(physical, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _validation_id(source_id, config_dict):
    return hashlib.sha256((source_id+_physical_config_hash(config_dict)).encode()).hexdigest()


def make_case_plan(coverage_seeds, boundary_scene_seeds, boundary_starts, tracking_seeds):
    if boundary_starts < 1:
        raise ValueError('boundary_starts must be positive')
    jobs = []
    for seed in coverage_seeds:
        jobs.append({'case_id': f'coverage-random-{seed}', 'task': 'coverage',
                     'kind': 'random', 'seed': int(seed)})
    for seed in boundary_scene_seeds:
        for index in range(boundary_starts):
            jobs.append({'case_id': f'coverage-boundary-{seed}-{index:02d}',
                         'task': 'coverage', 'kind': 'boundary', 'seed': int(seed),
                         'start_index': index, 'start_count': boundary_starts})
    for seed in tracking_seeds:
        jobs.append({'case_id': f'tracking-random-{seed}', 'task': 'tracking',
                     'kind': 'random', 'seed': int(seed)})
    if len({job['case_id'] for job in jobs}) != len(jobs):
        raise ValueError('Duplicate seeds create duplicate validation cases')
    return jobs


def boundary_start(roi, index, count):
    if count < 1 or not 0 <= index < count:
        raise ValueError('Invalid boundary point index')
    x0, y0, x1, y1 = roi
    width, height = x1-x0, y1-y0
    arc = 2*(width+height)*index/count
    if arc < height:
        return FixedWingState(x0, y0+arc, math.pi/2)
    if arc < height+width:
        return FixedWingState(x0+arc-height, y1, 0.)
    if arc < 2*height+width:
        return FixedWingState(x1, y1-(arc-height-width), -math.pi/2)
    return FixedWingState(x1-(arc-2*height-width), y0, math.pi)


def _initial_boundary_rejection(scene, config):
    """Only reject explicit hard initial-state violations, without expert trials."""
    uav = scene['uav_initial']
    bounds = config.world_bounds
    if not bounds[0] <= uav['x'] <= bounds[2] or not bounds[1] <= uav['y'] <= bounds[3]:
        return {'reason': 'initial_uav_out_of_bounds'}
    if abs(uav['bank']) > config.uav_config.bank_limit:
        return {'reason': 'initial_uav_bank_limit'}
    for cloud in scene['clouds']:
        clearance = math.hypot(uav['x']-cloud['x'], uav['y']-cloud['y'])-cloud['radius']
        if clearance <= config.uav_cloud_margin:
            return {'reason': 'initial_uav_cloud_collision', 'cloud_id': cloud['id'],
                    'surface_clearance_m': clearance, 'required_clearance_m': config.uav_cloud_margin}
    return None


def execute_case(job, config_dict, output_dir, trajectory_every=60., coverage_success_ratio=1.):
    """One serial physical rollout. Suitable for a ProcessPoolExecutor worker."""
    _worker_init()
    import numpy as np
    from config import Config
    from env_coverage import CoverageEnv
    from env_tracking import TrackingEnv
    from expert_coverage import CoverageExpert
    from expert_tracking import TrackingExpert

    config = Config.from_dict(config_dict)
    out = Path(output_dir)
    source_id, source_hashes = _source_fingerprint()
    global _PROCESS_SOURCE_ID
    if _PROCESS_SOURCE_ID is None:
        _PROCESS_SOURCE_ID = source_id
    elif _PROCESS_SOURCE_ID != source_id:
        raise RuntimeError('Physical/validation source changed inside a worker; restart the campaign with the new source version')
    if job.get('expected_source_id', source_id) != source_id:
        raise RuntimeError('Campaign source version changed after preregistration; restart with the new source version')
    config_hash = _physical_config_hash(config_dict)
    validation_id = _validation_id(source_id, config_dict)
    case_id = job['case_id']
    stem = f'{case_id}-{validation_id[:10]}'
    result_path = out/'results'/f'{stem}.json'
    env = (CoverageEnv if job['task'] == 'coverage' else TrackingEnv)(config)
    actor = (CoverageExpert if job['task'] == 'coverage' else TrackingExpert)(config)
    if config.potential_enabled:
        from expert_field import FieldCoverageExpert, FieldTrackingExpert
        actor=(FieldCoverageExpert if job['task']=='coverage' else FieldTrackingExpert)(config)
    scene = deepcopy(job.get('scene'))
    scene_file = out/'scenes'/f'{case_id}.json'
    if scene is None and scene_file.exists():
        scene = json.loads(scene_file.read_text(encoding='utf-8'))
    if scene is None and job.get('scene_file'):
        scene = json.loads(Path(job['scene_file']).read_text(encoding='utf-8'))
    if job['kind'] == 'boundary':
        if scene is None:
            raise ValueError('Boundary case requires a frozen ROI scene')
        scene['uav_initial'] = asdict(boundary_start(scene['roi'], job['start_index'], job['start_count']))
        scene['radar_pan_initial'] = 0.
        rejection = _initial_boundary_rejection(scene, config)
        if rejection:
            excluded = out/'excluded'/f'{stem}.json'
            _write_json(excluded, {'job': job, 'scene': scene, 'exclusion': rejection}, compact=True)
            result = {'case_id': case_id, 'task': job['task'], 'kind': job['kind'],
                      'seed': job['seed'], 'status': 'excluded_initial_state', 'passed': False,
                      'exclusion': rejection, 'source_id': source_id,
                      'physical_config_hash': config_hash, 'validation_id': validation_id,
                      'excluded_scene': str(excluded.relative_to(out))}
            _write_json(result_path, result)
            return result
    frames, actions, executed = [], [], []
    failure = None
    start = time.perf_counter()
    try:
        obs, _ = env.reset(seed=job['seed'], options={'scene': scene} if scene else None)
        scene = env.scene
        # Preserve the exact geometry before the first policy action.
        _write_json(scene_file, scene, compact=True)
        actor.reset()
        frames.append(env.render())
        terminated = truncated = False
        while not (terminated or truncated):
            try:
                action = np.asarray(actor.act(obs), dtype=np.float32)
            except Exception as error:
                failure = {'type': type(error).__name__, 'message': str(error),
                           'step': len(actions), 't': env.t,
                           'traceback': traceback.format_exc()}
                break
            obs, _, terminated, truncated, _ = env.step(action)
            actions.append(action.tolist())
            frame = env.render()
            frames.append(frame)
            speed = math.hypot(frame['uav']['vx'], frame['uav']['vy'])
            actual=[frame['uav']['vx']/speed, frame['uav']['vy']/speed]
            if config.potential_enabled:
                limit=config.coverage_pan_limit if job['task']=='coverage' else config.pan_limit
                actual.append(frame['radar']['pan']/limit)
            executed.append(actual)
        summary = env.summary()
        summary.update(steps=len(actions), terminated=bool(terminated or failure),
                       truncated=bool(truncated), wall_seconds=time.perf_counter()-start)
        if failure:
            status = 'expert_planning_failure' if failure['type'] == 'ExpertPlanningError' else 'runtime_failure'
            summary['termination_reason'] = status
            summary['planning_failure'] = failure
        elif terminated:
            status = 'physical_failure'
        elif job['task'] == 'coverage' and summary['coverage_ratio'] < coverage_success_ratio:
            status = 'coverage_incomplete'
        elif job['task'] == 'tracking' and summary['visible_fraction'] < 1.:
            status = 'visibility_failure'
        elif summary['termination_reason'] != 'time_limit':
            status = 'runtime_failure'
        else:
            status = 'passed'
    except Exception as error:
        status = 'initialization_failure' if not frames else 'runtime_failure'
        failure = {'type': type(error).__name__, 'message': str(error),
                   'step': len(actions), 'traceback': traceback.format_exc()}
        summary = {'time_seconds': frames[-1]['t'] if frames else 0., 'steps': len(actions),
                   'termination_reason': status, 'terminated': True, 'truncated': False,
                   'planning_failure': failure, 'wall_seconds': time.perf_counter()-start}
    passed = status == 'passed'
    action_sha256 = hashlib.sha256(np.asarray(actions, dtype='<f4').tobytes()).hexdigest()
    metadata = {'task': job['task'], 'seed': job['seed'], 'actor': 'expert',
                'case_id': case_id, 'kind': job['kind'], 'expert_valid': passed,
                'source_id': source_id, 'source_hashes': source_hashes,
                'physical_config_hash': config_hash, 'validation_id': validation_id,
                'config': config.to_dict(), 'obs_schema': 'obs-field-v2' if config.potential_enabled else 'obs-v1',
                'action_sha256': action_sha256, 'action_hash_encoding': 'row-major little-endian float32',
                'start_index': job.get('start_index'), 'start_count': job.get('start_count')}
    result = {'case_id': case_id, 'task': job['task'], 'kind': job['kind'],
              'seed': job['seed'], 'status': status, 'passed': passed,
              'action_sha256': action_sha256,
              'source_id': source_id, 'summary': summary,
              'physical_config_hash': config_hash, 'validation_id': validation_id,
              'scene_file': str(scene_file.relative_to(out)) if scene_file.exists() else None,
              'initial_uav': scene.get('uav_initial') if scene else None}
    if passed:
        # A representative trace is explicitly downsampled, never a full replay.
        sampled = [frames[0]]
        next_time = trajectory_every
        for frame in frames[1:-1]:
            if frame['t'] >= next_time-1e-8:
                sampled.append(frame)
                next_time += trajectory_every
        if len(frames) > 1:
            sampled.append(frames[-1])
        path = out/'representative'/f'{stem}.json'
        _write_json(path, {'schema_version': 'expert-validation-representative-v1',
                          'metadata': metadata, 'scene_file': result['scene_file'],
                          'trajectory_every_seconds': trajectory_every,
                          'frames': sampled, 'summary': summary}, compact=True)
        result['representative'] = str(path.relative_to(out))
    else:
        path = out/'failures'/f'{stem}.json'
        _write_json(path, {'schema_version': 'maritime-replay-v1', 'task': job['task'],
                          'actor': 'expert', 'seed': job['seed'], 'scene': scene,
                          'metadata': metadata, 'frames': frames, 'actions': actions,
                          'executed_actions': executed, 'summary': summary}, compact=True)
        result['replay'] = str(path.relative_to(out))
    _write_json(result_path, result)
    env.close()
    return result


def run_validation(config=None, *, output_dir='outputs/diagnostics/expert-randomized',
                   coverage_seeds=DEFAULT_COVERAGE_SEEDS,
                   boundary_scene_seeds=DEFAULT_BOUNDARY_SCENE_SEEDS,
                   boundary_starts=8, tracking_seeds=DEFAULT_TRACKING_SEEDS,
                   workers=4, coverage_seconds=10800., tracking_seconds=1800.,
                   trajectory_every=60., coverage_success_ratio=1., resume=True):
    from config import Config
    from env_coverage import CoverageEnv

    if workers < 1 or trajectory_every <= 0 or not 0 < coverage_success_ratio <= 1:
        raise ValueError('Invalid validation worker count, sampling interval or coverage criterion')
    config = config if config is not None else Config()
    if isinstance(config, dict):
        config = Config.from_dict(config)
    config = replace(config, coverage_episode_seconds=coverage_seconds,
                     tracking_episode_seconds=tracking_seconds)
    out = Path(output_dir).resolve()
    jobs = make_case_plan(coverage_seeds, boundary_scene_seeds, boundary_starts, tracking_seeds)
    source_id, source_hashes = _source_fingerprint()
    config_hash = _physical_config_hash(config.to_dict())
    validation_id = _validation_id(source_id, config.to_dict())
    manifest_path = out/'manifest.json'
    manifest = {'schema_version': 'expert-validation-manifest-v1',
                'created_utc': datetime.now(timezone.utc).isoformat(),
                'cases': jobs, 'config': config.to_dict(),
                'criteria': {'coverage_ratio': coverage_success_ratio,
                             'tracking_visible_fraction': 1., 'full_duration': True,
                             'no_collision_or_world_boundary_failure': True},
                'boundary_exclusion_rule': 'Only hard initial cloud collision, world bounds or bank limits; never expert rollout outcomes.',
                'workers': workers, 'threads_per_worker': 1,
                'trajectory_every_seconds': trajectory_every,
                'source_id': source_id, 'source_hashes': source_hashes}
    manifest.update(physical_config_hash=config_hash, validation_id=validation_id)
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding='utf-8'))
        if previous['cases'] != jobs or _physical_config_hash(previous['config']) != config_hash or previous['criteria'] != manifest['criteria']:
            raise ValueError('Existing frozen manifest has different cases/configuration/criteria; use a different output directory')
    else:
        # Case selection is permanently recorded before scene preparation or results.
        _write_json(manifest_path, manifest)
    _write_json(out/'runs'/f'{validation_id}.json', manifest)
    boundary_files = {}
    for seed in boundary_scene_seeds:
        path = out/'scenes'/f'coverage-boundary-roi-{seed}.json'
        if not path.exists():
            env = CoverageEnv(config)
            env.reset(seed=int(seed))
            _write_json(path, env.scene, compact=True)
            env.close()
        boundary_files[int(seed)] = str(path)
    pending, results = [], []
    for job in jobs:
        job = dict(job)
        job['expected_source_id'] = source_id
        if job['kind'] == 'boundary':
            job['scene_file'] = boundary_files[job['seed']]
        result_path = out/'results'/f"{job['case_id']}-{validation_id[:10]}.json"
        if resume and result_path.exists():
            results.append(json.loads(result_path.read_text(encoding='utf-8')))
        else:
            pending.append(job)

    def save_summary():
        ordered = sorted(results, key=lambda r:r['case_id'])
        eligible = [r for r in ordered if r['status'] != 'excluded_initial_state']
        excluded = [r for r in ordered if r['status'] == 'excluded_initial_state']
        failures = [r for r in eligible if not r['passed']]
        report = {'schema_version': 'expert-validation-summary-v1',
                  'source_id': source_id, 'physical_config_hash': config_hash,
                  'validation_id': validation_id, 'planned_case_count': len(jobs),
                  'completed_case_count': len(results), 'eligible_case_count': len(eligible),
                  'excluded_initial_state_count': len(excluded), 'failure_count': len(failures),
                  'passed_case_count': sum(r['passed'] for r in eligible),
                  'complete': len(results) == len(jobs),
                  'all_eligible_passed': len(results) == len(jobs) and not failures,
                  'results': ordered,
                  'scope': 'Finite preregistered simulation tests, not a proof for every mathematical state or ROI.'}
        _write_json(out/'suite-summary.json', report)
        return report

    save_summary()
    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init) as pool:
        futures = {pool.submit(execute_case, job, config.to_dict(), str(out), trajectory_every,
                               coverage_success_ratio): job for job in pending}
        for future in as_completed(futures):
            job = futures[future]
            try:
                result = future.result()
            except Exception as error:
                # A crashed worker/setup is retained as a failed case, never dropped.
                result = {'case_id': job['case_id'], 'task': job['task'], 'kind': job['kind'],
                          'seed': job['seed'], 'status': 'worker_failure', 'passed': False,
                          'source_id': source_id,
                          'physical_config_hash': config_hash, 'validation_id': validation_id,
                          'error': {'type': type(error).__name__, 'message': str(error),
                                    'traceback': traceback.format_exc()}}
                _write_json(out/'results'/f"{job['case_id']}-{validation_id[:10]}.json", result)
            results.append(result)
            report = save_summary()
            print(json.dumps({'case_id': result['case_id'], 'status': result['status'],
                              'completed': len(results), 'planned': len(jobs),
                              'failures': report['failure_count'],
                              'coverage_ratio': result.get('summary', {}).get('coverage_ratio'),
                              'time_seconds': result.get('summary', {}).get('time_seconds')},
                             ensure_ascii=False), flush=True)
    return save_summary()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', default='outputs/diagnostics/expert-randomized')
    parser.add_argument('--config-json', default='{}')
    parser.add_argument('--coverage-seeds', type=int, nargs='*', default=list(DEFAULT_COVERAGE_SEEDS))
    parser.add_argument('--boundary-scene-seeds', type=int, nargs='*', default=list(DEFAULT_BOUNDARY_SCENE_SEEDS))
    parser.add_argument('--boundary-starts', type=int, default=8)
    parser.add_argument('--tracking-seeds', type=int, nargs='*', default=list(DEFAULT_TRACKING_SEEDS))
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--coverage-seconds', type=float, default=10800.)
    parser.add_argument('--tracking-seconds', type=float, default=1800.)
    parser.add_argument('--trajectory-every', type=float, default=60.)
    parser.add_argument('--coverage-success-ratio', type=float, default=1.)
    parser.add_argument('--resume', action=argparse.BooleanOptionalAction, default=True)
    args = vars(parser.parse_args(argv))
    args['config'] = json.loads(args.pop('config_json'))
    report = run_validation(**args)
    return 0 if report['all_eligible_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
