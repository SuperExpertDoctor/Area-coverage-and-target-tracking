"""Bounded, deterministic process collection with authoritative resumable manifests.

Only a contiguous seed prefix may contribute accepted episodes. Worker speed and
worker count therefore cannot alter the first N physically valid demonstrations.
Canonical episode tensors stay on disk; futures carry small summaries only.
"""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import queue
import time
import traceback

from rollouts import json_value, run_episode, write_json, load_episode, load_replay
from episode_io import archive_fingerprint

SCHEMA = 'maritime-expert-collection-v1'
_EVENT_QUEUE = None
_SOURCE_NAMES = ('collection.py', 'rollouts.py', 'episode_io.py', 'config.py', 'dynamics.py',
    'geometry.py', 'navigation.py', 'scenario.py', 'observations.py',
    'env_base.py', 'env_coverage.py', 'env_tracking.py', 'expert_coverage.py',
    'expert_tracking.py', 'expert_field.py', 'potential_field.py')


def _source_fingerprint():
    digest = hashlib.sha256()
    for name in _SOURCE_NAMES:
        digest.update(name.encode())
        digest.update((Path(__file__).parent / name).read_bytes())
    return digest.hexdigest()


def _artifact(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}


def _check_artifacts(result):
    if not result.get('artifacts'):
        raise ValueError(f"Collection artifact evidence is missing for seed {result['seed']}")
    for expected in result['artifacts']:
        path = Path(expected['path'])
        if not path.is_file() or _artifact(path) != expected:
            raise ValueError(f'Collection artifact is missing or changed: {path}')


def _peak_rss_bytes():
    try:
        if os.name == 'nt':
            import ctypes
            from ctypes import wintypes
            class Counters(ctypes.Structure):
                _fields_ = [('cb', wintypes.DWORD), ('faults', wintypes.DWORD)] + [
                    (name, ctypes.c_size_t) for name in ('peak', 'working', 'pool_peak',
                    'pool', 'nonpaged_peak', 'nonpaged', 'pagefile', 'pagefile_peak')]
            counters = Counters()
            counters.cb = ctypes.sizeof(counters)
            process = ctypes.windll.kernel32.GetCurrentProcess
            process.restype = wintypes.HANDLE
            memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
            memory_info.argtypes = (wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD)
            if memory_info(process(), ctypes.byref(counters), counters.cb):
                return int(counters.peak)
        else:
            import resource
            import sys
            value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            return int(value if sys.platform == 'darwin' else value * 1024)
    except (ImportError, OSError, AttributeError):
        pass
    return None


def _init_worker(events):
    global _EVENT_QUEUE
    _EVENT_QUEUE = events
    # Display events are disposable; never wait on their feeder at process exit
    # if the parent was interrupted while the bounded queue was full.
    events.cancel_join_thread()
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[key] = '1'


class _QueuePublisher:
    """Lossy display transport: queue saturation must never slow physics."""
    def __init__(self, attempt, seed):
        self.attempt, self.seed = attempt, seed
        self.dropped_events = 0

    def publish(self, payload):
        if _EVENT_QUEUE is None:
            return
        event = {'attempt': self.attempt, 'seed': self.seed, 'payload': payload}
        try:
            _EVENT_QUEUE.put_nowait(event)
        except (queue.Full, ValueError, OSError):
            self.dropped_events += 1


def _task_rejection(task, summary, config):
    expected = float(config[f'{task}_episode_seconds'])
    if abs(float(summary.get('time_seconds', -1.)) - expected) > 1e-6:
        return 'episode_duration_incomplete'
    if task == 'coverage' and summary.get('coverage_ratio', 0.) < 1. - 1e-9:
        return 'coverage_incomplete'
    if task == 'tracking' and (summary.get('visible_fraction', 0.) < 1. - 1e-9
                              or summary.get('first_loss_time') is not None):
        return 'tracking_incomplete'
    return None


def _failure(job, error, status, record=None):
    result = {'seed': job['seed'], 'attempt': job['attempt'], 'status': status,
              'json': None, 'npz': None, 'summary': {},
              'error': {'type': type(error).__name__, 'message': str(error),
                        'traceback': ''.join(traceback.format_exception(
                            type(error), error, error.__traceback__))[-12000:]}}
    if record is not None:
        result.update(npz=str((Path(job['output_dir']) / f"episode-{job['seed']}.npz").resolve()),
                      summary=record.get('summary', {}))
    failure_path = Path(job['output_dir']) / 'failures' / f"episode-{job['seed']}.json"
    write_json(failure_path, {'task': job['task'], 'seed': job['seed'],
               'identity_hash': job['identity_hash'], 'status': status,
               'error': result['error'], 'summary': result['summary']})
    result['failure_json'] = str(failure_path.resolve())
    result['artifacts'] = [_artifact(failure_path)]
    if result['npz']:
        result['artifacts'].append(_artifact(result['npz']))
    return result


def _collect_one(job):
    """Import simulation only in workers; never import torch or own a server."""
    env = None
    phase = 'initialization_failure'
    publisher = _QueuePublisher(job['attempt'], job['seed'])
    start = time.perf_counter()
    path = Path(job['output_dir']) / f"episode-{job['seed']}.npz"
    previous_archive = archive_fingerprint(path)
    try:
        if job.get('source_sha256') and _source_fingerprint() != job['source_sha256']:
            raise RuntimeError('Collection source identity changed after manifest creation')
        from config import Config
        config = Config.from_dict(job['config'])
        if job['task'] == 'coverage':
            from env_coverage import CoverageEnv as Environment
            from expert_coverage import CoverageExpert as Expert
            if config.potential_enabled:
                from expert_field import FieldCoverageExpert as Expert
        else:
            from env_tracking import TrackingEnv as Environment
            from expert_tracking import TrackingExpert as Expert
            if config.potential_enabled:
                from expert_field import FieldTrackingExpert as Expert
        env = Environment(config=config, render_mode='json')
        actor = Expert(config)
        record, path = run_episode(env, actor, seed=job['seed'], actor_name='expert',
            output_dir=job['output_dir'], name=f"episode-{job['seed']}",
            record_observations=True, replay_every=job['replay_every'],
            render_every=job['replay_every'], progress_every=job['progress_every'],
            server=publisher if job['stream_events'] else None)
        phase = 'worker_error'
        summary = record['summary']
        valid = bool(record['metadata']['expert_valid'] and summary['steps'] > 0)
        status = ('valid' if valid else 'planning_failure'
                  if summary.get('planning_failure') else 'rejected')
        rejection = (_task_rejection(job['task'], summary, job['config'])
                     if valid and job.get('require_task_success') else None)
        if rejection:
            status = 'task_incomplete'
        cache = path.with_suffix('.npz')
        result = {'seed': job['seed'], 'attempt': job['attempt'], 'status': status,
                  'json': None, 'npz': str(cache.resolve()),
                  'summary': summary, 'scene_id': record['metadata']['scene_id'],
                  'scene_group_id': record['metadata']['scene_group_id'],
                  'artifacts': [_artifact(path)]}
        if rejection:
            result['rejection_reason'] = rejection
        if summary.get('planning_failure'):
            result['error'] = summary['planning_failure']
    except Exception as error:
        record = None
        if env is not None and env.scene is not None:
            phase = 'worker_error'
            if path.is_file() and archive_fingerprint(path)!=previous_archive:
                record = load_replay(path)
        result = _failure(job, error, phase, record)
    finally:
        if env is not None:
            env.close()
    result.update(worker_pid=os.getpid(), worker_wall_seconds=time.perf_counter() - start,
                  worker_peak_rss_bytes=_peak_rss_bytes(), dropped_events=publisher.dropped_events)
    write_json(job['result_path'], {'identity_hash': job['identity_hash'], 'result': result})
    return result


def collect_expert_episodes(task, config, output_dir, *, episodes, seed, workers,
                            attempt_multiplier, replay_every, progress_every,
                            server=None, resume=True, require_task_success=False):
    """Collect exactly ``episodes`` successes from the increasing seed prefix.

    ``accepted-manifest.json`` is authoritative even during an interruption.
    Identical source/configuration/seed/budget/cadence may resume; worker count
    and progress cadence can change without changing the selected data. Invalid
    attempts stay explicit. Insufficient valid episodes raises RuntimeError and
    leaves complete=false. Existing unrelated artifacts are never enumerated as
    training data. Use a new output directory for a new collection identity.
    """
    for key, value in {'episodes': episodes, 'seed': seed, 'workers': workers,
                       'attempt_multiplier': attempt_multiplier,
                       'replay_every': replay_every, 'progress_every': progress_every}.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < (0 if key == 'seed' else 1):
            raise ValueError(f'{key} must be a positive integer' if key != 'seed' else 'seed must be nonnegative')
    if task not in ('coverage', 'tracking'):
        raise ValueError('task must be coverage or tracking')
    config_dict = json_value(config)
    if not isinstance(config_dict, dict):
        raise ValueError('config must be a Config or configuration dictionary')
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / 'accepted-manifest.json'
    identity = {'schema_version': SCHEMA, 'task': task, 'config': config_dict,
                'source_sha256': _source_fingerprint(), 'seed': seed, 'episodes': episodes,
                'attempt_multiplier': attempt_multiplier, 'replay_every': replay_every,
                'require_task_success': bool(require_task_success)}
    identity_hash = hashlib.sha256(json.dumps(identity, sort_keys=True,
                                    separators=(',', ':')).encode()).hexdigest()
    previous = None
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding='utf-8'))
        if not resume:
            raise ValueError(f'Collection already exists; resume or choose a new output directory: {manifest_path}')
        if previous.get('identity_hash') != identity_hash:
            raise ValueError(f'Collection identity mismatch; use a new output directory: {manifest_path}')
    elif any(out.glob('episode-*.npz')) or any(out.glob('episode-*.json')) or (out / '.collection-attempts').exists():
        raise ValueError('Collection identity manifest is missing; use a new output directory')
    max_attempts = episodes * attempt_multiplier
    results = {}
    for result in (previous or {}).get('attempts', []):
        _check_artifacts(result)
        results[result['attempt']] = result
    for path in (out / '.collection-attempts').glob('episode-*.json'):
        recovered = json.loads(path.read_text(encoding='utf-8'))
        if recovered.get('identity_hash') != identity_hash:
            raise ValueError(f'Collection identity mismatch in attempt sidecar: {path}')
        result = recovered['result']
        if result['attempt'] not in results:
            _check_artifacts(result)
            results[result['attempt']] = result
    start = time.perf_counter()
    prior_seconds = (previous or {}).get('metrics', {}).get('wall_seconds', 0.)
    metrics = {'max_in_flight': 0, 'max_buffered_results': 0, 'workers': workers,
               'event_queue_capacity': max(4, workers * 4)}
    manifest = {}
    contiguous = 0
    active_visual = None
    scene_events = {}

    def save(status='collecting'):
        nonlocal manifest, contiguous
        accepted = []
        contiguous = 0
        while contiguous in results and len(accepted) < episodes:
            result = results[contiguous]
            if result['status'] == 'valid':
                accepted.append(result)
            contiguous += 1
        rejected = [r for _, r in sorted(results.items()) if r['status'] != 'valid']
        selected = {r['attempt'] for r in accepted}
        unused = [r for _, r in sorted(results.items())
                  if r['status'] == 'valid' and r['attempt'] not in selected]
        complete = len(accepted) == episodes
        metrics.update(wall_seconds=prior_seconds + time.perf_counter() - start,
                       accepted_count=len(accepted), rejected_count=len(rejected),
                       attempted_count=len(results), parent_peak_rss_bytes=_peak_rss_bytes())
        metrics['accepted_per_second'] = len(accepted) / max(metrics['wall_seconds'], 1e-9)
        manifest = {'schema_version': SCHEMA, 'task': task, 'complete': complete,
                    'status': 'complete' if complete else status,
                    'manifest_path': str(manifest_path), 'identity': identity,
                    'identity_hash': identity_hash, 'config': config_dict,
                    'collection_seed': seed, 'target_episodes': episodes,
                    'max_attempts': max_attempts, 'next_ordered_attempt': contiguous,
                    'accepted': accepted, 'rejected': rejected, 'unused': unused,
                    'attempts': [r for _, r in sorted(results.items())],
                    'paths': [r['npz'] for r in accepted], 'metrics': dict(metrics),
                    'updated_at': datetime.now(timezone.utc).isoformat()}
        write_json(manifest_path, manifest)
        if server:
            server.publish({'type': 'collection_progress', 'task': task,
                'accepted': len(accepted), 'target': episodes, 'rejected': len(rejected),
                'attempted': len(results), 'complete': complete,
                'manifest_path': str(manifest_path), 'status': manifest['status']})

    save()
    if manifest['complete']:
        return manifest
    context = multiprocessing.get_context('spawn')
    events = context.Queue(maxsize=metrics['event_queue_capacity'])
    pending = {}

    def drain_events():
        nonlocal active_visual
        # Only one episode owns the displayed scene. Other workers still report
        # progress, while their scene dictionaries are bounded by the window.
        display = min((job['attempt'] for job in pending.values()), default=None)
        if display != active_visual:
            active_visual = display
            if server and display in scene_events:
                server.publish(scene_events[display])
        for _ in range(metrics['event_queue_capacity'] * 2):
            try:
                event = events.get_nowait()
            except queue.Empty:
                break
            attempt, payload = event['attempt'], event['payload']
            payload = dict(payload, collection_seed=event['seed'], collection_attempt=attempt)
            if payload['type'] == 'scene':
                scene_events[attempt] = payload
            if server and (attempt == active_visual or payload['type'] == 'progress'):
                if payload['type'] != 'frame' or attempt in scene_events:
                    server.publish(payload)
        for attempt in list(scene_events):
            if attempt < contiguous:
                del scene_events[attempt]

    try:
        with ProcessPoolExecutor(max_workers=workers, mp_context=context,
                initializer=_init_worker, initargs=(events,)) as executor:
            while True:
                pending_attempts = {job['attempt'] for job in pending.values()}
                if not manifest['complete']:
                    for attempt in range(contiguous, min(contiguous + workers, max_attempts)):
                        if attempt in results or attempt in pending_attempts:
                            continue
                        job = {'task': task, 'config': config_dict, 'seed': seed + attempt,
                               'attempt': attempt, 'output_dir': str(out),
                               'identity_hash': identity_hash, 'replay_every': replay_every,
                               'source_sha256': identity['source_sha256'],
                               'require_task_success': bool(require_task_success),
                               'progress_every': progress_every, 'stream_events': server is not None,
                               'result_path': str(out / '.collection-attempts' / f'episode-{seed + attempt}.json')}
                        pending[executor.submit(_collect_one, job)] = job
                metrics['max_in_flight'] = max(metrics['max_in_flight'], len(pending))
                drain_events()
                if not pending:
                    break
                finished, _ = wait(pending, timeout=.1, return_when=FIRST_COMPLETED)
                for future in finished:
                    job = pending.pop(future)
                    try:
                        result = future.result()
                    except Exception as error:
                        result = _failure(job, error, 'worker_error')
                    results[job['attempt']] = result
                    write_json(job['result_path'], {'identity_hash': identity_hash, 'result': result})
                    metrics['max_buffered_results'] = max(metrics['max_buffered_results'],
                        sum(index >= contiguous for index in results))
                    save()
                drain_events()
            drain_events()
    except BaseException:
        # Each worker also writes its own result sidecar, so even completion
        # during parent shutdown can be recovered by the next invocation.
        raise
    finally:
        events.close()
        events.join_thread()
    save('insufficient_valid_episodes')
    if not manifest['complete']:
        raise RuntimeError(f"{task}: only {len(manifest['accepted'])}/{episodes} valid expert episodes; inspect {manifest_path}")
    return manifest
