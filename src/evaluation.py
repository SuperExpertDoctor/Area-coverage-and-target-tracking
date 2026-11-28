"""Evaluate one frozen policy on independent, auditable simulator episodes.

The simulator and success criterion remain shared with formal runs. This module
only schedules episodes, retains failures and writes incremental aggregate data.
"""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from dataclasses import asdict, is_dataclass
import hashlib
import math
import multiprocessing
from pathlib import Path

from .episode_io import archive_fingerprint


def _default_runner(job, server=None):
    # Lazy import keeps formal_run free to import evaluate_policy itself.
    from .formal_run import _test_rollout
    return _test_rollout(job) if server is None else _test_rollout(job, server=server)


def _provenance(job):
    return dict(checkpoint=deepcopy(job['checkpoint']),
                model_sha256=job['checkpoint']['sha256'],
                model_checkpoint=job['checkpoint']['source'],
                model_epoch=job['checkpoint']['epoch'])


def _failure_result(job, error):
    from .formal_run import atomic_json
    path = Path(job['output_dir']) / (job['name'] + '.failure.json')
    result = dict(task=job['task'], seed=job['seed'], mode=job['mode'],
                  path=str(path.resolve()), success=False, runtime_failure=True,
                  summary=dict(terminated=True, termination_reason='test_runtime_failure',
                               error=dict(type=type(error).__name__, message=str(error))),
                  **_provenance(job))
    artifact = Path(job['output_dir']) / (job['name'] + '.npz')
    if artifact.is_file() and archive_fingerprint(artifact)!=job['previous_episode_fingerprint']:
        result['episode_path'] = str(artifact.resolve())
    atomic_json(path, result)
    return result


def _evaluate_job(job, runner=None, server=None):
    """A picklable worker; every ordinary exception becomes a retained case."""
    from .formal_run import atomic_json, task_success
    from .rollouts import json_value
    try:
        result = (_default_runner(job, server) if runner is None else runner(job))
        if not isinstance(result, dict) or not isinstance(result.get('summary'), dict):
            raise ValueError('Evaluation runner returned no episode summary')
        for key in ('task', 'seed', 'mode'):
            if result.get(key) != job[key]:
                raise ValueError(f'Evaluation runner returned a different {key}')
        if result.get('model_sha256', job['checkpoint']['sha256']) != job['checkpoint']['sha256']:
            raise ValueError('Evaluation runner returned a different model SHA256')
        if result.get('scene_group_id') in job['excluded_scene_groups']:
            raise ValueError('Evaluation scene overlaps collected train/validation scenes')
        summary = json_value(deepcopy(result['summary']))
        runtime_failure = summary.get('termination_reason') in (
            'test_runtime_failure', 'runtime_failure')
        path = Path(result['path']).resolve()
        expected = (Path(job['output_dir']) / (job['name'] + '.npz')).resolve()
        fresh=expected.is_file() and archive_fingerprint(expected)!=job['previous_episode_fingerprint']
        if not runtime_failure and (path != expected or not fresh):
            raise ValueError('Evaluation episode must archive interactions at ' + str(expected))
        result = dict(json_value(deepcopy(result)), path=str(path), summary=summary,
                      success=not runtime_failure and task_success(
                          job['task'], summary,
                          job['config'][f'{job["task"]}_episode_seconds']),
                      runtime_failure=runtime_failure, **_provenance(job))
        if runtime_failure:
            # The shared runner can already have written a failure sidecar. Keep
            # the same sidecar complete with the frozen model identity as well.
            failure_path = Path(job['output_dir']) / (job['name'] + '.failure.json')
            result['path'] = str(failure_path.resolve())
            result.pop('episode_path',None)
            if fresh:
                result['episode_path'] = str(expected)
            atomic_json(failure_path, result)
        return result
    except Exception as error:
        return _failure_result(job, error)


def _metrics(rows):
    values = {}
    for row in rows:
        for key, value in row['summary'].items():
            if (isinstance(value, (int, float)) and not isinstance(value, bool)
                    and math.isfinite(value)):
                values.setdefault(key, []).append(float(value))
    return {key: dict(count=len(items), mean=sum(items) / len(items),
                      min=min(items), max=max(items))
            for key, items in sorted(values.items())}


def evaluate_policy(task, config, checkpoint, output_dir, *, episodes=30,
                    seed=940000, workers=1, mode='none', excluded_scene_groups=(),
                    replay_every=60, progress_every=300, inference_device='cpu',
                    inference_num_threads=1, runner=None, server=None):
    """Run a fixed set of fresh inference episodes and return their report.

    ``output_dir`` is the exact ``eval/<task>/<mode>`` directory. ``checkpoint``
    is the frozen descriptor returned by ``freeze_latest``. An optional
    ``runner(job)`` may replace the shared rollout for lightweight tests; runners
    used with process workers must be picklable. No outcome selects a checkpoint.

    Rates always divide by the requested episode count. Interrupted runs retain
    ``complete=False`` and their pending count; ordinary runtime errors remain
    failed rows and do not stop the remaining independent cases.
    """
    from .formal_run import atomic_json
    from .rollouts import json_value
    for name, value, minimum in (
            ('episodes', episodes, 1), ('seed', seed, 0), ('workers', workers, 1),
            ('replay_every', replay_every, 1), ('progress_every', progress_every, 1),
            ('inference_num_threads', inference_num_threads, 1)):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f'Invalid evaluation {name}')
    if task not in ('coverage', 'tracking') or mode not in ('none', 'forecast'):
        raise ValueError('Invalid evaluation task or mode')
    config = (asdict(config) if is_dataclass(config) else
              config.to_dict() if hasattr(config, 'to_dict') else dict(config))
    config = json_value(config)
    duration = config.get(f'{task}_episode_seconds')
    if (isinstance(duration, bool) or not isinstance(duration, (int, float))
            or not math.isfinite(duration) or duration <= 0):
        raise ValueError('Evaluation duration must be finite and positive')
    checkpoint = deepcopy(dict(checkpoint))
    if not all(key in checkpoint for key in ('path', 'source', 'sha256', 'epoch')):
        raise ValueError('Evaluation requires frozen checkpoint provenance')
    checkpoint['path'] = str(Path(checkpoint['path']).resolve())
    checkpoint['source'] = str(Path(checkpoint['source']).resolve())
    if hashlib.sha256(Path(checkpoint['path']).read_bytes()).hexdigest() != checkpoint['sha256']:
        raise ValueError('Evaluation frozen checkpoint SHA256 mismatch')
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    excluded = sorted(set(excluded_scene_groups), key=str)
    seeds = list(range(seed, seed + episodes))
    jobs = [dict(task=task, config=deepcopy(config), actor='policy', seed=episode_seed,
                 mode=mode, checkpoint=deepcopy(checkpoint), output_dir=str(output_dir),
                 name=f'episode-{episode_seed}', excluded_scene_groups=excluded,
                 previous_episode_fingerprint=archive_fingerprint(output_dir/f'episode-{episode_seed}.npz'),
                 replay_every=replay_every, progress_every=progress_every,
                 inference_device=inference_device,
                 inference_num_threads=inference_num_threads)
            for episode_seed in seeds]
    rows = []

    def persist():
        successes = sum(row['success'] for row in rows)
        failed = [row for row in rows if not row['success']]
        report = dict(schema='maritime-policy-evaluation-v1', task=task, mode=mode,
                      output_dir=str(output_dir), checkpoint=deepcopy(checkpoint),
                      model_sha256=checkpoint['sha256'], model_epoch=checkpoint['epoch'],
                      seeds=seeds, seed_start=seed, simulation=deepcopy(config),
                      excluded_scene_groups=excluded, requested_episodes=episodes,
                      completed_episodes=len(rows), pending_episodes=episodes - len(rows),
                      success_count=successes, failure_count=len(failed),
                      runtime_failure_count=sum(row['runtime_failure'] for row in rows),
                      success_rate=successes / episodes, win_rate=successes / episodes,
                      complete=len(rows) == episodes, episodes=deepcopy(rows),
                      summaries=[deepcopy(row['summary']) for row in rows],
                      failures=deepcopy(failed), metrics=_metrics(rows),
                      termination_reasons=dict(Counter(
                          row['summary'].get('termination_reason') or 'unspecified'
                          for row in rows)),
                      test_outcomes_used_for_weight_selection=False)
        atomic_json(output_dir / 'episodes.json', report['episodes'])
        atomic_json(output_dir / 'summary.json', report)
        if server is not None:
            server.publish(dict(type='evaluation_status', task=task, mode=mode,
                                complete=report['complete'], requested_episodes=episodes,
                                completed_episodes=len(rows), success_count=successes,
                                failure_count=len(failed), success_rate=report['success_rate']))
        return report

    report = persist()
    if workers == 1:
        for job in jobs:
            rows.append(_evaluate_job(job, runner, server))
            report = persist()
    else:
        with ProcessPoolExecutor(max_workers=workers,
                                 mp_context=multiprocessing.get_context('spawn')) as pool:
            futures = [pool.submit(_evaluate_job, job, runner) for job in jobs]
            for job, future in zip(jobs, futures):
                try:
                    row = future.result()
                except Exception as error:
                    row = _failure_result(job, error)
                rows.append(row)
                report = persist()
    return report
