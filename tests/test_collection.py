"""Deterministic selection and durable collection; real physics is checked separately."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import importlib
import json
from pathlib import Path
import sys
import threading
import time

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


def _module():
    return importlib.import_module('collection')


def _config():
    from config import Config
    return replace(Config(), potential_enabled=True, coverage_episode_seconds=2.,
                   tracking_episode_seconds=2.)


def _fake_worker(module, rejected=(), delays=None, errors=(), events=None):
    def worker(job):
        seed = job['seed']
        time.sleep((delays or {}).get(seed, 0.))
        if events is not None: events.append(seed)
        if seed in errors: raise RuntimeError('worker process disappeared')
        valid = seed not in rejected
        folder = Path(job['output_dir'])
        path = folder / f'episode-{seed}.json'
        module.write_json(path, {'seed': seed, 'fixture': True})
        cache = path.with_suffix('.npz')
        if valid: np.savez_compressed(cache, actions=np.ones((2, 3), np.float32))
        return {'seed': seed, 'attempt': job['attempt'], 'status': 'valid' if valid else 'rejected',
                'json': str(path.resolve()), 'npz': str(cache.resolve()) if valid else None,
                'scene_id': f'scene-{seed}', 'scene_group_id': f'group-{seed}',
                'summary': {'steps': 2, 'terminated': not valid,
                            'termination_reason': 'time_limit' if valid else 'uav_cloud_collision'},
                'artifacts': [module._artifact(path)] + ([module._artifact(cache)] if valid else [])}
    return worker


def _setup_fake(monkeypatch, **kwargs):
    module = _module()
    def executor(**settings):
        settings.pop('mp_context', None)
        return ThreadPoolExecutor(**settings)
    monkeypatch.setattr(module, 'ProcessPoolExecutor', executor)
    monkeypatch.setattr(module, '_collect_one', _fake_worker(module, **kwargs))
    return module


def _collect(module, path, **kwargs):
    settings = dict(episodes=2, seed=10, workers=3, attempt_multiplier=3,
                    replay_every=4, progress_every=2)
    settings.update(kwargs)
    return module.collect_expert_episodes('coverage', _config(), path, **settings)


def test_first_successes_follow_seed_order_despite_out_of_order_completion(tmp_path, monkeypatch):
    completed = []
    module = _setup_fake(monkeypatch, rejected={10}, delays={10: .12, 11: .07}, events=completed)
    result = _collect(module, tmp_path)
    assert completed[0] == 12
    assert [item['seed'] for item in result['accepted']] == [11, 12]
    assert [item['seed'] for item in result['rejected']] == [10]
    manifest = json.loads(Path(result['manifest_path']).read_text(encoding='utf-8'))
    assert manifest['complete'] is True
    assert manifest['paths'] == [item['npz'] for item in manifest['accepted']]
    assert len(manifest['accepted']) == 2
    assert manifest['metrics']['max_in_flight'] <= 3
    assert manifest['metrics']['max_buffered_results'] <= 3
    json.dumps(result, allow_nan=False)


def test_resume_recovers_incremental_manifest_without_recollecting_accepted_seed(tmp_path, monkeypatch):
    completed = []
    module = _setup_fake(monkeypatch, events=completed)
    original_write = module.write_json
    def interrupt_after_first(path, value):
        original_write(path, value)
        if Path(path).name == 'accepted-manifest.json' and len(value['accepted']) == 1:
            raise KeyboardInterrupt('simulated parent interruption')
    monkeypatch.setattr(module, 'write_json', interrupt_after_first)
    with pytest.raises(KeyboardInterrupt): _collect(module, tmp_path, workers=1)
    interrupted = json.loads((tmp_path / 'accepted-manifest.json').read_text(encoding='utf-8'))
    assert interrupted['complete'] is False
    assert len(interrupted['accepted']) == 1
    monkeypatch.setattr(module, 'write_json', original_write)
    result = _collect(module, tmp_path, workers=1)
    assert [item['seed'] for item in result['accepted']] == [10, 11]
    assert completed.count(10) == 1
    assert len(completed) == 2


def test_completed_resume_rejects_changed_config_source_or_seed(tmp_path, monkeypatch):
    module = _setup_fake(monkeypatch)
    result = _collect(module, tmp_path)
    monkeypatch.setattr(module, '_collect_one', lambda job: pytest.fail('completed resume started a worker'))
    assert _collect(module, tmp_path)['accepted'] == result['accepted']
    with pytest.raises(ValueError, match='identity'):
        _collect(module, tmp_path, seed=11)
    with pytest.raises(ValueError, match='identity'):
        module.collect_expert_episodes('coverage', replace(_config(), field_scan_weight=2.),
            tmp_path, episodes=2, seed=10, workers=3, attempt_multiplier=3,
            replay_every=4, progress_every=2)
    monkeypatch.setattr(module, '_source_fingerprint', lambda: 'changed-source')
    with pytest.raises(ValueError, match='identity'): _collect(module, tmp_path)


def test_resume_detects_missing_or_modified_training_artifacts(tmp_path, monkeypatch):
    module = _setup_fake(monkeypatch)
    result = _collect(module, tmp_path)
    Path(result['accepted'][0]['npz']).write_bytes(b'changed')
    with pytest.raises(ValueError, match='artifact'): _collect(module, tmp_path)


def test_failures_are_explicit_and_insufficient_collection_never_completes(tmp_path, monkeypatch):
    module = _setup_fake(monkeypatch, rejected={11}, errors={10})
    with pytest.raises(RuntimeError, match='0/2.*accepted-manifest'):
        _collect(module, tmp_path, attempt_multiplier=1, workers=2)
    manifest = json.loads((tmp_path / 'accepted-manifest.json').read_text(encoding='utf-8'))
    assert manifest['complete'] is False
    assert manifest['status'] == 'insufficient_valid_episodes'
    assert [item['status'] for item in manifest['rejected']] == ['worker_error', 'rejected']
    failed = manifest['rejected'][0]
    assert failed['error']['type'] == 'RuntimeError'
    assert Path(failed['failure_json']).is_file()
    assert manifest['paths'] == []


def test_submission_window_stays_bounded_when_lowest_seed_is_slow(tmp_path, monkeypatch):
    completed = []
    module = _setup_fake(monkeypatch, delays={10: .2}, events=completed)
    _collect(module, tmp_path, episodes=5, workers=2)
    assert completed[:2] == [11, 10]
    assert sorted(completed)[:5] == list(range(10, 15))


@pytest.mark.parametrize('settings', [{'episodes': 0}, {'workers': 0}, {'replay_every': 0},
                                     {'progress_every': 0}, {'attempt_multiplier': 0}])
def test_invalid_collection_settings_do_not_start_workers(tmp_path, monkeypatch, settings):
    module = _setup_fake(monkeypatch)
    with pytest.raises(ValueError): _collect(module, tmp_path, **settings)


@pytest.mark.parametrize('task,summary,reason', [
    ('coverage', {'time_seconds': 2., 'coverage_ratio': .99}, 'coverage_incomplete'),
    ('coverage', {'time_seconds': 2., 'coverage_ratio': 1.}, None),
    ('coverage', {'time_seconds': 1., 'coverage_ratio': 1.}, 'episode_duration_incomplete'),
    ('tracking', {'time_seconds': 2., 'visible_fraction': 1., 'first_loss_time': None}, None),
    ('tracking', {'time_seconds': 2., 'visible_fraction': .9, 'first_loss_time': .2}, 'tracking_incomplete'),
])
def test_task_success_requires_full_duration_and_objective(task, summary, reason):
    module = _module()
    assert module._task_rejection(task, summary, _config().to_dict()) == reason


def test_real_process_collection_keeps_parent_scene_and_frames_together(tmp_path):
    module = _module()
    class Server:
        def __init__(self): self.events = []
        def publish(self, event): self.events.append(event)
    server = Server()
    result = module.collect_expert_episodes('tracking', _config(), tmp_path,
        episodes=2, seed=60, workers=2, attempt_multiplier=2,
        replay_every=1, progress_every=1, server=server, require_task_success=True)
    assert len(result['accepted']) == 2
    assert result['complete']
    current_scene = None
    frame_count = 0
    for event in server.events:
        if event['type'] == 'scene': current_scene = event['collection_seed']
        if event['type'] == 'frame':
            frame_count += 1
            assert event['collection_seed'] == current_scene
    assert frame_count > 0
    for accepted in result['accepted']:
        record = module.load_episode(accepted['npz'])
        assert accepted['json'] is None
        assert not Path(accepted['npz']).with_suffix('.json').exists()
        assert len(record['actions']) == 2
        assert record['frame_action_indices'] == [0, 1, 2]
        assert record['metadata']['expert_valid']
        assert accepted['summary']['time_seconds'] == 2.


def test_safe_but_incomplete_coverage_retains_labels_and_explicit_rejection(tmp_path):
    module = _module()
    with pytest.raises(RuntimeError, match='0/1'):
        module.collect_expert_episodes('coverage', _config(), tmp_path,
            episodes=1, seed=40, workers=1, attempt_multiplier=1,
            replay_every=5, progress_every=2, require_task_success=True)
    manifest = json.loads((tmp_path / 'accepted-manifest.json').read_text(encoding='utf-8'))
    rejected = manifest['rejected'][0]
    assert rejected['status'] == 'task_incomplete'
    assert rejected['rejection_reason'] == 'coverage_incomplete'
    record = module.load_episode(rejected['npz'])
    assert record['metadata']['expert_valid'] is True
    assert Path(rejected['npz']).is_file()
    assert manifest['paths'] == []


def _job(tmp_path):
    return {'task': 'coverage', 'config': _config().to_dict(), 'seed': 40, 'attempt': 0,
            'output_dir': str(tmp_path), 'identity_hash': 'test-identity',
            'replay_every': 2, 'progress_every': 300, 'stream_events': False,
            'result_path': str(tmp_path / '.collection-attempts' / 'episode-40.json')}


def test_worker_preserves_initialization_error_evidence(tmp_path, monkeypatch):
    module = _module()
    from env_coverage import CoverageEnv
    def invalid_reset(self, **kwargs): raise ValueError('scene generation exhausted')
    monkeypatch.setattr(CoverageEnv, 'reset', invalid_reset)
    result = module._collect_one(_job(tmp_path))
    assert result['status'] == 'initialization_failure'
    assert result['error']['type'] == 'ValueError'
    assert result['json'] is None
    evidence = json.loads(Path(result['failure_json']).read_text(encoding='utf-8'))
    assert evidence['error']['message'] == 'scene generation exhausted'
    assert result['npz'] is None


@pytest.mark.parametrize('unexpected', [False, True])
def test_worker_distinguishes_planning_failure_from_unexpected_error(tmp_path, monkeypatch, unexpected):
    module = _module()
    from expert_field import FieldCoverageExpert
    class PlanningError(Exception): pass
    def fail(self, obs):
        raise (RuntimeError('unexpected bug') if unexpected else PlanningError('no feasible continuation'))
    monkeypatch.setattr(FieldCoverageExpert, 'act', fail)
    result = module._collect_one(_job(tmp_path))
    assert result['status'] == ('worker_error' if unexpected else 'planning_failure')
    assert result['json'] is None
    assert result['summary']['planning_failure']['type'] == ('RuntimeError' if unexpected else 'PlanningError')
    assert Path(result['npz'])==(tmp_path/'episode-40.npz').resolve()
    assert module.load_episode(result['npz'])['metadata']['expert_valid'] is False
    assert Path(_job(tmp_path)['result_path']).is_file()


def test_event_queue_saturation_drops_display_events_without_blocking(monkeypatch):
    import queue
    module = _module()
    events = queue.Queue(maxsize=1)
    monkeypatch.setattr(module, '_EVENT_QUEUE', events)
    publisher = module._QueuePublisher(0, 10)
    for step in range(100): publisher.publish({'type': 'frame', 'frame': {'t': step}})
    assert events.qsize() == 1
    assert publisher.dropped_events == 99


def test_resume_recovers_sidecars_completed_before_manifest_update(tmp_path, monkeypatch):
    module = _setup_fake(monkeypatch)
    result = _collect(module, tmp_path, workers=1)
    path = Path(result['manifest_path'])
    interrupted = dict(result, complete=False, status='collecting', accepted=[], rejected=[],
                       attempts=[], paths=[], unused=[], next_ordered_attempt=0)
    module.write_json(path, interrupted)
    monkeypatch.setattr(module, '_collect_one', lambda job: pytest.fail('sidecar seed recollected'))
    resumed = _collect(module, tmp_path, workers=2)
    assert [r['seed'] for r in resumed['accepted']] == [10, 11]
    assert resumed['complete']


def test_worker_rejects_source_changed_after_parent_manifest_creation(tmp_path):
    module = _module()
    job = dict(_job(tmp_path), source_sha256='different-source')
    result = module._collect_one(job)
    assert result['status'] == 'initialization_failure'
    assert 'source identity' in result['error']['message'].lower()
    assert result['json'] is None
