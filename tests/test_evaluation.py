"""Independent inference episodes keep fixed denominators and model provenance."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pytest


def frozen_checkpoint(tmp_path):
    path = tmp_path / 'frozen' / 'coverage-last.pt'
    path.parent.mkdir()
    path.write_bytes(b'immutable latest checkpoint bytes')
    return dict(path=str(path), source=str(tmp_path / 'models' / 'last.pt'),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(), epoch=7)


def successful_episode(job):
    """Write a real lightweight interaction artifact without loading a network."""
    path = Path(job['output_dir']) / (job['name'] + '.npz')
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, actions=np.asarray([[0., 1., .5]], dtype=np.float32),
                        executed_actions=np.asarray([[0., 1., .5]], dtype=np.float32))
    task = job['task']
    summary = dict(time_seconds=job['config'][f'{task}_episode_seconds'], steps=10,
                   terminated=False, termination_reason='time_limit',
                   coverage_ratio=1., visible_fraction=1., first_loss_time=None)
    return dict(task=task, seed=job['seed'], mode=job['mode'], path=str(path),
                summary=summary, success=True, scene_group_id=f'new-layout-{job["seed"]}')


def evaluate(*args, **kwargs):
    from src.evaluation import evaluate_policy
    return evaluate_policy(*args, **kwargs)


def config():
    return dict(coverage_episode_seconds=10800., tracking_episode_seconds=1800.)


def parallel_episode(job):
    if job['seed'] == 100:
        time.sleep(.1)
    if job['seed'] == 101:
        raise RuntimeError('independent worker failure')
    return successful_episode(job)


def test_default_evaluation_runs_thirty_ordered_fresh_policy_episodes(tmp_path):
    checkpoint = frozen_checkpoint(tmp_path)
    output = tmp_path / 'experiment' / 'eval' / 'coverage' / 'none'
    jobs = []

    def runner(job):
        jobs.append(job)
        return successful_episode(job)

    report = evaluate('coverage', config(), checkpoint, output, runner=runner)

    assert len(jobs) == 30
    assert report['seeds'] == list(range(940000, 940030))
    assert [row['seed'] for row in report['episodes']] == report['seeds']
    assert report['complete'] is True
    assert report['requested_episodes'] == report['completed_episodes'] == 30
    assert report['success_count'] == 30
    assert report['failure_count'] == report['pending_episodes'] == 0
    assert report['success_rate'] == report['win_rate'] == 1.
    assert report['checkpoint'] == checkpoint
    assert report['model_sha256'] == checkpoint['sha256']
    assert report['model_epoch'] == 7
    assert report['metrics']['coverage_ratio']['mean'] == 1.
    assert report['metrics']['time_seconds']['count'] == 30
    for job, row in zip(jobs, report['episodes']):
        assert job['actor'] == 'policy'
        assert job['checkpoint'] == checkpoint
        assert job['name'] == f'episode-{job["seed"]}'
        assert Path(row['path']) == output / f'episode-{job["seed"]}.npz'
        assert row['checkpoint'] == checkpoint
        assert row['model_sha256'] == checkpoint['sha256']
        with np.load(row['path'], allow_pickle=False) as artifact:
            assert artifact['actions'].shape == (1, 3)
    assert json.loads((output / 'summary.json').read_text(encoding='utf-8')) == report
    assert json.loads((output / 'episodes.json').read_text(encoding='utf-8')) == report['episodes']
    assert not list(output.glob('*.tmp'))


def test_runtime_failures_are_retained_and_never_removed_from_denominator(tmp_path):
    checkpoint = frozen_checkpoint(tmp_path)
    output = tmp_path / 'eval' / 'tracking' / 'none'

    def runner(job):
        if job['seed'] == 950001:
            raise RuntimeError('inference crashed')
        result = successful_episode(job)
        if job['seed'] == 950002:
            result['summary']['first_loss_time'] = 2.
        return result

    report = evaluate('tracking', config(), checkpoint, output,
                      episodes=3, seed=950000, runner=runner)

    assert report['complete'] is True
    assert report['success_count'] == 1
    assert report['failure_count'] == 2
    assert report['runtime_failure_count'] == 1
    assert report['success_rate'] == report['win_rate'] == pytest.approx(1 / 3)
    assert len(report['summaries']) == 3
    assert len(report['failures']) == 2
    failed = report['episodes'][1]
    assert failed['summary']['error'] == dict(type='RuntimeError', message='inference crashed')
    assert failed['summary']['termination_reason'] == 'test_runtime_failure'
    assert json.loads(Path(failed['path']).read_text(encoding='utf-8')) == failed
    assert report['episodes'][2]['success'] is False
    assert report['metrics']['visible_fraction']['count'] == 2


@pytest.mark.parametrize('shared_failure',[False,True])
def test_failed_rerun_does_not_attribute_previous_npz_to_current_attempt(tmp_path,shared_failure):
    checkpoint=frozen_checkpoint(tmp_path)
    output=tmp_path/'eval'
    first=evaluate('tracking',config(),checkpoint,output,episodes=1,runner=successful_episode)
    previous=Path(first['episodes'][0]['path'])
    original=previous.read_bytes()
    def runner(job):
        if not shared_failure:raise RuntimeError('initialization failed before interactions')
        return dict(task=job['task'],seed=job['seed'],mode=job['mode'],
                    path=str(output/(job['name']+'.failure.json')),
                    episode_path=str(previous),summary=dict(terminated=True,
                    termination_reason='test_runtime_failure'))
    report=evaluate('tracking',config(),checkpoint,output,episodes=1,runner=runner)
    assert report['runtime_failure_count']==1
    assert 'episode_path' not in report['episodes'][0]
    assert previous.read_bytes()==original


def test_existing_npz_is_not_evidence_of_a_successful_new_evaluation(tmp_path):
    checkpoint=frozen_checkpoint(tmp_path)
    output=tmp_path/'eval'
    first=evaluate('tracking',config(),checkpoint,output,episodes=1,runner=successful_episode)
    report=evaluate('tracking',config(),checkpoint,output,episodes=1,
                    runner=lambda job:first['episodes'][0])
    assert report['success_count']==0
    assert report['runtime_failure_count']==1
    assert 'episode_path' not in report['episodes'][0]


def test_progress_is_persisted_before_next_episode_and_interrupt_remains_incomplete(tmp_path):
    checkpoint = frozen_checkpoint(tmp_path)
    output = tmp_path / 'eval'
    observed = []

    def runner(job):
        progress = json.loads((output / 'summary.json').read_text(encoding='utf-8'))
        observed.append(progress)
        if job['seed'] == 12:
            raise KeyboardInterrupt()
        return successful_episode(job)

    with pytest.raises(KeyboardInterrupt):
        evaluate('coverage', config(), checkpoint, output, episodes=3, seed=10, runner=runner)

    assert [row['completed_episodes'] for row in observed] == [0, 1, 2]
    assert all(row['complete'] is False for row in observed)
    progress = json.loads((output / 'summary.json').read_text(encoding='utf-8'))
    assert progress['completed_episodes'] == 2
    assert progress['pending_episodes'] == 1
    assert progress['success_rate'] == pytest.approx(2 / 3)
    assert progress['complete'] is False
    assert len(json.loads((output / 'episodes.json').read_text(encoding='utf-8'))) == 2


def test_excluded_scene_groups_are_forwarded_and_returned_overlap_is_failure(tmp_path):
    checkpoint = frozen_checkpoint(tmp_path)
    excluded = ('training-layout', 'validation-layout')
    seen = []

    def runner(job):
        seen.append(job)
        result = successful_episode(job)
        result['scene_group_id'] = 'validation-layout'
        return result

    report = evaluate('coverage', config(), checkpoint, tmp_path / 'eval',
                      episodes=1, excluded_scene_groups=excluded, runner=runner,
                      mode='forecast', replay_every=15, progress_every=90,
                      inference_device='cpu', inference_num_threads=2)

    assert seen[0]['excluded_scene_groups'] == list(excluded)
    assert seen[0]['mode'] == 'forecast'
    assert seen[0]['replay_every'] == 15
    assert seen[0]['progress_every'] == 90
    assert seen[0]['inference_num_threads'] == 2
    assert report['success_rate'] == 0.
    assert report['runtime_failure_count'] == 1
    assert 'overlap' in report['failures'][0]['summary']['error']['message'].lower()


def test_success_is_recomputed_from_full_task_summary_not_worker_claim(tmp_path):
    checkpoint = frozen_checkpoint(tmp_path)

    def runner(job):
        result = successful_episode(job)
        result['summary']['time_seconds'] -= 1.
        return result

    report = evaluate('coverage', config(), checkpoint, tmp_path / 'eval',
                      episodes=1, runner=runner)
    assert report['success_count'] == 0
    assert report['failure_count'] == 1
    assert report['runtime_failure_count'] == 0


def test_tracking_supports_explicit_forecast_evaluation(tmp_path):
    checkpoint = frozen_checkpoint(tmp_path)
    report = evaluate('tracking', config(), checkpoint, tmp_path / 'eval',
                      episodes=1, mode='forecast', runner=successful_episode)
    assert report['mode'] == report['episodes'][0]['mode'] == 'forecast'
    assert report['complete'] is True
    assert report['success_count'] == 1


def test_mutated_frozen_checkpoint_is_rejected_before_running(tmp_path):
    checkpoint = frozen_checkpoint(tmp_path)
    Path(checkpoint['path']).write_bytes(b'changed bytes')

    def unexpected(job):
        pytest.fail('mutated frozen model must never run')

    with pytest.raises(ValueError, match='SHA256'):
        evaluate('coverage', config(), checkpoint, tmp_path / 'eval', runner=unexpected)


def test_process_workers_keep_seed_order_and_other_results_after_failure(tmp_path):
    checkpoint = frozen_checkpoint(tmp_path)
    report = evaluate('coverage', config(), checkpoint, tmp_path / 'eval',
                      episodes=3, seed=100, workers=2, runner=parallel_episode)
    assert [row['seed'] for row in report['episodes']] == [100, 101, 102]
    assert [row['success'] for row in report['episodes']] == [True, False, True]
    assert report['success_rate'] == pytest.approx(2 / 3)
    assert report['runtime_failure_count'] == 1
    assert report['complete'] is True


def test_default_runner_imports_shared_rollout_and_forwards_serial_live_server(tmp_path, monkeypatch):
    from src import formal_run
    checkpoint = frozen_checkpoint(tmp_path)
    calls = []

    class Server:
        def __init__(self):
            self.messages = []

        def publish(self, message):
            self.messages.append(message)

    server = Server()

    def rollout(job, server=None):
        calls.append((job, server))
        return successful_episode(job)

    monkeypatch.setattr(formal_run, '_test_rollout', rollout)
    report = evaluate('coverage', config(), checkpoint, tmp_path / 'eval',
                      episodes=1, server=server)
    assert calls[0][1] is server
    assert report['success_count'] == 1
    assert [message['completed_episodes'] for message in server.messages] == [0, 1]


def test_shared_runtime_failure_gets_full_checkpoint_provenance(tmp_path):
    checkpoint = frozen_checkpoint(tmp_path)
    output = tmp_path / 'eval'

    def runner(job):
        return dict(task=job['task'], seed=job['seed'], mode=job['mode'],
                    path=str(output / (job['name'] + '.failure.json')), success=False,
                    summary=dict(terminated=True, termination_reason='test_runtime_failure',
                                 error=dict(type='ValueError', message='invalid scene')))

    report = evaluate('coverage', config(), checkpoint, output, episodes=1, runner=runner)
    failure = report['failures'][0]
    assert failure['runtime_failure'] is True
    assert failure['checkpoint'] == checkpoint
    assert failure['summary']['error']['message'] == 'invalid scene'
    assert json.loads(Path(failure['path']).read_text(encoding='utf-8')) == failure


def test_full_config_report_roundtrips_as_json(tmp_path):
    from config import Config
    checkpoint = frozen_checkpoint(tmp_path)
    output = tmp_path / 'eval'
    report = evaluate('coverage', Config(), checkpoint, output,
                      episodes=1, runner=successful_episode)
    assert json.loads((output / 'summary.json').read_text(encoding='utf-8')) == report


@pytest.mark.parametrize('changes', [dict(seed=123), dict(model_sha256='different-model')])
def test_worker_identity_mismatch_is_retained_as_failure(tmp_path, changes):
    checkpoint = frozen_checkpoint(tmp_path)

    def runner(job):
        return dict(successful_episode(job), **changes)

    report = evaluate('coverage', config(), checkpoint, tmp_path / 'eval',
                      episodes=1, runner=runner)
    assert report['success_count'] == 0
    assert report['runtime_failure_count'] == 1


@pytest.mark.parametrize('options', [{'episodes': 0}, {'episodes': True}, {'workers': 0},
                                   {'seed': -1}, {'replay_every': 0},
                                   {'progress_every': 0}, {'inference_num_threads': 0}])
def test_invalid_evaluation_settings_fail_before_inference(tmp_path, options):
    checkpoint = frozen_checkpoint(tmp_path)
    with pytest.raises(ValueError):
        evaluate('coverage', config(), checkpoint, tmp_path / 'eval',
                 runner=successful_episode, **options)
