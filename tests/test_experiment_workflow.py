"""CLI contract for separated experiment artifacts and post-training tests."""
import importlib
import json
import os
from pathlib import Path
import sys
import types

import pytest
import torch


@pytest.mark.parametrize('existing_side',['outputs','tf-logs'])
@pytest.mark.parametrize('mode',['formal','collect','train','evaluate'])
def test_existing_experiment_is_rejected_before_any_side_effect(tmp_path,monkeypatch,existing_side,mode):
    import main
    existing=tmp_path/existing_side/'reserved'
    existing.mkdir(parents=True)
    marker=existing/'keep.txt'
    marker.write_text('original experiment',encoding='utf-8')
    other=tmp_path/('tf-logs' if existing_side=='outputs' else 'outputs')/'reserved'
    original_cache=sys.pycache_prefix
    monkeypatch.setattr(importlib.import_module('src.train'),'train_policy',
                        lambda *a,**k:pytest.fail('collision must be detected before training'))
    monkeypatch.setattr(importlib.import_module('src.formal_run'),'run_formal',
                        lambda *a,**k:pytest.fail('collision must be detected before formal execution'))
    monkeypatch.setattr(importlib.import_module('src.collection'),'collect_expert_episodes',
                        lambda *a,**k:pytest.fail('collision must be detected before collection'))
    with pytest.raises(FileExistsError,match='reserved'):
        main.main(['--mode',mode,'--experiment-name','reserved',
            '--output-dir',str(tmp_path/'outputs'),'--tf-log-dir',str(tmp_path/'tf-logs'),
            '--no-visualize','--no-save-video','--device','cpu'])
    assert marker.read_text(encoding='utf-8')=='original experiment'
    assert list(existing.iterdir())==[marker]
    assert not other.exists()
    assert sys.pycache_prefix==original_cache


def test_serve_can_view_an_existing_experiment_without_rewriting_configuration(tmp_path,monkeypatch):
    import main
    from visual_server import VisualServer
    root=tmp_path/'outputs'/'existing'
    (root/'logs').mkdir(parents=True)
    config=root/'logs'/'run-config.json'
    config.write_text('{"original":true}',encoding='utf-8')
    class Server:
        url='http://127.0.0.1:12345/'
        def close(self):pass
    monkeypatch.setattr(VisualServer,'start',lambda self:Server())
    monkeypatch.setattr(main.time,'sleep',lambda seconds:(_ for _ in ()).throw(KeyboardInterrupt()))
    main.main(['--mode','serve','--experiment-name','existing',
        '--output-dir',str(tmp_path/'outputs'),'--tf-log-dir',str(tmp_path/'tf-logs'),
        '--device','cpu'])
    assert config.read_text(encoding='utf-8')=='{"original":true}'
    assert not (root/'logs'/'configs').exists()
    assert not (tmp_path/'tf-logs').exists()


def test_cli_no_longer_exposes_training_resume():
    from main import parser
    with pytest.raises(SystemExit):
        parser().parse_args(['--resume-checkpoint','old.pt'])


def test_name_reservation_preserves_a_racing_process_and_rolls_back_own_empty_directory(tmp_path,monkeypatch):
    from main import reserve_experiment
    output,tf=tmp_path/'outputs'/'race',tmp_path/'tf-logs'/'race'
    original_mkdir=Path.mkdir
    def mkdir(path,*args,**kwargs):
        if path==tf:
            original_mkdir(path,parents=True)
            (path/'other-process.txt').write_text('preserve',encoding='utf-8')
        return original_mkdir(path,*args,**kwargs)
    monkeypatch.setattr(Path,'mkdir',mkdir)
    with pytest.raises(FileExistsError):reserve_experiment(output,tf)
    assert not output.exists()
    assert (tf/'other-process.txt').read_text(encoding='utf-8')=='preserve'


def test_name_collision_checks_literal_paths_before_resolving_links(tmp_path,monkeypatch):
    import main
    output=tmp_path/'outputs'/'linked'
    target=tmp_path/'missing-target'
    original_resolve=Path.resolve
    original_is_symlink=Path.is_symlink
    # Simulate a dangling link on hosts without Windows symlink privileges.
    monkeypatch.setattr(Path,'resolve',lambda path,*a,**k:
                        target if path==output else original_resolve(path,*a,**k))
    monkeypatch.setattr(Path,'is_symlink',lambda path:
                        path==output or original_is_symlink(path))
    with pytest.raises(FileExistsError,match='linked'):
        main.main(['--mode','train','--experiment-name','linked',
            '--output-dir',str(tmp_path/'outputs'),'--tf-log-dir',str(tmp_path/'tf-logs'),
            '--no-visualize','--device','cpu'])
    assert not target.exists()
    assert not (tmp_path/'tf-logs'/'linked').exists()


def test_evaluation_defaults_and_aliases_are_thirty_episodes():
    from main import parser
    defaults = parser().parse_args([])
    assert defaults.formal_test_episodes == 30
    assert defaults.validation_batch_size == defaults.batch_size == 4
    assert defaults.formal_coverage_safety_layer == 'none'
    values = parser().parse_args(['--eval-episodes', '7', '--eval-seed', '900',
                                  '--eval-workers', '2', '--validation-batch-size', '3'])
    assert (values.formal_test_episodes, values.formal_test_seed,
            values.formal_test_workers, values.validation_batch_size) == (7, 900, 2, 3)


def test_train_writes_only_four_experiment_directories_and_evaluates_latest(tmp_path, monkeypatch):
    main = importlib.import_module('main')
    trainer = importlib.import_module('src.train')
    monkeypatch.setattr(sys, 'pycache_prefix', sys.pycache_prefix)
    for name in ('PYTHONPYCACHEPREFIX', 'MPLCONFIGDIR', 'XDG_CACHE_HOME'):
        monkeypatch.setenv(name, str(tmp_path / 'old-cache'))
    training_calls, evaluations = [], []

    def train(task, dataset, weights, tf_logs, **kwargs):
        training_calls.append((dataset, weights, tf_logs, kwargs))
        weights.mkdir(parents=True, exist_ok=True)
        for tag in ('last', 'best'):
            torch.save({'task': task, 'epoch': 2 if tag == 'last' else 1}, weights / f'{tag}.pt')
        return dict(best_checkpoint=str(weights / 'best.pt'), last_checkpoint=str(weights / 'last.pt'),
                    initial_validation_mse=1., best_validation_mse=.5, best_epoch=1)

    def evaluate(task, config, checkpoint, destination, **kwargs):
        evaluations.append((task, checkpoint, destination, kwargs))
        destination.mkdir(parents=True, exist_ok=True)
        return dict(episodes=kwargs['episodes'], complete=True, success_rate=.5, rows=[])

    monkeypatch.setattr(trainer, 'train_policy', train)
    monkeypatch.setitem(sys.modules, 'src.evaluation', types.SimpleNamespace(evaluate_policy=evaluate))
    output, tf_root = tmp_path / 'outputs', tmp_path / 'tf-logs'
    report = main.main(['--mode', 'train', '--task', 'tracking', '--experiment-name', 'trial',
        '--output-dir', str(output), '--tf-log-dir', str(tf_root), '--run-name', 'test',
        '--no-visualize', '--no-save-video', '--no-closed-loop-selection', '--device', 'cpu'])
    root = output / 'trial'
    assert training_calls[0][:3] == (root / 'datasets' / 'tracking', root / 'weights' / 'tracking',
                                    tf_root / 'trial' / 'tracking' / 'test')
    assert training_calls[0][3]['log_dir'] == root / 'logs' / 'training' / 'tracking'
    assert training_calls[0][3]['validation_batch_size'] == 4
    assert set(path.name for path in root.iterdir()) <= {'datasets', 'weights', 'logs', 'eval'}
    assert (root / 'logs' / 'run-config.json').is_file()
    assert Path(os.environ['XDG_CACHE_HOME']) == root / 'logs' / 'cache'
    assert len(evaluations) == 1
    task, checkpoint, destination, settings = evaluations[0]
    assert checkpoint['epoch'] == 2
    assert Path(checkpoint['path']).parent == root / 'weights' / 'frozen-latest'
    assert destination == root / 'eval' / task / 'none'
    assert settings['episodes'] == 30
    assert report['tasks'][task]['evaluation']['success_rate'] == .5


def test_nonpositive_validation_batch_size_is_rejected_before_training(tmp_path, monkeypatch):
    import main
    trainer = importlib.import_module('src.train')
    monkeypatch.setattr(trainer, 'train_policy', lambda *a, **k: pytest.fail('must reject invalid size'))
    monkeypatch.setattr(sys, 'pycache_prefix', sys.pycache_prefix)
    for name in ('PYTHONPYCACHEPREFIX', 'MPLCONFIGDIR', 'XDG_CACHE_HOME'):
        monkeypatch.setenv(name, str(tmp_path / 'cache'))
    with pytest.raises(ValueError, match='positive'):
        main.main(['--mode', 'train', '--task', 'tracking', '--no-visualize', '--device', 'cpu',
                   '--output-dir', str(tmp_path), '--validation-batch-size', '0'])


def test_formal_pipeline_separates_logs_weights_and_thirty_npz_tests(tmp_path, monkeypatch):
    from config import Config
    from main import parser
    from src import collection, formal_run, train
    from src.episode_io import write_episode

    args = parser().parse_args(['--mode', 'formal', '--task', 'tracking', '--run-name', 'test',
        '--eval-workers', '1', '--no-browser-recording'])
    args.output_dir = tmp_path / 'experiment'
    args.tf_log_dir = tmp_path / 'tf-logs'
    cfg = Config(potential_enabled=True)
    training_calls = []

    def rollout(job, server=None):
        summary = dict(time_seconds=job['config']['tracking_episode_seconds'],
                       visible_fraction=1., first_loss_time=None, terminated=False)
        record = dict(task='tracking', actor=job['actor'], seed=job['seed'],
            scene={'config_snapshot': job['config']}, frames=[{'t': 0.}, {'t': summary['time_seconds']}],
            metadata={'scene_group_id': f"scene-{job['seed']}"}, summary=summary)
        path = write_episode(Path(job['output_dir']) / job['name'], record)
        return dict(task='tracking', seed=job['seed'], mode=job.get('mode','expert'),
                    path=str(path), summary=summary, success=True,
                    scene_group_id=record['metadata']['scene_group_id'])

    def collect(task, config, destination, **kwargs):
        destination.mkdir(parents=True, exist_ok=True)
        manifest = destination / 'accepted-manifest.json'
        manifest.write_text(json.dumps({'paths': []}), encoding='utf-8')
        return dict(manifest_path=str(manifest), complete=True, metrics={}, identity_hash='test')

    def fit(task, dataset, weights, tf_logs, **kwargs):
        training_calls.append((weights, kwargs['log_dir']))
        weights.mkdir(parents=True, exist_ok=True)
        torch.save(dict(task=task, epoch=2), weights / 'last.pt')
        return dict(last_checkpoint=str(weights / 'last.pt'))

    monkeypatch.setattr(formal_run, '_rollout', rollout)
    monkeypatch.setattr(collection, 'collect_expert_episodes', collect)
    monkeypatch.setattr(train, 'train_policy', fit)
    report = formal_run.run_formal(args, cfg, ['tracking'], None, {})
    root = args.output_dir
    assert training_calls == [(root / 'weights' / 'tracking', root / 'logs' / 'training' / 'tracking')]
    assert set(path.name for path in root.iterdir()) == {'datasets', 'weights', 'logs', 'eval'}
    summary = json.loads((root / 'eval' / 'tracking' / 'none' / 'summary.json').read_text())
    assert summary['requested_episodes'] == summary['completed_episodes'] == 30
    assert summary['success_rate'] == 1.
    assert len(list((root / 'eval' / 'tracking' / 'none').glob('episode-*.npz'))) == 30
    assert len(report['tasks']['tracking']['tests']) == 30
    assert (root / 'logs' / 'formal-status.json').is_file()
    assert (root / 'logs' / 'formal-protocol.json').is_file()
    assert not list(root.rglob('*.tmp'))


def test_external_checkpoint_retains_scene_exclusions_without_dataset_files(tmp_path, monkeypatch):
    import main
    from src import evaluation
    from tests.test_main import cli_args
    checkpoint = tmp_path / 'external.pt'
    split = dict(train_scene_group_ids=['train-layout'], validation_scene_group_ids=['val-layout'],
                 episodes=[dict(scene_group_id='third-layout', seed=4)],
                 fixed_scene_groups={'train':['historical-layout'],'validation':['historical-val']},
                 train_paths=['gone.npz'], validation_paths=['also-gone.npz'])
    torch.save(dict(task='tracking', epoch=2, dataset_split=split), checkpoint)
    calls=[]
    def evaluate(*args, **kwargs):
        calls.append(kwargs)
        return dict(complete=True, success_rate=0., episodes=[])
    monkeypatch.setattr(evaluation,'evaluate_policy',evaluate)
    main.main(['--mode','evaluate','--checkpoint-path',str(checkpoint)] + cli_args(tmp_path,monkeypatch))
    assert calls[0]['excluded_scene_groups'] == ['historical-layout','historical-val','third-layout','train-layout','val-layout']


def test_external_checkpoint_rejects_collected_seed_in_final_test(tmp_path, monkeypatch):
    import main
    from src import evaluation
    from tests.test_main import cli_args
    checkpoint = tmp_path / 'external.pt'
    torch.save(dict(task='tracking',epoch=2,dataset_split={
        'episodes':[dict(scene_group_id='train-layout', seed=950000)]}),checkpoint)
    monkeypatch.setattr(evaluation,'evaluate_policy',lambda *a,**k:pytest.fail('must reject seed leakage'))
    with pytest.raises(ValueError,match='overlap'):
        main.main(['--mode','evaluate','--checkpoint-path',str(checkpoint)] + cli_args(tmp_path,monkeypatch))
