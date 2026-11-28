"""CLI integration checks replace expensive collection, training and inference."""
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import torch


def training_result(model_dir):
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(dict(task=model_dir.name, epoch=1), model_dir / 'last.pt')
    return dict(best_checkpoint=str(model_dir / 'best.pt'), last_checkpoint=str(model_dir / 'last.pt'),
                initial_validation_mse=1., best_validation_mse=.5, best_epoch=1)


@pytest.fixture(autouse=True)
def lightweight_inference_evaluation(monkeypatch):
    # These CLI tests isolate option routing; actual 30-case scheduling and
    # artifact provenance are exercised in test_experiment_workflow/evaluation.
    evaluation = importlib.import_module('src.evaluation')
    monkeypatch.setattr(evaluation, 'evaluate_policy', lambda *a, **k:
        dict(complete=True, requested_episodes=k['episodes'], success_rate=.5, episodes=[]))


def cli_args(tmp_path, monkeypatch):
    # main owns runtime cache paths; restore the test process settings afterward.
    monkeypatch.setattr(sys, 'pycache_prefix', sys.pycache_prefix)
    monkeypatch.setenv('MPLCONFIGDIR', str(tmp_path / 'cache' / 'matplotlib'))
    monkeypatch.setenv('XDG_CACHE_HOME', str(tmp_path / 'cache'))
    monkeypatch.setenv('PYTHONPYCACHEPREFIX', str(tmp_path / 'cache' / 'python'))
    return ['--task', 'tracking', '--output-dir', str(tmp_path),
            '--tf-log-dir', str(tmp_path / 'tf-logs'), '--run-name', 'interface-test',
            '--device', 'cpu', '--no-visualize', '--no-save-video']


def test_named_experiments_separate_artifacts_training_paths_and_caches(tmp_path, monkeypatch):
    main = importlib.import_module('main')
    train = importlib.import_module('src.train')
    output_base = tmp_path / 'outputs'
    tf_base = tmp_path / 'tf-logs'
    monkeypatch.setattr(main, 'PROJECT_ROOT', tmp_path)
    arguments = cli_args(tmp_path, monkeypatch) + [
        '--mode', 'train', '--potential-field', '--no-closed-loop-selection',
        '--output-dir', str(output_base), '--tf-log-dir', str(tf_base)]
    calls = []

    def capture_training(task, dataset, model_dir, tf_dir, **kwargs):
        calls.append((task, dataset, model_dir, tf_dir, kwargs['metadata']['command']))
        return training_result(model_dir)

    monkeypatch.setattr(train, 'train_policy', capture_training)
    for name in ('experiment-a', '试验乙.v2'):
        main.main(arguments + ['--experiment-name', name])
        output = output_base / name
        tf_output = tf_base / name
        persisted = json.loads((output / 'logs' / 'run-config.json').read_text(encoding='utf-8'))
        archived = json.loads((output / 'logs' / 'configs' / 'interface-test-train.json').read_text(encoding='utf-8'))
        assert persisted == archived == json.loads(json.dumps(calls[-1][4]))
        assert persisted['experiment_name'] == name
        assert persisted['output_dir'] == str(output)
        assert persisted['tf_log_dir'] == str(tf_output)
        assert calls[-1][:4] == ('tracking', output / 'datasets' / 'tracking',
                                output / 'weights' / 'tracking', tf_output / 'tracking' / 'interface-test')
        assert (output / 'logs' / 'training' / 'tracking-report.json').is_file()
        assert (output / 'logs' / 'reports' / 'train-report.json').is_file()
        assert (output / 'logs' / 'validation-report.json').is_file()
        assert sys.pycache_prefix == os.environ['PYTHONPYCACHEPREFIX'] == str(output / 'logs' / 'cache' / 'python')
        assert os.environ['MPLCONFIGDIR'] == str(output / 'logs' / 'cache' / 'matplotlib')
        assert os.environ['XDG_CACHE_HOME'] == str(output / 'logs' / 'cache')
    assert len(calls) == 2
    assert not (output_base / 'run-config.json').exists()
    assert not (output_base / 'field-v2').exists()


@pytest.mark.parametrize('potential_field', [False, True])
def test_omitting_experiment_name_preserves_legacy_output_paths(tmp_path, monkeypatch, potential_field):
    main = importlib.import_module('main')
    train = importlib.import_module('src.train')
    monkeypatch.setattr(main, 'PROJECT_ROOT', tmp_path)
    assert main.parser().parse_args([]).experiment_name is None
    output_base = tmp_path / 'outputs'
    tf_base = tmp_path / 'tf-logs'
    output = output_base / 'field-v2' if potential_field else output_base
    calls = []

    def capture_training(task, dataset, model_dir, tf_dir, **kwargs):
        calls.append((model_dir, tf_dir))
        return training_result(model_dir)

    monkeypatch.setattr(train, 'train_policy', capture_training)
    main.main(cli_args(tmp_path, monkeypatch) + [
        '--mode', 'train', '--potential-field' if potential_field else '--no-potential-field',
        '--no-closed-loop-selection', '--output-dir', str(output_base), '--tf-log-dir', str(tf_base)])
    assert calls == [(output / 'weights' / 'tracking', tf_base / 'tracking' / 'interface-test')]
    persisted = json.loads((output / 'logs' / 'run-config.json').read_text(encoding='utf-8'))
    assert persisted['experiment_name'] is None
    assert persisted['output_dir'] == str(output)


@pytest.mark.parametrize('name', [
    '', ' ', '.', '..', '../escape', '..\\escape', 'a/b', 'a\\b', '/absolute',
    '\\absolute', r'C:\absolute', 'C:relative', '//server/share',
    'trailing.', 'trailing ', ' leading', 'bad:name', 'bad?name', 'bad*name',
    'bad<name', 'bad>name', 'bad|name', 'bad"name', 'bad\x00name', 'bad\nname',
    'bad\x7fname', 'CON', 'nul.txt', 'AUX.log', 'PRN', 'COM1', 'com9.log',
    'LPT1', 'lpt9.txt', 'COM¹', 'LPT².log', 'CONIN$', 'CONOUT$', 'CLOCK$',
    'x' * 256,
])
def test_invalid_experiment_name_fails_before_creating_outputs(tmp_path, monkeypatch, capsys, name):
    main = importlib.import_module('main')
    output = tmp_path / 'must-not-be-created'
    arguments = cli_args(output, monkeypatch)
    with pytest.raises(SystemExit) as error:
        main.main(arguments + ['--mode', 'train', '--experiment-name', name])
    assert error.value.code == 2
    assert 'experiment-name must be one safe directory name' in capsys.readouterr().err
    assert not output.exists()


@pytest.mark.parametrize('experiment_name', [None, 'another-experiment'])
def test_run_script_passes_experiment_and_base_directories_without_shared_startup_caches(tmp_path, experiment_name):
    bash = shutil.which('bash')
    if os.name == 'nt':
        git_bash = Path(os.environ.get('ProgramFiles', 'C:/Program Files')) / 'Git' / 'bin' / 'bash.exe'
        if git_bash.is_file():
            bash = str(git_bash)
    if not bash:
        pytest.skip('Bash is unavailable')
    script = tmp_path / 'run.sh'
    script.write_bytes((Path(__file__).resolve().parents[1] / 'run.sh').read_bytes())
    capture = tmp_path / 'captured-cli.json'
    (tmp_path / 'main.py').write_text(
        'import json, os, pathlib, sys\n'
        'pathlib.Path(os.environ["CLI_CAPTURE"]).write_text(json.dumps({\n'
        '    "args": sys.argv[1:], "no_bytecode": sys.dont_write_bytecode,\n'
        '    "python_cache": os.environ.get("PYTHONPYCACHEPREFIX"),\n'
        '    "matplotlib_cache": os.environ.get("MPLCONFIGDIR")}))\n', encoding='utf-8')
    environment = dict(os.environ, PYTHON=sys.executable.replace('\\', '/'), CLI_CAPTURE=str(capture))
    for key in ('EXPERIMENT_NAME', 'PYTHONPYCACHEPREFIX', 'MPLCONFIGDIR'):
        environment.pop(key, None)
    if experiment_name is not None:
        environment['EXPERIMENT_NAME'] = experiment_name
    subprocess.run([bash, str(script).replace('\\', '/'), 'train'], env=environment,
                   capture_output=True, text=True, check=True, timeout=30)
    captured = json.loads(capture.read_text(encoding='utf-8'))
    arguments = captured['args']
    assert arguments[arguments.index('--experiment-name') + 1] == (experiment_name or 'formal')
    assert arguments[arguments.index('--output-dir') + 1] == 'outputs'
    assert arguments[arguments.index('--tf-log-dir') + 1] == 'tf-logs'
    assert captured['no_bytecode'] is True
    assert captured['python_cache'] is None
    assert captured['matplotlib_cache'] is None


@pytest.mark.parametrize('requested,cuda_available,expected', [
    ('cpu', False, 'cpu'), ('cuda', False, 'cuda'),
    ('auto', False, 'cpu'), ('auto', True, 'cuda'),
])
def test_main_passes_resolved_inference_device_to_dagger_collection(
        tmp_path, monkeypatch, requested, cuda_available, expected):
    main = importlib.import_module('main')
    collector = importlib.import_module('collect_dagger')
    train = importlib.import_module('src.train')
    checkpoint = tmp_path / 'source.pt'
    checkpoint.write_bytes(b'fixture; torch.load is replaced')
    manifest = tmp_path / 'accepted-manifest.json'
    manifest.write_text(json.dumps({'task': 'tracking', 'complete': True, 'paths': []}), encoding='utf-8')
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: cuda_available)
    monkeypatch.setattr(torch, 'load', lambda *args, **kwargs: {'model_config': {}})
    calls = []

    def collect(*args, **kwargs):
        calls.append(kwargs)
        return manifest, {'accepted': ['labeled-prefix.json'], 'rejected': []}

    def unexpected_training(*args, **kwargs):
        pytest.fail('collect-only must not train')

    monkeypatch.setattr(collector, 'collect_dagger', collect)
    monkeypatch.setattr(train, 'train_policy', unexpected_training)
    report = main.main(['--mode', 'dagger', '--dagger-collect-only',
        '--checkpoint-path', str(checkpoint), '--dataset-manifest', str(manifest),
        '--inference-device', requested] + cli_args(tmp_path, monkeypatch))
    assert len(calls) == 1
    assert calls[0].get('device') == expected
    assert 'training' not in report['tasks']['tracking']
    assert 'policy' not in report['tasks']['tracking']


def test_main_missing_explicit_checkpoint_does_not_use_a_fallback(tmp_path, monkeypatch):
    main = importlib.import_module('main')
    policy = importlib.import_module('src.policy_tracking')
    fallback = tmp_path / 'models' / 'tracking' / 'best.pt'
    fallback.parent.mkdir(parents=True)
    fallback.write_bytes(b'fallback must not be loaded')

    def unexpected_load(*args, **kwargs):
        pytest.fail('a missing explicit checkpoint must not select another model')

    monkeypatch.setattr(policy.NeuralPolicy, 'from_checkpoint', unexpected_load)
    with pytest.raises(FileNotFoundError, match='Explicit checkpoint is missing'):
        main.main(['--mode', 'evaluate', '--checkpoint-path', str(tmp_path / 'missing.pt')]
                  + cli_args(tmp_path, monkeypatch))


def test_main_fresh_cloud_encoder_preserves_only_requested_split_reference(tmp_path, monkeypatch):
    main = importlib.import_module('main')
    train = importlib.import_module('src.train')
    reference = tmp_path / 'historical-slots.pt'
    calls = []

    def capture_training(*args, **kwargs):
        calls.append(kwargs)
        return training_result(args[2])

    monkeypatch.setattr(train, 'train_policy', capture_training)
    main.main(['--mode', 'train', '--potential-field', '--no-closed-loop-selection',
               '--field-cloud-geometry-encoder', 'shared_pool_v1',
               '--field-steering-loss-weight', '0.05',
               '--split-reference-checkpoint', str(reference)] + cli_args(tmp_path, monkeypatch))
    assert len(calls) == 1
    assert calls[0]['model_config']['cloud_geometry_encoder'] == 'shared_pool_v1'
    assert calls[0]['split_reference_checkpoint'] == reference
    assert calls[0]['initial_checkpoint'] is None
    assert calls[0]['field_steering_loss_weight'] == .05


def test_field_evaluation_requires_latest_model_instead_of_falling_back_to_best(tmp_path, monkeypatch):
    main = importlib.import_module('main')
    policy = importlib.import_module('src.policy_tracking')
    candidate = tmp_path / 'weights' / 'tracking' / 'best.pt'
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(b'unvalidated or withdrawn model')

    def unexpected_load(*args, **kwargs):
        pytest.fail('default field deployment must not fall back to an offline best candidate')

    monkeypatch.setattr(policy.NeuralPolicy, 'from_checkpoint', unexpected_load)
    with pytest.raises(FileNotFoundError, match='last.pt'):
        main.main(['--mode', 'evaluate', '--potential-field'] + cli_args(tmp_path, monkeypatch))


@pytest.mark.parametrize('task,expected', [('coverage', 'shared_pool_v1'), ('tracking', 'slots')])
def test_main_selects_task_specific_default_cloud_encoder(tmp_path, monkeypatch, task, expected):
    main = importlib.import_module('main')
    train = importlib.import_module('src.train')
    calls = []

    def capture(*args, **kwargs):
        calls.append(kwargs)
        return training_result(args[2])

    monkeypatch.setattr(train, 'train_policy', capture)
    main.main(['--mode', 'train', '--potential-field', '--no-closed-loop-selection']
              + cli_args(tmp_path, monkeypatch) + ['--task', task])
    assert len(calls) == 1
    assert calls[0]['model_config']['cloud_geometry_encoder'] == expected


def test_constraint_mode_defaults_to_pure_and_requires_explicit_field_evaluation(tmp_path, monkeypatch):
    main = importlib.import_module('main')
    assert main.parser().parse_args([]).policy_safety_layer == 'none'
    for extra in (['--mode', 'collect', '--potential-field'], ['--mode', 'evaluate', '--no-potential-field']):
        with pytest.raises(ValueError, match='constraint'):
            main.main(['--policy-safety-layer', 'forecast'] + extra + cli_args(tmp_path, monkeypatch))


def test_main_inference_evaluation_uses_fresh_seeds_and_requested_physics(tmp_path, monkeypatch):
    from dataclasses import replace
    main = importlib.import_module('main')
    evaluation = importlib.import_module('src.evaluation')
    from config import Config
    cfg = replace(Config(), potential_enabled=True, uav_speed=55.)
    checkpoint = tmp_path / 'candidate.pt'
    torch.save(dict(task='tracking',epoch=1), checkpoint)
    calls = []
    def evaluate(task, config, checkpoint, destination, **kw):
        assert task == 'tracking'
        assert config.uav_speed == 55.
        assert checkpoint['epoch'] == 1
        calls.append(kw)
        return dict(complete=True, requested_episodes=kw['episodes'], success_rate=.5, episodes=[])
    monkeypatch.setattr(evaluation, 'evaluate_policy', evaluate)
    report = main.main(['--mode', 'evaluate', '--potential-field', '--policy-safety-layer', 'forecast',
                       '--checkpoint-path', str(checkpoint), '--config-json', json.dumps(cfg.to_dict())]
                      + cli_args(tmp_path, monkeypatch))
    assert len(calls) == 1
    assert calls[0]['episodes'] == 30 and calls[0]['seed'] == 950000
    assert calls[0]['mode'] == 'forecast'
    assert report['tasks']['tracking']['control_mode'] == 'forecast_constraint'
