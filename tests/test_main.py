"""CLI integration checks replace expensive collection, training and inference."""
import importlib
import json
import sys

import pytest
import torch


def cli_args(tmp_path, monkeypatch):
    # main owns runtime cache paths; restore the test process settings afterward.
    monkeypatch.setattr(sys, 'pycache_prefix', sys.pycache_prefix)
    monkeypatch.setenv('MPLCONFIGDIR', str(tmp_path / 'cache' / 'matplotlib'))
    monkeypatch.setenv('XDG_CACHE_HOME', str(tmp_path / 'cache'))
    return ['--task', 'tracking', '--output-dir', str(tmp_path),
            '--tf-log-dir', str(tmp_path / 'tf-logs'), '--run-name', 'interface-test',
            '--device', 'cpu', '--no-visualize', '--no-save-video']


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
        return dict(best_checkpoint=str(tmp_path / 'new.pt'), initial_validation_mse=1.,
                    best_validation_mse=.5, best_epoch=1)

    monkeypatch.setattr(train, 'train_policy', capture_training)
    main.main(['--mode', 'train', '--potential-field', '--no-closed-loop-selection',
               '--field-cloud-geometry-encoder', 'shared_pool_v1',
               '--split-reference-checkpoint', str(reference)] + cli_args(tmp_path, monkeypatch))
    assert len(calls) == 1
    assert calls[0]['model_config']['cloud_geometry_encoder'] == 'shared_pool_v1'
    assert calls[0]['split_reference_checkpoint'] == reference
    assert calls[0]['initial_checkpoint'] is None


def test_field_evaluation_requires_selected_model_instead_of_falling_back_to_best(tmp_path, monkeypatch):
    main = importlib.import_module('main')
    policy = importlib.import_module('src.policy_tracking')
    candidate = tmp_path / 'models' / 'tracking' / 'best.pt'
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(b'unvalidated or withdrawn model')

    def unexpected_load(*args, **kwargs):
        pytest.fail('default field deployment must not fall back to an offline best candidate')

    monkeypatch.setattr(policy.NeuralPolicy, 'from_checkpoint', unexpected_load)
    with pytest.raises(FileNotFoundError, match='selected.pt'):
        main.main(['--mode', 'evaluate', '--potential-field'] + cli_args(tmp_path, monkeypatch))


@pytest.mark.parametrize('task,expected', [('coverage', 'shared_pool_v1'), ('tracking', 'slots')])
def test_main_selects_task_specific_default_cloud_encoder(tmp_path, monkeypatch, task, expected):
    main = importlib.import_module('main')
    train = importlib.import_module('src.train')
    calls = []

    def capture(*args, **kwargs):
        calls.append(kwargs)
        return dict(best_checkpoint=str(tmp_path / 'new.pt'), initial_validation_mse=1.,
                    best_validation_mse=.5, best_epoch=1)

    monkeypatch.setattr(train, 'train_policy', capture)
    main.main(['--mode', 'train', '--potential-field', '--no-closed-loop-selection']
              + cli_args(tmp_path, monkeypatch) + ['--task', task])
    assert len(calls) == 1
    assert calls[0]['model_config']['cloud_geometry_encoder'] == expected
