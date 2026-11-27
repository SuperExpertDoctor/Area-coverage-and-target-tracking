"""Atomic persistence, bounded training telemetry and exact epoch resumption."""
import copy
import json
import os
import random
import weakref
from pathlib import Path

import numpy as np
import pytest
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from src import train


def field_episode(path, scene, length=5):
    vector = np.zeros((length + 1, 111), np.float32)
    vector[:, 2] = 1
    vector[:, 6] = int(scene[-1]) / 10
    actions = np.tile([np.cos(.4), np.sin(.4), .3], (length, 1)).astype(np.float32)
    np.savez_compressed(path, vector=vector, potential=np.ones((length + 1, 4, 33, 33), np.float32),
                        field_context=np.zeros((length + 1, 16), np.float32), actions=actions,
                        metadata=json.dumps(dict(task='coverage', scene_id=scene, scene_group_id=scene,
                            dataset_version='bc-dataset-field-v2', observation_schema='obs-field-v2',
                            action_schema='action-field-v2')))


def setup_run(tmp_path):
    dataset = tmp_path / 'data'; dataset.mkdir()
    for i in range(4): field_episode(dataset / f'{i}.npz', f'scene-{i}', length=3+i)
    return dataset, dict(batch_size=2, sequence_length=2, seed=17,
                         model_config=dict(potential_field=True, hidden_size=8),
                         progress_every=1, progress_seconds=30., load_workers=2)


def assert_tree_equal(first, second):
    if torch.is_tensor(first): torch.testing.assert_close(first, second, atol=0, rtol=0)
    elif isinstance(first, np.ndarray): np.testing.assert_array_equal(first, second)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first: assert_tree_equal(first[key], second[key])
    elif isinstance(first, (tuple, list)):
        assert len(first) == len(second)
        for a, b in zip(first, second): assert_tree_equal(a, b)
    else: assert first == second


@pytest.fixture
def deterministic_runtime(monkeypatch):
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    previous = torch.are_deterministic_algorithms_enabled()
    warning_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        with torch.backends.cudnn.flags(deterministic=True, benchmark=False):
            yield
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=warning_only)


def test_checkpoint_write_failure_preserves_previous_valid_file(tmp_path, monkeypatch):
    path = tmp_path / 'last.pt'
    train._atomic_torch_save(path, {'epoch': 4})
    original = path.read_bytes()
    def interrupted_save(value, stream):
        stream.write(b'incomplete checkpoint')
        raise OSError('disk interrupted')
    monkeypatch.setattr(torch, 'save', interrupted_save)
    with pytest.raises(OSError, match='interrupted'):
        train._atomic_torch_save(path, {'epoch': 5})
    assert path.read_bytes() == original
    assert [p.name for p in tmp_path.iterdir()] == ['last.pt']


def test_atomic_publication_retries_transient_windows_read_lock(tmp_path, monkeypatch):
    path = tmp_path/'status.json'
    train._write_json(path, {'epoch': 0})
    attempts = []
    replace = os.replace
    def temporarily_locked(source, target):
        attempts.append(1)
        if len(attempts) <= 2:
            assert json.loads(path.read_text()) == {'epoch': 0}
            raise PermissionError('Windows reader holds destination open')
        replace(source, target)
    monkeypatch.setattr(os, 'replace', temporarily_locked)
    train._write_json(path, {'epoch': 1})
    assert len(attempts) == 3
    assert json.loads(path.read_text()) == {'epoch': 1}
    assert [item.name for item in tmp_path.iterdir()] == ['status.json']


def test_atomic_publication_permanent_failure_is_bounded(tmp_path, monkeypatch):
    path = tmp_path/'status.json'
    train._write_json(path, {'epoch': 0})
    pauses = []
    monkeypatch.setattr(train.time, 'sleep', pauses.append)
    def locked(*args): raise PermissionError('persistent lock')
    monkeypatch.setattr(os, 'replace', locked)
    with pytest.raises(PermissionError, match='persistent'):
        train._write_json(path, {'epoch': 1})
    assert len(pauses) == 7 and sum(pauses) <= 1.3
    assert json.loads(path.read_text()) == {'epoch': 0}
    assert [item.name for item in tmp_path.iterdir()] == ['status.json']


def test_json_write_failure_preserves_previous_snapshot(tmp_path, monkeypatch):
    path = tmp_path / 'status.json'
    replacements = []
    replace = os.replace
    def observe(source, target):
        replacements.append(Path(target))
        replace(source, target)
    monkeypatch.setattr(os, 'replace', observe)
    train._write_json(path, {'epoch': 1})
    assert replacements == [path]
    with pytest.raises(ValueError): train._write_json(path, {'loss': float('nan')})
    assert json.loads(path.read_text()) == {'epoch': 1}
    assert [p.name for p in tmp_path.iterdir()] == ['status.json']


@pytest.mark.parametrize('progress_options', [dict(progress_every=1),
                                             dict(progress_every=99999, progress_seconds=1.e-12)])
def test_parallel_episode_loading_and_progress_preserve_tbptt_and_metrics(tmp_path, progress_options):
    dataset, _ = setup_run(tmp_path)
    from src.datasets import SequenceDataset
    data = SequenceDataset(sorted(dataset.glob('*.npz')), task='coverage', observation_schema='obs-field-v2')
    torch.manual_seed(1)
    model = train.create_policy('coverage', dict(potential_field=True, hidden_size=8))
    baseline = copy.deepcopy(model)
    first = train._run_epoch(baseline, data, 2, 2, torch.device('cpu'),
                             optimizer=torch.optim.Adam(baseline.parameters()))
    events = []
    second = train._run_epoch(model, data, 2, 2, torch.device('cpu'),
                              optimizer=torch.optim.Adam(model.parameters()), load_workers=2,
                              progress_callback=events.append, **progress_options)
    assert first == second
    assert_tree_equal(baseline.state_dict(), model.state_dict())
    chunks = [event for event in events if event['phase'] == 'chunks']
    assert len(chunks) == 5
    assert chunks[-1]['processed_steps'] == sum(range(3, 7))
    assert chunks[-1]['metrics'] == second
    assert chunks[-1]['gradient_norm'] >= 0
    assert events[0]['phase'] == 'loading'
    assert chunks[-1]['episodes_completed'] == 4


def test_status_is_available_during_epoch_zero_and_history_is_incremental(tmp_path, monkeypatch):
    dataset, options = setup_run(tmp_path)
    snapshots = []
    write_json = train._write_json
    def observe(path, value):
        if Path(path).name in {'training-status.json', 'history.json'}:
            snapshots.append((Path(path).name, copy.deepcopy(value)))
        write_json(path, value)
    monkeypatch.setattr(train, '_write_json', observe)
    output = tmp_path / 'models'
    result = train.train_policy('coverage', dataset, output, tmp_path/'tf', epochs=2, **options)
    assert any(name == 'training-status.json' and value['epoch'] == 0 and
               value['phase'] == 'initial_train' and value['processed_steps'] > 0
               for name, value in snapshots)
    assert any(name == 'training-status.json' and value['epoch'] == 0 and
               value['train_metrics'] == result['initial_train'] and
               value['validation_metrics'] == result['initial_validation'] for name, value in snapshots)
    histories = [value for name, value in snapshots if name == 'history.json']
    assert any(len(value['history']) == 1 for value in histories)
    assert histories[-1]['history'] == result['history']
    status = json.loads((output / 'training-status.json').read_text())
    assert status['state'] == 'complete' and status['epoch'] == 2
    assert status['validation_available'] is True


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA unavailable'))])
def test_exact_resume_matches_uninterrupted_optimizer_rng_history_and_tf(tmp_path, device, deterministic_runtime):
    dataset, options = setup_run(tmp_path)
    options['device'] = device
    uninterrupted = train.train_policy('coverage', dataset, tmp_path/'full', tmp_path/'full-tf', epochs=3, **options)
    full_state = torch.load(uninterrupted['last_checkpoint'], weights_only=False)
    first = train.train_policy('coverage', dataset, tmp_path/'resume', tmp_path/'resume-tf', epochs=1, **options)
    random.random(); np.random.random(7); torch.rand(11)
    resumed = train.train_policy('coverage', dataset, tmp_path/'resume', tmp_path/'resume-tf', epochs=3,
                                 resume_checkpoint=first['last_checkpoint'], **options)
    state = torch.load(resumed['last_checkpoint'], weights_only=False)
    for key in ['model_state_dict', 'optimizer_state_dict', 'dataset_split']:
        assert_tree_equal(full_state[key], state[key])
    assert_tree_equal(full_state['resume_state']['rng'], state['resume_state']['rng'])
    assert resumed['history'] == uninterrupted['history']
    assert resumed['best_epoch'] == uninterrupted['best_epoch']
    assert resumed['initial_train'] == uninterrupted['initial_train']
    assert resumed['resume_checkpoint'] == str(Path(first['last_checkpoint']).resolve())
    events = EventAccumulator(str(tmp_path/'resume-tf')).Reload()
    assert set(events.Tags()['scalars']) == {
        f'{subset}/{metric}' for subset in ['Train', 'Validation'] for metric in ['Loss', 'Action_MSE', 'Action_MAE']}
    for tag in events.Tags()['scalars']:
        assert [item.step for item in events.Scalars(tag)] == [0, 1, 2, 3]


def test_resume_rejects_changed_data_config_and_warm_start_combination(tmp_path):
    dataset, options = setup_run(tmp_path)
    result = train.train_policy('coverage', dataset, tmp_path/'models', tmp_path/'tf', epochs=1, **options)
    checkpoint = result['last_checkpoint']
    with pytest.raises(ValueError, match='mutually exclusive'):
        train.train_policy('coverage', dataset, tmp_path/'models', tmp_path/'tf', epochs=2,
                           resume_checkpoint=checkpoint, initial_checkpoint=checkpoint, **options)
    changed = dict(options, sequence_length=3)
    with pytest.raises(ValueError, match='sequence_length'):
        train.train_policy('coverage', dataset, tmp_path/'models', tmp_path/'tf', epochs=2,
                           resume_checkpoint=checkpoint, **changed)
    with (dataset/'0.npz').open('ab') as stream: stream.write(b'changed bytes')
    with pytest.raises(ValueError, match='dataset'):
        train.train_policy('coverage', dataset, tmp_path/'models', tmp_path/'tf', epochs=2,
                           resume_checkpoint=checkpoint, **options)


def test_rng_snapshot_restores_python_numpy_torch_and_shuffle_generator():
    rng = np.random.default_rng(7)
    saved = train._capture_rng_state(rng)
    expected = (random.random(), np.random.rand(), torch.rand(3), rng.permutation(9))
    expected_cuda = torch.rand(3, device='cuda') if torch.cuda.is_available() else None
    train._restore_rng_state(saved, rng)
    actual = (random.random(), np.random.rand(), torch.rand(3), rng.permutation(9))
    actual_cuda = torch.rand(3, device='cuda') if torch.cuda.is_available() else None
    assert_tree_equal(expected, actual)
    assert_tree_equal(expected_cuda, actual_cuda)


@pytest.mark.parametrize('failure', ['prediction', 'gradient'])
def test_nonfinite_training_fails_with_status_and_retains_last_valid_checkpoint(tmp_path, monkeypatch, failure):
    dataset, options = setup_run(tmp_path)
    create = train.create_policy
    def create_failing(*args, **kwargs):
        model = create(*args, **kwargs)
        if failure == 'prediction':
            forward = model.forward
            def bad_forward(*args, **kwargs):
                action, memory = forward(*args, **kwargs)
                return action * float('nan') if model.training else action, memory
            model.forward = bad_forward
        else:
            next(model.parameters()).register_hook(lambda gradient: gradient * float('nan'))
        return model
    monkeypatch.setattr(train, 'create_policy', create_failing)
    output = tmp_path/'models'
    with pytest.raises((FloatingPointError, RuntimeError), match='nonfinite|non-finite'):
        train.train_policy('coverage', dataset, output, tmp_path/'tf', epochs=1, **options)
    status = json.loads((output/'training-status.json').read_text())
    assert status['state'] == 'failed' and status['epoch'] == 1
    assert 'nonfinite' in status['error'] or 'non-finite' in status['error']
    state = torch.load(output/'last.pt', weights_only=False)
    assert state['epoch'] == 0
    assert all(torch.isfinite(value).all() for value in state['model_state_dict'].values())


def test_nonfinite_loss_is_rejected_before_optimizer_step():
    class Extreme(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.weight = torch.nn.Parameter(torch.tensor(1.e30))
        def forward(self, obs, hidden=None):
            return self.weight.expand(1, 1, 2), torch.zeros(1, 1, 1)
    model = Extreme()
    item = dict(vector=torch.zeros(1, 134), actions=torch.zeros(1, 2))
    optimizer = torch.optim.Adam(model.parameters())
    with pytest.raises(FloatingPointError, match='nonfinite.*loss'):
        train._run_epoch(model, [item], 1, 1, torch.device('cpu'), optimizer=optimizer)
    assert not optimizer.state


def test_next_batch_releases_previous_episode_storage():
    references = []
    class Streaming:
        def __len__(self): return 2
        def __getitem__(self, index):
            if index:
                assert references[0]() is None, 'previous full episode is still resident'
            vector = torch.zeros(3, 134); vector[:, 2] = 1
            references.append(weakref.ref(vector))
            return dict(vector=vector, actions=torch.tensor([[1., 0.]]).repeat(3, 1))
    model = train.create_policy('tracking', dict(hidden_size=8))
    train._run_epoch(model, Streaming(), 1, 2, torch.device('cpu'))


def test_resume_recovers_recorded_best_after_interrupted_checkpoint_pair(tmp_path):
    dataset, options = setup_run(tmp_path)
    result = train.train_policy('coverage', dataset, tmp_path/'models', tmp_path/'tf', epochs=1, **options)
    original_best = Path(result['best_checkpoint']).read_bytes()
    # A newer best may be published before a crash leaves last at the old epoch.
    Path(result['best_checkpoint']).write_bytes(b'newer best interrupted before last')
    resumed = train.train_policy('coverage', dataset, tmp_path/'models', tmp_path/'tf', epochs=1,
                                 resume_checkpoint=result['last_checkpoint'], **options)
    assert Path(resumed['best_checkpoint']).read_bytes() == original_best
    assert torch.load(resumed['last_checkpoint'], weights_only=False)['epoch'] == 1


def test_resume_rejects_changed_runtime_configuration(tmp_path, monkeypatch):
    dataset, options = setup_run(tmp_path)
    result = train.train_policy('coverage', dataset, tmp_path/'models', tmp_path/'tf', epochs=1, **options)
    monkeypatch.setattr(torch.backends.cudnn, 'benchmark', not torch.backends.cudnn.benchmark)
    with pytest.raises(ValueError, match='runtime'):
        train.train_policy('coverage', dataset, tmp_path/'models', tmp_path/'tf', epochs=2,
                           resume_checkpoint=result['last_checkpoint'], **options)


def test_preparation_failure_publishes_failed_status(tmp_path):
    output = tmp_path/'models'
    with pytest.raises(ValueError, match='no valid'):
        train.train_policy('coverage', tmp_path/'missing-data', output, tmp_path/'tf', epochs=1)
    status = json.loads((output/'training-status.json').read_text())
    assert status['state'] == 'failed'
    assert 'no valid' in status['error']
