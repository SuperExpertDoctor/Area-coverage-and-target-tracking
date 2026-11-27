"""Accepted DAgger manifests remain usable after moving to a new directory."""
import importlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch


def test_training_pan_jitter_is_seeded_guarded_and_preserves_original_layout():
    from copy import deepcopy
    import math
    from config import Config
    from env_tracking import TrackingEnv
    from scenario import validate_scene
    from geometry import wrap
    collector = importlib.import_module('collect_dagger')
    cfg = Config(potential_enabled=True)
    env = TrackingEnv(cfg)
    env.reset(seed=22000)
    original = deepcopy(env.scene)
    jitter = math.radians(20)
    first, metadata = collector.jitter_tracking_initial_pan(
        original, cfg, jitter, np.random.default_rng(7))
    same, _ = collector.jitter_tracking_initial_pan(original, cfg, jitter, np.random.default_rng(7))
    assert first == same
    assert original == env.scene
    assert first['ship_route'] == original['ship_route']
    assert first['clouds'] == original['clouds'] and first['islands'] == original['islands']
    assert first['uav_initial'] == original['uav_initial']
    assert first['radar_pan_initial'] != original['radar_pan_initial']
    ship, uav = first['ship_initial'], first['uav_initial']
    bearing = wrap(math.atan2(ship['y'] - uav['y'], ship['x'] - uav['x']) - uav['heading'])
    assert abs(wrap(first['radar_pan_initial'] - bearing)) <= jitter + 1e-9
    assert abs(first['radar_pan_initial']) <= cfg.pan_limit
    assert metadata['requested_radians'] == pytest.approx(jitter)
    assert metadata['actual_initial_pan'] == first['radar_pan_initial']
    assert validate_scene(first, cfg, 'tracking')


@pytest.mark.parametrize('value', [-.1, float('nan'), float('inf')])
def test_initial_pan_jitter_rejects_invalid_values_before_loading_checkpoint(tmp_path, value):
    collector = importlib.import_module('collect_dagger')
    with pytest.raises(ValueError, match='jitter'):
        collector.collect_dagger('tracking', 'missing.pt', 'missing.json', tmp_path,
                                 initial_pan_jitter=value)


def test_dagger_pan_jitter_keeps_layout_split_and_json_npz_teacher_provenance(tmp_path):
    import math
    from config import Config
    from env_tracking import TrackingEnv
    from expert_field import FieldTrackingExpert
    from rollouts import run_episode
    from src.train import create_policy
    from src.datasets import read_episode_metadata, SequenceDataset
    collector = importlib.import_module('collect_dagger')
    cfg = Config(potential_enabled=True)
    original, source = run_episode(TrackingEnv(cfg), FieldTrackingExpert(cfg), seed=22000,
                                  actor_name='expert', output_dir=tmp_path / 'original',
                                  name='source', record_observations=True, max_steps=1)
    source_bytes = source.read_bytes()
    manifest = source.parent / 'accepted-manifest.json'
    manifest.write_text(json.dumps(dict(complete=True, task='tracking',
                                        paths=[str(source.with_suffix('.npz'))])), encoding='utf-8')
    model = create_policy('tracking', {'potential_field': True, 'field_frame': 'body',
                                      'pan_residual': True})
    checkpoint = tmp_path / 'student.pt'
    torch.save(dict(checkpoint_version='bc-field-v2', task='tracking',
                    observation_schema='obs-field-v2', action_schema='action-field-v2',
                    model_config=model.model_config, model_state_dict=model.state_dict(),
                    dataset_split={'train_paths': [str(source.with_suffix('.npz'))]}), checkpoint)
    target, report = collector.collect_dagger('tracking', checkpoint, manifest, tmp_path / 'augmented',
        episodes=1, beta=0, max_steps=1, initial_pan_jitter=math.radians(20), seed=99)
    assert len(report['accepted']) == 1 and source.read_bytes() == source_bytes
    record = json.loads(Path(report['accepted'][0]).read_text(encoding='utf-8'))
    metadata = read_episode_metadata(Path(report['accepted'][0]).with_suffix('.npz'))
    assert metadata['training_augmentation'] == record['metadata']['training_augmentation']
    assert metadata['scene_group_id'] == original['metadata']['scene_group_id']
    assert metadata['scene_id'] != original['metadata']['scene_id']
    assert record['scene']['ship_route'] == original['scene']['ship_route']
    assert record['metadata']['mixture_beta'] == 0
    data = SequenceDataset([Path(report['accepted'][0]).with_suffix('.npz')], 'tracking',
                           observation_schema='obs-field-v2')[0]
    teacher = FieldTrackingExpert(cfg)
    env = TrackingEnv(cfg)
    obs, _ = env.reset(seed=22000, options={'scene': record['scene']})
    teacher.reset()
    np.testing.assert_allclose(data['actions'][0], teacher.act(obs), atol=1e-6)


@pytest.mark.parametrize('relative', [True, False])
def test_dagger_manifest_resolves_paths_against_the_original_manifest(tmp_path, monkeypatch, relative):
    collector = importlib.import_module('collect_dagger')
    from src.datasets import split_episodes

    original_dir = tmp_path / 'original'
    original_dir.mkdir()
    source = original_dir / 'episode.npz'
    metadata = {'task': 'tracking', 'scene_id': 'training-scene',
                'scene_group_id': 'training-layout', 'expert_valid': True}
    np.savez(source, metadata=np.asarray(json.dumps(metadata)))
    source.with_suffix('.json').write_text(json.dumps({
        'seed': 7, 'scene': {'config_snapshot': {}}}), encoding='utf-8')
    manifest = original_dir / 'accepted-manifest.json'
    manifest.write_text(json.dumps({'task': 'tracking', 'complete': True,
        'paths': [source.name if relative else str(source)]}), encoding='utf-8')
    monkeypatch.setattr(torch, 'load', lambda *args, **kwargs: {
        'dataset_split': {'train_paths': [str(source)]}})
    monkeypatch.setattr(collector.NeuralPolicy, 'from_checkpoint',
                        lambda *args, **kwargs: object())

    def labeled_rollout(*args, output_dir, name, **kwargs):
        target = Path(output_dir) / f'{name}.json'
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez(target.with_suffix('.npz'), metadata=np.asarray(json.dumps(metadata)))
        return {'metadata': {'expert_valid': True}}, target

    monkeypatch.setattr(collector, 'run_episode', labeled_rollout)
    target, _ = collector.collect_dagger('tracking', 'checkpoint.pt', manifest,
                                         tmp_path / 'augmented', episodes=1)
    result = json.loads(target.read_text(encoding='utf-8'))
    assert all(Path(path).is_absolute() for path in result['paths'])
    split = split_episodes(target, 'tracking', validation_fraction=0)
    assert str(source.resolve()) in split['train_paths']
    assert len(split['train_paths']) == 2


@pytest.mark.parametrize('device', [None, 'cuda'])
def test_dagger_student_uses_the_requested_inference_device(tmp_path, monkeypatch, device):
    collector = importlib.import_module('collect_dagger')
    source = tmp_path / 'episode.npz'
    np.savez(source, metadata=np.asarray(json.dumps({
        'task': 'tracking', 'scene_id': 'scene', 'scene_group_id': 'layout'})))
    source.with_suffix('.json').write_text(json.dumps({
        'seed': 7, 'scene': {'config_snapshot': {}}}), encoding='utf-8')
    manifest = tmp_path / 'accepted-manifest.json'
    manifest.write_text(json.dumps({'task': 'tracking', 'complete': True,
                                    'paths': [str(source)]}), encoding='utf-8')
    monkeypatch.setattr(torch, 'load', lambda *args, **kwargs: {
        'dataset_split': {'train_paths': [str(source)]}})
    loaded_devices = []

    def load_student(checkpoint, *, device):
        loaded_devices.append(device)
        return object()

    monkeypatch.setattr(collector.NeuralPolicy, 'from_checkpoint', load_student)
    monkeypatch.setattr(collector, 'run_episode', lambda *args, **kwargs: (
        {'metadata': {'expert_valid': False}}, tmp_path / 'rejected.json'))
    options = {} if device is None else {'device': device}
    collector.collect_dagger('tracking', 'checkpoint.pt', manifest,
                              tmp_path / 'augmented', episodes=1, **options)
    assert loaded_devices == ['cpu' if device is None else device]
