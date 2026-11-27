"""Accepted DAgger manifests remain usable after moving to a new directory."""
import importlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch


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
