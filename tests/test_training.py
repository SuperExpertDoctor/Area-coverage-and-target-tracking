"""Real sequence learning, scene isolation and deployable recurrent policies."""

import importlib
import importlib.util
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch


def module(name):
    assert importlib.util.find_spec(f"src.{name}") is not None, f"{name} is not implemented"
    return importlib.import_module(f"src.{name}")


def episode(path, task, scene_id, sign=1, length=10):
    dimension = 111 if task == "coverage" else 134
    vector = np.zeros((length + 1, dimension), dtype=np.float32)
    vector[:, 2] = 1
    vector[0, 6] = sign  # initial bank cue, preserved by body-coordinate preprocessing
    actions = np.tile([np.cos(0.65), sign * np.sin(0.65)], (length, 1)).astype(np.float32)
    arrays = dict(vector=vector, actions=actions, executed_actions=-actions,
                  metadata=json.dumps(dict(task=task, scene_id=scene_id, seed=0, schema_version="obs-v1")))
    if task == "coverage":
        arrays["coverage_map"] = np.zeros((length + 1, 3, 64, 64), np.float32)
        arrays["coverage_map"][:, 0] = 1
    np.savez_compressed(path, **arrays)


@pytest.mark.parametrize("task,dimension", [("coverage", 111), ("tracking", 134)])
def test_causal_policy_chunking_preserves_history_and_unit_actions(task, dimension):
    train = module("train")
    torch.set_num_threads(1)
    model = train.create_policy(task)
    vector = torch.randn(2, 7, dimension)
    vector[..., 2:4] = torch.tensor([1., 0.])
    obs = {"vector": vector}
    if task == "coverage":
        obs["coverage_map"] = torch.rand(2, 7, 3, 64, 64)
    full, hidden = model(obs)
    first, memory = model({key: value[:, :3] for key, value in obs.items()})
    last, chunk_hidden = model({key: value[:, 3:] for key, value in obs.items()}, memory)
    assert full.shape == (2, 7, 2) and hidden.shape == (1, 2, 128)
    torch.testing.assert_close(torch.cat([first, last], dim=1), full)
    torch.testing.assert_close(chunk_hidden, hidden)
    torch.testing.assert_close(full.norm(dim=-1), torch.ones(2, 7))
    assert torch.isfinite(full).all()


def test_zero_head_fallback_and_policy_reset():
    train = module("train")
    policy_module = module("policy_tracking")
    model = train.create_policy("tracking", {"heading_residual": False})
    with torch.no_grad():
        for parameter in model.action_head.parameters():
            parameter.zero_()
    obs = {"vector": np.zeros(134, np.float32)}
    obs["vector"][2:4] = [0.6, 0.8]
    wrapper = policy_module.NeuralPolicy(model)
    np.testing.assert_allclose(wrapper.act(obs), [0.6, 0.8], atol=1e-6)
    assert wrapper.hidden is not None
    wrapper.reset()
    assert wrapper.hidden is None
    np.testing.assert_allclose(wrapper.act(obs), [0.6, 0.8], atol=1e-6)
    obs["vector"][0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        wrapper.act(obs)


def test_heading_residual_can_command_opposite_direction():
    model = module("train").create_policy("tracking", {"heading_residual": True})
    with torch.no_grad():
        for parameter in model.action_head.parameters():
            parameter.zero_()
        model.action_head[-1].bias[0] = -2
    vector = torch.zeros(1, 1, 134)
    vector[..., 2] = 1
    actions, _ = model({"vector": vector})
    assert actions[0, 0, 0] < -.99, "the heading residual must allow a full direction reversal"


def test_zero_correction_preserves_twenty_random_headings():
    model = module("train").create_policy("tracking", {"heading_residual": True})
    with torch.no_grad():
        for parameter in model.action_head.parameters():
            parameter.zero_()
    vector = torch.zeros(20, 1, 134)
    vector[:, 0, :2] = torch.arange(40).reshape(20, 2) / 40
    angle = torch.rand(20, generator=torch.Generator().manual_seed(47)) * 2 * torch.pi
    vector[:, 0, 2:4] = torch.stack([angle.cos(), angle.sin()], dim=-1)
    actions, _ = model({"vector": vector})
    torch.testing.assert_close(actions, vector[..., 2:4], atol=1e-6, rtol=1e-6)


def test_body_action_head_rotates_with_current_heading():
    model = module("train").create_policy("tracking", {"action_frame": "body"})
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.action_head[-1].bias.copy_(torch.tensor([.2, -.1]))
    angle = torch.linspace(-torch.pi, torch.pi, 20)
    vector = torch.zeros(20, 1, 134)
    vector[:, 0, 2:4] = torch.stack([angle.cos(), angle.sin()], dim=-1)
    actions, _ = model({"vector": vector})
    local_x = actions[:, 0, 0] * angle.cos() + actions[:, 0, 1] * angle.sin()
    local_y = -actions[:, 0, 0] * angle.sin() + actions[:, 0, 1] * angle.cos()
    torch.testing.assert_close(local_x, local_x[0].expand_as(local_x), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(local_y, local_y[0].expand_as(local_y), atol=1e-6, rtol=1e-6)
    assert local_y[0] < 0


def test_body_rotation_roundoff_remains_inside_strict_action_box():
    model = module("train").create_policy("tracking", {"action_frame": "body"})
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.action_head[-1].bias.copy_(torch.tensor([-.0018463134765625, -.03722048178315163]))
    vector = torch.zeros(1, 1, 134)
    vector[0, 0, 2:4] = torch.tensor([.07434283941984177, -.9972327351570129])
    actions, _ = model({"vector": vector})
    assert (actions.abs() <= 1).all(), "float32 rotation must satisfy the strict environment Box"
    torch.testing.assert_close(actions.norm(dim=-1), torch.ones(1, 1), atol=1e-6, rtol=1e-6)


def test_body_feature_transform_preserves_local_ship_and_obstacle_geometry():
    model = module("train").create_policy("tracking", {"feature_frame": "body"})
    angles = torch.linspace(-torch.pi, torch.pi, 20)
    vector = torch.zeros(20, 1, 134)
    vector[:, 0, :2] = torch.arange(40).reshape(20, 2) / 40
    heading = torch.stack([angles.cos(), angles.sin()], dim=-1)
    vector[:, 0, 2:4] = heading
    def rotate(local):
        return torch.stack([heading[:, 0] * local[0] - heading[:, 1] * local[1],
                            heading[:, 1] * local[0] + heading[:, 0] * local[1]], dim=-1)
    for begin, local in [(4, [1., 0.]), (8, [.9, -.1]), (14, [.5, -.8]),
                          (94, [.1, -.7]), (97, [.07, -.05]), (99, [.6, .8])]:
        vector[:, 0, begin:begin + 2] = rotate(local)
    transformed = model.preprocess_vector(vector)
    torch.testing.assert_close(transformed[..., :2], torch.zeros_like(transformed[..., :2]))
    for begin, local in [(2, [1., 0.]), (4, [1., 0.]), (8, [.9, -.1]), (14, [.5, -.8]),
                          (94, [.1, -.7]), (97, [.07, -.05]), (99, [.6, .8])]:
        torch.testing.assert_close(transformed[:, 0, begin:begin + 2],
                                   torch.tensor(local).expand(20, 2), atol=1e-6, rtol=1e-6)
    assert transformed.shape == vector.shape


def test_coverage_ego_raster_locates_aircraft_and_heading_without_changing_obs():
    model = module("train").create_policy("coverage", {"ego_map_channels": True})
    vector = torch.zeros(1, 1, 111)
    vector[..., 2:4] = torch.tensor([0., 1.])
    x_fraction, y_fraction = 20.5 / 64, 40.5 / 64
    vector[..., 94:98] = torch.tensor([-x_fraction, 1-x_fraction, -y_fraction, 1-y_fraction])
    vector[..., :2] = .7
    vector[..., 54:94] = 3
    obs = {"vector": vector, "coverage_map": torch.zeros(1, 1, 3, 64, 64)}
    original = vector.clone()
    channels = model.build_ego_channels(obs)
    assert channels.shape == (1, 1, 3, 64, 64)
    assert tuple(torch.nonzero(channels[0, 0, 0] == channels[0, 0, 0].max())[0]) == (40, 20)
    assert channels[0, 0, 0, 40, 20] == pytest.approx(1.)
    torch.testing.assert_close(channels[0, 0, 1], torch.zeros(64, 64))
    torch.testing.assert_close(channels[0, 0, 2], channels[0, 0, 0])
    features = model.preprocess_vector(vector)
    torch.testing.assert_close(features[..., :2], torch.zeros_like(features[..., :2]))
    torch.testing.assert_close(features[..., 54:94], torch.zeros_like(features[..., 54:94]))
    actions, _ = model(obs)
    assert actions.shape == (1, 1, 2) and torch.isfinite(actions).all()
    torch.testing.assert_close(vector, original)
    assert model.map_encoder[0].in_channels == 6


@pytest.mark.parametrize("mode", ["gaussian", "body_coordinates", "world_coordinates"])
def test_coverage_raster_upgrade_is_explicit_and_preserves_original_map_weights(tmp_path, mode):
    train = module("train")
    coverage = module("policy_coverage")
    model = train.create_policy("coverage", {"ego_map_channels": False})
    model.set_vector_normalization(torch.arange(111) / 111, torch.ones(111))
    source = tmp_path / "source.pt"
    target = tmp_path / "raster.pt"
    checkpoint = dict(checkpoint_version="bc-v1", observation_schema="obs-v1", task="coverage",
                      model_config=model.model_config, model_state_dict=model.state_dict(),
                      normalization={"method": "train_feature_standardization", "fit_subset": "train"},
                      optimizer_state_dict={"old": "not reusable after a shape change"}, epoch=12)
    torch.save(checkpoint, source)
    source_bytes = source.read_bytes()
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    coverage.upgrade_coverage_checkpoint(source, target, ego_map_sigma=2., ego_map_mode=mode)
    upgraded = torch.load(target, map_location="cpu", weights_only=False)
    state = upgraded["model_state_dict"]
    torch.testing.assert_close(state["map_encoder.0.weight"][:, :3], model.map_encoder[0].weight)
    torch.testing.assert_close(state["map_encoder.0.weight"][:, 3:], torch.zeros(16, 3, 5, 5))
    assert upgraded["model_config"]["ego_map_channels"] is True
    assert upgraded["model_config"]["ego_map_mode"] == mode
    assert upgraded["optimizer_state_dict"] == {}
    assert upgraded["validation_metrics"] is None  # old metrics no longer describe this architecture
    masked = [0, 1] + list(range(54, 94))
    torch.testing.assert_close(state["vector_mean"][masked], torch.zeros(len(masked)))
    torch.testing.assert_close(state["vector_std"][masked], torch.ones(len(masked)))
    torch.testing.assert_close(state["vector_mean"][6], model.vector_mean[6])
    conversion = upgraded["architecture_conversion"]
    snapshot = Path(conversion["source_checkpoint"])
    assert snapshot == target.parent / "source-checkpoints" / f"{source_hash}.pt"
    assert snapshot.read_bytes() == source_bytes
    assert conversion["requested_source_checkpoint"] == str(source.resolve())
    assert conversion["source_checkpoint_sha256"] == source_hash
    assert conversion["ego_map_mode"] == mode
    assert conversion["added_map_channels"] == (
        ["ego_gaussian", "ego_gaussian_cos_heading", "ego_gaussian_sin_heading"] if mode == "gaussian"
        else [f"cell_{'body' if mode == 'body_coordinates' else 'world'}_x_over_20000m",
              f"cell_{'body' if mode == 'body_coordinates' else 'world'}_y_over_20000m", "ego_gaussian"])
    original = torch.load(source, map_location="cpu", weights_only=False)
    assert original["model_state_dict"]["map_encoder.0.weight"].shape[1] == 3
    loaded = module("policy_tracking").NeuralPolicy.from_checkpoint(target)
    obs = {"vector": np.zeros(111, np.float32), "coverage_map": np.zeros((3, 64, 64), np.float32)}
    obs["vector"][2] = 1
    obs["vector"][94:98] = [-.5, .5, -.5, .5]
    assert np.isfinite(loaded.act(obs)).all()
    with pytest.raises(ValueError, match="different"):
        coverage.upgrade_coverage_checkpoint(source, source)
    with pytest.raises(ValueError, match="already"):
        coverage.upgrade_coverage_checkpoint(target, tmp_path / "invalid.pt", ego_map_mode=mode)
    # Overwriting or deleting the requested input cannot erase the original
    # three-channel model referenced by the converted checkpoint's provenance.
    source.write_bytes(b"production best checkpoint was replaced")
    assert snapshot.read_bytes() == source_bytes
    source.unlink()
    retained = module("policy_tracking").NeuralPolicy.from_checkpoint(snapshot)
    assert retained.model.map_encoder[0].in_channels == 3
    assert retained.checkpoint_metadata["sha256"] == source_hash
    assert hashlib.sha256(snapshot.read_bytes()).hexdigest() == source_hash
    # Content-addressed snapshots must never be silently overwritten if corrupt.
    source.write_bytes(source_bytes)
    snapshot.write_bytes(b"external corruption")
    with pytest.raises(ValueError, match="snapshot has been modified"):
        coverage.upgrade_coverage_checkpoint(source, tmp_path / "another.pt", ego_map_mode=mode)


@pytest.mark.parametrize("mode", ["body_coordinates", "world_coordinates"])
def test_dense_ego_coordinates_have_physical_scale_heading_boundaries_and_batch_axes(mode):
    train = module("train")
    torch.set_num_threads(1)
    model = train.create_policy("coverage", {"ego_map_channels": True, "ego_map_mode": mode})
    vector = torch.zeros(2, 3, 111)
    cases = [
        (-10000, 10000, -5000, 5000, 0),
        (-10000, 10000, -5000, 5000, math.pi / 2),
        (60000, 80000, -5000, 5000, 0),  # UAV far outside the ROI
        (0, 10000, 0, 20000, math.pi / 2),  # UAV on southwest boundary
        (-12000, 0, -8000, 0, math.pi),  # UAV on northeast boundary
        (-7500, 7500, -2500, 2500, math.pi / 4),
    ]
    for index, (left, right, bottom, top, angle) in enumerate(cases):
        value = vector[index // 3, index % 3]
        value[2:4] = torch.tensor([math.cos(angle), math.sin(angle)])
        value[94:98] = torch.tensor([left, right, bottom, top]) / 20000
    original = vector.clone()
    obs = {"vector": vector, "coverage_map": torch.zeros(2, 3, 3, 64, 64)}
    channels = model.build_ego_channels(obs)
    assert channels.shape == (2, 3, 3, 64, 64) and torch.isfinite(channels).all()
    for index, (left, right, bottom, top, angle) in enumerate(cases):
        for iy, ix in [(0, 0), (31, 32), (63, 63)]:
            dx = left + (ix + .5) / 64 * (right - left)
            dy = bottom + (iy + .5) / 64 * (top - bottom)
            expected = torch.tensor([(math.cos(angle) * dx + math.sin(angle) * dy) / 20000,
                                     (-math.sin(angle) * dx + math.cos(angle) * dy) / 20000]
                                    if mode == "body_coordinates" else [dx / 20000, dy / 20000])
            torch.testing.assert_close(channels[index // 3, index % 3, :2, iy, ix], expected,
                                       atol=1e-6, rtol=1e-6)
    assert channels[0, 2, 0].min() > 3, "outside-ROI coordinates must remain dense and unclipped"
    assert channels[0, 2, 2].max() < 1e-20
    gaussian = train.create_policy("coverage", {"ego_map_channels": True}).build_ego_channels(obs)
    torch.testing.assert_close(channels[:, :, 2], gaussian[:, :, 0], rtol=0, atol=0)
    actions, hidden = model(obs)
    assert actions.shape == (2, 3, 2) and hidden.shape == (1, 2, 128)
    torch.testing.assert_close(actions.norm(dim=-1), torch.ones(2, 3))
    torch.testing.assert_close(vector, original)


def test_legacy_gaussian_checkpoint_without_mode_retains_identical_inference(tmp_path):
    model = module("train").create_policy("coverage", {"ego_map_channels": True})
    config = dict(model.model_config)
    config.pop("ego_map_mode", None)
    checkpoint = dict(checkpoint_version="bc-v1", observation_schema="obs-v1", task="coverage",
                      model_config=config, model_state_dict=model.state_dict())
    path = tmp_path / "legacy-gaussian.pt"
    torch.save(checkpoint, path)
    policy_module = module("policy_tracking")
    loaded, direct = policy_module.NeuralPolicy.from_checkpoint(path), policy_module.NeuralPolicy(model)
    assert loaded.model.ego_map_mode == "gaussian"
    assert loaded.checkpoint_metadata["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    obs = {"vector": np.zeros(111, np.float32), "coverage_map": np.zeros((3, 64, 64), np.float32)}
    obs["vector"][2:4] = [.6, .8]
    obs["vector"][94:98] = [-.2, .3, -.4, .6]
    for _ in range(3):
        np.testing.assert_allclose(loaded.act(obs), direct.act(obs), atol=1e-7)


def test_coverage_ego_mode_rejects_unknown_coordinate_encoding():
    with pytest.raises(ValueError, match="ego_map_mode"):
        module("train").create_policy("coverage", {"ego_map_mode": "unsupported"})


@pytest.mark.parametrize("frame", ["world", "body"])
@pytest.mark.parametrize("ego_map", [False, True])
def test_optional_coverage_command_mask_removes_only_command_features(frame, ego_map):
    train = module("train")
    config = dict(feature_frame=frame, action_frame=frame, ego_map_channels=ego_map,
                  mask_previous_command=True, ego_map_mode="world_coordinates")
    model = train.create_policy("coverage", config)
    baseline = train.create_policy("coverage", {**config, "mask_previous_command": False})
    vector = torch.randn(2, 4, 111)
    vector[..., 2:4] = torch.tensor([.6, .8])
    vector[..., 94:98] = torch.tensor([-.2, .3, -.4, .6])
    original = vector.clone()
    transformed = model.preprocess_vector(vector)
    expected = baseline.preprocess_vector(vector).clone()
    expected[..., 8:10] = 0
    torch.testing.assert_close(transformed, expected)
    torch.testing.assert_close(transformed[..., 6:8], vector[..., 6:8])
    torch.testing.assert_close(vector, original)
    obs = dict(vector=vector, coverage_map=torch.rand(2, 4, 3, 64, 64))
    different = vector.clone()
    different[..., 8:10] = torch.tensor([0., -1.])
    first, hidden1 = model(obs)
    second, hidden2 = model({**obs, "vector": different})
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    torch.testing.assert_close(hidden1, hidden2, rtol=0, atol=0)
    assert model.model_config["mask_previous_command"] is True
    assert "mask_previous_command" not in train.create_policy("tracking").model_config


def test_masked_coverage_checkpoint_roundtrip_and_legacy_default(tmp_path):
    train = module("train")
    policy_module = module("policy_tracking")
    for masked in [False, True]:
        model = train.create_policy("coverage", {"mask_previous_command": masked})
        config = dict(model.model_config)
        if not masked:
            config.pop("mask_previous_command")  # checkpoints predating this option
        path = tmp_path / f"masked-{masked}.pt"
        torch.save(dict(checkpoint_version="bc-v1", observation_schema="obs-v1", task="coverage",
                        model_config=config, model_state_dict=model.state_dict()), path)
        loaded = policy_module.NeuralPolicy.from_checkpoint(path)
        direct = policy_module.NeuralPolicy(model)
        assert loaded.model.mask_previous_command is masked
        obs = {"vector": np.zeros(111, np.float32), "coverage_map": np.zeros((3, 64, 64), np.float32)}
        obs["vector"][2:4] = [.6, .8]
        obs["vector"][8:10] = [-1., 0.]
        for _ in range(3):
            np.testing.assert_allclose(loaded.act(obs), direct.act(obs), atol=1e-7)


def test_cli_command_mask_is_explicit_and_fresh_training_receives_it(tmp_path, monkeypatch):
    import main
    import sys
    assert main.parser().parse_args([]).coverage_mask_previous_command is False
    assert main.parser().parse_args(["--coverage-mask-previous-command"]).coverage_mask_previous_command is True
    assert main.parser().parse_args(["--no-coverage-mask-previous-command"]).coverage_mask_previous_command is False
    def capture(task, *args, **kwargs):
        assert task == "coverage" and kwargs["model_config"]["mask_previous_command"] is True
        assert kwargs["initial_checkpoint"] is None
        raise RuntimeError("captured fresh model config")
    monkeypatch.setattr(module("train"), "train_policy", capture)
    previous_cache = sys.pycache_prefix
    try:
        with pytest.raises(RuntimeError, match="captured fresh model config"):
            main.main(["--mode", "train", "--task", "coverage", "--output-dir", str(tmp_path),
                       "--device", "cpu", "--coverage-mask-previous-command", "--no-visualize"])
    finally:
        sys.pycache_prefix = previous_cache


def test_cli_warm_start_explicitly_inherits_legacy_command_features(tmp_path, monkeypatch, capsys):
    import main
    import sys
    checkpoint = tmp_path / "old.pt"
    torch.save({"model_config": {}}, checkpoint)
    def capture(task, *args, **kwargs):
        assert kwargs["model_config"].get("mask_previous_command", False) is False
        assert kwargs["initial_checkpoint"] == checkpoint
        raise RuntimeError("captured inherited config")
    monkeypatch.setattr(module("train"), "train_policy", capture)
    previous_cache = sys.pycache_prefix
    try:
        with pytest.raises(RuntimeError, match="captured inherited config"):
            main.main(["--mode", "train", "--task", "coverage", "--output-dir", str(tmp_path),
                       "--device", "cpu", "--no-visualize", "--warm-start",
                       "--checkpoint-path", str(checkpoint), "--coverage-mask-previous-command"])
    finally:
        sys.pycache_prefix = previous_cache
    assert "mask_previous_command=False inherited from checkpoint" in capsys.readouterr().out


def test_scene_groups_never_leak_and_dataset_uses_command_labels(tmp_path):
    datasets = module("datasets")
    for index, scene in enumerate(["repeat", "repeat", "b", "c", "d"]):
        episode(tmp_path / f"episode{index}.npz", "tracking", scene)
    split = datasets.split_episodes(tmp_path, "tracking", validation_fraction=0.4, seed=7)
    same = datasets.split_episodes(tmp_path, "tracking", validation_fraction=0.4, seed=7)
    assert split == same
    assert set(split["train_scene_ids"]).isdisjoint(split["validation_scene_ids"])
    assert set(split["train_paths"]).isdisjoint(split["validation_paths"])
    dataset = datasets.SequenceDataset(split["train_paths"], task="tracking")
    item = dataset[0]
    assert item["vector"].shape == (10, 134)
    np.testing.assert_allclose(item["actions"][0], [np.cos(.65), np.sin(.65)])
    batch = datasets.pad_episodes([item, {key: value[:3] for key, value in item.items()}])
    assert batch["mask"].sum().item() == 13
    assert batch["vector"].shape == (2, 10, 134)


def test_split_requires_two_independent_scenes(tmp_path):
    datasets = module("datasets")
    episode(tmp_path / "only.npz", "tracking", "one")
    with pytest.raises(ValueError, match="scene"):
        datasets.split_episodes(tmp_path, "tracking", validation_fraction=.2, seed=0)


def test_static_layout_group_overrides_different_initial_scene_ids(tmp_path):
    datasets = module("datasets")
    for index in range(6):
        path = tmp_path / f"episode{index}.npz"
        episode(path, "tracking", f"different-initial-{index}")
        with np.load(path, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data}
        metadata = json.loads(arrays["metadata"].item())
        metadata["scene_group_id"] = f"layout-{index // 2}"
        arrays["metadata"] = json.dumps(metadata)
        np.savez_compressed(path, **arrays)
    split = datasets.split_episodes(tmp_path, "tracking", validation_fraction=.34, seed=4)
    training = set(split["train_paths"])
    for index in range(0, 6, 2):
        assert (str(tmp_path / f"episode{index}.npz") in training) == (
            str(tmp_path / f"episode{index+1}.npz") in training)


def test_accepted_manifest_excludes_stale_unlisted_episode_caches(tmp_path):
    datasets = module("datasets")
    paths = []
    for index in range(4):
        path = tmp_path / f"episode{index}.npz"
        episode(path, "tracking", f"scene{index}")
        paths.append(str(path))
    episode(tmp_path / "stale-other-task.npz", "coverage", "stale")
    manifest = tmp_path / "accepted-manifest.json"
    manifest.write_text(json.dumps(dict(paths=paths, task="tracking", complete=True)), encoding="utf-8")
    split = datasets.split_episodes(manifest, "tracking", validation_fraction=.25, seed=0)
    assert set(split["train_paths"] + split["validation_paths"]) == set(paths)
    assert split["source_manifest"] == str(manifest.resolve())


@pytest.mark.parametrize("defect", ["incomplete", "duplicate", "missing"])
def test_invalid_accepted_manifest_is_rejected(tmp_path, defect):
    datasets = module("datasets")
    path = tmp_path / "episode.npz"
    episode(path, "tracking", "scene")
    value = dict(paths=[str(path)], complete=True)
    if defect == "incomplete":
        value["complete"] = False
    elif defect == "duplicate":
        value["paths"] *= 2
    else:
        value["paths"] = [str(tmp_path / "not-found.npz")]
    manifest = tmp_path / "accepted-manifest.json"
    manifest.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match=defect):
        datasets.split_episodes(manifest, "tracking", validation_fraction=0, seed=0)


def test_training_normalization_does_not_amplify_narrow_physical_states(tmp_path):
    train = module("train")
    datasets = module("datasets")
    path = tmp_path / "episode.npz"
    episode(path, "tracking", "scene")
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data}
    arrays["vector"][:, 6] = np.arange(11) * .0001
    arrays["vector"][:, 7] = .2
    np.savez_compressed(path, **arrays)
    dataset = datasets.SequenceDataset([path], task="tracking")
    _, std = train._fit_vector_normalization(dataset, train.create_policy("tracking"), minimum_std=.1)
    assert std[6] == pytest.approx(.1)
    assert std[7] == pytest.approx(1.), "exactly constant physical fields retain neutral scaling"


def test_warm_start_retains_checkpoint_weights_and_normalization(tmp_path):
    train = module("train")
    data = tmp_path / "data"
    data.mkdir()
    for index in range(4):
        episode(data / f"ep{index}.npz", "tracking", f"scene{index}", sign=(-1) ** index)
    first = train.train_policy("tracking", data, tmp_path / "first", tmp_path / "logs1",
                               epochs=1, batch_size=2, sequence_length=5, seed=4,
                               model_config={"hidden_size": 32})
    checkpoint = torch.load(first["best_checkpoint"], map_location="cpu", weights_only=False)
    # A changed collection distribution would alter fitted stats. Warm-start
    # must keep the coordinate system in which the existing weights were learned.
    for path in data.glob("*.npz"):
        with np.load(path, allow_pickle=False) as arrays:
            values = {key: arrays[key] for key in arrays}
        values["vector"][:, 7] += .4
        np.savez_compressed(path, **values)
    dataset = module("datasets").SequenceDataset(first["split"]["validation_paths"], task="tracking")
    expected = train.create_policy("tracking", checkpoint["model_config"])
    expected.load_state_dict(checkpoint["model_state_dict"])
    baseline = train._run_epoch(expected, dataset, 2, 5, torch.device("cpu"))
    result = train.train_policy("tracking", data, tmp_path / "warm", tmp_path / "logs2",
                                epochs=1, batch_size=2, sequence_length=5, seed=4,
                                initial_checkpoint=first["best_checkpoint"], learning_rate=1e-8)
    assert result["initial_validation_mse"] == pytest.approx(baseline["action_mse"], abs=1e-8)
    last = torch.load(result["last_checkpoint"], map_location="cpu", weights_only=False)
    for key in ["vector_mean", "vector_std"]:
        torch.testing.assert_close(last["model_state_dict"][key], checkpoint["model_state_dict"][key])
    assert result["model_config"]["hidden_size"] == 32
    assert last["normalization"]["fit_subset"] == "initial_checkpoint"
    assert last["train_config"]["initial_checkpoint"] == result["initial_checkpoint"]
    assert Path(result["initial_checkpoint"]).read_bytes() == Path(first["best_checkpoint"]).read_bytes()
    # The new Adam has only this new epoch's update count, not the source run's.
    assert max(float(value["step"]) for value in last["optimizer_state_dict"]["state"].values()) == 4
    with pytest.raises(ValueError, match="config"):
        train.train_policy("tracking", data, tmp_path / "bad", tmp_path / "logs3",
                           epochs=1, initial_checkpoint=first["best_checkpoint"],
                           model_config={"hidden_size": 64})
    checkpoint["task"] = "coverage"
    wrong = tmp_path / "wrong-task.pt"
    torch.save(checkpoint, wrong)
    with pytest.raises(ValueError, match="task"):
        train.train_policy("tracking", data, tmp_path / "bad-task", tmp_path / "logs4",
                           epochs=1, initial_checkpoint=wrong)
    checkpoint["task"] = "tracking"
    checkpoint["observation_schema"] = "unknown"
    torch.save(checkpoint, wrong)
    with pytest.raises(ValueError, match="schema"):
        train.train_policy("tracking", data, tmp_path / "bad-schema", tmp_path / "logs5",
                           epochs=1, initial_checkpoint=wrong)
    checkpoint["observation_schema"] = "obs-v1"
    checkpoint["model_state_dict"]["action_head.2.weight"] = torch.zeros(3, 64)
    torch.save(checkpoint, wrong)
    with pytest.raises(ValueError, match="weights"):
        train.train_policy("tracking", data, tmp_path / "bad-shape", tmp_path / "logs6",
                           epochs=1, initial_checkpoint=wrong)
    checkpoint["model_state_dict"]["action_head.2.weight"] = torch.zeros(2, 64)
    checkpoint["dataset_split"]["train_scene_group_ids"].append(
        checkpoint["dataset_split"]["validation_scene_group_ids"][0])
    torch.save(checkpoint, wrong)
    with pytest.raises(ValueError, match="overlap"):
        train.train_policy("tracking", data, tmp_path / "bad-overlap", tmp_path / "logs7",
                           epochs=1, initial_checkpoint=wrong)


def test_warm_start_pins_scene_groups_when_seed_changes_and_data_is_added(tmp_path):
    train = module("train")
    datasets = module("datasets")
    data = tmp_path / "data"
    data.mkdir()
    for index in range(4):
        episode(data / f"old{index}.npz", "tracking", f"oldscene{index}", length=3)
    first = train.train_policy("tracking", data, tmp_path / "first", tmp_path / "logs1",
                               epochs=1, seed=4, batch_size=2)
    original_train = set(first["split"]["train_scene_group_ids"])
    original_val = set(first["split"]["validation_scene_group_ids"])
    assert set(datasets.split_episodes(data, "tracking", seed=0)["validation_scene_group_ids"]) != original_val
    for index in range(2):
        episode(data / f"new{index}.npz", "tracking", f"newscene{index}", length=3)
    # Another initial condition of a historical validation layout stays validation.
    duplicate = data / "repeat-val.npz"
    episode(duplicate, "tracking", "different-initial-condition", length=3)
    with np.load(duplicate, allow_pickle=False) as values:
        arrays = {key: values[key] for key in values}
    meta = json.loads(arrays["metadata"].item())
    meta["scene_group_id"] = next(iter(original_val))
    arrays["metadata"] = json.dumps(meta)
    np.savez_compressed(duplicate, **arrays)
    warm = train.train_policy("tracking", data, tmp_path / "warm", tmp_path / "logs2",
                              epochs=1, seed=0, batch_size=2,
                              initial_checkpoint=first["best_checkpoint"])
    training = set(warm["split"]["train_scene_group_ids"])
    validation = set(warm["split"]["validation_scene_group_ids"])
    assert original_train <= training and original_val <= validation
    assert training.isdisjoint(validation)
    assert str(duplicate.resolve()) in warm["split"]["validation_paths"]
    assert training | validation == original_train | original_val | {"newscene0", "newscene1"}


def test_warm_start_keeps_immutable_hashed_input_when_overwriting_same_best(tmp_path):
    train = module("train")
    data = tmp_path / "data"
    data.mkdir()
    for index in range(3):
        episode(data / f"ep{index}.npz", "tracking", f"scene{index}", length=3)
    models = tmp_path / "models"
    first = train.train_policy("tracking", data, models, tmp_path / "logs1", epochs=1, batch_size=2)
    source = Path(first["best_checkpoint"])
    original_bytes = source.read_bytes()
    original_hash = hashlib.sha256(original_bytes).hexdigest()
    warm = train.train_policy("tracking", data, models, tmp_path / "logs2", epochs=1, batch_size=2,
                              initial_checkpoint=source)
    snapshot = Path(warm["initial_checkpoint"])
    assert snapshot != source, "warm-start provenance cannot reference its overwritten output"
    assert snapshot.read_bytes() == original_bytes
    assert warm["initial_checkpoint_sha256"] == original_hash
    assert hashlib.sha256(snapshot.read_bytes()).hexdigest() == original_hash
    assert source.read_bytes() != original_bytes
    last = torch.load(warm["last_checkpoint"], map_location="cpu", weights_only=False)
    assert last["initial_checkpoint"] == str(snapshot)
    assert last["initial_checkpoint_sha256"] == original_hash
    assert last["normalization"]["source_checkpoint"] == str(snapshot)
    assert last["train_config"]["initial_checkpoint"] == str(snapshot)
    original_source = tmp_path / "original-input.pt"
    original_source.write_bytes(original_bytes)
    snapshot.write_bytes(b"external corruption")
    with pytest.raises(ValueError, match="snapshot has been modified"):
        train.train_policy("tracking", data, models, tmp_path / "logs3", epochs=1,
                           initial_checkpoint=original_source)


def test_training_prints_epoch_losses_and_validation_mse(tmp_path, capsys):
    train = module("train")
    data = tmp_path / "data"
    data.mkdir()
    for index in range(2):
        episode(data / f"ep{index}.npz", "tracking", f"scene{index}", length=2)
    result = train.train_policy("tracking", data, tmp_path / "models", tmp_path / "logs", epochs=2,
                                correction_weight_max=4, correction_threshold=.1)
    assert result["train_config"]["correction_weight_max"] == 4
    checkpoint = torch.load(result["last_checkpoint"], map_location="cpu", weights_only=False)
    assert checkpoint["loss_definition"]["validation"] == "unweighted"
    output = capsys.readouterr().out
    for epoch in [1, 2]:
        lines = [line for line in output.splitlines() if f"epoch={epoch}/2" in line]
        assert len(lines) == 1
        assert all(key in lines[0] for key in ["tracking", "train_loss=", "val_loss=", "val_mse="])


class _AngleTablePolicy(torch.nn.Module):
    """Independent unit-direction parameters make loss-gradient ratios measurable."""

    def __init__(self):
        super().__init__()
        self.angles = torch.nn.Parameter(torch.zeros(2))

    def forward(self, obs, hidden=None):
        angle = self.angles[obs["vector"][..., 0].long()]
        action = torch.stack([angle.cos(), angle.sin()], dim=-1)
        return action, torch.zeros(1, len(angle), 1)


def _correction_episode(first_target_matches=False):
    vector = torch.zeros(2, 134)
    vector[:, 0] = torch.tensor([0., 1.])
    vector[:, 2] = 1
    vector[0, 8:10] = torch.tensor([1., 0.]) if first_target_matches else torch.tensor([0., 1.])
    vector[1, 8:10] = torch.tensor([1., 0.])
    actions = torch.tensor([[1., 0.], [0., 1.]]) if first_target_matches else torch.tensor([[0., 1.], [0., 1.]])
    return dict(vector=vector, actions=actions)


def test_correction_weights_increase_only_corrective_gradient_relative_to_other_steps():
    train = module("train")
    item = _correction_episode()
    plain, weighted = _AngleTablePolicy(), _AngleTablePolicy()
    train._run_epoch(plain, [item], 1, 8, torch.device("cpu"),
                     optimizer=torch.optim.SGD(plain.parameters(), lr=.1), grad_clip=100)
    train._run_epoch(weighted, [item], 1, 8, torch.device("cpu"),
                     optimizer=torch.optim.SGD(weighted.parameters(), lr=.1), grad_clip=100,
                     correction_weight_max=4, correction_threshold=.1)
    torch.testing.assert_close(plain.angles, torch.tensor([.1, .1]))
    torch.testing.assert_close(weighted.angles, torch.tensor([.04, .16]))


def test_weighted_loss_excludes_padding_and_reports_raw_action_metrics_and_validation():
    train = module("train")
    item = _correction_episode(first_target_matches=True)
    short = {key: value[:1] for key, value in item.items()}
    model = _AngleTablePolicy()
    # Two chunks test aggregation by total valid weight, rather than averaging
    # chunk losses with different correction counts or counting tail padding.
    trained = train._run_epoch(model, [item, short], 2, 1, torch.device("cpu"),
                               optimizer=torch.optim.SGD(model.parameters(), lr=0), grad_clip=100,
                               correction_weight_max=4, correction_threshold=.1)
    assert trained["loss"] == pytest.approx(4 / 3)
    assert trained["action_mse"] == pytest.approx(1 / 3)
    assert trained["action_mae"] == pytest.approx(1 / 3)
    assert trained["steps"] == 3
    validation = train._run_epoch(model, [item, short], 2, 1, torch.device("cpu"),
                                  correction_weight_max=4, correction_threshold=.1)
    assert validation["loss"] == pytest.approx(2 / 3)
    assert validation["action_mse"] == pytest.approx(trained["action_mse"])
    assert validation["action_mae"] == pytest.approx(trained["action_mae"])


def test_correction_weight_one_is_exactly_the_original_unweighted_behavior():
    train = module("train")
    first, second = _AngleTablePolicy(), _AngleTablePolicy()
    item = _correction_episode(first_target_matches=True)
    original = train._run_epoch(first, [item], 1, 8, torch.device("cpu"),
                                optimizer=torch.optim.SGD(first.parameters(), lr=.1), grad_clip=100)
    disabled = train._run_epoch(second, [item], 1, 8, torch.device("cpu"),
                                optimizer=torch.optim.SGD(second.parameters(), lr=.1), grad_clip=100,
                                correction_weight_max=1, correction_threshold=0)
    assert original == disabled
    assert torch.equal(first.angles, second.angles)


@pytest.mark.parametrize("options", [{"correction_weight_max": .5}, {"correction_weight_max": float("inf")},
                                      {"correction_threshold": -.1}, {"correction_threshold": float("nan")}])
def test_invalid_correction_weight_configuration_rejected_before_loading_data(tmp_path, options):
    with pytest.raises(ValueError, match="correction"):
        module("train").train_policy("tracking", tmp_path / "absent", tmp_path / "models", tmp_path / "logs",
                                     epochs=1, **options)


@pytest.mark.parametrize("task", ["tracking", "coverage"])
def test_temporal_behavior_cloning_learns_and_checkpoint_roundtrips(tmp_path, task):
    train = module("train")
    policy_module = module("policy_tracking")
    data = tmp_path / "data"
    data.mkdir()
    for index in range(12):
        episode(data / f"episode{index:02}.npz", task, f"scene{index}", sign=1 if index % 2 else -1)
    result = train.train_policy(task, data, tmp_path / "models", tmp_path / "logs",
                                epochs=18, batch_size=4, sequence_length=5,
                                learning_rate=.003, validation_fraction=.25,
                                seed=3, num_threads=1)
    assert result["best_validation_mse"] < .7 * result["initial_validation_mse"]
    assert len(result["history"]) == 18
    checkpoint = torch.load(result["best_checkpoint"], map_location="cpu", weights_only=False)
    assert checkpoint["task"] == task
    assert checkpoint["observation_schema"] == "obs-v1"
    assert checkpoint["normalization"] and checkpoint["optimizer_state_dict"]
    loaded = policy_module.NeuralPolicy.from_checkpoint(result["best_checkpoint"])
    original = train.create_policy(task, checkpoint["model_config"])
    original.load_state_dict(checkpoint["model_state_dict"])
    direct = policy_module.NeuralPolicy(original)
    dimension = 111 if task == "coverage" else 134
    obs = {"vector": np.zeros(dimension, np.float32)}
    obs["vector"][2] = 1
    if task == "coverage":
        obs["coverage_map"] = np.zeros((3, 64, 64), np.float32)
    for _ in range(3):
        np.testing.assert_allclose(loaded.act(obs), direct.act(obs), atol=1e-7)
    loaded.reset()
    direct.reset()
    np.testing.assert_allclose(loaded.act(obs), direct.act(obs), atol=1e-7)
    assert (tmp_path / "models" / "dataset_split.json").exists()
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    accumulator = EventAccumulator(str(tmp_path / "logs")).Reload()
    assert set(accumulator.Tags()["scalars"]) == {
        f"{subset}/{metric}" for subset in ["Train", "Validation"]
        for metric in ["Loss", "Action_MSE", "Action_MAE"]}


def test_two_task_models_do_not_share_parameters():
    train = module("train")
    coverage, tracking = train.create_policy("coverage"), train.create_policy("tracking")
    assert not {p.data_ptr() for p in coverage.parameters()} & {p.data_ptr() for p in tracking.parameters()}
