"""Field-v2 protocol isolation, causal imitation and deployable checkpoints."""

import json

import numpy as np
import pytest
import torch

from src import datasets, train
from src.policy_tracking import load_policy


def episode(path, task="coverage", scene="layout", length=5, sign=1):
    vector = np.zeros((length + 1, 111 if task == "coverage" else 134), np.float32)
    vector[:, 2] = 1
    potential = np.zeros((length + 1, 4, 33, 33), np.float32)
    potential[:, 0, :, 17:] = sign
    potential[:, 3] = 1
    context = np.zeros((length + 1, 16), np.float32)
    context[:, 7] = sign
    actions = np.tile([np.cos(.4), sign * np.sin(.4), sign * .6], (length, 1)).astype(np.float32)
    metadata = dict(task=task, scene_id=scene, scene_group_id=scene,
                    dataset_version="bc-dataset-field-v2", observation_schema="obs-field-v2",
                    action_schema="action-field-v2")
    np.savez_compressed(path, vector=vector, potential=potential, field_context=context,
                        actions=actions, executed_actions=actions[:, :2], metadata=json.dumps(metadata))


def observation(task="coverage", batch=2, time=7):
    vector = torch.randn(batch, time, 111 if task == "coverage" else 134)
    vector[..., 2:4] = torch.tensor([1., 0.])
    return dict(vector=vector, potential=torch.randn(batch, time, 4, 33, 33),
                field_context=torch.randn(batch, time, 16))


@pytest.mark.parametrize("task", ["coverage", "tracking"])
def test_field_factory_causality_unit_xy_pan_and_field_gradients(task):
    torch.set_num_threads(1)
    model = train.create_policy(task, {"potential_field": True, "hidden_size": 32,
                                       "ego_map_channels": True, "feature_frame": "body"})
    assert model.__class__.__name__ == "FieldPolicy"
    obs = observation(task)
    full, memory = model(obs)
    first, hidden = model({key: value[:, :3] for key, value in obs.items()})
    last, hidden = model({key: value[:, 3:] for key, value in obs.items()}, hidden)
    assert full.shape == (2, 7, 3) and memory.shape == (1, 2, 32)
    torch.testing.assert_close(torch.cat([first, last], 1), full)
    torch.testing.assert_close(hidden, memory)
    torch.testing.assert_close(full[..., :2].norm(dim=-1), torch.ones(2, 7))
    assert (full[..., 2].abs() <= 1).all()
    (full - torch.tensor([0., 1., .8])).square().mean().backward()
    assert model.field_encoder[0].weight.grad.abs().sum() > 0
    assert model.pan_head[-1].weight.grad.abs().sum() > 0
    changed = {**obs, "potential": torch.zeros_like(obs["potential"])}
    assert not torch.allclose(model(changed)[0], full)


def test_field_dataset_rejects_protocol_mixing_and_preserves_pan_labels(tmp_path):
    path = tmp_path / "field.npz"
    episode(path)
    data = datasets.SequenceDataset([path], task="coverage", observation_schema="obs-field-v2")
    assert data.action_dim == 3
    item = data[0]
    assert set(item) == {"vector", "potential", "field_context", "actions"}
    assert item["actions"][0, 2] == pytest.approx(.6)
    with pytest.raises(ValueError, match="schema|protocol"):
        datasets.SequenceDataset([path], task="coverage", observation_schema="obs-v1")[0]
    with np.load(path, allow_pickle=False) as values:
        arrays = dict(values)
    meta = json.loads(arrays["metadata"].item())
    meta.pop("dataset_version")
    arrays["metadata"] = json.dumps(meta)
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="version"):
        data[0]


def test_three_component_metrics_exclude_tail_padding_and_keep_recurrent_state():
    class Constant(torch.nn.Module):
        def forward(self, obs, hidden=None):
            batch, time = obs["vector"].shape[:2]
            initial = torch.zeros(1, batch, 1) if hidden is None else hidden
            self.states.append(initial.clone())
            return torch.tensor([1., 0., 0.]).expand(batch, time, 3), initial + time

    model = Constant()
    model.states = []
    item = dict(vector=torch.zeros(5, 111), potential=torch.zeros(5, 4, 33, 33),
                field_context=torch.zeros(5, 16), actions=torch.tensor([1., 0., 1.]).expand(5, 3))
    short = {key: value[:2] for key, value in item.items()}
    result = train._run_epoch(model, [item, short], 2, 2, torch.device("cpu"))
    assert result["loss"] == pytest.approx(1)
    assert result["action_mse"] == pytest.approx(1 / 3)
    assert result["action_mae"] == pytest.approx(1 / 3)
    assert result["steps"] == 7
    assert [float(hidden[0, 0, 0]) for hidden in model.states] == [0, 2, 4]


@pytest.mark.parametrize("task", ["coverage", "tracking"])
def test_field_training_roundtrip_scene_split_and_exact_six_tensorboard_tags(tmp_path, task):
    folder = tmp_path / "data"
    folder.mkdir()
    for index in range(8):
        episode(folder / f"{index}.npz", task, f"layout{index // 2}",
                length=3 + index % 3, sign=(-1) ** index)
    result = train.train_policy(task, folder, tmp_path / "models", tmp_path / "tf",
                                epochs=4, batch_size=2, sequence_length=2, learning_rate=.003,
                                model_config={"potential_field": True, "hidden_size": 32}, seed=9)
    assert result["best_validation_mse"] < result["initial_validation_mse"]
    split = result["split"]
    assert set(split["train_scene_group_ids"]).isdisjoint(split["validation_scene_group_ids"])
    state = torch.load(result["best_checkpoint"], weights_only=False)
    assert state["checkpoint_version"] == "bc-field-v2"
    assert state["observation_schema"] == "obs-field-v2" and state["action_schema"] == "action-field-v2"
    loaded = load_policy(result["best_checkpoint"])
    from src.policy_field import FieldNeuralPolicy
    direct_model = train.create_policy(task, state["model_config"])
    direct_model.load_state_dict(state["model_state_dict"])
    direct = FieldNeuralPolicy(direct_model)
    obs = {key: value[0, 0].numpy() for key, value in observation(task, 1, 1).items()}
    for _ in range(3):
        np.testing.assert_allclose(loaded.act(obs), direct.act(obs), atol=1e-7)
    loaded.reset()
    direct.reset()
    np.testing.assert_allclose(loaded.act(obs), direct.act(obs), atol=1e-7)
    with pytest.raises(ValueError, match="potential|schema|protocol"):
        loaded.act({"vector": obs["vector"]})
    state["action_schema"] = "action-v1"
    invalid = tmp_path / "mismatch.pt"
    torch.save(state, invalid)
    with pytest.raises(ValueError, match="schema"):
        load_policy(invalid)
    warm = train.train_policy(task, folder, tmp_path / "warm", tmp_path / "tf-warm",
                              epochs=1, batch_size=2, sequence_length=2,
                              initial_checkpoint=result["best_checkpoint"], learning_rate=1e-8,
                              seed=12)
    assert warm["model_config"]["potential_field"] is True
    assert warm["initial_validation_mse"] == pytest.approx(result["best_validation_mse"], abs=1e-8)
    assert warm["split"]["train_scene_group_ids"] == split["train_scene_group_ids"]
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    tags = EventAccumulator(str(tmp_path / "tf")).Reload().Tags()["scalars"]
    assert set(tags) == {f"{subset}/{metric}" for subset in ["Train", "Validation"]
                         for metric in ["Loss", "Action_MSE", "Action_MAE"]}


def test_field_and_legacy_warm_start_cannot_cross_protocol(tmp_path):
    legacy = train.create_policy("tracking")
    path = tmp_path / "v1.pt"
    torch.save(dict(checkpoint_version="bc-v1", observation_schema="obs-v1", task="tracking",
                    model_config=legacy.model_config, model_state_dict=legacy.state_dict()), path)
    from src.policy_field import FieldNeuralPolicy
    with pytest.raises(ValueError, match="field|protocol"):
        FieldNeuralPolicy.from_checkpoint(path)
    with pytest.raises(ValueError, match="protocol|schema|config"):
        train.train_policy("tracking", tmp_path / "absent", tmp_path / "models", tmp_path / "tf",
                           epochs=1, initial_checkpoint=path, model_config={"potential_field": True})


def test_factory_accepts_named_model_config_and_never_shares_task_weights():
    coverage = train.create_policy("coverage", model_config={"potential_field": True})
    tracking = train.create_policy("tracking", model_config={"potential_field": True})
    assert not {parameter.data_ptr() for parameter in coverage.parameters()} & {
        parameter.data_ptr() for parameter in tracking.parameters()}
    original = observation("coverage", 1, 2)
    modified = {**original, "vector": original["vector"].clone()}
    modified["vector"][..., 8:10] = 9
    torch.testing.assert_close(coverage(original)[0], coverage(modified)[0], atol=0, rtol=0)


def test_field_protocol_requires_explicit_observation_schema_even_with_other_v2_tags(tmp_path):
    path = tmp_path / "field.npz"
    episode(path)
    with np.load(path, allow_pickle=False) as data:
        arrays = dict(data)
    meta = json.loads(arrays["metadata"].item())
    meta.pop("observation_schema")
    arrays["metadata"] = json.dumps(meta)
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="schema|protocol"):
        datasets.split_episodes(tmp_path, "coverage", validation_fraction=0,
                                observation_schema="obs-field-v2")


def test_body_field_frame_rotates_goal_raster_and_action_with_observed_heading():
    model = train.create_policy("tracking", {"potential_field": True, "field_frame": "body"})
    obs = observation("tracking", 1, 1)
    obs["vector"].zero_()
    obs["vector"][..., 3] = 1  # aircraft points north
    obs["field_context"].zero_()
    obs["field_context"][..., 7] = 2  # public goal is east
    obs["potential"].zero_()
    obs["potential"][0, 0, 0, 16, 24] = 1
    original = {key: value.clone() for key, value in obs.items()}
    torch.testing.assert_close(model.preprocess_context(obs)[..., 7:9], torch.tensor([[[0., -2.]]]))
    raster = model.preprocess_field(obs)
    assert raster[0, 0, 0, 8, 16] == pytest.approx(1., abs=1e-6)
    assert raster[0, 0, 0].sum() == pytest.approx(1., abs=1e-5)
    physical = model.preprocess_vector(obs["vector"])
    torch.testing.assert_close(physical[..., 2:4], torch.tensor([[[1., 0.]]]))
    with torch.no_grad():
        for parameter in model.action_head.parameters():
            parameter.zero_()
    action, _ = model(obs)
    torch.testing.assert_close(action[..., :2], obs["vector"][..., 2:4], atol=1e-6, rtol=1e-6)
    assert model.model_config["field_frame"] == "body"
    for key in obs:
        torch.testing.assert_close(obs[key], original[key], atol=0, rtol=0)


def test_pan_residual_holds_observed_physical_pan_and_allows_reversal():
    model = train.create_policy("tracking", {"potential_field": True, "field_frame": "body",
                                           "pan_residual": True})
    obs = observation("tracking", 1, 1)
    obs["field_context"][..., 3] = .2
    obs["field_context"][..., 5] = .4
    with torch.no_grad():
        for parameter in model.pan_head.parameters():
            parameter.zero_()
    action, _ = model(obs)
    assert action[0, 0, 2].item() == pytest.approx(.5)
    with torch.no_grad():
        model.pan_head[-1].bias.fill_(-3)
    assert model(obs)[0][0, 0, 2].item() < -.95
    assert model.model_config["pan_residual"] is True


def test_tracking_goal_pan_reference_uses_common_geometry_not_accumulated_actual_pan():
    model = train.create_policy("tracking", {"potential_field": True, "field_frame": "body",
                                           "pan_residual": True, "pan_reference": "goal"})
    obs = observation("tracking", 1, 1)
    obs["vector"].zero_()
    obs["vector"][..., 2] = 1
    obs["field_context"].zero_()
    obs["field_context"][..., 8] = 1  # target north of an east-facing UAV
    obs["field_context"][..., 5] = 2 / 3  # mechanical limit +/-120deg
    obs["field_context"][..., 3] = -.3
    with torch.no_grad():
        for parameter in model.pan_head.parameters():
            parameter.zero_()
    action, _ = model(obs)
    assert action[0, 0, 2].item() == pytest.approx(.75)
    obs["field_context"][..., 3] = .3
    assert model(obs)[0][0, 0, 2].item() == pytest.approx(.75)
    with torch.no_grad():
        model.pan_head[-1].bias.fill_(-5)
    assert model(obs)[0][0, 0, 2].item() < -.99
    assert model.model_config["pan_reference"] == "goal"


def test_coverage_goal_orbit_reference_approaches_public_goal_and_turns_clockwise():
    model = train.create_policy("coverage", {"potential_field": True, "field_frame": "body",
        "direction_reference": "goal_orbit", "reference_orbit_radius": 1700., "reference_radial_gain": 2.5})
    obs = observation("coverage", 1, 1)
    obs["vector"].zero_()
    obs["vector"][..., 2] = 1
    obs["vector"][..., 4] = 1
    obs["vector"][..., 98] = 1  # speed/reference_speed
    obs["vector"][..., 101] = 1  # roll time constant
    obs["vector"][..., 103] = 1  # heading time constant
    obs["vector"][..., 105] = 1  # decision dt
    obs["field_context"].zero_()
    obs["field_context"][..., 7] = 2  # target6000m east
    obs["field_context"][..., 10] = 1
    with torch.no_grad():
        for parameter in model.action_head.parameters():
            parameter.zero_()
    action, _ = model(obs)
    assert action[0, 0, 0] > .98
    assert action[0, 0, 1] > 0
    obs["field_context"][..., 7] = (-1700. + (220/3.6)*2.5) / 3000.
    action, _ = model(obs)
    torch.testing.assert_close(action[0, 0, :2], torch.tensor([0., -1.]), atol=1e-5, rtol=1e-5)
    obs["field_context"][..., 10] = 0
    torch.testing.assert_close(model(obs)[0][0, 0, :2], torch.tensor([1., 0.]), atol=1e-5, rtol=1e-5)
    assert model.model_config["direction_reference"] == "goal_orbit"


def test_coverage_goal_pan_reference_accounts_for_right_mount_and_mechanical_limits():
    model = train.create_policy("coverage", {"potential_field": True, "pan_residual": True,
                                           "pan_reference": "goal"})
    obs = observation("coverage", 1, 1)
    obs["vector"].zero_()
    obs["vector"][..., 2] = 1
    obs["field_context"].zero_()
    obs["field_context"][..., 7] = 1
    obs["field_context"][..., 4] = -.5
    obs["field_context"][..., 5] = 1/3
    with torch.no_grad():
        for parameter in model.pan_head.parameters():
            parameter.zero_()
    assert model(obs)[0][0, 0, 2].item() == pytest.approx(.99)
    obs["field_context"][..., 7] = 0
    obs["field_context"][..., 8] = -1
    assert model(obs)[0][0, 0, 2].item() == pytest.approx(0., abs=1e-6)


def test_tracking_orbit_reference_preserves_fixed_speed_and_uses_observed_target_motion_radius():
    model = train.create_policy("tracking", {"potential_field": True, "field_frame": "body",
        "direction_reference": "goal_orbit", "reference_orbit_radius": 1700., "reference_radial_gain": 3.})
    obs = observation("tracking", 1, 1)
    obs["vector"].zero_()
    obs["vector"][..., 2] = 1
    obs["vector"][..., 4] = 1
    obs["vector"][..., 106] = 1
    speed = 220/3.6
    obs["vector"][..., 97] = 5/speed
    obs["vector"][..., 131] = 2000/3000
    obs["field_context"].zero_()
    obs["field_context"][..., 7] = -2000/3000
    obs["field_context"][..., 10] = 1
    with torch.no_grad():
        for parameter in model.action_head.parameters():
            parameter.zero_()
    action, _ = model(obs)
    expected = torch.tensor([5/speed, -np.sqrt(speed**2-25)/speed], dtype=torch.float32)
    torch.testing.assert_close(action[0, 0, :2], expected, atol=1e-6, rtol=1e-6)
    obs["vector"][..., 131] = 1  # current2000m should move outward toward observed3000m
    adjusted, _ = model(obs)
    assert adjusted[0, 0, 0] > .7
    torch.testing.assert_close(adjusted[..., :2].norm(dim=-1), torch.ones(1, 1))
