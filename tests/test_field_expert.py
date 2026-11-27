"""Public-observation guidance, field sensitivity and physical safety checks."""
import copy
import math

import numpy as np
import pytest

from config import Config


def observation(task='coverage'):
    v = np.zeros(111 if task == 'coverage' else 134, dtype=np.float32)
    v[:4] = [.5, .5, 1., 0.]
    v[4] = 1.
    v[10:14] = .5
    if task == 'coverage':
        v[94:98] = [-.5, .5, -.5, .5]
    else:
        v[94:97] = [0., -2/3, 2/3]
        v[97] = Config().ship_speed / (220/3.6)
    radius = 6000.
    axis = np.linspace(-radius, radius, 33)
    xx, yy = np.meshgrid(axis, axis)
    potential = np.zeros((4, 33, 33), dtype=np.float32)
    potential[0:2] = 1.
    potential[3] = np.hypot(xx, yy) <= radius
    context = np.zeros(16, dtype=np.float32)
    context[[0, 1, 2, 6, 9, 10, 11, 12, 13]] = [2., 1., 3., math.pi/6, 1., 1., 1., 1., 10.]
    context[4] = -.5 if task == 'coverage' else 0.
    context[5] = 1/3 if task == 'coverage' else 2/3
    context[7:9] = [0., -1700/3000 if task == 'coverage' else -2/3]
    context[15] = task == 'tracking'
    return {'vector':v, 'potential':potential, 'field_context':context}


@pytest.mark.parametrize('task', ['coverage', 'tracking'])
def test_experts_return_world_unit_direction_and_explicit_normalized_pan(task):
    from expert_field import FieldCoverageExpert, FieldTrackingExpert
    expert = (FieldCoverageExpert if task == 'coverage' else FieldTrackingExpert)(Config())
    action = expert.act(observation(task))
    assert action.shape == (3,)
    assert action.dtype == np.float32
    assert np.all(np.isfinite(action))
    assert np.linalg.norm(action[:2]) == pytest.approx(1., abs=1e-6)
    assert abs(action[2]) <= 1.


def test_coverage_prefers_positive_radar_pixels_over_recently_scanned_pixels():
    from expert_field import FieldCoverageExpert
    left, right = observation(), observation()
    axis = np.linspace(-6000, 6000, 33)
    xx, yy = np.meshgrid(axis, axis)
    left['potential'][0] = np.where(xx < -500, 1., -1.)
    right['potential'][0] = np.where(xx > 500, 1., -1.)
    a = FieldCoverageExpert(Config()).act(left)
    b = FieldCoverageExpert(Config()).act(right)
    assert a[2] < b[2] - .05


def test_coverage_public_goal_changes_guidance_in_uniform_local_field():
    from expert_field import FieldCoverageExpert
    a, b = observation(), observation()
    a['field_context'][7:9] = [2., 0.]
    b['field_context'][7:9] = [-2., 0.]
    first = FieldCoverageExpert(Config()).act(a)
    second = FieldCoverageExpert(Config()).act(b)
    assert np.linalg.norm(first[:2]-second[:2]) > .5


def test_scan_preference_integral_changes_coverage_motion_without_hard_obstacles():
    from expert_field import FieldCoverageExpert
    north, south = observation(), observation()
    yy = np.meshgrid(np.linspace(-6000, 6000, 33), np.linspace(-6000, 6000, 33))[1]
    north['potential'][0] = np.where(yy > 0, 1., -1.)
    south['potential'][0] = np.where(yy < 0, 1., -1.)
    a = FieldCoverageExpert(Config()).act(north)
    b = FieldCoverageExpert(Config()).act(south)
    assert np.linalg.norm(a[:2]-b[:2]) > .05


@pytest.mark.parametrize('task', ['coverage', 'tracking'])
def test_hard_local_field_changes_motion_even_without_vector_cloud_slots(task):
    from expert_field import FieldCoverageExpert, FieldTrackingExpert
    cls = FieldCoverageExpert if task == 'coverage' else FieldTrackingExpert
    clear, blocked = observation(task), observation(task)
    blocked['potential'][2, 15:19, 18:23] = 1.
    blocked['potential'][0, 15:19, 18:23] = -2.
    a, b = cls(Config()).act(clear), cls(Config()).act(blocked)
    assert np.linalg.norm(a[:2]-b[:2]) > 1e-3


def test_tracking_pan_uses_current_world_bearing_and_predictive_relative_motion():
    from expert_field import FieldTrackingExpert
    obs = observation('tracking')
    obs['field_context'][7:9] = [2/3, 2/3]
    obs['vector'][94:96] = [2/3, 2/3]
    action = FieldTrackingExpert(Config()).act(obs)
    assert .3 < action[2] < .5
    obs['field_context'][3] = .2
    assert FieldTrackingExpert(Config()).act(obs)[2] == pytest.approx(action[2], abs=1e-6)


def test_tracking_scan_history_does_not_push_aircraft_from_its_orbit():
    from expert_field import FieldTrackingExpert
    new, seen = observation('tracking'), observation('tracking')
    seen['potential'][0] = -1.
    np.testing.assert_array_equal(FieldTrackingExpert(Config()).act(new), FieldTrackingExpert(Config()).act(seen))


def test_clear_tracking_retains_verified_relative_motion_guidance():
    from expert_field import FieldTrackingExpert
    from expert_tracking import TrackingExpert
    obs = observation('tracking')
    np.testing.assert_allclose(FieldTrackingExpert(Config()).act(obs)[:2],
                               TrackingExpert(Config()).act(obs), atol=1e-7)


def test_tracking_pan_respects_sensor_installation_frame():
    from expert_field import FieldTrackingExpert
    obs = observation('tracking')
    obs['field_context'][7:9] = [2/3, 2/3]
    obs['vector'][94:96] = [2/3, 2/3]
    c = Config()
    initial = FieldTrackingExpert(c).act(obs)[2]
    obs['field_context'][4] = .5
    shifted = FieldTrackingExpert(c).act(obs)[2]
    assert shifted == pytest.approx(initial-.75, abs=1e-6)


def test_coverage_obstacle_command_rolls_out_clear_with_finite_roll():
    from dynamics import FixedWingModel
    from expert_field import FieldCoverageExpert
    from expert_tracking import decode_ego
    obs = observation()
    obs['vector'][14:19] = [2000/3000, 0., 500/3000, 1500/3000, 1.]
    c = Config()
    command = FieldCoverageExpert(c).act(obs)
    model = FixedWingModel(c.uav_config)
    state = decode_ego(obs['vector'], c)
    for _ in range(60):
        state = model.step_velocity(state, command[:2], .5)
        assert math.hypot(state.x-152000, state.y-150000) > 500+c.uav_cloud_margin
        assert abs(state.bank) <= c.uav_config.bank_limit


@pytest.mark.parametrize('task', ['coverage', 'tracking'])
def test_reset_and_replay_are_deterministic_and_inputs_are_not_mutated(task):
    from expert_field import FieldCoverageExpert, FieldTrackingExpert
    cls = FieldCoverageExpert if task == 'coverage' else FieldTrackingExpert
    obs = observation(task)
    saved = copy.deepcopy(obs)
    expert = cls(Config())
    first = expert.act(obs)
    expert.reset()
    np.testing.assert_array_equal(expert.act(obs), first)
    for key in obs:
        np.testing.assert_array_equal(obs[key], saved[key])


def test_impossible_cloud_encounter_is_reported_by_physical_forecast():
    from expert_field import FieldCoverageExpert
    from expert_tracking import ExpertPlanningError
    obs = observation()
    obs['vector'][14:19] = [450/3000, 0., 200/3000, 250/3000, 1.]
    with pytest.raises(ExpertPlanningError, match='No safe physical'):
        FieldCoverageExpert(Config()).act(obs)


def test_experts_reject_legacy_observations_instead_of_silently_ignoring_field():
    from expert_field import FieldCoverageExpert
    obs = observation()
    del obs['potential']
    with pytest.raises(ValueError, match='potential'):
        FieldCoverageExpert(Config()).act(obs)
