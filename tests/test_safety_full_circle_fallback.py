"""A real safe 30-second direction missed by both earlier candidate sets."""
import json
import math
from pathlib import Path

import numpy as np
import pytest

from config import Config
from dynamics import FixedWingModel, FixedWingState
from expert_tracking import ExpertPlanningError, SafetyGuidance
from geometry import in_bounds, segment_circle_collision, unit, wrap


FIXTURES = Path(__file__).parent/'fixtures'


def snapshot():
    return json.loads((FIXTURES/'coverage-cloud-cluster-12029-safety.json').read_text('utf-8'))


def test_real_t8925_full_circle_finds_safe_constant_command():
    data = snapshot()
    c = Config.from_dict(data['config'])
    calls = []
    guidance = SafetyGuidance(c, preferred_side=-1)
    original = guidance._forecast_clearance

    def record(state, action, clouds):
        clearance = original(state, action, clouds)
        calls.append((np.asarray(action).copy(), clearance))
        return clearance

    guidance._forecast_clearance = record
    command = guidance.direction(FixedWingState(**data['state']), data['desired'],
                                 data['clouds'], obstacle_buffer=data['soft_buffer'])
    # The unchanged original and union sets all fail at this real public state.
    prior = data['provenance']['old_candidate_count']
    assert all(clearance is None for _, clearance in calls[:prior])
    assert prior < len(calls) <= prior+c.expert_safety_fallback_heading_samples
    assert calls[-1][1] is not None
    np.testing.assert_array_equal(command, calls[-1][0])
    generic_heading = math.atan2(calls[0][0][1], calls[0][0][0])
    angles = sorted(data['state']['heading']+np.linspace(-math.pi, math.pi,
        c.expert_safety_fallback_heading_samples, endpoint=False),
        key=lambda angle: abs(wrap(angle-generic_heading)))
    for (actual, _), angle in zip(calls[prior:], angles):
        np.testing.assert_array_equal(actual, unit(angle))

    # Independently check exact, unquantized state at the native physics step,
    # retaining the original extra safety margin and curvature bound.
    model = FixedWingModel(c.uav_config)
    state = FixedWingState(**data['actual_state'])
    dt = c.physics_dt
    curve = c.uav_speed*model.config.max_yaw_rate*dt*dt/8+c.expert_safety_extra_margin
    for _ in range(round(c.expert_safety_horizon/dt)):
        nxt = model.step_velocity(state, command, dt)
        assert in_bounds((nxt.x, nxt.y), c.world_bounds, curve)
        assert abs(nxt.bank) <= c.uav_config.bank_limit
        for cloud in data['actual_clouds']:
            assert not segment_circle_collision((state.x, state.y), (nxt.x, nxt.y),
                (cloud['x'], cloud['y']), cloud['radius']+c.uav_cloud_margin+curve)
        state = nxt


@pytest.mark.parametrize('rotation', [.137, .731, -1.193])
def test_real_fallback_rotates_with_public_state_away_from_world_edges(rotation):
    data = snapshot()
    c = Config.from_dict(data['config'])
    state = FixedWingState(**data['state'])
    original = SafetyGuidance(c, preferred_side=-1).direction(
        state, data['desired'], data['clouds'], obstacle_buffer=data['soft_buffer'])
    matrix = np.array([[math.cos(rotation), -math.sin(rotation)],
                       [math.sin(rotation), math.cos(rotation)]])
    origin = np.array([state.x, state.y])
    clouds = []
    for cloud in data['clouds']:
        point = origin+matrix@(np.array([cloud['x'], cloud['y']])-origin)
        clouds.append(dict(cloud, x=point[0], y=point[1]))
    rotated = SafetyGuidance(c, preferred_side=-1).direction(
        FixedWingState(state.x, state.y, wrap(state.heading+rotation), state.bank),
        matrix@data['desired'], clouds, obstacle_buffer=data['soft_buffer'])
    np.testing.assert_allclose(rotated, matrix@original, atol=2e-6, rtol=0)


def test_union_feasible_command_remains_bitexact():
    data = json.loads((FIXTURES/'coverage-cloud-cluster-12153-safety.json').read_text('utf-8'))
    expected = snapshot()['prior_union_expected_command_12153']
    command = SafetyGuidance(Config.from_dict(data['config']), preferred_side=-1).direction(
        FixedWingState(**data['state']), data['desired'], data['clouds'],
        obstacle_buffer=data['soft_buffer'])
    np.testing.assert_array_equal(command, np.asarray(expected, dtype=np.float32))


def test_full_circle_still_rejects_state_inside_hard_cloud():
    c = Config()
    state = FixedWingState(150000., 150000., 0., 0.)
    with pytest.raises(ExpertPlanningError, match='No safe physical'):
        SafetyGuidance(c).direction(state, [1., 0.],
            [{'id':0, 'x':150100., 'y':150000., 'radius':500.}])


def test_fallback_sample_default_and_minimum_are_explicit():
    assert Config().expert_safety_fallback_heading_samples == 145
    assert Config(expert_safety_fallback_heading_samples=8).expert_safety_fallback_heading_samples == 8


@pytest.mark.parametrize('value', [0, 7, -1, 8.5, float('nan'), float('inf'), True])
def test_fallback_samples_must_be_finite_integer_at_least_eight(value):
    with pytest.raises(ValueError, match='expert_safety_fallback_heading_samples'):
        Config(expert_safety_fallback_heading_samples=value)


def test_run_sh_exposes_full_circle_sample_count():
    script = (Path(__file__).parents[1]/'run.sh').read_text('utf-8')
    embedded = json.JSONDecoder().raw_decode(script[script.index('\n{')+1:])[0]
    assert embedded.get('expert_safety_fallback_heading_samples') == 145
