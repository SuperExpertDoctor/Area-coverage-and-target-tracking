"""Real seed12153 regressions: a missed safe command and an unseen-cell limit cycle."""
import json
import math
from pathlib import Path

import numpy as np
import pytest

from config import Config
from dynamics import FixedWingModel, FixedWingState
from expert_field import FieldCoverageExpert, FieldTrackingExpert
from expert_tracking import SafetyGuidance
from geometry import in_bounds, segment_circle_collision

FIXTURES = Path(__file__).parent/'fixtures'


def fixture(name):
    return json.loads((FIXTURES/f'coverage-cloud-cluster-12153-{name}.json').read_text('utf-8'))


def assert_physical_forecast(state, action, clouds, config):
    model = FixedWingModel(config.uav_config)
    dt = config.physics_dt
    curve = config.uav_speed*model.config.max_yaw_rate*dt*dt/8
    for _ in range(round(config.expert_safety_horizon/dt)):
        nxt = model.step_velocity(state, action, dt)
        assert in_bounds((nxt.x, nxt.y), config.world_bounds, curve)
        assert abs(nxt.bank) <= config.uav_config.bank_limit
        for ob in clouds:
            assert not segment_circle_collision((state.x, state.y), (nxt.x, nxt.y),
                (ob['x'], ob['y']), ob['radius']+config.uav_cloud_margin+curve)
        state = nxt


def test_real_t7569_missed_candidate_returns_physically_safe_command():
    data = fixture('safety')
    c = Config.from_dict(data['config'])
    state = FixedWingState(**data['state'])
    command = SafetyGuidance(c, preferred_side=-1).direction(
        state, data['desired'], data['clouds'], obstacle_buffer=data['soft_buffer'])
    assert np.linalg.norm(command) == pytest.approx(1., abs=1e-6)
    assert_physical_forecast(FixedWingState(**data['actual_state']), command, data['clouds'], c)


def test_existing_feasible_guidance_command_remains_bitexact():
    data = fixture('safety')['legacy_feasible_case']
    command = SafetyGuidance(Config(), preferred_side=-1).direction(
        FixedWingState(**data['state']), data['desired'], data['clouds'])
    np.testing.assert_array_equal(command, np.asarray(data['expected_command'], np.float32))


def restore_loop_snapshot():
    from env_coverage import CoverageEnv
    from observations import build_observation
    data = fixture('loop')
    c = Config.from_dict(data['config'])
    env = CoverageEnv(c)
    env.reset(seed=12153, options={'scene':data['scene']})
    env.physics_tick = data['physics_tick']
    env.previous_action = np.asarray(data['previous_action'], dtype=np.float32)
    env.last_seen_tick[:] = data['last_seen_tick']
    env.scan_count[:] = data['scan_count']
    env.public_goal = np.asarray(data['public_goal'])
    env.public_goal_priority = data['public_goal_priority']
    env._goal_selected_tick = data['goal_selected_tick']
    env._goal_deadline = data['goal_deadline']
    env._goal_progress_bucket = data['goal_progress_bucket']
    env.potential_field.last_seen = {(int(x), int(y)):t for x, y, t in data['potential_last_seen']}
    env.potential_field._cache_key = None
    return env, build_observation(env)


def test_real_stable_loop_finishes_six_unseen_cells_without_hard_collision():
    env, obs = restore_loop_snapshot()
    try:
        initial_missing = int(np.sum(env.sea_mask & (env.last_seen_tick < 0)))
        assert initial_missing == 6
        actor = FieldCoverageExpert(env.config)
        actor.reset()
        for _ in range(720):
            obs, _, terminated, truncated, info = env.step(actor.act(obs))
            assert not terminated and not truncated
            assert not info['uav_collision'] and not info['ship_collision'] and not info['out_of_bounds']
            if env.coverage_ratio == 1.:
                break
        assert env.coverage_ratio == 1., 'Real cloud-cluster limit cycle leaves sea cells unseen'
    finally:
        env.close()


def test_coverage_soft_buffer_default_and_zero_are_supported():
    assert Config().field_coverage_obstacle_buffer == 200.
    assert Config(field_coverage_obstacle_buffer=0.).field_coverage_obstacle_buffer == 0.


@pytest.mark.parametrize('value', [-1., float('nan'), float('inf')])
def test_coverage_soft_buffer_must_be_finite_nonnegative(value):
    with pytest.raises(ValueError, match='field_coverage_obstacle_buffer'):
        Config(field_coverage_obstacle_buffer=value)


def test_run_sh_exposes_coverage_soft_buffer():
    script = (Path(__file__).parents[1]/'run.sh').read_text('utf-8')
    embedded = json.JSONDecoder().raw_decode(script[script.index('\n{')+1:])[0]
    assert embedded.get('field_coverage_obstacle_buffer') == 200.


def test_coverage_soft_buffer_does_not_change_tracking_command():
    from env_tracking import TrackingEnv
    c = Config(potential_enabled=True)
    env = TrackingEnv(c)
    try:
        obs, _ = env.reset(seed=17)
        low = FieldTrackingExpert(Config(potential_enabled=True, field_coverage_obstacle_buffer=0.)).act(obs)
        high = FieldTrackingExpert(Config(potential_enabled=True, field_coverage_obstacle_buffer=1300.)).act(obs)
        np.testing.assert_array_equal(low, high)
    finally:
        env.close()
