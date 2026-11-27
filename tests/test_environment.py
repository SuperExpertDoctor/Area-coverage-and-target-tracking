import copy
import math

import numpy as np
import pytest


def test_environment_contract_and_replay():
    from config import Config
    from env_coverage import CoverageEnv
    from env_tracking import TrackingEnv
    for env_class, size in [(CoverageEnv, 111), (TrackingEnv, 134)]:
        env = env_class(Config(coverage_episode_seconds=3, tracking_episode_seconds=3))
        obs, info = env.reset(seed=13)
        assert obs['vector'].shape == (size,)
        assert obs['vector'].dtype == np.float32
        assert env.observation_space.contains(obs)
        initial = copy.deepcopy(obs)
        scene = copy.deepcopy(env.scene)
        assert len(scene['ship_route']) > 100
        assert scene['world_bounds'] == [0, 0, 300000, 300000]
        if env.task == 'coverage':
            assert obs['coverage_map'].shape == (3, 64, 64)
        other = env_class(env.config)
        replay, _ = other.reset(options={'scene': scene})
        for key in obs:
            np.testing.assert_array_equal(obs[key], replay[key])
        action = obs['vector'][2:4]
        result = env.step(action)
        replay_result = other.step(action)
        np.testing.assert_array_equal(result[0]['vector'], replay_result[0]['vector'])
        np.testing.assert_array_equal(initial['vector'], obs['vector'])
        assert result[4]['elapsed_seconds'] == pytest.approx(1)
        assert not result[2]
        env.step(action)
        result = env.step(action)
        assert result[3]
        with pytest.raises(RuntimeError):
            env.step(action)


def test_tracking_initialization_and_expert_stays_visible():
    from config import Config
    from env_tracking import TrackingEnv
    from expert_tracking import TrackingExpert
    env = TrackingEnv(Config(tracking_episode_seconds=180))
    obs, info = env.reset(seed=7)
    assert 1600 <= info['distance'] <= 2400
    assert info['target_visible']
    assert env.summary()['min_distance'] == pytest.approx(info['distance'])
    assert env.summary()['max_distance'] == pytest.approx(info['distance'])
    assert env.summary()['mean_distance'] == pytest.approx(info['distance'])
    expert = TrackingExpert(env.config)
    expert.reset()
    last_heading = env.uav.heading
    for _ in range(180):
        action = expert.act(obs)
        assert np.linalg.norm(action) == pytest.approx(1, abs=1e-6)
        obs, _, terminated, _, info = env.step(action)
        assert not terminated
        assert info['target_visible']
        turn = (env.uav.heading-last_heading+math.pi)%(2*math.pi)-math.pi
        assert abs(turn) <= env.uav_model.config.max_yaw_rate * 1.001
        last_heading = env.uav.heading
    assert env.summary()['visible_fraction'] == 1


def test_coverage_history_land_exclusion_and_action_validation():
    from config import Config
    from env_coverage import CoverageEnv
    env = CoverageEnv(Config(coverage_episode_seconds=8))
    obs, _ = env.reset(seed=2)
    last_seen = env.last_seen_tick.copy()
    for action in ([float('nan'), 0], [1.01, 0], [1.00000001, 0], [0, 0, 1]):
        with pytest.raises(ValueError):
            env.step(action)
    before = (env.uav.x, env.uav.y)
    obs, _, terminated, _, info = env.step([0, 0])
    assert not terminated
    assert math.dist(before, (env.uav.x, env.uav.y)) == pytest.approx(env.config.uav_speed, rel=.001)
    assert np.all(env.last_seen_tick[last_seen >= 0] >= last_seen[last_seen >= 0])
    assert np.all(obs['coverage_map'][:, ~env.sea_mask] == 0)
    assert info['coverage_ratio'] >= 0
    scene = copy.deepcopy(env.scene)
    scene['uav_initial']['x'] = -1
    with pytest.raises(ValueError):
        env.reset(options={'scene': scene})


def test_collision_uses_segment_and_clouds_do_not_block_ships():
    from geometry import segment_circle_collision
    assert segment_circle_collision((-2, 0), (2, 0), (0, 0), 1)
    assert not segment_circle_collision((-2, 2), (2, 2), (0, 0), 1)


def test_ship_route_physical_displacement_and_render_callback():
    from env_tracking import TrackingEnv
    frames = []
    env = TrackingEnv(render_callback=frames.append)
    obs, _ = env.reset(seed=5)
    start = (env.ship.x, env.ship.y)
    env.step(obs['vector'][2:4])
    assert math.dist(start, (env.ship.x, env.ship.y)) == pytest.approx(env.config.ship_speed, rel=.001)
    frame = env.render()
    assert frames[-1] is frame
    assert frame['t'] == pytest.approx(1)
    assert all(key in frame for key in ('uav', 'ship', 'radar', 'metrics', 'coverage'))


def test_coverage_expert_handles_successive_clouds_after_first_pass():
    from config import Config
    from env_coverage import CoverageEnv
    from expert_coverage import CoverageExpert
    env = CoverageEnv(Config(coverage_episode_seconds=4200))
    expert = CoverageExpert(env.config)
    obs, _ = env.reset(seed=0)
    for _ in range(4200):
        obs, _, terminated, _, _ = env.step(expert.act(obs))
        assert not terminated
    assert env.coverage_ratio > .95
    assert env.rescan_ratio > .5


def test_ship_astar_bypass_is_smooth_and_physically_clears_an_island():
    from config import Config
    from dynamics import ShipModel, ShipState
    from navigation import island_bypass, ShipNavigator, smooth_polyline
    config = Config()
    island = {'id': 0, 'x': 20000., 'y': 10000., 'radius': 1500.}
    route = island_bypass((10000., 10000.), (30000., 10000.), [island], config)
    route = smooth_polyline(route, 600., config.ship_min_turn_radius, 25.)
    assert np.max(np.linalg.norm(np.diff(route, axis=0), axis=1)) <= 25.01
    nav = ShipNavigator(route)
    model = ShipModel(config.ship_config)
    state = ShipState(*route[0], math.atan2(*(route[1]-route[0])[::-1]))
    clearance = math.inf
    for _ in range(3600):
        previous_heading = state.heading
        state = model.step_heading(state, nav.heading_command(state), 1.)
        clearance = min(clearance, math.hypot(state.x-island['x'], state.y-island['y'])-island['radius'])
        turn = (state.heading-previous_heading+math.pi)%(2*math.pi)-math.pi
        assert abs(turn) <= config.ship_config.max_yaw_rate*1.001
        if state.x > 29500:
            break
    assert state.x > 29500
    assert clearance > config.ship_island_margin


@pytest.mark.parametrize('override', [
    {'coverage_range': -1}, {'navigation_grid_resolution': 0},
    {'tracking_half_angle': 4}, {'ship_route_spacing': 30},
    {'age_scale': 0}, {'max_scene_attempts': 0},
])
def test_invalid_physical_and_sensor_configuration_is_rejected(override):
    from config import Config
    with pytest.raises(ValueError):
        Config(**override)


def test_coverage_cloud_bypass_keeps_fixed_right_sensor_toward_cloud_water():
    from config import Config
    from dynamics import FixedWingModel, FixedWingState
    from expert_tracking import SafetyGuidance
    from geometry import wrap
    config = Config()
    guidance = SafetyGuidance(config, preferred_side=-1)
    state = FixedWingState(100000., 100000., 0.)
    cloud = {'id': 0, 'x': 104500., 'y': 100000., 'radius': 700.}
    model = FixedWingModel(config.uav_config)
    visible = 0
    clearance = math.inf
    for _ in range(350):
        desired = np.array([cloud['x']-state.x, cloud['y']-state.y])
        state = model.step_velocity(state, guidance.direction(state, desired, [cloud]), 1.)
        distance = math.hypot(state.x-cloud['x'], state.y-cloud['y'])
        clearance = min(clearance, distance-cloud['radius'])
        error = wrap(math.atan2(cloud['y']-state.y, cloud['x']-state.x)-state.heading-config.coverage_offset)
        visible += int(distance <= config.coverage_range and abs(error) <= config.coverage_half_angle)
    assert clearance > config.uav_cloud_margin
    assert visible > 150


def test_scene_replay_rejects_a_ship_route_crossing_an_island():
    from env_tracking import TrackingEnv
    env = TrackingEnv()
    env.reset(seed=7)
    scene = copy.deepcopy(env.scene)
    point = scene['ship_route'][100]
    scene['islands'] = [{'id': 0, 'x': point[0], 'y': point[1], 'radius': 500.}]
    with pytest.raises(ValueError, match='route'):
        env.reset(options={'scene': scene})


def test_cloud_gap_keeps_reachable_current_heading_candidate():
    """Seed2000 has a safe northbound gap despite misleading tangential guidance."""
    from config import Config
    from dynamics import FixedWingModel, FixedWingState
    from expert_tracking import SafetyGuidance
    from geometry import segment_circle_collision
    config = Config()
    state = FixedWingState(145757.3081667885, 194336.68861683604,
                           2.3383979026788895, -.3635453094962078)
    clouds = [
        {'id': 0, 'x': 148917.15006254808, 'y': 188841.72757234218, 'radius': 531.849520125779},
        {'id': 1, 'x': 143533.25711474972, 'y': 195115.4728327928, 'radius': 746.3580620255435},
        {'id': 2, 'x': 146756.02187434552, 'y': 196492.8552678979, 'radius': 847.6830590464792},
    ]
    action = SafetyGuidance(config, preferred_side=-1).direction(
        state, [2.867286496995291, -.31060780867392046], clouds)
    assert np.linalg.norm(action) == pytest.approx(1., abs=1e-6)
    model = FixedWingModel(config.uav_config)
    for _ in range(300):
        future = model.step_velocity(state, action, .1)
        for cloud in clouds:
            assert not segment_circle_collision(
                (state.x, state.y), (future.x, future.y),
                (cloud['x'], cloud['y']), cloud['radius']+config.uav_cloud_margin)
        state = future


def test_tracking_keeps_radar_envelope_when_ship_bypasses_toward_cloud():
    """Seed5003's island bypass approaches a cloud outside the base spiral corridor."""
    from env_tracking import TrackingEnv
    from expert_tracking import TrackingExpert
    env = TrackingEnv()
    obs, _ = env.reset(seed=5003)
    expert = TrackingExpert(env.config)
    expert.reset()
    for _ in range(1800):
        obs, _, terminated, truncated, _ = env.step(expert.act(obs))
        assert not terminated
        assert env.target_visible
        if truncated:
            break
    summary = env.summary()
    assert summary['time_seconds'] == 1800
    assert summary['visible_fraction'] == 1
    assert summary['min_distance'] >= env.config.tracking_min_distance
    assert summary['max_distance'] <= env.config.tracking_range-env.config.tracking_range_guard


@pytest.mark.parametrize('buffer', [-1., float('nan'), float('inf')])
def test_safety_guidance_rejects_invalid_soft_buffer(buffer):
    from config import Config
    from dynamics import FixedWingState
    from expert_tracking import SafetyGuidance
    with pytest.raises(ValueError, match='obstacle_buffer'):
        SafetyGuidance(Config()).direction(FixedWingState(100000., 100000., 0.), [1., 0.], [],
                                          obstacle_buffer=buffer)
