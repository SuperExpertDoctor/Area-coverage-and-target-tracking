"""Physical invariants and independent analytic solutions, not planner behavior."""

import importlib
import importlib.util
import math

import pytest


def module():
    assert importlib.util.find_spec("dynamics") is not None, "Motion module is not implemented"
    return importlib.import_module("dynamics")


@pytest.mark.parametrize("kind", ["uav", "ship"])
def test_straight_motion_has_analytic_endpoint(kind):
    d = module()
    if kind == "uav":
        model = d.FixedWingModel(d.FixedWingConfig(speed=10))
        state = d.FixedWingState(3, 4, math.pi / 2)
        result = model.step_bank(state, 0, 5)
    else:
        model = d.ShipModel(d.ShipConfig(speed=10))
        state = d.ShipState(3, 4, math.pi / 2)
        result = model.step_rudder(state, 0, 5)
    assert (result.x, result.y) == pytest.approx((3, 54), abs=1e-9)
    assert state.x == 3 and state.y == 4  # advancing must not mutate rollout history


def test_uav_steady_turn_matches_analytic_circle():
    d = module()
    model = d.FixedWingModel()
    bank = math.atan(model.config.speed**2 / (9.80665 * 2000))
    state = d.FixedWingState(0, 0, 0, bank)
    duration = (math.pi / 2) * 2000 / model.config.speed
    result = model.step_bank(state, bank, duration)
    assert (result.x, result.y, result.heading) == pytest.approx(
        (2000, 2000, math.pi / 2), abs=1e-6
    )


def test_uav_turn_response_is_finite_and_respects_roll_rate():
    d = module()
    model = d.FixedWingModel()
    state = d.FixedWingState(0, 0, 0)
    result = model.step_velocity(state, (0, 1), 0.1)
    assert 0 < result.bank <= math.radians(1.5) + 1e-12
    assert 0 < result.heading < math.radians(0.1)


def test_uav_oscillating_commands_cannot_break_speed_or_curvature_limits():
    d = module()
    model = d.FixedWingModel()
    state = d.FixedWingState(0, 0, 0)
    for i in range(600):
        previous = state
        state = model.step_velocity(state, (0, 1 if i // 80 % 2 == 0 else -1), 0.1)
        assert math.hypot(*model.velocity(state)) == pytest.approx(61.11111111111111)
        assert abs(model.yaw_rate(state)) <= 61.11111111111111 / 1000 + 1e-12
        assert abs(state.bank - previous.bank) <= math.radians(15) * 0.1 + 1e-12


def test_ship_nomoto_step_matches_independent_analytic_response():
    d = module()
    model = d.ShipModel()
    delta = math.radians(10)
    state = d.ShipState(0, 0, 0, 0, delta)
    result = model.step_rudder(state, delta, 20)
    # Defaults parameterize a 300 m steady turn radius at 35 degree rudder.
    r_inf = (6.173333333333333 / 300) * (10 / 35)
    assert result.yaw_rate == pytest.approx(r_inf * (1 - math.exp(-1)), abs=1e-11)
    assert result.heading == pytest.approx(r_inf * 20 * math.exp(-1), abs=1e-10)


def test_ship_steady_turn_matches_analytic_circle():
    d = module()
    model = d.ShipModel()
    delta = math.radians(17.5)
    rate = 6.173333333333333 / 600
    state = d.ShipState(0, 0, 0, rate, delta)
    result = model.step_rudder(state, delta, (math.pi / 2) / rate)
    assert (result.x, result.y, result.heading) == pytest.approx((600, 600, math.pi / 2), abs=1e-6)


def test_ship_rudder_response_and_turn_rate_remain_bounded():
    d = module()
    model = d.ShipModel()
    state = d.ShipState(0, 0, 0)
    for i in range(1000):
        previous = state
        state = model.step_rudder(state, 10 if i < 500 else -10, 0.1)
        assert abs(state.rudder) <= math.radians(35) + 1e-12
        assert abs(state.rudder - previous.rudder) <= math.radians(3) * 0.1 + 1e-12
        assert abs(state.yaw_rate) <= 6.173333333333333 / 300 + 1e-12
        assert math.hypot(*model.velocity(state)) == pytest.approx(6.173333333333333)
    assert state.yaw_rate < 0


@pytest.mark.parametrize("kind", ["uav", "ship"])
def test_heading_commands_use_shortest_turn_across_angle_boundary(kind):
    d = module()
    start, target = math.radians(179), math.radians(-179)
    if kind == "uav":
        model = d.FixedWingModel()
        state = model.step_velocity(d.FixedWingState(0, 0, start), (math.cos(target), math.sin(target)), 0.2)
        assert state.bank > 0
    else:
        model = d.ShipModel()
        state = model.step_heading(d.ShipState(0, 0, start), target, 0.2)
        assert state.rudder > 0


def test_velocity_command_magnitude_does_not_change_uav_motion():
    d = module()
    model = d.FixedWingModel()
    state = d.FixedWingState(0, 0, 0)
    assert model.step_velocity(state, (1, 1), 1) == model.step_velocity(state, (100, 100), 1)


def test_zero_velocity_command_means_heading_hold_not_hover():
    d = module()
    model = d.FixedWingModel()
    result = model.step_velocity(d.FixedWingState(0, 0, 0), (0, 0), 2)
    assert (result.x, result.y) == pytest.approx((122.22222222222222, 0), abs=1e-9)


@pytest.mark.parametrize("kind", ["uav", "ship"])
def test_integration_does_not_depend_on_external_step_partition(kind):
    d = module()
    if kind == "uav":
        model = d.FixedWingModel()
        initial = d.FixedWingState(0, 0, 0)
        advance = lambda state, dt: model.step_velocity(state, (1, 0.3), dt)
    else:
        model = d.ShipModel()
        initial = d.ShipState(0, 0, 0)
        advance = lambda state, dt: model.step_heading(state, 0.3, dt)
    whole = advance(initial, 4)
    split = initial
    for _ in range(40):
        split = advance(split, 0.1)
    assert (whole.x, whole.y, whole.heading) == pytest.approx((split.x, split.y, split.heading), abs=1e-7)


@pytest.mark.parametrize("kind", ["uav", "ship"])
def test_heading_controller_settles_without_changing_speed(kind):
    d = module()
    if kind == "uav":
        model = d.FixedWingModel()
        state = d.FixedWingState(0, 0, 0)
        for _ in range(40):
            state = model.step_velocity(state, (math.cos(0.3), math.sin(0.3)), 1)
        assert abs(state.bank) < 1e-5
    else:
        model = d.ShipModel()
        state = d.ShipState(0, 0, 0)
        for _ in range(600):
            state = model.step_heading(state, 0.3, 1)
        assert abs(state.yaw_rate) < 1e-6
    assert state.heading == pytest.approx(0.3, abs=1e-4)


@pytest.mark.parametrize("kwargs", [{"speed": 0}, {"min_turn_radius": -1}, {"roll_time_constant": 0}, {"max_bank": math.pi / 2}, {"speed": math.nan}])
def test_uav_rejects_invalid_physical_parameters(kwargs):
    d = module()
    with pytest.raises(ValueError):
        d.FixedWingConfig(**kwargs)


@pytest.mark.parametrize("kwargs", [{"speed": -1}, {"min_turn_radius": 0}, {"nomoto_time_constant": 0}, {"max_rudder_rate": 0}, {"rudder_time_constant": math.inf}])
def test_ship_rejects_invalid_physical_parameters(kwargs):
    d = module()
    with pytest.raises(ValueError):
        d.ShipConfig(**kwargs)


def test_invalid_state_action_and_time_are_rejected_before_propagation():
    d = module()
    uav, ship = d.FixedWingModel(), d.ShipModel()
    with pytest.raises(ValueError):
        uav.step_bank(d.FixedWingState(0, 0, 0, math.radians(45)), 0, 1)
    with pytest.raises(ValueError):
        ship.step_rudder(d.ShipState(0, 0, 0, 1), 0, 1)
    with pytest.raises(ValueError):
        uav.step_velocity(d.FixedWingState(0, 0, 0), (math.nan, 1), 1)
    with pytest.raises(ValueError):
        ship.step_heading(d.ShipState(0, 0, 0), 0, -1)
    with pytest.raises(ValueError):
        uav.step_bank(d.FixedWingState(math.inf, 0, 0), 0, 1)
    with pytest.raises(ValueError):
        uav.step_bank(d.FixedWingState(0, 0, 0), 0, math.inf)


def test_zero_elapsed_time_preserves_complete_state():
    d = module()
    u = d.FixedWingState(1, 2, 0.3, 0.1)
    s = d.ShipState(1, 2, 0.3, 0.01, 0.1)
    assert d.FixedWingModel().step_bank(u, 0.2, 0) == u
    assert d.ShipModel().step_rudder(s, 0.2, 0) == s
