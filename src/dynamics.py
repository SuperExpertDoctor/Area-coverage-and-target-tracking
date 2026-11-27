"""Planar constant-speed models with finite steering response.

SI units throughout. x points east, y north, angles are counterclockwise
from +x; positive bank/rudder means LEFT. No wind, current, sideslip or
altitude dynamics. Model defaults are simulation parameters, not identified
CH-3 / Xiang Yang Hong 01 maneuvering coefficients. See docs/kinematics.md.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import math
from typing import Callable, Sequence


GRAVITY = 9.80665


def _finite(*values: float) -> None:
    if not all(math.isfinite(v) for v in values):
        raise ValueError("All states, commands and times must be finite")


def _clip(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def _positive_config(config: object) -> None:
    for item in fields(config):
        value = getattr(config, item.name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{item.name} must be finite and positive")


def _check_bound(value: float, limit: float, name: str) -> None:
    _finite(value)
    if abs(value) > limit + 1e-12:
        raise ValueError(f"{name} exceeds this model's physical limit")


def _integrate(
    initial: tuple[float, ...],
    derivative: Callable[[tuple[float, ...]], tuple[float, ...]],
    duration: float,
    max_step: float,
) -> tuple[float, ...]:
    """RK4 with bounded internal steps; command feedback is evaluated at stages."""
    _finite(duration)
    if duration < 0:
        raise ValueError("Elapsed time must be nonnegative")
    if duration == 0:
        return initial
    count = math.ceil(duration / max_step)
    h = duration / count
    values = initial
    for _ in range(count):
        k1 = derivative(values)
        k2 = derivative(tuple(v + h * k / 2 for v, k in zip(values, k1)))
        k3 = derivative(tuple(v + h * k / 2 for v, k in zip(values, k2)))
        k4 = derivative(tuple(v + h * k for v, k in zip(values, k3)))
        values = tuple(
            v + h * (a + 2 * b + 2 * c + d) / 6
            for v, a, b, c, d in zip(values, k1, k2, k3, k4)
        )
    _finite(*values)
    return values


@dataclass(frozen=True)
class FixedWingConfig:
    speed: float = 220 / 3.6
    min_turn_radius: float = 1000.0
    max_bank: float = math.radians(25)
    roll_time_constant: float = 1.0
    max_roll_rate: float = math.radians(15)
    heading_time_constant: float = 1.0
    integration_step: float = 0.05

    def __post_init__(self) -> None:
        _positive_config(self)
        if self.max_bank >= math.pi / 2:
            raise ValueError("max_bank must be smaller than pi/2 radians")

    @property
    def bank_limit(self) -> float:
        return min(self.max_bank, math.atan(self.speed**2 / (GRAVITY * self.min_turn_radius)))

    @property
    def max_yaw_rate(self) -> float:
        return GRAVITY * math.tan(self.bank_limit) / self.speed


@dataclass(frozen=True)
class FixedWingState:
    x: float
    y: float
    heading: float
    bank: float = 0.0


class FixedWingModel:
    """Coordinated-turn kinematics, augmented with a bounded roll actuator."""

    def __init__(self, config: FixedWingConfig | None = None):
        self.config = config if config is not None else FixedWingConfig()

    def _validate(self, state: FixedWingState) -> None:
        _finite(state.x, state.y, state.heading, state.bank)
        _check_bound(state.bank, self.config.bank_limit, "bank")

    def velocity(self, state: FixedWingState) -> tuple[float, float]:
        self._validate(state)
        return (self.config.speed * math.cos(state.heading), self.config.speed * math.sin(state.heading))

    def yaw_rate(self, state: FixedWingState) -> float:
        self._validate(state)
        return GRAVITY * math.tan(state.bank) / self.config.speed

    def _advance(
        self,
        state: FixedWingState,
        bank_target: Callable[[tuple[float, ...]], float],
        dt: float,
    ) -> FixedWingState:
        self._validate(state)
        cfg = self.config

        def derivative(values: tuple[float, ...]) -> tuple[float, ...]:
            _, _, heading, bank = values
            target = _clip(bank_target(values), cfg.bank_limit)
            return (
                cfg.speed * math.cos(heading),
                cfg.speed * math.sin(heading),
                GRAVITY * math.tan(bank) / cfg.speed,
                _clip((target - bank) / cfg.roll_time_constant, cfg.max_roll_rate),
            )

        result = _integrate(
            (state.x, state.y, state.heading, state.bank), derivative, dt,
            min(cfg.integration_step, cfg.roll_time_constant / 10, cfg.heading_time_constant / 10),
        )
        if dt == 0:
            return state
        next_state = FixedWingState(result[0], result[1], _wrap(result[2]), result[3])
        self._validate(next_state)
        return next_state

    def step_bank(self, state: FixedWingState, bank_command: float, dt: float) -> FixedWingState:
        """Advance under a requested bank, respecting both bank and roll-rate limits."""
        _finite(bank_command)
        return self._advance(state, lambda _: bank_command, dt)

    def step_velocity(self, state: FixedWingState, desired_velocity: Sequence[float], dt: float) -> FixedWingState:
        """Track a world-frame velocity DIRECTION. Magnitude never changes speed.

        A zero vector requests holding the initial heading, not a stop. The
        direction is a setpoint, not an instantaneous or guaranteed end velocity.
        The fixed command time scale is independent of physics integration dt.
        """
        if isinstance(desired_velocity, (str, bytes)) or len(desired_velocity) != 2:
            raise ValueError("desired_velocity must have two finite components")
        vx, vy = float(desired_velocity[0]), float(desired_velocity[1])
        _finite(vx, vy)
        target_heading = math.atan2(vy, vx) if vx != 0 or vy != 0 else state.heading

        def bank_target(values: tuple[float, ...]) -> float:
            rate = _wrap(target_heading - values[2]) / self.config.heading_time_constant
            rate = _clip(rate, self.config.max_yaw_rate)
            return math.atan(self.config.speed * rate / GRAVITY)

        return self._advance(state, bank_target, dt)


@dataclass(frozen=True)
class ShipConfig:
    speed: float = 12 * 1852 / 3600
    min_turn_radius: float = 300.0
    nomoto_time_constant: float = 20.0
    max_rudder: float = math.radians(35)
    max_rudder_rate: float = math.radians(3)
    rudder_time_constant: float = 1.0
    heading_time_constant: float = 30.0
    yaw_damping: float = 0.8
    integration_step: float = 0.05

    def __post_init__(self) -> None:
        _positive_config(self)
        if self.max_rudder >= math.pi / 2:
            raise ValueError("max_rudder must be smaller than pi/2 radians")

    @property
    def max_yaw_rate(self) -> float:
        return self.speed / self.min_turn_radius

    @property
    def nomoto_gain(self) -> float:
        """Simulation K [1/s], parameterized by the full-rudder steady radius."""
        return self.max_yaw_rate / self.max_rudder


@dataclass(frozen=True)
class ShipState:
    x: float
    y: float
    heading: float
    yaw_rate: float = 0.0
    rudder: float = 0.0


class ShipModel:
    """Constant surge, zero sway; first-order Nomoto yaw and finite rudder motion."""

    def __init__(self, config: ShipConfig | None = None):
        self.config = config if config is not None else ShipConfig()

    def _validate(self, state: ShipState) -> None:
        _finite(state.x, state.y, state.heading, state.yaw_rate, state.rudder)
        _check_bound(state.yaw_rate, self.config.max_yaw_rate, "yaw_rate")
        _check_bound(state.rudder, self.config.max_rudder, "rudder")

    def velocity(self, state: ShipState) -> tuple[float, float]:
        self._validate(state)
        return (self.config.speed * math.cos(state.heading), self.config.speed * math.sin(state.heading))

    def _advance(
        self,
        state: ShipState,
        rudder_target: Callable[[tuple[float, ...]], float],
        dt: float,
    ) -> ShipState:
        self._validate(state)
        cfg = self.config

        def derivative(values: tuple[float, ...]) -> tuple[float, ...]:
            _, _, heading, rate, rudder = values
            target = _clip(rudder_target(values), cfg.max_rudder)
            return (
                cfg.speed * math.cos(heading),
                cfg.speed * math.sin(heading),
                rate,
                (cfg.nomoto_gain * rudder - rate) / cfg.nomoto_time_constant,
                _clip((target - rudder) / cfg.rudder_time_constant, cfg.max_rudder_rate),
            )

        result = _integrate(
            (state.x, state.y, state.heading, state.yaw_rate, state.rudder), derivative, dt,
            min(cfg.integration_step, cfg.rudder_time_constant / 10, cfg.nomoto_time_constant / 10, cfg.heading_time_constant / 10),
        )
        if dt == 0:
            return state
        next_state = ShipState(result[0], result[1], _wrap(result[2]), result[3], result[4])
        self._validate(next_state)
        return next_state

    def step_rudder(self, state: ShipState, rudder_command: float, dt: float) -> ShipState:
        """Advance with a requested rudder; actual rudder and yaw respond gradually."""
        _finite(rudder_command)
        return self._advance(state, lambda _: rudder_command, dt)

    def step_heading(self, state: ShipState, desired_heading: float, dt: float) -> ShipState:
        """Track a path follower's heading setpoint; A* itself does not move the ship."""
        _finite(desired_heading)

        def rudder_target(values: tuple[float, ...]) -> float:
            error = _wrap(desired_heading - values[2])
            rate_command = error / self.config.heading_time_constant - self.config.yaw_damping * values[3]
            rate_command = _clip(rate_command, self.config.max_yaw_rate)
            return rate_command / self.config.nomoto_gain

        return self._advance(state, rudder_target, dt)
