"""Optional finite-horizon command constraint, distinct from pure policy execution.

The check assumes a fixed XY command over the forecast horizon. It uses shared
observations only, and is a conservative simulation heuristic, not a global or
recursive safety certificate. Neither task experts nor target planners are used.
"""
import math

import numpy as np

from dynamics import FixedWingModel
from expert_tracking import ExpertPlanningError, SafetyGuidance, decode_clouds, decode_ego
from geometry import in_bounds, segment_circle_collision
from observations import observation_metadata


class NoFeasibleAction(RuntimeError):
    """Neither raw, generic nor searched commands passed the finite forecast."""


class ForecastConstrainedPolicy:
    def __init__(self, base_policy, config, task):
        if task not in ('coverage', 'tracking'):
            raise ValueError('task must be coverage or tracking')
        if not config.potential_enabled:
            raise ValueError('Forecast constraints require three-action obs-field-v2 Config')
        if config.decision_dt > config.expert_safety_horizon:
            raise ValueError('decision_dt cannot exceed the forecast horizon')
        self.base_policy = base_policy
        self.config = config
        self.task = task
        self.model = FixedWingModel(config.uav_config)
        self.safety = SafetyGuidance(config, preferred_side=-1 if task == 'coverage' else None)
        self._schema = observation_metadata(task, True)
        self._clear_history()

    @property
    def checkpoint_metadata(self):
        return getattr(self.base_policy, 'checkpoint_metadata', None)

    @property
    def execution_metadata(self):
        return dict(mode='forecast_constraint', heuristic_constraint=True,
                    constraint_version='forecast-v2-candidate-search',
                    task_expert_fallback=False, task=self.task,
                    horizon=float(self.config.expert_safety_horizon),
                    projection_dt=float(self.config.expert_safety_projection_dt),
                    checked_actions=self._checked, interventions=self._interventions,
                    failed_actions=self._failed, forecast_checks=self._forecasts,
                    candidate_repairs=self._candidate_repairs,
                    reachable_disk_checks=self._fast_checks)

    def _clear_history(self):
        self.last_raw_action = None
        self.last_applied_action = None
        self.last_intervened = False
        self._checked = self._interventions = self._failed = 0
        self._forecasts = self._fast_checks = 0
        self._candidate_repairs = 0

    def _candidate_repair(self,uav,raw,clouds,previous):
        """Search public geometric commands; never query a task expert/goal.

        A generic tangent heuristic may miss a narrow feasible interval between
        clouds. Independently forecast nearby directions and the previous safe
        command before declaring failure. Angular offsets remain explicit in
        the existing run.sh configuration.
        """
        angle=(uav.heading if math.hypot(float(raw[0]),float(raw[1]))<1e-6
               else math.atan2(float(raw[1]),float(raw[0])))
        offsets=set(self.config.field_heading_offsets)|set(self.config.expert_safety_heading_offsets)
        candidates=[np.array([math.cos(base+offset),math.sin(base+offset)],dtype=np.float32)
                    for base in (angle,uav.heading) for offset in sorted(offsets)]
        if previous is not None:
            candidates.append(previous[:2].copy())
        def distance(xy):
            delta=math.atan2(float(xy[1]),float(xy[0]))-angle
            return abs(math.atan2(math.sin(delta),math.cos(delta)))
        for xy in sorted(candidates,key=distance):
            if self._safe(uav,xy,clouds):
                self._candidate_repairs+=1
                return xy
        return None

    def reset(self):
        self.base_policy.reset()
        self.safety.reset()
        self._clear_history()

    def _decode(self, obs):
        """Reject incompatible shared physics before consuming recurrent state."""
        c = self.config
        vector = np.asarray(obs['vector'])
        context = np.asarray(obs['field_context'])
        potential = np.asarray(obs['potential'])
        if (vector.shape != (self._schema['vector_size'],) or context.shape != (16,)
                or potential.shape != (4, 33, 33)
                or not all(np.isfinite(a).all() for a in (vector, context, potential))):
            raise ValueError('Expected finite obs-field-v2 vector, potential and field_context')
        start, end = self._schema['slices']['params']
        expected = [getattr(c, name) / scale for name, scale in
                    zip(self._schema['parameter_order'], self._schema['parameter_divisors'])]
        if not np.allclose(vector[start:end], expected, rtol=2e-6, atol=2e-7):
            raise ValueError('Observation physics parameters do not match Config')
        fixed_context = [c.potential_radius / 3000, c.potential_period / 3600,
                         c.potential_decay,
                         (c.coverage_offset if self.task == 'coverage' else 0.) / math.pi,
                         (c.coverage_pan_limit if self.task == 'coverage' else c.pan_limit) / math.pi,
                         c.max_pan_rate, float(self.task == 'tracking')]
        if not np.allclose(context[[0, 1, 2, 4, 5, 6, 15]], fixed_context,
                           rtol=2e-6, atol=2e-7):
            raise ValueError('Observation field context does not match Config/task')
        if abs(float(vector[6])) > 1. or not np.isclose(np.linalg.norm(vector[2:4]), 1., atol=2e-6):
            raise ValueError('Observation contains an invalid physical heading/bank')
        uav = decode_ego(vector, c)
        b = c.world_bounds
        boundary = [(uav.x-b[0])/300000, (b[2]-uav.x)/300000,
                    (uav.y-b[1])/300000, (b[3]-uav.y)/300000]
        if not np.allclose(vector[10:14], boundary, rtol=2e-6, atol=2e-7):
            raise ValueError('Observation world boundaries do not match Config')
        clouds = decode_clouds(vector, uav)
        if any(ob['radius'] < 0 for ob in clouds):
            raise ValueError('Cloud radii must be nonnegative')
        return uav, clouds

    def _safe(self, uav, command, clouds):
        """Independent fixed-command rollout with a conservative curved tube."""
        self._forecasts += 1
        c = self.config
        count = math.ceil(c.expert_safety_horizon / c.expert_safety_projection_dt)
        dt = c.expert_safety_horizon / count
        margin = c.uav_speed * self.model.config.max_yaw_rate * dt*dt / 8 + c.expert_safety_extra_margin
        # Any constant-speed trajectory lies in this reachable disk regardless
        # of bank/roll response. This avoids integration only when all commands
        # are clear of every cloud and the total world boundary.
        reach = c.uav_speed * c.expert_safety_horizon + margin
        if (in_bounds((uav.x, uav.y), c.world_bounds, reach)
                and all(math.hypot(uav.x-ob['x'], uav.y-ob['y']) >
                        reach + ob['radius'] + c.uav_cloud_margin for ob in clouds)):
            self._fast_checks += 1
            return True
        if not in_bounds((uav.x, uav.y), c.world_bounds, margin):
            return False
        direction = np.asarray(command, dtype=float)
        norm = float(np.linalg.norm(direction))
        direction = direction / norm if norm >= 1e-6 else np.array([math.cos(uav.heading), math.sin(uav.heading)])
        state = uav
        for _ in range(count):
            nxt = self.model.step_velocity(state, direction, dt)
            if not in_bounds((nxt.x, nxt.y), c.world_bounds, margin):
                return False
            for ob in clouds:
                if segment_circle_collision((state.x, state.y), (nxt.x, nxt.y),
                        (ob['x'], ob['y']), ob['radius'] + c.uav_cloud_margin + margin):
                    return False
            state = nxt
        return True

    def act(self, obs):
        uav, clouds = self._decode(obs)
        raw = np.asarray(self.base_policy.act(obs), dtype=np.float32)
        if raw.shape != (3,) or not np.isfinite(raw).all() or np.any(np.abs(raw) > 1.):
            raise ValueError('Base policy action must be a finite float32 three-vector in [-1,1]')
        raw = raw.copy()
        previous=None if self.last_applied_action is None else self.last_applied_action.copy()
        self.last_raw_action = raw.copy()
        self.last_applied_action = None
        self.last_intervened = False
        self._checked += 1
        if (not in_bounds((uav.x, uav.y), self.config.world_bounds)
                or any(math.hypot(uav.x-ob['x'], uav.y-ob['y']) <=
                       ob['radius'] + self.config.uav_cloud_margin for ob in clouds)):
            self._failed += 1
            raise NoFeasibleAction('Initial observation already violates a hard physical constraint')
        if self._safe(uav, raw[:2], clouds):
            applied = raw
        else:
            c = self.config
            buffer = c.expert_obstacle_buffer if self.task == 'coverage' else min(c.expert_obstacle_buffer, c.tracking_range_guard)
            try:
                xy = np.asarray(self.safety.direction(uav, raw[:2], clouds, obstacle_buffer=buffer), dtype=np.float32)
            except ExpertPlanningError:
                xy=None
            if (xy is None or xy.shape != (2,) or not np.isfinite(xy).all() or np.any(np.abs(xy) > 1.)
                    or not self._safe(uav, xy, clouds)):
                xy=self._candidate_repair(uav,raw,clouds,previous)
            if xy is None:
                self._failed += 1
                raise NoFeasibleAction('No searched command passed the independent physical forecast')
            applied = raw.copy()
            applied[:2] = xy
            self.last_intervened = True
            self._interventions += 1
        self.last_applied_action = applied.copy()
        return applied.copy()
