"""Observation-only, field-aware guidance with explicit radar commands.

Coverage uses the environment's public timestamp-derived goal rather than a
private route. The local field ranks nearby commands and radar sectors; scanned
pixels are preferences, never aircraft collision geometry. Tracking retains the
verified moving-orbit law and limits field perturbations to its range reserve.
"""
import math

import numpy as np

from config import Config
from expert_tracking import SafetyGuidance, decode_clouds, decode_ego
from geometry import unit, wrap


def _normalized(vector, heading=0.):
    norm = float(np.linalg.norm(vector))
    return np.asarray(vector, dtype=float)/norm if norm > 1e-9 else unit(heading).astype(float)


class _FieldExpert:
    task = None

    def __init__(self, config=None):
        self.config = config if config is not None else Config()
        self.safety = SafetyGuidance(self.config, preferred_side=-1 if self.task == 'coverage' else None)

    def reset(self):
        self.safety.reset()

    def _read(self, obs):
        shapes = {'vector':(111 if self.task == 'coverage' else 134,),
                  'potential':(4, 33, 33), 'field_context':(16,)}
        for key, shape in shapes.items():
            if key not in obs or np.shape(obs[key]) != shape or not np.all(np.isfinite(obs[key])):
                raise ValueError(f'obs-field-v2 requires finite {key}{shape}')
        context = np.asarray(obs['field_context'], dtype=float)
        if context[0] <= 0 or context[5] <= 0:
            raise ValueError('field radius and radar pan limit must be positive')
        field = np.asarray(obs['potential'], dtype=float)
        axis = np.linspace(-context[0]*3000, context[0]*3000, field.shape[-1])
        xx, yy = np.meshgrid(axis, axis)
        return obs['vector'], field, context, xx, yy

    def _obstacle_direction(self, field, xx, yy, heading):
        """Near hard pixels contribute repulsion and a deterministic tangent.

        Circular validity excludes padding. Islands never enter this mask. The
        geometric hard checks still use exact public cloud circles and world
        bounds, avoiding an unsafe conversion of raster cells into tiny circles.
        """
        c = self.config
        distance = np.hypot(xx, yy)
        mask = (field[2] > .5) & (field[3] > .5) & (distance < c.field_obstacle_influence)
        if not np.any(mask):
            return np.zeros(2)
        weights = np.maximum(0., 1-distance[mask]/c.field_obstacle_influence)**2
        away = -np.array([np.sum(xx[mask]*weights), np.sum(yy[mask]*weights)])
        radial = _normalized(away, heading+math.pi)
        tangent = np.array([-radial[1], radial[0]])
        side = -1 if self.task == 'coverage' else (1 if tangent@unit(heading) >= 0 else -1)
        strength = float(np.max(weights))
        return strength*(radial+side*tangent)

    def _safe(self, vector, uav, desired):
        c = self.config
        limit = c.tracking_range_guard if self.task == 'tracking' else c.field_coverage_obstacle_buffer
        buffer = min(c.expert_obstacle_buffer, limit)
        return self.safety.direction(uav, desired, decode_clouds(vector, uav), obstacle_buffer=buffer)


class FieldCoverageExpert(_FieldExpert):
    task = 'coverage'

    def _radar(self, field, context, xx, yy, heading, position, horizon):
        c = self.config
        px, py = position
        dx, dy = xx-px, yy-py
        distance = np.hypot(dx, dy)
        bearing = np.arctan2(dy, dx)
        offset, limit, current = context[4]*math.pi, context[5]*math.pi, context[3]*math.pi
        goal = context[7:9]*3000-position
        goal_pan = np.clip(wrap(math.atan2(goal[1], goal[0])-heading-offset), -limit, limit)
        pans = np.unique(np.r_[np.linspace(-limit, limit, c.field_pan_samples), goal_pan, current])
        # Score the physically reachable boresight, while returning its requested
        # setpoint. This prevents commanding distant pans for imaginary gain.
        actual = current+np.clip(pans-current, -context[6]*horizon, context[6]*horizon)
        errors = (bearing[None]-heading-offset-actual[:, None, None]+math.pi)%(2*math.pi)-math.pi
        valid = ((field[1] > .5) & (field[2] < .5) & (field[3] > .5) & (distance <= c.coverage_range))
        sectors = valid[None] & (np.abs(errors) <= c.coverage_half_angle)
        cell_area = (2*context[0]*3000/(field.shape[-1]-1))**2
        area = c.coverage_half_angle*c.coverage_range**2/cell_area
        utility = np.sum(sectors*field[0][None], axis=(1, 2))/area
        goal_bearing = math.atan2(goal[1], goal[0])
        goal_error = (goal_bearing-heading-offset-actual+math.pi)%(2*math.pi)-math.pi
        goal_reward = np.cos(goal_error)*context[9]*context[10]
        slew = np.abs(pans-current)/limit
        score = c.field_scan_weight*utility+c.field_pan_goal_weight*goal_reward-c.field_pan_slew_weight*slew
        best = int(np.argmax(score))
        return float(pans[best]), float(score[best])

    def act(self, obs):
        v, field, context, xx, yy = self._read(obs)
        c = self.config
        u = decode_ego(v, c)
        velocity = c.uav_speed*unit(u.heading).astype(float)
        delay = c.uav_roll_time_constant+c.uav_heading_time_constant+.5*c.decision_dt
        relative = -context[7:9]*3000+velocity*delay
        if context[10] > .5:
            distance = max(1., float(np.linalg.norm(relative)))
            radial = relative/distance
            desired = np.array([radial[1], -radial[0]])-c.coverage_repair_radial_gain*(distance-c.coverage_repair_orbit_radius)/c.coverage_repair_orbit_radius*radial
        else:
            desired = unit(u.heading).astype(float)
        desired = _normalized(desired, u.heading)
        desired = _normalized(desired+c.field_obstacle_weight*self._obstacle_direction(field, xx, yy, u.heading), u.heading)
        angle = math.atan2(desired[1], desired[0])
        roi = np.array([float(v[94])*20000, float(v[96])*20000,
                        float(v[95])*20000, float(v[97])*20000])
        times = np.linspace(c.field_prediction_seconds/c.field_path_samples,
                            c.field_prediction_seconds, c.field_path_samples)
        best = None
        for shift in c.field_heading_offsets:
            command = unit(angle+shift).astype(float)
            # Candidate utility uses a bounded coordinated-turn approximation.
            # Exact finite-roll collision projection is applied to the winner.
            yaw = np.clip(wrap(angle+shift-u.heading)/max(c.uav_heading_time_constant, c.field_prediction_seconds),
                          -c.uav_config.max_yaw_rate, c.uav_config.max_yaw_rate)
            headings = u.heading+yaw*times
            midpoint = u.heading+yaw*times/2
            position = c.uav_speed*times[:, None]*np.column_stack((np.cos(midpoint), np.sin(midpoint)))
            indices = np.rint((position/(context[0]*3000)+1)*(field.shape[-1]-1)/2).astype(int)
            indices = np.clip(indices, 0, field.shape[-1]-1)
            samples = field[0, indices[:, 1], indices[:, 0]]
            preference = float(np.mean(np.minimum(samples, 0.)))
            hard = float(np.mean(field[2, indices[:, 1], indices[:, 0]]))
            outside = np.maximum(roi[:2]-position, 0.)+np.maximum(position-roi[2:], 0.)
            roi_cost = float(np.mean(np.linalg.norm(outside, axis=1)))/c.coverage_range
            _, radar_score = self._radar(field, context, xx, yy, float(headings[-1]), position[-1], c.field_prediction_seconds)
            score = (c.field_guidance_weight*float(command@desired)+radar_score+
                     c.field_preference_weight*preference-c.field_roi_weight*roi_cost-
                     c.field_obstacle_weight*hard-c.field_turn_weight*abs(wrap(angle+shift-u.heading)))
            if best is None or score > best[0]:
                best = (score, command)
        direction = _normalized(self._safe(v, u, best[1]), u.heading)
        # Use current physical heading for the current command, not the requested
        # direction (which can require tens of seconds of finite-bank turning).
        pan, _ = self._radar(field, context, xx, yy, u.heading, np.zeros(2), c.decision_dt)
        return np.array([*direction, np.clip(pan/(context[5]*math.pi), -1., 1.)], dtype=np.float32)


class FieldTrackingExpert(_FieldExpert):
    task = 'tracking'

    def act(self, obs):
        v, field, context, xx, yy = self._read(obs)
        c = self.config
        u = decode_ego(v, c)
        ship_velocity = v[97:99].astype(float)*(220/3.6)
        velocity = c.uav_speed*unit(u.heading).astype(float)
        delay = c.uav_roll_time_constant+c.uav_heading_time_constant+.5*c.decision_dt
        relative = -v[94:96].astype(float)*3000+(velocity-ship_velocity)*delay
        distance = max(1., float(np.linalg.norm(relative)))
        radial = relative/distance
        tangent = np.array([radial[1], -radial[0]])
        normal = _normalized(tangent-c.tracking_radial_gain*(distance-c.tracking_desired_distance)/c.tracking_desired_distance*radial, u.heading)
        dot = float(ship_velocity@normal)
        speed = -dot+math.sqrt(dot*dot+c.uav_speed**2-float(ship_velocity@ship_velocity))
        desired = _normalized(ship_velocity+speed*normal, u.heading)
        repulsion = c.field_obstacle_weight*self._obstacle_direction(field, xx, yy, u.heading)
        norm = float(np.linalg.norm(repulsion))
        if norm > c.field_tracking_repulsion_limit:
            repulsion *= c.field_tracking_repulsion_limit/norm
        direction = _normalized(self._safe(v, u, desired+repulsion), u.heading)
        to_ship = context[7:9]*3000
        relative_velocity = ship_velocity-velocity
        bearing_rate = float(to_ship[0]*relative_velocity[1]-to_ship[1]*relative_velocity[0])/max(1., float(to_ship@to_ship))
        yaw_rate = float(v[7])*c.uav_config.max_yaw_rate
        pan = wrap(math.atan2(to_ship[1], to_ship[0])-u.heading-context[4]*math.pi+
                   (bearing_rate-yaw_rate)*c.decision_dt/2)
        return np.array([*direction, np.clip(pan/(context[5]*math.pi), -1., 1.)], dtype=np.float32)
