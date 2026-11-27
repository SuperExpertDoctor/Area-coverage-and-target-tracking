"""Moving-orbit guidance with physical short-horizon collision checks.

This is a deterministic heuristic, not the full research MPC in the design.
It receives only the same obs-v1 arrays available to the learned policy.
"""
import math
import numpy as np
from config import Config
from dynamics import FixedWingModel,FixedWingState
from geometry import wrap,unit,in_bounds,segment_circle_collision


class ExpertPlanningError(RuntimeError):
    pass


def decode_ego(vector,config):
    return FixedWingState(float(vector[0])*300000,float(vector[1])*300000,math.atan2(vector[3],vector[2]),float(vector[6])*config.uav_config.bank_limit)


def decode_clouds(vector,uav):
    return [{'x':uav.x+float(row[0])*3000,'y':uav.y+float(row[1])*3000,'radius':float(row[2])*3000,'id':i} for i,row in enumerate(vector[14:54].reshape(8,5)) if row[4]>.5]


class SafetyGuidance:
    def __init__(self,config,preferred_side=None):
        self.config=config
        self.model=FixedWingModel(config.uav_config)
        self.sides={}
        self.preferred_side=preferred_side

    def reset(self):self.sides={}

    def _forecast_clearance(self,uav,action,clouds):
        """Use the same finite-roll forecast for every candidate direction."""
        c=self.config;state=uav;clearance=math.inf
        count=int(math.ceil(c.expert_safety_horizon/c.expert_safety_projection_dt))
        projection_dt=c.expert_safety_horizon/count
        for _ in range(count):
            nxt=self.model.step_velocity(state,action,projection_dt)
            curve_margin=c.uav_speed*self.model.config.max_yaw_rate*projection_dt**2/8+c.expert_safety_extra_margin
            if not in_bounds((nxt.x,nxt.y),c.world_bounds,curve_margin):return None
            for ob in clouds:
                # Bound departure of the physical curve from each forecast chord.
                radius=ob['radius']+c.uav_cloud_margin+curve_margin
                if segment_circle_collision((state.x,state.y),(nxt.x,nxt.y),(ob['x'],ob['y']),radius):return None
                clearance=min(clearance,math.hypot(nxt.x-ob['x'],nxt.y-ob['y'])-radius)
            state=nxt
        return clearance

    def direction(self,uav,desired,clouds,obstacle_buffer=None):
        c=self.config
        soft_buffer=c.expert_obstacle_buffer if obstacle_buffer is None else obstacle_buffer
        if not math.isfinite(soft_buffer) or soft_buffer<0:
            raise ValueError('obstacle_buffer must be finite and nonnegative')
        direction=np.asarray(desired,dtype=float)
        norm=float(np.linalg.norm(direction))
        direction=direction/norm if norm>=1e-6 else unit(uav.heading).astype(float)
        requested_heading=math.atan2(direction[1],direction[0])
        p=np.array([uav.x,uav.y])
        # Resolve overlapping guidance zones from far to near: a distant cloud
        # must never overwrite the avoidance direction of an imminent cloud.
        for ob in sorted(clouds,key=lambda o:math.hypot(uav.x-o['x'],uav.y-o['y']),reverse=True):
            radial=p-np.array([ob['x'],ob['y']]);distance=float(np.linalg.norm(radial));radial/=max(1.,distance)
            hard=ob['radius']+c.uav_cloud_margin
            guard=hard+soft_buffer
            forward=float(direction@radial)
            if distance<guard+c.expert_safety_enter_extension and forward<c.expert_safety_forward_threshold:
                side=self.sides.get(ob['id'])
                if side is None:
                    left=np.array([-radial[1],radial[0]])
                    heading=unit(uav.heading)
                    side=self.preferred_side if self.preferred_side is not None else (1 if left@heading>=0 else -1)
                    self.sides[ob['id']]=side
                tangent=side*np.array([-radial[1],radial[0]])
                avoidance=tangent+radial*max(-c.expert_safety_inward_gain,c.expert_safety_radial_gain*(guard-distance)/guard)
                avoidance/=np.linalg.norm(avoidance)
                weight=min(1.,max(0.,(guard+c.expert_safety_enter_extension-distance)/c.expert_safety_enter_extension))
                direction=(1-weight)*direction+weight*avoidance
                norm=float(np.linalg.norm(direction))
                direction=direction/norm if norm>=1e-6 else unit(uav.heading).astype(float)
            elif distance>guard+c.expert_safety_exit_extension:self.sides.pop(ob['id'],None)
        # Total world boundaries are hard, whereas the ROI is intentionally permeable.
        b=c.world_bounds
        inward=np.zeros(2)
        buffer=c.expert_world_buffer_turn_factor*c.uav_min_turn_radius
        for distance,axis,sign in ((uav.x-b[0],0,1),(b[2]-uav.x,0,-1),(uav.y-b[1],1,1),(b[3]-uav.y,1,-1)):
            if distance<buffer:inward[axis]+=sign*max(0.,(buffer-distance)/buffer)*c.expert_world_inward_gain
        direction+=inward
        norm=float(np.linalg.norm(direction))
        direction=direction/norm if norm>=1e-6 else unit(uav.heading).astype(float)
        near=[ob for ob in clouds if math.hypot(uav.x-ob['x'],uav.y-ob['y'])<ob['radius']+c.uav_cloud_margin+c.expert_safety_activation_distance]
        if near or np.any(inward):
            heading=math.atan2(direction[1],direction[0])
            feasible=[]
            candidates=[(shift,unit(heading+shift)) for shift in c.expert_safety_heading_offsets]
            candidates.extend((abs(wrap(uav.heading+turn-heading)),unit(uav.heading+turn)) for turn in (-math.pi/2,math.pi/2))
            # Overlapping tangential fields can point away from a traversable
            # gap. If their candidates all fail, also test holding the current
            # heading; this still rolls out through the same physical checks.
            candidates.append((None,unit(uav.heading)))
            for shift,action in candidates:
                if shift is None:
                    if feasible:break
                    shift=abs(wrap(uav.heading-heading))
                clearance=self._forecast_clearance(uav,action,clouds)
                if clearance is not None:
                    feasible.append((abs(shift)-c.expert_safety_clearance_reward*min(clearance,c.expert_safety_clearance_cap),action))
                    if shift==0:return action
            if not feasible:
                # A cloud deflection can hide feasible headings from the original
                # shifted set. Search public current/requested references only
                # after that set fails, retaining the identical hard forecast.
                offsets=tuple(dict.fromkeys((*c.field_heading_offsets,*c.expert_safety_heading_offsets)))
                for reference in (uav.heading,requested_heading):
                    for offset in offsets:
                        angle=reference+offset;action=unit(angle)
                        clearance=self._forecast_clearance(uav,action,clouds)
                        if clearance is not None:
                            shift=abs(wrap(angle-heading))
                            feasible.append((shift-c.expert_safety_clearance_reward*min(clearance,c.expert_safety_clearance_cap),action))
            if not feasible:raise ExpertPlanningError(f'No safe physical {c.expert_safety_horizon:g}-second candidate at current state')
            return min(feasible,key=lambda item:item[0])[1]
        return np.array(direction,dtype=np.float32)


class TrackingExpert:
    def __init__(self,config=None):
        self.config=config if config is not None else Config()
        self.safety=SafetyGuidance(self.config)

    def reset(self):self.safety.reset()

    def act(self,obs):
        v=obs['vector'];c=self.config;u=decode_ego(v,c)
        relative=-v[94:96].astype(float)*3000
        ship_velocity=v[97:99].astype(float)*(220/3.6)
        # Preview the field by the actuator and heading response delay, using only
        # current relative motion. This changes the command, never the state.
        delay=c.uav_roll_time_constant+c.uav_heading_time_constant+.5*c.decision_dt
        velocity=c.uav_speed*np.array([math.cos(u.heading),math.sin(u.heading)])
        relative+=(velocity-ship_velocity)*delay
        distance=max(1.,float(np.linalg.norm(relative)))
        er=relative/distance;tangent=np.array([er[1],-er[0]])
        n=tangent-c.tracking_radial_gain*(distance-c.tracking_desired_distance)/c.tracking_desired_distance*er
        n/=np.linalg.norm(n)
        dot=float(ship_velocity@n)
        speed=-dot+math.sqrt(dot*dot+c.uav_speed**2-float(ship_velocity@ship_velocity))
        desired=ship_velocity+speed*n
        # A coverage-sized soft detour can leave the radar envelope when the
        # ship's island bypass approaches a cloud. Spend at most the tracking
        # reserve on this early vector-field repulsion; the hard cloud margin
        # and complete physical safety forecast remain unchanged.
        return self.safety.direction(u,desired,decode_clouds(v,u),
                                    obstacle_buffer=min(c.expert_obstacle_buffer,c.tracking_range_guard))
