"""Gymnasium decisions backed by the original finite-response physical models."""
from copy import deepcopy
from dataclasses import asdict
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from config import Config
from dynamics import FixedWingModel, FixedWingState, ShipModel, ShipState
from geometry import in_bounds, segment_circle_collision, wrap
from navigation import ShipNavigator
from observations import build_observation, observation_space, observation_metadata
from scenario import sample_coverage_scene, sample_tracking_scene, validate_scene


class BaseMaritimeEnv(gym.Env):
    metadata={'render_modes':['json','human'],'render_fps':10}
    task=None

    def __init__(self,config=None,render_mode=None,render_callback=None):
        self.config=config if config is not None else Config()
        if isinstance(self.config,dict):self.config=Config.from_dict(self.config)
        self.render_mode=render_mode
        self.render_callback=render_callback
        self.uav_model=FixedWingModel(self.config.uav_config)
        self.ship_model=ShipModel(self.config.ship_config)
        self.observation_space=observation_space(self.task)
        self.action_space=spaces.Box(-1.,1.,(2,),dtype=np.float32)
        self.observation_schema=observation_metadata(self.task)
        self._done=True
        self.scene=None

    @property
    def t(self):return self.physics_tick*self.config.physics_dt

    @property
    def scene_spec(self):return self.scene

    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed)
        if options and 'scene' in options:
            scene=options['scene']
            if hasattr(scene,'to_dict'):scene=scene.to_dict()
            scene=deepcopy(scene)
            validate_scene(scene,self.config,self.task)
        else:
            sampler=sample_coverage_scene if self.task=='coverage' else sample_tracking_scene
            scene=sampler(self.config,self.np_random,int(seed if seed is not None else self.np_random.integers(0,2**31)))
        self.scene=scene
        self.uav=FixedWingState(**scene['uav_initial'])
        self.ship=ShipState(**scene['ship_initial'])
        self.ship_navigator=ShipNavigator(scene['ship_route'],self.config.ship_lookahead,self.config.ship_route_local_search_points)
        self.physics_tick=0
        self.radar_pan=float(scene['radar_pan_initial'])
        self.previous_action=np.array([math.cos(self.uav.heading),math.sin(self.uav.heading)],dtype=np.float32)
        self.termination_reason=None
        self.uav_collision=False;self.ship_collision=False;self.out_of_bounds=False
        self.first_loss_time=None;self.visible_ticks=0;self.total_tracking_ticks=0
        self.min_distance=math.inf;self.max_distance=0.;self.distance_sum=0.
        self._done=False
        self._reset_task()
        self._update_radar(0.)
        self.min_distance=self.max_distance=self.distance
        self._update_task(initial=True)
        return build_observation(self),self._info(0.)

    def _reset_task(self):pass

    def _update_task(self,initial=False):pass

    def _update_radar(self,dt):
        dx,dy=self.ship.x-self.uav.x,self.ship.y-self.uav.y
        self.distance=math.hypot(dx,dy)
        bearing=math.atan2(dy,dx)
        if self.task=='tracking':
            desired=max(-self.config.pan_limit,min(self.config.pan_limit,wrap(bearing-self.uav.heading)))
            self.radar_pan+=max(-self.config.max_pan_rate*dt,min(self.config.max_pan_rate*dt,desired-self.radar_pan))
            self.bearing_error=wrap(bearing-self.uav.heading-self.radar_pan)
            self.target_visible=self.distance<=self.config.tracking_range and abs(self.bearing_error)<=self.config.tracking_half_angle
        else:
            self.bearing_error=wrap(bearing-self.uav.heading-self.config.coverage_offset)
            self.target_visible=self.distance<=self.config.coverage_range and abs(self.bearing_error)<=self.config.coverage_half_angle

    def _safety(self,old_uav,old_ship,dt):
        c=self.config
        uc=c.uav_speed*self.uav_model.config.max_yaw_rate*dt*dt/8
        sc=c.ship_speed*self.ship_model.config.max_yaw_rate*dt*dt/8
        self.uav_collision=any(segment_circle_collision((old_uav.x,old_uav.y),(self.uav.x,self.uav.y),(ob['x'],ob['y']),ob['radius']+c.uav_cloud_margin+uc) for ob in self.scene['clouds'])
        self.ship_collision=any(segment_circle_collision((old_ship.x,old_ship.y),(self.ship.x,self.ship.y),(ob['x'],ob['y']),ob['radius']+c.ship_island_margin+sc) for ob in self.scene['islands'])
        self.out_of_bounds=not in_bounds((self.uav.x,self.uav.y),c.world_bounds,uc) or not in_bounds((self.ship.x,self.ship.y),c.world_bounds,sc)
        if self.uav_collision:self.termination_reason='uav_cloud_collision'
        elif self.ship_collision:self.termination_reason='ship_island_collision'
        elif self.out_of_bounds:self.termination_reason='world_boundary'

    def step(self,action):
        if self._done:raise RuntimeError('reset() is required before step() or after episode end')
        command=np.asarray(action,dtype=float)
        if command.shape!=(2,) or not np.isfinite(command).all() or np.any(np.abs(command)>1.):
            raise ValueError('action must be a finite two-vector inside [-1,1]')
        norm=float(np.linalg.norm(command))
        command=command/norm if norm>=1e-6 else np.array([math.cos(self.uav.heading),math.sin(self.uav.heading)])
        start=self.t;c=self.config
        for _ in range(int(round(c.decision_dt/c.physics_dt))):
            if self.t>=self.scene['episode_seconds']-1e-8:break
            old_uav,old_ship=self.uav,self.ship
            try:
                heading=self.ship_navigator.heading_command(self.ship)
                self.ship=self.ship_model.step_heading(self.ship,heading,c.physics_dt)
                self.uav=self.uav_model.step_velocity(self.uav,command,c.physics_dt)
            except (ValueError,OverflowError):
                self.termination_reason='nonfinite_physical_state';break
            self.physics_tick+=1
            self._update_radar(c.physics_dt)
            self._safety(old_uav,old_ship,c.physics_dt)
            self._update_task()
            if self.termination_reason:break
        self.previous_action=np.array(command,dtype=np.float32)
        terminated=self.termination_reason is not None
        truncated=not terminated and self.t>=self.scene['episode_seconds']-1e-8
        if truncated:self.termination_reason='time_limit'
        self._done=terminated or truncated
        return build_observation(self),0.,terminated,truncated,self._info(self.t-start)

    def _info(self,elapsed):
        result={'time_seconds':self.t,'physics_tick':self.physics_tick,'elapsed_seconds':elapsed,'executed_action':np.array(self.uav_model.velocity(self.uav),dtype=np.float32)/self.config.uav_speed,'termination_reason':self.termination_reason,'uav_collision':self.uav_collision,'ship_collision':self.ship_collision,'out_of_bounds':self.out_of_bounds}
        if self.task=='coverage':result.update(coverage_ratio=self.coverage_ratio,mean_age=self.mean_age,max_age=self.max_age,first_full_coverage_time=self.first_full_coverage_time,rescan_ratio=self.rescan_ratio)
        else:result.update(target_visible=self.target_visible,distance=self.distance,bearing_error=self.bearing_error,radar_pan=self.radar_pan,first_loss_time=self.first_loss_time)
        return result

    def summary(self):
        result={'task':self.task,'time_seconds':self.t,'termination_reason':self.termination_reason,'uav_collision':self.uav_collision,'ship_collision':self.ship_collision,'out_of_bounds':self.out_of_bounds}
        if self.task=='coverage':result.update(coverage_ratio=self.coverage_ratio,mean_age=self.mean_age,max_age=self.max_age,first_full_coverage_time=self.first_full_coverage_time,rescan_ratio=self.rescan_ratio)
        else:result.update(visible_fraction=self.visible_ticks/self.total_tracking_ticks if self.total_tracking_ticks else float(self.target_visible),first_loss_time=self.first_loss_time,min_distance=self.min_distance if math.isfinite(self.min_distance) else self.distance,max_distance=self.max_distance,mean_distance=self.distance_sum/self.total_tracking_ticks if self.total_tracking_ticks else self.distance,distance=self.distance)
        return result

    def render(self):
        if self.scene is None:return None
        u=asdict(self.uav);s=asdict(self.ship)
        u.update(zip(('vx','vy'),self.uav_model.velocity(self.uav)))
        s.update(zip(('vx','vy'),self.ship_model.velocity(self.ship)))
        c=self.config
        coverage={'ratio':self.coverage_ratio,'mean_age':self.mean_age,'max_age':self.max_age,'rescan_ratio':self.rescan_ratio} if self.task=='coverage' else {'ratio':0.,'mean_age':0.,'max_age':0.}
        if self.task=='coverage':
            stride=c.render_grid_stride
            coverage['grid']=np.where(self.sea_mask,self.ages(),-1.)[::stride,::stride].tolist()
            coverage['seen']=(self.last_seen_tick>=0)[::stride,::stride].tolist()
            coverage['grid_bounds']=self.scene['roi']
            coverage['grid_max_age']=max(1.,self.max_age)
        frame={'t':self.t,'task':self.task,'uav':u,'ship':s,'radar':{'range':c.coverage_range if self.task=='coverage' else c.tracking_range,'half_angle':c.coverage_half_angle if self.task=='coverage' else c.tracking_half_angle,'pan':self.radar_pan,'offset':c.coverage_offset if self.task=='coverage' else 0.,'visible':self.target_visible},'coverage':coverage,'metrics':{'distance':self.distance,'bearing_error':self.bearing_error,'visible':self.target_visible,'coverage_ratio':coverage['ratio'],'mean_age':coverage['mean_age'],'max_age':coverage['max_age']},'termination_reason':self.termination_reason}
        if self.render_callback is not None:self.render_callback(frame)
        return frame

    def close(self):pass
