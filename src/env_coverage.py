"""Persistent right-side radar coverage; a first pass never terminates."""
import numpy as np
from env_base import BaseMaritimeEnv
from geometry import sector_mask,water_grid


class CoverageEnv(BaseMaritimeEnv):
    task='coverage'

    def _reset_task(self):
        self.grid_x,self.grid_y,self.sea_mask=water_grid(self.scene['roi'],self.scene['islands'],self.config.coverage_grid)
        self.last_seen_tick=np.full(self.sea_mask.shape,-1,dtype=np.int32)
        self.scan_count=np.zeros(self.sea_mask.shape,dtype=np.int32)
        self.first_full_coverage_time=None

    def _update_task(self,initial=False):
        c=self.config
        heading=self.uav.heading+c.coverage_offset+(self.radar_pan if c.potential_enabled else 0.)
        hit=sector_mask(self.grid_x,self.grid_y,(self.uav.x,self.uav.y),heading,c.coverage_range,c.coverage_half_angle)&self.sea_mask
        new_sweep=hit&((self.last_seen_tick<0)|((self.physics_tick-self.last_seen_tick)*c.physics_dt>c.coverage_rescan_gap_seconds))
        self.scan_count[new_sweep]+=1
        self.last_seen_tick[hit]=self.physics_tick
        if c.potential_enabled:self.potential_field.mark_scan((self.uav.x,self.uav.y),heading,c.coverage_range,c.coverage_half_angle,self.t)
        if self.first_full_coverage_time is None and np.all(self.last_seen_tick[self.sea_mask]>=0):self.first_full_coverage_time=self.t

    def ages(self):return np.where(self.sea_mask,np.where(self.last_seen_tick>=0,(self.physics_tick-self.last_seen_tick)*self.config.physics_dt,self.t),0.)

    def coverage_map(self):return np.stack((self.sea_mask,self.sea_mask&(self.last_seen_tick>=0),self.ages()/self.config.age_scale)).astype(np.float32)

    @property
    def coverage_ratio(self):return float(np.mean(self.last_seen_tick[self.sea_mask]>=0))

    @property
    def mean_age(self):return float(np.mean(self.ages()[self.sea_mask]))

    @property
    def max_age(self):return float(np.max(self.ages()[self.sea_mask]))

    @property
    def rescan_ratio(self):return float(np.mean(self.scan_count[self.sea_mask]>=2))
