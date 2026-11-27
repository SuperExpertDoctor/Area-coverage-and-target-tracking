"""Independent mobile-target tracking with rate-limited physical radar pan."""
from env_base import BaseMaritimeEnv


class TrackingEnv(BaseMaritimeEnv):
    task='tracking'

    def _update_task(self,initial=False):
        if initial:return
        self.total_tracking_ticks+=1
        self.visible_ticks+=int(self.target_visible)
        self.min_distance=min(self.min_distance,self.distance)
        self.max_distance=max(self.max_distance,self.distance)
        self.distance_sum+=self.distance
        if not self.target_visible and self.first_loss_time is None:self.first_loss_time=self.t
