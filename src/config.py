"""Discoverable SI-unit configuration shared by the two independent tasks."""
from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
from pathlib import Path

from dynamics import FixedWingConfig, ShipConfig


@dataclass
class Config:
    world_bounds: tuple = (0., 0., 300000., 300000.)
    roi_min_size: float = 10000.
    roi_max_size: float = 20000.
    roi_outer_buffer: float = 6000.
    coverage_grid: int = 64
    cloud_min_count: int = 2
    cloud_max_count: int = 6
    max_clouds: int = 8
    cloud_min_radius: float = 500.
    cloud_max_radius: float = 1000.
    island_min_count: int = 1
    island_max_count: int = 4
    max_islands: int = 8
    island_min_radius: float = 500.
    island_max_radius: float = 2000.
    min_water_fraction: float = .70
    uav_cloud_margin: float = 200.
    ship_island_margin: float = 100.
    navigation_grid_resolution: float = 250.
    decision_dt: float = 1.
    physics_dt: float = .1
    coverage_episode_seconds: float = 10800.
    tracking_episode_seconds: float = 1800.
    max_scene_attempts: int = 200
    per_scene_start_attempts: int = 100
    uav_speed: float = 220 / 3.6
    uav_min_turn_radius: float = 1000.
    uav_max_bank: float = math.radians(25)
    uav_roll_time_constant: float = 1.
    uav_max_roll_rate: float = math.radians(15)
    uav_heading_time_constant: float = 1.
    uav_integration_step: float = .05
    ship_speed: float = 12 * 1852 / 3600
    ship_min_turn_radius: float = 300.
    ship_nomoto_time_constant: float = 20.
    ship_max_rudder: float = math.radians(35)
    ship_max_rudder_rate: float = math.radians(3)
    ship_rudder_time_constant: float = 1.
    ship_heading_time_constant: float = 30.
    ship_yaw_damping: float = .8
    ship_integration_step: float = .05
    coverage_range: float = 3000.
    coverage_half_angle: float = math.radians(30)
    coverage_offset: float = -math.pi / 2
    tracking_range: float = 3000.
    tracking_half_angle: float = math.radians(30)
    pan_limit: float = math.radians(120)
    max_pan_rate: float = math.radians(30)
    potential_enabled: bool = False
    potential_radius: float = 6000.
    potential_period: float = 3600.
    potential_decay: float = 3.
    coverage_pan_limit: float = math.pi / 3
    potential_goal_hold_seconds: float = 180.
    potential_goal_radius: float = 1500.
    potential_unseen_bonus: float = 2.
    field_heading_offsets: tuple = (0., -.18, .18, -.36, .36)
    field_prediction_seconds: float = 8.
    field_path_samples: int = 4
    field_guidance_weight: float = 2.
    field_scan_weight: float = 1.4
    field_preference_weight: float = .2
    field_roi_weight: float = .3
    field_turn_weight: float = .05
    field_obstacle_weight: float = 1.
    field_obstacle_influence: float = 2500.
    field_coverage_obstacle_buffer: float = 200.
    field_tracking_repulsion_limit: float = .12
    field_pan_samples: int = 9
    field_pan_goal_weight: float = .6
    field_pan_slew_weight: float = .05
    visual_show_potential: bool = True
    tracking_desired_distance: float = 2000.
    tracking_initial_min_distance: float = 1600.
    tracking_initial_max_distance: float = 2400.
    tracking_min_distance: float = 1300.
    tracking_range_guard: float = 200.
    tracking_angle_guard: float = math.radians(5)
    age_scale: float = 3600.
    ship_lookahead: float = 700.
    ship_route_spacing: float = 25.
    ship_spiral_turns: float = 4.
    ship_spiral_inner_radius: float = 30000.
    ship_spiral_outer_radius: float = 125000.
    coverage_lane_spacing: float = 2200.
    coverage_sweep_timeout: float = 5400.
    expert_lookahead: float = 650.
    expert_obstacle_buffer: float = 1300.
    tracking_radial_gain: float = 3.
    coverage_outer_offset: float = 700.
    coverage_corner_radius_factor: float = 1.35
    coverage_path_spacing: float = 100.
    coverage_center_radius_factor: float = 1.4
    coverage_center_path_points: int = 140
    coverage_min_rectangle_factor: float = 2.8
    coverage_repair_timeout: float = 360.
    coverage_repair_min_seconds: float = 160.
    coverage_repair_capture_distance: float = 2500.
    coverage_repair_hit_radius: float = 1500.
    coverage_repair_orbit_radius: float = 1700.
    coverage_repair_radial_gain: float = 2.5
    coverage_unseen_weight: float = 20.
    coverage_age_weight: float = 1.
    coverage_distance_scale: float = 15000.
    coverage_recent_goal_factor: float = .1
    coverage_recent_goal_count: int = 4
    coverage_completion_index_margin: int = 12
    coverage_route_local_search_points: int = 65
    coverage_planning_blocks: int = 8
    coverage_rescan_gap_seconds: float = 60.
    render_grid_stride: int = 4
    expert_safety_horizon: float = 30.
    expert_safety_projection_dt: float = 5.
    expert_safety_activation_distance: float = 2200.
    expert_safety_enter_extension: float = 1700.
    expert_safety_exit_extension: float = 2200.
    expert_safety_radial_gain: float = 2.5
    expert_safety_inward_gain: float = .2
    expert_safety_forward_threshold: float = .45
    expert_world_buffer_turn_factor: float = 2.5
    expert_world_inward_gain: float = 3.
    expert_safety_heading_offsets: tuple = (0., -.4, .4, -.8, .8, -1.4, 1.4)
    expert_safety_clearance_reward: float = .00005
    expert_safety_clearance_cap: float = 1000.
    expert_safety_extra_margin: float = 10.
    ship_planning_radius_factor: float = 2.
    ship_route_extra_margin: float = 500.
    ship_astar_padding: float = 2000.
    ship_bypass_smoothing_radius: float = 600.
    ship_bypass_corner_fraction: float = .4
    ship_route_local_search_points: int = 80
    ship_spiral_generation_points: int = 6000
    ship_spiral_world_radius_factor: float = .42
    ship_spiral_inner_ratio: float = .4
    ship_bypass_lead_distance: float = 1000.
    coverage_cloud_padding: float = 2500.
    tracking_island_sampling_extent: float = 14000.
    tracking_cloud_sampling_extent: float = 18000.
    tracking_cloud_corridor_clearance: float = 3500.
    tracking_corridor_sample_stride: int = 20
    tracking_corridor_lookahead_distance: float = 2500.
    visual_grid_cell_size: float = 250.
    visual_grid_fade_seconds: float = 600.
    visual_grid_min_alpha: float = .025
    visual_grid_max_alpha: float = .55
    visual_grid_min_spacing_px: float = 12.
    visual_uav_length: float = 9.
    visual_uav_wingspan: float = 18.
    visual_ship_length: float = 83.
    visual_ship_beam: float = 16.
    visual_uav_min_pixels: float = 12.
    visual_ship_min_pixels: float = 8.
    visual_trail_seconds: float = 60.
    visual_show_grid: bool = True
    visual_show_trail: bool = True
    visual_show_radar_age: bool = True
    visual_trail_style: str = 'meteor'

    def __post_init__(self):
        self.world_bounds = tuple(float(v) for v in self.world_bounds)
        if len(self.world_bounds) != 4 or not all(math.isfinite(v) for v in self.world_bounds):
            raise ValueError('world_bounds must contain four finite coordinates')
        if self.world_bounds[2] <= self.world_bounds[0] or self.world_bounds[3] <= self.world_bounds[1]:
            raise ValueError('world_bounds must have positive area')
        for field in fields(self):
            value = getattr(self, field.name)
            if field.type is bool and not isinstance(value,bool):
                raise ValueError(f'{field.name} must be a boolean')
            if isinstance(value, (int, float)) and not math.isfinite(value):
                raise ValueError(f'{field.name} must be finite')
            if field.type is int:
                if isinstance(value,bool) or int(value)!=value:raise ValueError(f'{field.name} must be an integer')
                setattr(self,field.name,int(value))
        self.expert_safety_heading_offsets=tuple(float(v) for v in self.expert_safety_heading_offsets)
        self.field_heading_offsets=tuple(float(v) for v in self.field_heading_offsets)
        if not self.field_heading_offsets or not all(math.isfinite(v) and abs(v)<=math.pi for v in self.field_heading_offsets):
            raise ValueError('Field heading offsets must be finite angles in [-pi,pi]')
        if not self.expert_safety_heading_offsets or not all(math.isfinite(v) and abs(v)<=math.pi for v in self.expert_safety_heading_offsets):
            raise ValueError('Safety heading offsets must be finite angles in [-pi,pi]')
        nonnegative={'cloud_min_count','cloud_max_count','island_min_count','island_max_count','uav_cloud_margin','ship_island_margin','tracking_range_guard','tracking_angle_guard','roi_outer_buffer','expert_obstacle_buffer','field_coverage_obstacle_buffer','potential_unseen_bonus','field_tracking_repulsion_limit','field_guidance_weight','field_scan_weight','field_preference_weight','field_roi_weight','field_turn_weight','field_obstacle_weight','field_pan_goal_weight','field_pan_slew_weight'}
        for field in fields(self):
            value=getattr(self,field.name)
            if isinstance(value,bool) or not isinstance(value,(int,float)) or field.name=='coverage_offset':continue
            if value<0 or (value==0 and field.name not in nonnegative):
                raise ValueError(f'{field.name} must be positive (or a permitted zero margin/count)')
        for name in ('cloud_min_count','cloud_max_count','island_min_count','island_max_count','max_scene_attempts','per_scene_start_attempts','coverage_grid','max_clouds','max_islands'):
            if int(getattr(self,name))!=getattr(self,name):raise ValueError(f'{name} must be an integer')
        if not 0<self.min_water_fraction<=1:raise ValueError('min_water_fraction must be in (0,1]')
        if not 0<self.visual_grid_min_alpha<=self.visual_grid_max_alpha<=1:
            raise ValueError('Visual grid alpha must satisfy 0 < min <= max <= 1')
        if self.visual_trail_style not in ('meteor','full'):
            raise ValueError('visual_trail_style must be meteor or full')
        if not 0<self.coverage_half_angle<math.pi/2 or not 0<self.tracking_half_angle<math.pi/2 or not 0<self.pan_limit<math.pi:
            raise ValueError('Invalid radar angular limits')
        if not 0<self.coverage_pan_limit<math.pi:
            raise ValueError('coverage_pan_limit must be in (0,pi)')
        if self.ship_route_spacing>25:raise ValueError('ship_route_spacing may not exceed25m')
        if 64 % self.coverage_planning_blocks:raise ValueError('coverage_planning_blocks must divide64')
        if self.expert_safety_projection_dt>self.expert_safety_horizon:raise ValueError('Safety projection dt exceeds horizon')
        if not 0<self.ship_bypass_corner_fraction<.5:raise ValueError('ship_bypass_corner_fraction must be below.5')
        if self.cloud_min_radius>self.cloud_max_radius or self.island_min_radius>self.island_max_radius:
            raise ValueError('Obstacle radius ranges must be ordered')
        if not 0<self.tracking_initial_min_distance<=self.tracking_initial_max_distance<self.tracking_range:
            raise ValueError('Invalid tracking initial distance range')
        if self.tracking_range_guard>=self.tracking_range or self.tracking_angle_guard>=self.tracking_half_angle:
            raise ValueError('Tracking guards must be smaller than radar limits')
        if not math.isclose(self.coverage_offset,-math.pi/2):raise ValueError('obs-v1 coverage radar is fixed to the right side')
        if self.decision_dt <= 0 or self.physics_dt <= 0 or abs(round(self.decision_dt/self.physics_dt)-self.decision_dt/self.physics_dt) > 1e-8:
            raise ValueError('decision_dt must be a positive integer multiple of physics_dt')
        if self.coverage_grid != 64 or self.max_clouds != 8 or self.max_islands != 8:
            raise ValueError('obs-v1 fixes coverage_grid=64 and obstacle slots=8')
        if not 0 <= self.cloud_min_count <= self.cloud_max_count <= self.max_clouds or not 0 <= self.island_min_count <= self.island_max_count <= self.max_islands:
            raise ValueError('invalid obstacle counts')
        if not 0 < self.roi_min_size <= self.roi_max_size or self.coverage_episode_seconds <= 0 or self.tracking_episode_seconds <= 0:
            raise ValueError('sizes and episode durations must be positive')
        if not self.tracking_min_distance < self.tracking_desired_distance < self.tracking_range:
            raise ValueError('tracking distances must be ordered')
        if self.tracking_initial_min_distance<self.tracking_min_distance or self.tracking_initial_max_distance>self.tracking_range-self.tracking_range_guard:
            raise ValueError('Initial tracking distances must stay inside the guarded radar envelope')
        self.uav_config
        self.ship_config

    @property
    def uav_config(self):
        return FixedWingConfig(**{f.name: getattr(self, 'uav_'+f.name) for f in fields(FixedWingConfig)})

    @property
    def ship_config(self):
        return ShipConfig(**{f.name: getattr(self, 'ship_'+f.name) for f in fields(ShipConfig)})

    def to_dict(self):
        return json.loads(json.dumps(asdict(self)))

    @classmethod
    def from_dict(cls, value):
        return cls(**value)

    @property
    def config_hash(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True, separators=(',', ':')).encode()).hexdigest()


TaskConfig = Config


def load_config(path):
    return Config.from_dict(json.loads(Path(path).read_text(encoding='utf-8')))


def save_config(config, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(config.to_dict(), indent=2), encoding='utf-8')


def make_config(task=None):
    return Config()
