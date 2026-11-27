"""obs-v1 schema. All policy and expert inputs share these exact arrays."""
import math
import numpy as np
from gymnasium import spaces

SCHEMA_VERSION='obs-v1'
WORLD_SCALE=300000.
RADAR_SCALE=3000.
ROI_SCALE=20000.
VELOCITY_SCALE=220/3.6


def observation_metadata(task,potential_enabled=False):
    common={'ego':[0,10],'boundary':[10,14],'clouds':[14,54],'islands':[54,94]}
    common.update({'roi':[94,98],'params':[98,111]} if task=='coverage' else {'ship':[94,103],'radar':[103,106],'params':[106,134]})
    uav_names=['uav_speed','uav_min_turn_radius','uav_max_bank','uav_roll_time_constant','uav_max_roll_rate','uav_heading_time_constant','uav_integration_step']
    uav_scales=[VELOCITY_SCALE,RADAR_SCALE,math.pi,1.,1.,1.,1.]
    if task=='coverage':
        parameter_names=uav_names+['decision_dt','physics_dt','coverage_range','coverage_half_angle','uav_cloud_margin','age_scale']
        parameter_scales=uav_scales+[1.,1.,RADAR_SCALE,math.pi,RADAR_SCALE,3600.]
    else:
        parameter_names=uav_names+['ship_speed','ship_min_turn_radius','ship_nomoto_time_constant','ship_max_rudder','ship_max_rudder_rate','ship_rudder_time_constant','ship_heading_time_constant','ship_yaw_damping','ship_integration_step','decision_dt','physics_dt','tracking_range','tracking_half_angle','pan_limit','max_pan_rate','uav_cloud_margin','ship_island_margin','tracking_min_distance','tracking_desired_distance','tracking_range_guard','tracking_angle_guard']
        parameter_scales=uav_scales+[VELOCITY_SCALE,RADAR_SCALE,1.,math.pi,1.,1.,1.,1.,1.,1.,1.,RADAR_SCALE,math.pi,math.pi,1.,RADAR_SCALE,RADAR_SCALE,RADAR_SCALE,RADAR_SCALE,RADAR_SCALE,math.pi]
    metadata={'schema_version':SCHEMA_VERSION,'task':task,'vector_size':111 if task=='coverage' else 134,
            'slices':common,'obstacle_shape':[8,5],
            'ego_order':['x/world','y/world','cos_heading','sin_heading','vx/speed','vy/speed','bank/bank_limit','yaw_rate/max_yaw_rate','previous_action_x','previous_action_y'],
            'boundary_order':['left','right','bottom','top'],
            'obstacle_order':['dx/3000','dy/3000','radius/3000','surface_distance/3000','valid'],
            'roi_order':['left_minus_x','right_minus_x','bottom_minus_y','top_minus_y'],
            'ship_order':['dx/3000','dy/3000','distance/3000','vx/Vref','vy/Vref','cos_heading','sin_heading','yaw_rate/max_yaw_rate','rudder/max_rudder'],
            'radar_order':['pan/pan_limit','bearing_error/half_angle','visible'],
            'parameter_order':parameter_names,'parameter_divisors':parameter_scales,
            'coverage_map_channels':['sea_mask','ever_seen','age/age_scale'],
            'coverage_map_origin':'lower; row south to north, column west to east',
            'scales':{'world':WORLD_SCALE,'radar':RADAR_SCALE,'roi':ROI_SCALE,'velocity':VELOCITY_SCALE}}
    if potential_enabled:
        metadata.update(schema_version='obs-field-v2',action_schema='action-field-v2',potential_shape=[4,33,33],field_context_size=16,
            potential_channels=['preference','scannable_water','hard_obstacle','query_valid'],
            potential_origin='row south to north; column west to east; inclusive centers at +/-R',
            field_context_order=['radius/3000','period/3600','decay','pan/pi','offset/pi','pan_limit/pi','max_pan_rate',
                'goal_dx/3000','goal_dy/3000','goal_priority','goal_valid','unseen_fraction','mean_preference','cloud_surface_distance/3000','coverage_ratio','tracking'])
        metadata.pop('coverage_map_channels',None);metadata.pop('coverage_map_origin',None)
    return metadata


def observation_space(task,potential_enabled=False):
    result={'vector':spaces.Box(-np.inf,np.inf,(111 if task=='coverage' else 134,),dtype=np.float32)}
    if potential_enabled:
        result['potential']=spaces.Box(-2.,1.,(4,33,33),dtype=np.float32)
        result['field_context']=spaces.Box(-np.inf,np.inf,(16,),dtype=np.float32)
    elif task=='coverage':result['coverage_map']=spaces.Box(0,np.inf,(3,64,64),dtype=np.float32)
    return spaces.Dict(result)


def _uav_params(c):
    return [c.uav_speed/VELOCITY_SCALE,c.uav_min_turn_radius/RADAR_SCALE,c.uav_max_bank/math.pi,c.uav_roll_time_constant,c.uav_max_roll_rate,c.uav_heading_time_constant,c.uav_integration_step]


def build_observation(env):
    c=env.config;u=env.uav;b=c.world_bounds;m=env.uav_model
    vx,vy=m.velocity(u)
    result=[u.x/WORLD_SCALE,u.y/WORLD_SCALE,math.cos(u.heading),math.sin(u.heading),vx/c.uav_speed,vy/c.uav_speed,u.bank/m.config.bank_limit,m.yaw_rate(u)/m.config.max_yaw_rate,*env.previous_action,
            (u.x-b[0])/WORLD_SCALE,(b[2]-u.x)/WORLD_SCALE,(u.y-b[1])/WORLD_SCALE,(b[3]-u.y)/WORLD_SCALE]
    for name in ('clouds','islands'):
        obstacles=sorted(env.scene[name],key=lambda o:o['id'])
        for ob in obstacles:
            dx,dy=ob['x']-u.x,ob['y']-u.y
            result.extend([dx/RADAR_SCALE,dy/RADAR_SCALE,ob['radius']/RADAR_SCALE,(math.hypot(dx,dy)-ob['radius'])/RADAR_SCALE,1.])
        result.extend([0.]*(5*(8-len(obstacles))))
    if env.task=='coverage':
        roi=env.scene['roi']
        result.extend([(roi[0]-u.x)/ROI_SCALE,(roi[2]-u.x)/ROI_SCALE,(roi[1]-u.y)/ROI_SCALE,(roi[3]-u.y)/ROI_SCALE])
        result.extend(_uav_params(c)+[c.decision_dt,c.physics_dt,c.coverage_range/RADAR_SCALE,c.coverage_half_angle/math.pi,c.uav_cloud_margin/RADAR_SCALE,c.age_scale/3600])
        return _complete_observation(env,result)
    s=env.ship;svx,svy=env.ship_model.velocity(s)
    result.extend([(s.x-u.x)/RADAR_SCALE,(s.y-u.y)/RADAR_SCALE,env.distance/RADAR_SCALE,svx/VELOCITY_SCALE,svy/VELOCITY_SCALE,math.cos(s.heading),math.sin(s.heading),s.yaw_rate/env.ship_model.config.max_yaw_rate,s.rudder/c.ship_max_rudder])
    result.extend([env.radar_pan/c.pan_limit,env.bearing_error/c.tracking_half_angle,float(env.target_visible)])
    result.extend(_uav_params(c)+[c.ship_speed/VELOCITY_SCALE,c.ship_min_turn_radius/RADAR_SCALE,c.ship_nomoto_time_constant,c.ship_max_rudder/math.pi,c.ship_max_rudder_rate,c.ship_rudder_time_constant,c.ship_heading_time_constant,c.ship_yaw_damping,c.ship_integration_step,c.decision_dt,c.physics_dt,c.tracking_range/RADAR_SCALE,c.tracking_half_angle/math.pi,c.pan_limit/math.pi,c.max_pan_rate,c.uav_cloud_margin/RADAR_SCALE,c.ship_island_margin/RADAR_SCALE,c.tracking_min_distance/RADAR_SCALE,c.tracking_desired_distance/RADAR_SCALE,c.tracking_range_guard/RADAR_SCALE,c.tracking_angle_guard/math.pi])
    return _complete_observation(env,result)


def _complete_observation(env,vector):
    result={'vector':np.array(vector,dtype=np.float32)}
    if env.config.potential_enabled:
        potential=env.potential_snapshot()
        result.update(potential=potential.copy(),field_context=env.field_context(potential))
    elif env.task=='coverage':result['coverage_map']=env.coverage_map()
    return result
