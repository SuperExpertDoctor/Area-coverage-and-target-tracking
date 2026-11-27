"""Bounded reproducible scene sampling and complete JSON scene replay."""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import numpy as np

from dynamics import GRAVITY, FixedWingState, ShipState
from geometry import in_bounds, wrap, water_grid, line_clear
from navigation import global_spiral, route_around_islands


class SceneGenerationError(ValueError):
    pass


class SceneValidationError(ValueError):
    pass


@dataclass(frozen=True)
class CircleObstacle:
    id: int
    x: float
    y: float
    radius: float


@dataclass(frozen=True)
class SceneSpec:
    schema_version: str
    task: str
    seed: int
    world_bounds: tuple
    roi: tuple | None
    clouds: tuple
    islands: tuple
    uav_initial: dict
    ship_initial: dict
    ship_route: tuple
    radar_pan_initial: float
    episode_seconds: float
    config_snapshot: dict

    def to_dict(self):
        return json.loads(json.dumps(asdict(self)))

    @classmethod
    def from_dict(cls, scene):
        return cls(**scene)


def scene_hash(scene):
    if hasattr(scene,'to_dict'):scene=scene.to_dict()
    return hashlib.sha256(json.dumps(scene,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def validate_scene(scene,config,task=None):
    if isinstance(scene,SceneSpec):scene=scene.to_dict()
    try:
        if scene['schema_version']!='scene-v1' or (task and scene['task']!=task):
            raise SceneValidationError('Scene schema/task mismatch')
        if tuple(scene['world_bounds'])!=tuple(config.world_bounds):
            raise SceneValidationError('Scene/config world bounds mismatch')
        for name,max_count in [('clouds',config.max_clouds),('islands',config.max_islands)]:
            if len(scene[name])>max_count:raise SceneValidationError('Obstacle slot overflow')
            ids=[]
            for ob in scene[name]:
                if not all(math.isfinite(ob[k]) for k in ('x','y','radius')) or ob['radius']<=0:
                    raise SceneValidationError('Invalid obstacle')
                ids.append(ob['id'])
            if len(set(ids))!=len(ids):raise SceneValidationError('Duplicate obstacle id')
        u=FixedWingState(**scene['uav_initial']);s=ShipState(**scene['ship_initial'])
        from dynamics import FixedWingModel,ShipModel
        FixedWingModel(config.uav_config).velocity(u);ShipModel(config.ship_config).velocity(s)
        if not in_bounds((u.x,u.y),config.world_bounds) or not in_bounds((s.x,s.y),config.world_bounds):
            raise SceneValidationError('Initial state outside world')
        for state,obstacles,margin in [(u,scene['clouds'],config.uav_cloud_margin),(s,scene['islands'],config.ship_island_margin)]:
            if any(math.hypot(state.x-o['x'],state.y-o['y'])<=o['radius']+margin for o in obstacles):
                raise SceneValidationError('Initial obstacle collision')
        route=np.asarray(scene['ship_route'],dtype=float)
        if route.ndim!=2 or route.shape[1]!=2 or len(route)<2 or not np.isfinite(route).all():
            raise SceneValidationError('Invalid ship route')
        b=config.world_bounds
        if np.any(route[:,0]<b[0]) or np.any(route[:,0]>b[2]) or np.any(route[:,1]<b[1]) or np.any(route[:,1]>b[3]):
            raise SceneValidationError('Ship route outside world')
        route_segments=np.diff(route,axis=0)
        route_lengths2=np.sum(route_segments*route_segments,axis=1)
        if np.any(route_lengths2>25.01**2):
            raise SceneValidationError('Ship route samples exceed25m spacing')
        for ob in scene['islands']:
            relative=np.array([ob['x'],ob['y']])-route[:-1]
            fraction=np.clip(np.sum(relative*route_segments,axis=1)/np.maximum(route_lengths2,1e-12),0,1)
            closest=route[:-1]+route_segments*fraction[:,None]
            if np.any(np.sum((closest-np.array([ob['x'],ob['y']]))**2,axis=1)<=(ob['radius']+config.ship_island_margin)**2):
                raise SceneValidationError('Ship route crosses an inflated island')
        if math.dist(route[0],(s.x,s.y))>1:
            raise SceneValidationError('Ship initial state does not match route start')
        pan_limit=config.coverage_pan_limit if config.potential_enabled and scene['task']=='coverage' else config.pan_limit
        if not math.isfinite(scene['radar_pan_initial']) or abs(scene['radar_pan_initial'])>pan_limit:
            raise SceneValidationError('Invalid initial radar pan')
        if not math.isfinite(scene['episode_seconds']) or scene['episode_seconds']<=0:
            raise SceneValidationError('Invalid episode duration')
        if abs(round(scene['episode_seconds']/config.physics_dt)-scene['episode_seconds']/config.physics_dt)>1e-7:
            raise SceneValidationError('Episode duration must end on an integer physics tick')
        if scene['task']=='coverage':
            roi=scene['roi']
            if roi is None or len(roi)!=4 or roi[2]<=roi[0] or roi[3]<=roi[1] or not in_bounds(roi[:2],b) or not in_bounds(roi[2:],b):
                raise SceneValidationError('Invalid coverage ROI')
            if water_grid(roi,scene['islands'])[2].mean()<config.min_water_fraction:
                raise SceneValidationError('Insufficient sea area')
        elif scene['task']=='tracking':
            d=math.hypot(s.x-u.x,s.y-u.y)
            error=wrap(math.atan2(s.y-u.y,s.x-u.x)-u.heading-scene['radar_pan_initial'])
            if d<config.tracking_min_distance or d>config.tracking_range-config.tracking_range_guard or abs(error)>config.tracking_half_angle-config.tracking_angle_guard:
                raise SceneValidationError('Tracking target initially outside guarded radar envelope')
        else:raise SceneValidationError('Unknown task')
    except (KeyError,TypeError) as error:
        raise SceneValidationError(f'Malformed scene: {error}') from error
    return True


def _obstacles(rng,count,center,extent,rmin,rmax):
    return [{'id':i,'x':float(center[0]+rng.uniform(-extent[0],extent[0])),
             'y':float(center[1]+rng.uniform(-extent[1],extent[1])),
             'radius':float(rng.uniform(rmin,rmax))} for i in range(count)]


def _sample(config,rng,task,seed):
    rejected={}
    for _ in range(config.max_scene_attempts):
        try:
            b=config.world_bounds
            base=global_spiral(config,float(rng.uniform(-math.pi,math.pi)))
            if task=='coverage':
                w,h=rng.uniform(config.roi_min_size,config.roi_max_size,2)
                buffer=config.roi_outer_buffer
                cx=float(rng.uniform(b[0]+buffer+w/2,b[2]-buffer-w/2))
                cy=float(rng.uniform(b[1]+buffer+h/2,b[3]-buffer-h/2))
                roi=[cx-w/2,cy-h/2,cx+w/2,cy+h/2]
                islands=_obstacles(rng,int(rng.integers(config.island_min_count,config.island_max_count+1)),(cx,cy),(w/2,h/2),config.island_min_radius,config.island_max_radius)
                # Cloud placement retains broad maneuvering corridors, including around ROI corners.
                clouds=_obstacles(rng,int(rng.integers(config.cloud_min_count,config.cloud_max_count+1)),(cx,cy),(w/2+config.coverage_cloud_padding,h/2+config.coverage_cloud_padding),config.cloud_min_radius,config.cloud_max_radius)
                arc=float(rng.uniform(0,2*(w+h)))
                if arc<h:x,y,heading=roi[0],roi[1]+arc,math.pi/2
                elif arc<h+w:x,y,heading=roi[0]+arc-h,roi[3],0.
                elif arc<2*h+w:x,y,heading=roi[2],roi[3]-(arc-h-w),-math.pi/2
                else:x,y,heading=roi[2]-(arc-2*h-w),roi[1],math.pi
                uav=FixedWingState(float(x),float(y),heading)
            else:
                roi=None
                cx,cy=base[0]
                islands=_obstacles(rng,int(rng.integers(config.island_min_count,config.island_max_count+1)),(cx,cy),(config.tracking_island_sampling_extent,config.tracking_island_sampling_extent),config.island_min_radius,config.island_max_radius)
                # Clouds may lie close to the orbit corridor, but do not force an infeasible
                # radar-constrained orbit around a ship that travels through their center.
                clouds=[]
                count=int(rng.integers(config.cloud_min_count,config.cloud_max_count+1))
                corridor=base[:int((config.tracking_episode_seconds*config.ship_speed+config.tracking_corridor_lookahead_distance)/config.ship_route_spacing):config.tracking_corridor_sample_stride]
                for i in range(count):
                    for _ in range(config.per_scene_start_attempts):
                        ob=_obstacles(rng,1,(cx,cy),(config.tracking_cloud_sampling_extent,config.tracking_cloud_sampling_extent),config.cloud_min_radius,config.cloud_max_radius)[0]
                        if np.min(np.linalg.norm(corridor-np.array([ob['x'],ob['y']]),axis=1))>ob['radius']+config.uav_cloud_margin+config.tracking_cloud_corridor_clearance:
                            ob['id']=i;clouds.append(ob);break
                    else:raise SceneValidationError('Could not place tracking cloud corridor')
            route=route_around_islands(base,islands,config)
            heading=math.atan2(route[1,1]-route[0,1],route[1,0]-route[0,0])
            ship=ShipState(float(route[0,0]),float(route[0,1]),heading)
            if task=='tracking':
                d=float(rng.uniform(config.tracking_initial_min_distance,config.tracking_initial_max_distance))
                beta=float(rng.uniform(-math.pi,math.pi))
                er=np.array([math.cos(beta),math.sin(beta)])
                tangent=np.array([er[1],-er[0]])
                vs=config.ship_speed*np.array([math.cos(heading),math.sin(heading)])
                dot=float(vs@tangent)
                c=-dot+math.sqrt(dot*dot+config.uav_speed**2-config.ship_speed**2)
                vu=vs+c*tangent
                omega=-c*c/(d*(dot+c))
                uav=FixedWingState(ship.x+d*er[0],ship.y+d*er[1],math.atan2(vu[1],vu[0]),math.atan(config.uav_speed*omega/GRAVITY))
                pan=wrap(math.atan2(ship.y-uav.y,ship.x-uav.x)-uav.heading)
            else:pan=0.
            scene=SceneSpec('scene-v1',task,int(seed),tuple(config.world_bounds),tuple(roi) if roi else None,tuple(clouds),tuple(islands),asdict(uav),asdict(ship),tuple(map(tuple,route)),float(pan),config.coverage_episode_seconds if task=='coverage' else config.tracking_episode_seconds,config.to_dict()).to_dict()
            validate_scene(scene,config,task)
            return scene
        except (ValueError,OverflowError) as error:
            key=str(error);rejected[key]=rejected.get(key,0)+1
    raise SceneGenerationError(f'No valid {task} scene after {config.max_scene_attempts} attempts; rejections={rejected}')


def sample_coverage_scene(config,rng,seed=0):
    return _sample(config,rng,'coverage',seed)


def sample_tracking_scene(config,rng,seed=0):
    return _sample(config,rng,'tracking',seed)
