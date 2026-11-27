"""Clockwise sensor-aligned sweeps, followed by observation-driven missed-cell repair."""
import math
import numpy as np
from config import Config
from geometry import unit
from expert_tracking import SafetyGuidance,decode_ego,decode_clouds,ExpertPlanningError


def rounded_rectangle(bounds,radius,spacing=100.):
    x0,y0,x1,y1=bounds
    radius=min(radius,(x1-x0)/2,(y1-y0)/2)
    points=[]
    corners=[(x0+radius,y1-radius,math.pi,math.pi/2),(x1-radius,y1-radius,math.pi/2,0.),(x1-radius,y0+radius,0.,-math.pi/2),(x0+radius,y0+radius,-math.pi/2,-math.pi)]
    for i,(cx,cy,a,b) in enumerate(corners):
        arc=np.linspace(a,b,max(3,int(math.ceil(radius*math.pi/2/spacing))))
        curve=np.column_stack((cx+radius*np.cos(arc),cy+radius*np.sin(arc)))
        points.extend(curve)
        nxt=corners[(i+1)%4]
        end=np.array([nxt[0]+radius*math.cos(nxt[2]),nxt[1]+radius*math.sin(nxt[2])])
        count=max(1,int(math.ceil(np.linalg.norm(end-curve[-1])/spacing)))
        points.extend(curve[-1]+(end-curve[-1])*f for f in np.linspace(0,1,count,endpoint=False)[1:])
    return np.asarray(points)


class CoverageExpert:
    def __init__(self,config=None):
        self.config=config if config is not None else Config()
        self.safety=SafetyGuidance(self.config,preferred_side=-1)
        self.reset()

    def reset(self):
        self.safety.reset();self.route=None;self.index=0;self.stage='sweep';self.repair_goal=None;self.goal_steps=0;self.sweep_steps=0;self.recent_goals=[]

    def _prepare(self,roi,p):
        c=self.config
        outer=c.coverage_outer_offset
        bounds=np.array(roi)+np.array([-outer,-outer,outer,outer])
        loops=[]
        while min(bounds[2]-bounds[0],bounds[3]-bounds[1])>c.coverage_min_rectangle_factor*c.uav_min_turn_radius:
            loops.append(rounded_rectangle(bounds,c.coverage_corner_radius_factor*c.uav_min_turn_radius,c.coverage_path_spacing))
            bounds+=np.array([c.coverage_lane_spacing,c.coverage_lane_spacing,-c.coverage_lane_spacing,-c.coverage_lane_spacing])
        center=np.array([(roi[0]+roi[2])/2,(roi[1]+roi[3])/2])
        angles=np.linspace(math.pi, -math.pi,c.coverage_center_path_points)
        loops.append(center+c.coverage_center_radius_factor*c.uav_min_turn_radius*np.column_stack((np.cos(angles),np.sin(angles))))
        first=loops[0]
        start=int(np.argmin(np.sum((first-p)**2,axis=1)))
        loops[0]=np.concatenate((first[start:],first[:start],first[start:start+1]))
        self.route=np.concatenate(loops)
        self.index=0
        self.sweep_steps=0

    def _repair(self,obs,roi,u):
        c=self.config;data=obs['coverage_map'];sea=data[0]>.5;seen=data[1]>.5
        missing=sea&~seen
        ages=data[2]
        x=roi[0]+(np.arange(64)+.5)*(roi[2]-roi[0])/64
        y=roi[1]+(np.arange(64)+.5)*(roi[3]-roi[1])/64
        xx,yy=np.meshgrid(x,y)
        if self.repair_goal is None or self.goal_steps*c.decision_dt>c.coverage_repair_timeout or (self.goal_steps*c.decision_dt>c.coverage_repair_min_seconds and np.hypot(u.x-self.repair_goal[0],u.y-self.repair_goal[1])<c.coverage_repair_capture_distance and not np.any(missing&((xx-self.repair_goal[0])**2+(yy-self.repair_goal[1])**2<c.coverage_repair_hit_radius**2))):
            blocks=c.coverage_planning_blocks;stride=64//blocks
            score=np.zeros((blocks,blocks))
            for iy in range(blocks):
                for ix in range(blocks):
                    block=(slice(iy*stride,(iy+1)*stride),slice(ix*stride,(ix+1)*stride))
                    valid=sea[block]
                    if not np.any(valid):continue
                    px,py=float(xx[block].mean()),float(yy[block].mean())
                    value=c.coverage_unseen_weight*float(missing[block].sum())+c.coverage_age_weight*float(ages[block][valid].mean())
                    score[iy,ix]=value/(1+math.hypot(u.x-px,u.y-py)/c.coverage_distance_scale)
                    if (iy,ix) in self.recent_goals:score[iy,ix]*=c.coverage_recent_goal_factor
            iy,ix=np.unravel_index(np.argmax(score),score.shape)
            self.recent_goals=(self.recent_goals+[(int(iy),int(ix))])[-c.coverage_recent_goal_count:]
            block=(slice(iy*stride,(iy+1)*stride),slice(ix*stride,(ix+1)*stride))
            cells=missing[block] if np.any(missing[block]) else sea[block]
            self.repair_goal=np.array([xx[block][cells].mean(),yy[block][cells].mean()])
            self.goal_steps=0
        self.goal_steps+=1
        # A clockwise orbit points the fixed right-hand radar into the goal disk.
        relative=np.array([u.x,u.y])-self.repair_goal
        velocity=c.uav_speed*np.array([math.cos(u.heading),math.sin(u.heading)])
        relative+=velocity*(c.uav_roll_time_constant+c.uav_heading_time_constant+.5*c.decision_dt)
        distance=max(1.,np.linalg.norm(relative));er=relative/distance
        return np.array([er[1],-er[0]])-c.coverage_repair_radial_gain*(distance-c.coverage_repair_orbit_radius)/c.coverage_repair_orbit_radius*er

    def act(self,obs):
        v=obs['vector'];c=self.config;u=decode_ego(v,c);p=np.array([u.x,u.y])
        roi=[u.x+float(v[94])*20000,u.y+float(v[96])*20000,u.x+float(v[95])*20000,u.y+float(v[97])*20000]
        if self.route is None:self._prepare(roi,p)
        if self.stage=='repair' and not np.any((obs['coverage_map'][0]>.5)&(obs['coverage_map'][1]<.5)):
            # A completed first pass begins another physical sweep. Repeating the
            # full sensor-aligned pattern bounds revisit gaps more reliably than
            # remaining indefinitely in one locally oldest cloud-covered block.
            self._prepare(roi,p);self.stage='sweep';self.repair_goal=None
        if self.stage=='sweep':
            self.sweep_steps+=1
            stop=min(len(self.route),self.index+c.coverage_route_local_search_points)
            local=self.route[self.index:stop]
            self.index+=int(np.argmin(np.sum((local-p)**2,axis=1)))
            ahead=min(len(self.route)-1,self.index+int(c.expert_lookahead/c.coverage_path_spacing))
            desired=self.route[ahead]-p
            if self.index>=len(self.route)-c.coverage_completion_index_margin or self.sweep_steps*c.decision_dt>c.coverage_sweep_timeout:self.stage='repair'
        else:desired=self._repair(obs,roi,u)
        return self.safety.direction(u,desired,decode_clouds(v,u))
