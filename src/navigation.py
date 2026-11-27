"""Island-only A* bypasses and a forward-progress physical ship path follower."""
import heapq
import math
import numpy as np
from geometry import line_clear


def smooth_polyline(path, radius=600., min_radius=300., spacing=25., corner_fraction=.4):
    """Replace polyline corners by tangent circular arcs with bounded curvature.

    This constructs reference geometry only. The ship still follows through its
    Nomoto actuator; physical clearance is a separate check.
    """
    points=np.asarray(path,dtype=float)
    # A dense already-smooth reference should not acquire thousands of tiny
    # fillets. Retain its existing geometry if all discrete turns are gentle.
    segments=np.diff(points,axis=0)
    lengths=np.linalg.norm(segments,axis=1)
    if np.any(lengths<1e-8):
        keep=np.r_[True,lengths>1e-8]
        points=points[keep];segments=np.diff(points,axis=0);lengths=np.linalg.norm(segments,axis=1)
    if len(points)<2:raise ValueError('A path needs two distinct points')
    output=[points[0]]
    def line_to(end):
        start=np.asarray(output[-1]);distance=float(np.linalg.norm(end-start))
        count=max(1,int(math.ceil(distance/spacing)))
        output.extend(start+(end-start)*f for f in np.linspace(0,1,count+1)[1:])
    for i in range(1,len(points)-1):
        incoming=(points[i]-points[i-1])/lengths[i-1]
        outgoing=(points[i+1]-points[i])/lengths[i]
        turn=math.acos(float(np.clip(incoming@outgoing,-1,1)))
        if turn<1e-3:
            line_to(points[i]);continue
        tangent_factor=math.tan(turn/2)
        effective=min(radius,corner_fraction*min(lengths[i-1],lengths[i])/tangent_factor)
        if effective<min_radius-1e-6:
            # Dense samples of an existing smooth curve need no replacement.
            if turn<.12 and min(lengths[i-1],lengths[i])/max(turn,1e-9)>=min_radius:
                line_to(points[i]);continue
            raise ValueError('Insufficient path length for the ship minimum turn radius')
        sign=1 if incoming[0]*outgoing[1]-incoming[1]*outgoing[0]>0 else -1
        first=points[i]-incoming*effective*tangent_factor
        last=points[i]+outgoing*effective*tangent_factor
        center=first+sign*effective*np.array([-incoming[1],incoming[0]])
        line_to(first)
        angle=math.atan2(first[1]-center[1],first[0]-center[0])
        count=max(2,int(math.ceil(effective*turn/spacing)))
        arc=angle+sign*np.linspace(0,turn,count+1)[1:]
        output.extend(center+effective*np.array([math.cos(a),math.sin(a)]) for a in arc)
    line_to(points[-1])
    return np.asarray(output)


def astar_path(start_xy, goal_xy, blocked_grid, resolution, origin=(0., 0.)):
    blocked = np.asarray(blocked_grid, dtype=bool)
    def cell(p):
        return (int(round((p[0]-origin[0])/resolution)), int(round((p[1]-origin[1])/resolution)))
    start, goal = cell(start_xy), cell(goal_xy)
    ny, nx = blocked.shape
    def valid(c):
        return 0 <= c[0] < nx and 0 <= c[1] < ny and not blocked[c[1], c[0]]
    if not valid(start) or not valid(goal):
        raise ValueError('A* endpoint is blocked')
    queue, costs, parent = [(0., 0, start)], {start: 0.}, {}
    serial = 0
    while queue:
        _, _, cur = heapq.heappop(queue)
        if cur == goal:
            result = [cur]
            while cur in parent:
                cur = parent[cur]
                result.append(cur)
            result.reverse()
            return np.array([(origin[0]+x*resolution, origin[1]+y*resolution) for x,y in result])
        for dx,dy in ((1,0),(0,1),(-1,0),(0,-1),(1,1),(-1,1),(-1,-1),(1,-1)):
            nxt = cur[0]+dx, cur[1]+dy
            if not valid(nxt) or (dx and dy and (not valid((cur[0]+dx,cur[1])) or not valid((cur[0],cur[1]+dy)))):
                continue
            cost = costs[cur] + math.hypot(dx,dy)*resolution
            if cost < costs.get(nxt, math.inf):
                costs[nxt], parent[nxt] = cost, cur
                hx,hy = abs(goal[0]-nxt[0]),abs(goal[1]-nxt[1])
                heuristic = (max(hx,hy)+(math.sqrt(2)-1)*min(hx,hy))*resolution
                serial += 1
                heapq.heappush(queue,(cost+heuristic,serial,nxt))
    raise ValueError('A* endpoints are disconnected')


def island_bypass(start, end, islands, config):
    margin = config.ship_island_margin + config.ship_planning_radius_factor*config.ship_min_turn_radius + config.ship_speed*config.ship_nomoto_time_constant
    if line_clear(start, end, islands, margin):
        return np.array([start,end], dtype=float)
    res = config.navigation_grid_resolution
    relevant = [o for o in islands if not line_clear(start,end,[o],margin)]
    lo = np.minimum(start,end)-max(o['radius'] for o in relevant)-margin-config.ship_astar_padding
    hi = np.maximum(start,end)+max(o['radius'] for o in relevant)+margin+config.ship_astar_padding
    bounds = config.world_bounds
    lo = np.maximum(lo,np.array(bounds[:2])+res)
    hi = np.minimum(hi,np.array(bounds[2:])-res)
    xx,yy = np.meshgrid(np.arange(lo[0],hi[0]+res,res),np.arange(lo[1],hi[1]+res,res))
    blocked = np.zeros(xx.shape,dtype=bool)
    for o in islands:
        blocked |= (xx-o['x'])**2+(yy-o['y'])**2 <= (o['radius']+margin+res/math.sqrt(2))**2
    path=astar_path(start,end,blocked,res,lo)
    path[0],path[-1]=start,end
    simple=[path[0]]
    index=0
    while index<len(path)-1:
        nxt=len(path)-1
        while nxt>index+1 and not line_clear(path[index],path[nxt],islands,margin):
            nxt-=1
        simple.append(path[nxt]);index=nxt
    smooth=smooth_polyline(simple,max(config.ship_bypass_smoothing_radius,config.ship_planning_radius_factor*config.ship_min_turn_radius),config.ship_min_turn_radius,config.ship_route_spacing,config.ship_bypass_corner_fraction)
    if any(not line_clear(a,b,islands,config.ship_island_margin+config.ship_min_turn_radius) for a,b in zip(smooth[:-1],smooth[1:])):
        raise ValueError('Smoothed ship bypass violates island planning margin')
    return smooth


def global_spiral(config, phase=0.):
    """A many-hour reference route. Drawing it never advances the physical ship."""
    bounds=config.world_bounds
    center=np.array([(bounds[0]+bounds[2])/2,(bounds[1]+bounds[3])/2])
    outer=min(config.ship_spiral_outer_radius,config.ship_spiral_world_radius_factor*min(bounds[2]-bounds[0],bounds[3]-bounds[1]))
    inner=min(config.ship_spiral_inner_radius,outer*config.ship_spiral_inner_ratio)
    theta=np.linspace(0,config.ship_spiral_turns*2*math.pi,config.ship_spiral_generation_points)
    radii=inner+(outer-inner)*theta/theta[-1]
    xy=center+np.column_stack((radii*np.cos(theta+phase),radii*np.sin(theta+phase)))
    lengths=np.r_[0,np.cumsum(np.linalg.norm(np.diff(xy,axis=0),axis=1))]
    sample=np.arange(0,lengths[-1],config.ship_route_spacing)
    return np.column_stack((np.interp(sample,lengths,xy[:,0]),np.interp(sample,lengths,xy[:,1])))


def route_around_islands(route, islands, config):
    """Only locally obstructed portions enter A*; clear spiral points stay smooth."""
    margin=config.ship_island_margin+config.ship_planning_radius_factor*config.ship_min_turn_radius+config.ship_speed*config.ship_nomoto_time_constant+config.ship_route_extra_margin
    blocked=np.zeros(len(route),dtype=bool)
    for o in islands:
        blocked |= np.sum((route-np.array([o['x'],o['y']]))**2,axis=1)<(o['radius']+margin)**2
    if not np.any(blocked):
        return route
    output=[]; index=0
    while index<len(route):
        if not blocked[index]:
            output.append(route[index]);index+=1;continue
        lead=int(math.ceil(config.ship_bypass_lead_distance/config.ship_route_spacing))
        start=max(0,index-lead)
        stop=index
        while stop<len(route) and blocked[stop]:stop+=1
        stop=min(len(route)-1,stop+lead)
        if start==0 or stop==len(route)-1:
            raise ValueError('Island obstructs spiral endpoint')
        output=output[:max(0,len(output)-(index-start))]
        bypass=island_bypass(route[start],route[stop],islands,config)
        # Dense reference points keep progress local; Nomoto performs real turns.
        for a,b in zip(bypass[:-1],bypass[1:]):
            n=max(1,int(math.ceil(np.linalg.norm(b-a)/config.ship_route_spacing)))
            output.extend(a+(b-a)*f for f in np.linspace(0,1,n,endpoint=False))
        index=stop
    return np.asarray(output)


class ShipNavigator:
    def __init__(self, route, lookahead=700., local_search_points=80):
        self.route=np.asarray(route,dtype=float)
        self.lookahead=lookahead
        self.index=0
        self.local_search_points=local_search_points
        self.spacing=max(1.,float(np.median(np.linalg.norm(np.diff(self.route,axis=0),axis=1))))

    def heading_command(self,state):
        stop=min(len(self.route),self.index+self.local_search_points)
        points=self.route[self.index:stop]
        self.index+=int(np.argmin((points[:,0]-state.x)**2+(points[:,1]-state.y)**2))
        ahead=min(len(self.route)-1,self.index+max(1,int(self.lookahead/self.spacing)))
        p=self.route[ahead]
        return math.atan2(p[1]-state.y,p[0]-state.x)
