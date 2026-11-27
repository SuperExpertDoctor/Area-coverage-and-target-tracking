"""One bounded local preference raster backed by sparse scan timestamps."""
import numpy as np
from geometry import sector_mask


def scan_preference(age,period,decay):
    """Normalized signed exponential: freshly scanned -1, period-old +1."""
    fraction=np.clip(np.asarray(age,dtype=float)/period,0.,1.)
    value=1.-2.*(np.expm1(-decay*fraction)-np.expm1(-decay))/(-np.expm1(-decay))
    return np.where(fraction<=0.,-1.,np.where(fraction>=1.,1.,np.clip(value,-1.,1.)))


class LocalPotentialField:
    def __init__(self,config,roi,islands,clouds,tracking=False):
        self.config=config;self.roi=roi;self.islands=islands;self.clouds=clouds
        self.tracking=tracking;self.last_seen={}
        self.cache=np.zeros((4,33,33),dtype=np.float32)
        self._cache_key=None

    @property
    def spacing(self):return self.config.potential_radius/16.

    def _water(self,x,y):
        bounds=self.config.world_bounds if self.tracking else self.roi
        water=(x>=bounds[0])&(x<=bounds[2])&(y>=bounds[1])&(y<=bounds[3])
        for ob in self.islands:water&=(x-ob['x'])**2+(y-ob['y'])**2>ob['radius']**2
        return water

    def mark_scan(self,position,heading,radius,half_angle,time):
        if self.tracking:return
        spacing=self.spacing
        ix=np.arange(int(np.floor((position[0]-radius)/spacing)),int(np.floor((position[0]+radius)/spacing))+1)
        iy=np.arange(int(np.floor((position[1]-radius)/spacing)),int(np.floor((position[1]+radius)/spacing))+1)
        # A small user query radius must not enlarge a scan temporary beyond33².
        for start_y in range(0,len(iy),33):
            for start_x in range(0,len(ix),33):
                tile_x=ix[start_x:start_x+33];tile_y=iy[start_y:start_y+33]
                xx,yy=np.meshgrid((tile_x+.5)*spacing,(tile_y+.5)*spacing)
                hit=sector_mask(xx,yy,position,heading,radius,half_angle)&self._water(xx,yy)
                rows,cols=np.nonzero(hit)
                for row,col in zip(rows,cols):self.last_seen[(int(tile_x[col]),int(tile_y[row]))]=float(time)
        self._cache_key=None

    def snapshot(self,position,time,goal=None):
        key=(float(position[0]),float(position[1]),float(time),None if goal is None else tuple(goal))
        if key==self._cache_key:return self.cache
        c=self.config;radius=c.potential_radius
        offset=np.linspace(-radius,radius,33)
        xx,yy=np.meshgrid(position[0]+offset,position[1]+offset)
        valid=(xx-position[0])**2+(yy-position[1])**2<=radius**2+1e-6
        water=self._water(xx,yy)&valid
        if self.tracking:
            preference=np.ones((33,33)) if goal is None else np.clip(1.-np.hypot(xx-goal[0],yy-goal[1])/radius,-1.,1.)
        else:
            ix=np.floor(xx/self.spacing).astype(int);iy=np.floor(yy/self.spacing).astype(int)
            times=np.fromiter((self.last_seen.get((int(x),int(y)),-np.inf) for x,y in zip(ix.ravel(),iy.ravel())),dtype=float,count=33*33).reshape(33,33)
            preference=scan_preference(time-times,c.potential_period,c.potential_decay)
        preference=np.where(water,preference,-1.)
        b=c.world_bounds
        hard=(xx<b[0])|(xx>b[2])|(yy<b[1])|(yy>b[3])
        for ob in self.clouds:hard|=(xx-ob['x'])**2+(yy-ob['y'])**2<=(ob['radius']+c.uav_cloud_margin)**2
        hard&=valid
        self.cache=np.stack((np.where(valid,np.where(hard,-2.,preference),0.),water,hard,valid)).astype(np.float32)
        self._cache_key=key
        return self.cache
