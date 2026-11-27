"""Headless Matplotlib MP4/PNG export of the same frames served to JavaScript."""
from __future__ import annotations

import json
import math
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, Polygon, Rectangle, Wedge
from matplotlib.collections import LineCollection, PolyCollection
import numpy as np


def _meteor_segments(positions,times,index,seconds):
    """Clip the real time window, interpolating its boundary before fading."""
    now=times[index];cutoff=now-seconds
    start=max(1,int(np.searchsorted(times,cutoff,side='right')))
    segments=[];strength=[]
    for end in range(start,index+1):
        t0,t1=times[end-1:end+1]
        if t1<=t0 or t1<cutoff:continue
        begin=max(t0,cutoff)
        count=max(1,int(math.ceil((t1-begin)/(seconds/24))))
        boundaries=np.linspace(begin,t1,count+1)
        fractions=(boundaries-t0)/(t1-t0)
        points=positions[end-1]+fractions[:,None]*(positions[end]-positions[end-1])
        segments.extend(np.stack((points[:-1],points[1:]),axis=1))
        strength.extend((.5*(boundaries[:-1]+boundaries[1:])-cutoff)/seconds)
    return np.asarray(segments).reshape(-1,2,2),np.clip(strength,0,1)


def export_video(record, output, *, fps=24, max_frames=480, width=1440, height=900, visual_config=None):
    if fps <= 0 or max_frames < 2 or width < 320 or height < 240:
        raise ValueError('Invalid movie dimensions, frame count or frame rate')
    if isinstance(record, (str, Path)):
        record = json.loads(Path(record).read_text(encoding='utf-8'))
    frames, scene, task = record['frames'], record['scene'], record['task']
    from config import Config
    visual={key:value for key,value in Config().to_dict().items() if key.startswith('visual_')}
    visual.update({key:value for key,value in scene.get('config_snapshot',{}).items() if key.startswith('visual_')})
    visual.update({key:value for key,value in (visual_config or {}).items() if key.startswith('visual_')})
    if len(frames) < 2:
        raise ValueError('At least two actual rollout frames are required')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    indices = np.unique(np.linspace(0, len(frames) - 1, min(max_frames, len(frames))).astype(int))
    bg, fg, muted = '#071521', '#dcebf1', '#89a6b6'
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'text.color': fg, 'axes.labelcolor': muted,
                         'xtick.color': muted, 'ytick.color': muted})
    fig = plt.figure(figsize=(width/100, height/100), dpi=100, facecolor=bg)
    ax = fig.add_axes([.055, .12, .65, .77], facecolor='#102b3b')
    stats = fig.add_axes([.75, .09, .225, .80], facecolor=bg)
    stats.axis('off')
    title = 'PERSISTENT SEA COVERAGE' if task == 'coverage' else 'MOVING VESSEL TRACKING'
    fig.text(.055, .95, title, fontsize=21, weight='bold')
    fig.text(.055, .913, f'{record["actor"].upper()}  /  SEED {record["seed"]}  /  CONSTANT-SPEED PHYSICS', color=muted)
    duration=frames[-1]['t']-frames[0]['t']
    playback_factor=duration/max(1e-9,(len(indices)-1)/fps)
    fig.text(.055,.012,f'Actual simulation timestamps  |  Playback {playback_factor:.0f}x  |  Display decimated to {len(indices)} frames',color=muted,fontsize=8)
    roi = scene.get('roi')
    if task == 'coverage' and roi:
        bounds = [roi[0]-2500, roi[2]+2500, roi[1]-2500, roi[3]+2500]
    else:
        positions = np.array([[f['ship']['x'], f['ship']['y']] for f in frames])
        lo, hi = positions.min(axis=0)-4200, positions.max(axis=0)+4200
        center = (lo+hi)/2
        half = max((hi-lo).max()/2, 4200)
        bounds = [center[0]-half,center[0]+half,center[1]-half,center[1]+half]
    bounds = np.asarray(bounds)/1000
    ax.set(xlim=bounds[:2], ylim=bounds[2:], aspect='equal', xlabel='East [km]', ylabel='North [km]')
    ax.grid(color='#8eb3c3', alpha=.12, linewidth=.6)
    for spine in ax.spines.values():
        spine.set_color('#2c4d60')
    for obstacle in scene.get('islands', []):
        ax.add_patch(Circle((obstacle['x']/1000, obstacle['y']/1000), obstacle['radius']/1000,
                            color='#776e4e', alpha=.8, zorder=2))
    for obstacle in scene.get('clouds', []):
        ax.add_patch(Circle((obstacle['x']/1000, obstacle['y']/1000), obstacle['radius']/1000,
                            facecolor='#7d5e9d', edgecolor='#d6b5f2', alpha=.6, zorder=3))
    if roi:
        ax.add_patch(Rectangle((roi[0]/1000,roi[1]/1000),(roi[2]-roi[0])/1000,
                               (roi[3]-roi[1])/1000,fill=False,color='#44d8c6',ls='--',lw=1.4))
    age_image=None
    if task=='coverage' and roi and visual['visual_show_radar_age']:
        age_image=ax.imshow(np.zeros((1,1,4)),origin='lower',interpolation='nearest',
            extent=np.asarray([roi[0],roi[2],roi[1],roi[3]])/1000,zorder=1)
    route=np.asarray(scene.get('ship_route', []))
    if len(route):
        ax.plot(route[:,0]/1000,route[:,1]/1000,color='#708797',lw=.7,ls=':',alpha=.6)
    line_u,=ax.plot([],[],color='#64e2d1',lw=1.25,alpha=.8)
    meteor=LineCollection([],zorder=5)
    ax.add_collection(meteor)
    ship_meteor=LineCollection([],zorder=4.9)
    ax.add_collection(ship_meteor)
    passage=PolyCollection([],edgecolors='none',zorder=1.2)
    ax.add_collection(passage)
    line_s,=ax.plot([],[],color='#f5bd68',lw=1.5)
    uav=Polygon([[0,0]]*4,closed=True,facecolor='#c7fff5',edgecolor='#41d9c4',zorder=6)
    ship=Polygon([[0,0]]*5,closed=True,facecolor='#ffdd96',edgecolor='#ffa14f',zorder=6)
    radar=Wedge((0,0),1,0,60,facecolor='#39d1b1',edgecolor='#69ffe1',alpha=.17,zorder=4)
    ax.add_patch(radar);ax.add_patch(uav);ax.add_patch(ship)
    status=stats.text(0,.98,'',va='top',linespacing=1.35,fontsize=10)
    stats.text(0,.005,'AIR OBSTACLE   purple\nSEA OBSTACLE   sand\nUAV   cyan     VESSEL   amber',color=muted,fontsize=8,linespacing=1.2)
    overview=stats.inset_axes([0,.14,.98,.24])
    world_bounds=np.asarray(scene['world_bounds'])/1000
    overview.set(facecolor='#102b3b',xlim=(world_bounds[0],world_bounds[2]),ylim=(world_bounds[1],world_bounds[3]),aspect='equal')
    overview.tick_params(colors=muted,labelsize=7)
    if len(route): overview.plot(route[:,0]/1000,route[:,1]/1000,color='#a38055',lw=.7,alpha=.65)
    omark,=overview.plot([],[],'o',color='#64e2d1',ms=4)
    overview.set_title(f'{world_bounds[2]-world_bounds[0]:g} km × {world_bounds[3]-world_bounds[1]:g} km world',color=muted,fontsize=9)
    u_positions=np.array([[f['uav']['x'],f['uav']['y']] for f in frames])/1000
    s_positions=np.array([[f['ship']['x'],f['ship']['y']] for f in frames])/1000
    def glyph(position, heading, shape, dimensions):
        c,s=math.cos(heading),math.sin(heading)
        return (np.asarray(shape)*np.asarray(dimensions)) @ np.array([[c,s],[-s,c]]) + position
    fig.canvas.draw()
    km_per_pixel=(bounds[1]-bounds[0])/ax.get_window_extent().width
    def dimensions(length,beam,minimum_pixels):
        factor=max(1.,minimum_pixels*km_per_pixel*1000/max(length,beam))
        return np.array([length,beam])*factor/1000
    u_size=dimensions(visual['visual_uav_length'],visual['visual_uav_wingspan'],visual['visual_uav_min_pixels'])
    s_size=dimensions(visual['visual_ship_length'],visual['visual_ship_beam'],visual['visual_ship_min_pixels'])
    cell=visual['visual_grid_cell_size']/1000
    origin=np.asarray(scene['world_bounds'][:2])/1000
    world=np.asarray(scene['world_bounds'])/1000
    times=np.array([frame['t'] for frame in frames])
    visits={};processed=-1
    if visual['visual_show_grid']:
        spacing=cell
        while spacing/km_per_pixel<visual['visual_grid_min_spacing_px']:spacing*=2
        gx=np.arange(origin[0]+math.floor((bounds[0]-origin[0])/spacing)*spacing,bounds[1]+spacing,spacing)
        gy=np.arange(origin[1]+math.floor((bounds[2]-origin[1])/spacing)*spacing,bounds[3]+spacing,spacing)
        grid_lines=[[(x,bounds[2]),(x,bounds[3])] for x in gx]+[[(bounds[0],y),(bounds[1],y)] for y in gy]
        ax.add_collection(LineCollection(grid_lines,colors='#7bacc1',linewidths=.35,alpha=.18,zorder=1.1))
    final_png=output.with_suffix('.png')
    with imageio.get_writer(str(output),fps=fps,codec='libx264',quality=8,
                            macro_block_size=2, ffmpeg_log_level='error') as writer:
        for index in indices:
            f=frames[index];u=f['uav'];s=f['ship'];r=f['radar']
            if visual['visual_show_trail'] and visual['visual_trail_style']=='full':
                line_u.set_data(u_positions[:index+1,0],u_positions[:index+1,1])
                line_s.set_data(s_positions[:index+1,0],s_positions[:index+1,1])
            else:
                line_u.set_data([],[]);line_s.set_data([],[])
            if visual['visual_show_trail'] and visual['visual_trail_style']=='meteor':
                for positions,collection,color in [(u_positions,meteor,(.39,.89,.82)),
                                                   (s_positions,ship_meteor,(.96,.74,.41))]:
                    segments,strength=_meteor_segments(positions,times,index,visual['visual_trail_seconds'])
                    collection.set_segments(segments)
                    collection.set_color([(*color,float(a*.95)) for a in strength])
                    collection.set_linewidths(.3+1.7*strength)
            if visual['visual_show_grid']:
                while processed<index:
                    processed+=1
                    p=u_positions[processed];previous=u_positions[max(0,processed-1)]
                    count=max(1,int(math.ceil(np.linalg.norm(p-previous)/(cell*.5))))
                    for fraction in np.linspace(0,1,count+1):
                        q=previous+(p-previous)*fraction
                        if not(world[0]<=q[0]<=world[2] and world[1]<=q[1]<=world[3]):continue
                        key=tuple(np.minimum(np.floor((q-origin)/cell).astype(int),np.ceil((world[2:]-world[:2])/cell).astype(int)-1))
                        visits[key]=float(times[max(0,processed-1)]+fraction*(times[processed]-times[max(0,processed-1)]))
                vertices=[];colors=[]
                for (ix,iy),last in visits.items():
                    x,y=origin+np.array([ix,iy])*cell
                    if x+cell<bounds[0] or x>bounds[1] or y+cell<bounds[2] or y>bounds[3]:continue
                    vertices.append([(x,y),(x+cell,y),(x+cell,y+cell),(x,y+cell)])
                    freshness=max(0.,1-(f['t']-last)/visual['visual_grid_fade_seconds'])
                    alpha=visual['visual_grid_min_alpha']+(visual['visual_grid_max_alpha']-visual['visual_grid_min_alpha'])*freshness**1.5
                    colors.append((.13,.40,.78,alpha))
                passage.set_verts(vertices);passage.set_facecolors(colors)
            uav.set_xy(glyph(u_positions[index],math.atan2(u['vy'],u['vx']),[[.5,0],[-.5,.5],[-.25,0],[-.5,-.5]],u_size))
            ship.set_xy(glyph(s_positions[index],math.atan2(s['vy'],s['vx']),[[.5,0],[.15,.5],[-.5,.5],[-.5,-.5],[.15,-.5]],s_size))
            radar.set_center(u_positions[index]);radar.set_radius(r['range']/1000)
            center=math.degrees(u['heading']+r.get('offset',0)+r.get('pan',0))
            radar.set_theta1(center-math.degrees(r['half_angle']))
            radar.set_theta2(center+math.degrees(r['half_angle']))
            c=f.get('coverage',{});m=f.get('metrics',{})
            if age_image is not None:
                grid=np.asarray(c.get('grid',[]),dtype=float)
                if grid.ndim==2 and grid.size:
                    maximum=max(1.,c.get('grid_max_age',max(c.get('max_age',0),f['t'])))
                    age=np.clip(np.nan_to_num(grid)/maximum,0,1)
                    rgba=np.stack((64+119*age,189-61*age,170-79*age,72-47*age),axis=-1)/255
                    seen=np.asarray(c.get('seen',[]))
                    if seen.shape==grid.shape:rgba[~seen.astype(bool)]=np.array([113,134,147,16])/255
                    rgba[...,3]=np.where(np.isfinite(grid)&(grid>=0),rgba[...,3],0)
                    age_image.set_data(rgba)
                    gb=c.get('grid_bounds',roi)
                    age_image.set_extent(np.asarray([gb[0],gb[2],gb[1],gb[3]])/1000)
            summary=record['summary']
            text=f'ELAPSED\n{f["t"]/60:7.1f} min\n\n'
            if task=='coverage':
                text+=f'SEA CELLS SCANNED\n{100*c.get("ratio",0):7.2f} %\n\nMEAN / MAX AGE\n{c.get("mean_age",0):.0f} / {c.get("max_age",0):.0f} s\n\n'
            else:
                text+=f'TARGET RANGE\n{m.get("distance",0)/1000:7.3f} km\n\nACTUAL RADAR FOV\n{"VISIBLE" if r.get("visible") else "OUT OF VIEW"}\n\n'
            text+=f'UAV / VESSEL SPEED\n{math.hypot(u["vx"],u["vy"])*3.6:.0f} / {math.hypot(s["vx"],s["vy"])*3.6:.1f} km/h'
            status.set_text(text);omark.set_data([u['x']/1000],[u['y']/1000])
            fig.canvas.draw()
            rgba=np.asarray(fig.canvas.buffer_rgba())
            writer.append_data(rgba[:,:,:3])
        fig.savefig(final_png,facecolor=bg,dpi=100)
    plt.close(fig)
    return {'mp4':str(output.resolve()),'png':str(final_png.resolve()),
            'frames':len(indices),'fps':fps,'simulation_seconds':frames[-1]['t']}
