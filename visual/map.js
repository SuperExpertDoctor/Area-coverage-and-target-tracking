/* All geometry uses Cartesian metres, +Y north and CCW heading in radians. */
import {SparseVisitGrid, visualConfig, visitAlpha, adaptiveGridStep, modelSize, trailSegments, potentialColor, potentialCellBounds} from './layers.js';
export const finite = (v, fallback = 0) => Number.isFinite(Number(v)) ? Number(v) : fallback;
export const position = p => p && Number.isFinite(Number(p.x)) && Number.isFinite(Number(p.y));
export function velocityHeading(p) {
  const vx = finite(p?.vx), vy = finite(p?.vy);
  return Math.hypot(vx, vy) > 1e-6 ? Math.atan2(vy, vx) : finite(p?.heading);
}
export const speedOf = p => Math.hypot(finite(p?.vx), finite(p?.vy));
export const headingDegrees = p => ((velocityHeading(p) * 180 / Math.PI) % 360 + 360) % 360;

export class MaritimeMap {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.scene = {};
    this.frames = [];
    this.index = -1;
    this.view = 'mission';
    this.zoom = 1;
    this.pan = [0, 0];
    this.showAge = true;
    this.showPotential = true;
    this.visualOverrides = {};
    this.layerPreferences = {};
    this.config = visualConfig({});
    this.showGrid = this.config.visual_show_grid;
    this.showTrail = this.config.visual_show_trail;
    this.trailStyle = this.config.visual_trail_style;
    this.visits = new SparseVisitGrid();
    this.width = 1;
    this.height = 1;
    this.gridCache = null;
    this.models = {};
    for (const [type, url] of [['uav','/figs/uav-model.svg'],['ship','/figs/ship-model.svg']]) {
      const model = new Image();
      model.onload = () => {this.models[type] = model;this.dirty = true;};
      model.src = url;
    }
    this.dirty = true;
    this.onChange = () => {};
    new ResizeObserver(() => this.resize()).observe(canvas.parentElement);
    canvas.addEventListener('wheel', e => {
      e.preventDefault();
      this.adjustZoom(e.deltaY < 0 ? 1.2 : 1 / 1.2);
    }, {passive: false});
    canvas.addEventListener('pointerdown', e => {
      this.drag = {x: e.clientX, y: e.clientY, pan: [...this.pan]};
      canvas.setPointerCapture(e.pointerId);
    });
    canvas.addEventListener('pointermove', e => {
      if (!this.drag) return;
      this.pan = [this.drag.pan[0] - (e.clientX - this.drag.x) / this.scale,
                  this.drag.pan[1] + (e.clientY - this.drag.y) / this.scale];
      this.dirty = true;
    });
    const stop = () => this.drag = null;
    canvas.addEventListener('pointerup', stop);
    canvas.addEventListener('pointercancel', stop);
  }
  resize() {
    const rect = this.canvas.getBoundingClientRect();
    this.width = Math.max(1, rect.width); this.height = Math.max(1, rect.height);
    this.dpr = Math.min(devicePixelRatio || 1, 2);
    this.canvas.width = Math.round(this.width * this.dpr);
    this.canvas.height = Math.round(this.height * this.dpr);
    this.dirty = true;
    this.draw(); // Resizing clears the bitmap; repaint before the next browser paint.
  }
  setScene(scene) {
    this.scene = scene || {};this.frame = null;this.frames = [];this.index = -1;
    this.reset();this.gridCache = null;this.applyVisualConfig();
  }
  setVisualConfig(overrides={}) { this.visualOverrides = {...overrides};this.applyVisualConfig(); }
  applyVisualConfig() {
    this.config = visualConfig(this.scene, this.visualOverrides);
    this.showGrid = this.layerPreferences.showGrid ?? this.config.visual_show_grid;
    this.showTrail = this.layerPreferences.showTrail ?? this.config.visual_show_trail;
    this.showAge = this.layerPreferences.showAge ?? this.config.visual_show_radar_age;
    this.showPotential = this.layerPreferences.showPotential ?? this.config.visual_show_potential;
    this.trailStyle = this.layerPreferences.trailStyle ?? this.config.visual_trail_style;
    this.visits = new SparseVisitGrid(this.scene.world_bounds || [0,0,300000,300000],this.config.visual_grid_cell_size);
    this.visits.update(this.frames,this.index);this.dirty = true;
  }
  setLayerOptions(options) { Object.assign(this.layerPreferences, options);Object.assign(this,options);this.dirty = true; }
  reset() { this.zoom = 1; this.pan = [0, 0]; this.dirty = true; }
  setView(view) { this.view = view; this.reset(); }
  adjustZoom(factor) { this.zoom = Math.max(.5, Math.min(18, this.zoom * factor)); this.dirty = true; }
  setFrame(frame, frames, index) {
    this.frame = frame;this.frames = frames;this.index = index;
    this.visits.update(frames,index);this.dirty = true;
  }
  camera() {
    let bounds = this.scene.world_bounds;
    if (!Array.isArray(bounds) || bounds.length !== 4) bounds = [0, 0, 300000, 300000];
    let [x0, y0, x1, y1] = bounds.map(Number);
    if (!(x1 > x0 && y1 > y0)) [x0,y0,x1,y1] = [0,0,300000,300000];
    const entity = this.view === 'uav' ? this.frame?.uav : this.view === 'ship' ? this.frame?.ship : null;
    let cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
    let spanX = x1 - x0, spanY = y1 - y0;
    const roi = this.scene.roi;
    if (this.view === 'mission' && Array.isArray(roi) && roi.length === 4) {
      cx = (roi[0] + roi[2]) / 2; cy = (roi[1] + roi[3]) / 2;
      spanX = (roi[2] - roi[0]) * 1.55; spanY = (roi[3] - roi[1]) * 1.55;
    }
    if (position(entity)) {
      cx = Number(entity.x); cy = Number(entity.y);
      const span = Math.max(30000, finite(this.frame?.radar?.range, 15000) * 3);
      spanX = span; spanY = span;
    }
    this.scale = Math.max(1e-8,Math.min(Math.max(1,this.width - 90) / spanX, Math.max(1,this.height - 90) / spanY) * this.zoom);
    this.center = [cx + this.pan[0], cy + this.pan[1]];
    return [x0,y0,x1,y1];
  }
  xy(x, y) { return [this.width / 2 + (Number(x) - this.center[0]) * this.scale,
                     this.height / 2 - (Number(y) - this.center[1]) * this.scale]; }
  draw() {
    if (!this.dirty) return;
    this.dirty = false;
    const c = this.ctx, w = this.width, h = this.height;
    c.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    c.save(); // Tail line caps and other layer styles must not leak to a later frame.
    c.clearRect(0, 0, w, h);
    const bounds = this.camera();
    c.fillStyle = '#06152235'; c.fillRect(0,0,w,h);
    this.drawGrid(bounds);
    if (this.showAge) this.drawCoverage();
    if (this.showGrid) this.drawVisitGrid(bounds);
    if (this.showPotential) this.drawPotential();
    this.drawObstacles(this.scene.islands || [], 'island');
    this.drawObstacles(this.scene.clouds || [], 'cloud');
    this.drawTrails();
    this.drawRadar();
    if (position(this.frame?.ship)) this.drawVehicle(this.frame.ship, 'ship');
    if (position(this.frame?.uav)) this.drawVehicle(this.frame.uav, 'uav');
    const km = this.niceStep(100 / this.scale) / 1000;
    document.getElementById('scale-line').style.width = `${km * 1000 * this.scale}px`;
    document.getElementById('scale-label').textContent = `${km >= 1 ? km.toFixed(km % 1 ? 1 : 0) : km.toFixed(2)} km`;
    document.getElementById('coordinate').textContent = `X ${(this.center[0]/1000).toFixed(1)} · Y ${(this.center[1]/1000).toFixed(1)} km`;
    c.restore();
  }
  drawPotential() {
    const field=this.frame?.potential;
    if(!field?.values?.length||!Array.isArray(field.bounds))return;
    const c=this.ctx,b=field.bounds,rows=field.values.length,cols=field.values[0].length;
    if(this.scene.roi){
      const roi=this.scene.roi,p=this.xy(roi[0],roi[3]),q=this.xy(roi[2],roi[1]);
      c.save();c.beginPath();c.rect(0,0,this.width,this.height);c.rect(p[0],p[1],q[0]-p[0],q[1]-p[1]);c.fillStyle='#cd71ae0c';c.fill('evenodd');c.restore();
    }
    for(let row=0;row<rows;row++)for(let col=0;col<cols;col++) {
      const color=potentialColor(field.values[row][col],field.valid?.[row]?.[col]!==0);
      if(!color[3])continue;
      const rect=potentialCellBounds(field,row,col,this.scene.roi);
      if(!rect)continue;
      const p=this.xy(rect[0],rect[3]),q=this.xy(rect[2],rect[1]);
      const overlap=this.scene.roi&&field.values[row][col]>0?0:.3;
      c.fillStyle=`rgba(${color.join(',')})`;c.fillRect(p[0],p[1],q[0]-p[0]+overlap,q[1]-p[1]+overlap);
    }
    const p=this.frame.uav,r=field.radius||this.scene.config_snapshot?.potential_radius;
    if(p&&r){const a=this.xy(p.x,p.y);c.strokeStyle='#e9f0e466';c.lineWidth=1;c.setLineDash([3,5]);c.beginPath();c.arc(a[0],a[1],r*this.scale,0,2*Math.PI);c.stroke();c.setLineDash([]);}
    if(field.goal?.length===2){const a=this.xy(...field.goal);c.strokeStyle='#a2f5c5';c.lineWidth=1.5;c.beginPath();c.arc(a[0],a[1],5,0,2*Math.PI);c.moveTo(a[0]-9,a[1]);c.lineTo(a[0]+9,a[1]);c.moveTo(a[0],a[1]-9);c.lineTo(a[0],a[1]+9);c.stroke();}
  }
  niceStep(value) {
    const scale = 10 ** Math.floor(Math.log10(Math.max(value, .001)));
    return [1,2,5,10].map(x => x * scale).reduce((a,b) => Math.abs(a-value)<Math.abs(b-value)?a:b);
  }
  drawVisitGrid(bounds) {
    const c=this.ctx,base=this.config.visual_grid_cell_size;
    const step=adaptiveGridStep(base,this.scale,this.config.visual_grid_min_spacing_px);
    const left=Math.max(bounds[0],this.center[0]-this.width/this.scale/2),right=Math.min(bounds[2],this.center[0]+this.width/this.scale/2);
    const bottom=Math.max(bounds[1],this.center[1]-this.height/this.scale/2),top=Math.min(bounds[3],this.center[1]+this.height/this.scale/2);
    const a=this.xy(left,top),b=this.xy(right,bottom);
    if(left>=right||bottom>=top)return;
    c.save();c.beginPath();c.rect(a[0],a[1],b[0]-a[0],b[1]-a[1]);c.clip();
    // Only viewport grid lines are drawn; no dense world-sized array exists.
    c.beginPath();c.strokeStyle='#9cc5e021';c.lineWidth=.65;
    for(let x=bounds[0]+Math.ceil((left-bounds[0])/step)*step;x<=right;x+=step){const p=this.xy(x,0);c.moveTo(p[0],a[1]);c.lineTo(p[0],b[1]);}
    for(let y=bounds[1]+Math.ceil((bottom-bounds[1])/step)*step;y<=top;y+=step){const p=this.xy(0,y);c.moveTo(a[0],p[1]);c.lineTo(b[0],p[1]);}
    c.stroke();
    const now=this.frame?.t;
    for(const cell of this.visits.cells.values()) {
      const x=bounds[0]+cell.x*base,y=bounds[1]+cell.y*base;
      if(x+base<left||x>right||y+base<bottom||y>top)continue;
      const alpha=visitAlpha(cell.t,now,this.config);if(alpha<=0)continue;
      const p=this.xy(x,Math.min(bounds[3],y+base)),q=this.xy(Math.min(bounds[2],x+base),y);
      c.fillStyle=`rgba(35,111,184,${alpha})`;
      c.fillRect(p[0],p[1],Math.max(.7,q[0]-p[0]),Math.max(.7,q[1]-p[1]));
      if(base*this.scale>=5&&now-cell.t<30){c.strokeStyle=`rgba(109,184,231,${alpha*.65})`;c.lineWidth=.6;c.strokeRect(p[0]+.3,p[1]+.3,q[0]-p[0]-.6,q[1]-p[1]-.6);}
    }
    c.restore();
    const label=document.getElementById('grid-summary');
    if(label){label.textContent=`${base>=1000?`${base/1000} km`:`${base} m`} · ${this.config.visual_grid_fade_seconds}s 渐淡`;label.title=`UAV经过记录：${base}m采样；当前网格线间距${step}m，随缩放合并显示。此图层不表示雷达覆盖。`;}
  }
  drawGrid(bounds) {
    const c = this.ctx, w = this.width, h = this.height;
    const step = this.niceStep(95 / this.scale);
    const left = this.center[0] - w / this.scale / 2, right = this.center[0] + w / this.scale / 2;
    const bottom = this.center[1] - h / this.scale / 2, top = this.center[1] + h / this.scale / 2;
    c.lineWidth = 1; c.strokeStyle = '#819eb212'; c.font = '9px "IBM Plex Mono", Consolas, monospace';
    c.fillStyle = '#7794a867';
    c.beginPath();
    for (let x = Math.ceil(left/step)*step; x < right; x += step) {
      const [px] = this.xy(x, 0); c.moveTo(px,0);c.lineTo(px,h);
      if (px > 45 && px < w-70) c.fillText(`${Math.round(x/1000)}`,px+5,h-105);
    }
    for (let y = Math.ceil(bottom/step)*step; y < top; y += step) {
      const [,py] = this.xy(0,y);c.moveTo(0,py);c.lineTo(w,py);
      if (py>60 && py<h-110) c.fillText(`${Math.round(y/1000)}`,8,py-5);
    }
    c.stroke();
    const a = this.xy(bounds[0],bounds[3]), b = this.xy(bounds[2],bounds[1]);
    c.strokeStyle='#7297ad55';c.setLineDash([4,5]);c.strokeRect(a[0],a[1],b[0]-a[0],b[1]-a[1]);c.setLineDash([]);
    const roi = this.scene.roi;
    if (Array.isArray(roi) && roi.length === 4) {
      const p=this.xy(roi[0],roi[3]),q=this.xy(roi[2],roi[1]);
      c.strokeStyle='#70ddd64a';c.setLineDash([7,5]);c.strokeRect(p[0],p[1],q[0]-p[0],q[1]-p[1]);c.setLineDash([]);
      if (p[1]>38&&p[0]>0) {c.fillStyle='#9cc6cc';c.font='10px "Noto Sans SC",sans-serif';c.fillText('任务区域 ROI',p[0]+9,p[1]+17);}
    }
  }
  drawCoverage() {
    const coverage = this.frame?.coverage, grid = coverage?.grid;
    if (!grid) return;
    let rows, cols, values;
    if (Array.isArray(grid) && Array.isArray(grid[0])) {rows=grid.length;cols=grid[0].length;values=grid;}
    else if (grid.values && grid.rows && grid.cols) {rows=grid.rows;cols=grid.cols;values=grid.values;}
    else return;
    if (!rows || !cols || rows>512 || cols>512) return;
    if (this.gridCache?.grid !== grid) {
      const image=document.createElement('canvas');image.width=cols;image.height=rows;
      const ic=image.getContext('2d'), pixels=ic.createImageData(cols,rows);
      const max = Math.max(1, finite(coverage.grid_max_age, Math.max(finite(coverage.max_age),finite(this.frame.t),1)));
      for(let j=0;j<rows;j++) for(let i=0;i<cols;i++) {
        const value=Array.isArray(values[j])?values[j][i]:values[j*cols+i];
        if (!Number.isFinite(value) || value<0) continue;
        const k=((rows-1-j)*cols+i)*4;
        if (coverage.seen && coverage.seen[j]?.[i] === false) {pixels.data[k]=113;pixels.data[k+1]=134;pixels.data[k+2]=147;pixels.data[k+3]=16;continue;}
        const age=Math.min(1,Math.max(0,value/max));
        pixels.data[k]=Math.round(64+age*119);pixels.data[k+1]=Math.round(189-age*61);pixels.data[k+2]=Math.round(170-age*79);pixels.data[k+3]=Math.round(72-age*47);
      }
      ic.putImageData(pixels,0,0);this.gridCache={grid,image};
    }
    const bounds=coverage.grid_bounds||this.scene.roi||this.scene.world_bounds;
    if (!bounds || bounds.length!==4) return;
    const a=this.xy(bounds[0],bounds[3]),b=this.xy(bounds[2],bounds[1]);
    this.ctx.imageSmoothingEnabled=false;this.ctx.drawImage(this.gridCache.image,a[0],a[1],b[0]-a[0],b[1]-a[1]);this.ctx.imageSmoothingEnabled=true;
  }
  drawRoute() {
    const route=this.scene.ship_route;
    if (!Array.isArray(route) || route.length<2) return;
    const c=this.ctx;c.beginPath();c.strokeStyle='#f7bc7530';c.lineWidth=1;c.setLineDash([3,6]);
    route.forEach((p,i)=>{const a=Array.isArray(p)?this.xy(p[0],p[1]):this.xy(p.x,p.y);i?c.lineTo(...a):c.moveTo(...a);});c.stroke();c.setLineDash([]);
  }
  drawObstacles(items, type) {
    const c=this.ctx;
    items.forEach((obstacle,i)=>{
      const p=Array.isArray(obstacle)?{x:obstacle[0],y:obstacle[1],radius:obstacle[2]}:obstacle;
      if(!position(p)) return;
      const [x,y]=this.xy(p.x,p.y),r=finite(p.radius ?? p.r,1000)*this.scale;
      if(x+r<0||y+r<0||x-r>this.width||y-r>this.height||r<=0)return;
      c.save();c.translate(x,y);
      if(type==='cloud'){
        const grad=c.createRadialGradient(-r*.2,-r*.25,0,0,0,r);grad.addColorStop(0,'#c4d8e044');grad.addColorStop(.7,'#a2bdcc20');grad.addColorStop(1,'#b2ccda08');
        c.fillStyle=grad;c.beginPath();c.arc(0,0,r,0,Math.PI*2);c.fill();
        c.strokeStyle='#b3ccdb66';c.lineWidth=1;c.setLineDash([3,4]);c.stroke();c.setLineDash([]);
        c.strokeStyle='#c2d9e340';c.beginPath();c.ellipse(-r*.12,-r*.06,r*.62,r*.36,-.2,Math.PI*.8,Math.PI*2.05);c.stroke();
      }else{
        c.fillStyle='#55786d2b';c.strokeStyle='#7da89b70';c.lineWidth=1;c.beginPath();c.arc(0,0,r,0,Math.PI*2);c.fill();c.setLineDash([2,4]);c.stroke();c.setLineDash([]);
        c.beginPath();for(let k=0;k<40;k++){const a=k/40*Math.PI*2,rr=r*(.79+.08*Math.sin(k*.9+i)+.06*Math.cos(k*.53));k?c.lineTo(Math.cos(a)*rr,Math.sin(a)*rr):c.moveTo(Math.cos(a)*rr,Math.sin(a)*rr);}c.closePath();
        c.fillStyle='#4b655cbb';c.strokeStyle='#92b2a077';c.fill();c.stroke();
        c.beginPath();c.ellipse(-r*.1,-r*.04,r*.42,r*.25,.4,0,Math.PI*2);c.strokeStyle='#9aaf8f33';c.stroke();
      }
      if(r>13){c.font='9px "Noto Sans SC",sans-serif';c.fillStyle=type==='cloud'?'#b6ccd8':'#b3c2ae';c.textAlign='center';c.fillText(`${type==='cloud'?'云区 · AIR':'岛屿 · SEA'} ${p.id ?? i+1}`,0,r+14);}
      c.restore();
    });
  }
  drawTrails() {
    if(!this.showTrail||!this.frames.length||this.index<1)return;
    const c=this.ctx;
    for(const [key,rgb] of [['ship','247,188,117'],['uav','112,221,214']]) {
      const segments=trailSegments(this.frames,this.index,key,this.config.visual_trail_seconds,this.trailStyle);
      if(this.trailStyle==='history'){
        c.beginPath();const stride=Math.max(1,Math.floor(segments.length/1400));
        segments.forEach((s,i)=>{if(s.breakBefore)c.moveTo(...this.xy(s.from.x,s.from.y));if(i%stride===0||i===segments.length-1)c.lineTo(...this.xy(s.to.x,s.to.y));});
        c.strokeStyle=`rgba(${rgb},.65)`;c.lineWidth=1.15;c.lineJoin='round';c.stroke();
      }else{
        for(const segment of segments){
          const a=this.xy(segment.from.x,segment.from.y),b=this.xy(segment.to.x,segment.to.y);
          if(Math.hypot(a[0]-b[0],a[1]-b[1])<.01)continue;
          const gradient=c.createLinearGradient(...a,...b);
          gradient.addColorStop(0,`rgba(${rgb},${segment.alpha0*.85})`);gradient.addColorStop(1,`rgba(${rgb},${segment.alpha1*.85})`);
          c.beginPath();c.moveTo(...a);c.lineTo(...b);c.strokeStyle=gradient;c.lineWidth=.7+segment.alpha1*1.15;c.lineCap='round';c.stroke();
        }
      }
    }
  }
  drawRadar() {
    const radar=this.frame?.radar,p=this.frame?.uav;if(!radar||!position(p))return;
    const r=finite(radar.range)*this.scale;if(r<=0)return;
    const [x,y]=this.xy(p.x,p.y),center=velocityHeading(p)+finite(radar.offset)+finite(radar.pan),half=finite(radar.half_angle,Math.PI/6),c=this.ctx;
    const grad=c.createRadialGradient(x,y,0,x,y,r);grad.addColorStop(0,'#70ddd62b');grad.addColorStop(1,'#70ddd60a');
    c.beginPath();c.moveTo(x,y);c.arc(x,y,r,-center-half,-center+half);c.closePath();c.fillStyle=grad;c.fill();c.strokeStyle=radar.visible?'#8df2b4aa':'#70ddd67a';c.lineWidth=1;c.stroke();
    c.beginPath();c.moveTo(x,y);c.lineTo(x+Math.cos(center)*r,y-Math.sin(center)*r);c.strokeStyle='#70ddd64a';c.setLineDash([3,5]);c.stroke();c.setLineDash([]);
  }
  drawVehicle(p,type) {
    const c=this.ctx,[x,y]=this.xy(p.x,p.y),heading=velocityHeading(p),color=type==='uav'?'#90eee3':'#ffd197';
    const size=modelSize(type,this.scale,this.config);
    c.save();c.translate(x,y);c.rotate(-heading);c.shadowColor='#03111e';c.shadowBlur=3;c.shadowOffsetY=1;
    if(this.models[type]){
      // Crop transparent SVG margins so physical dimensions describe the model,
      // rather than its padded source viewBox.
      const source=type==='uav'?[43,11,128,178]:[13,17,193,56];
      c.drawImage(this.models[type],...source,-size.width/2,-size.height/2,size.width,size.height);
    }
    c.restore();
    if(x>0&&x<this.width-90&&y>25&&y<this.height-85){
      const labelY=y+(type==='uav'?-17:20),labelX=x+Math.max(13,size.width/2+7);
      c.font='500 10px "IBM Plex Mono",Consolas,monospace';c.fillStyle=color;c.fillText(type==='uav'?'CH-4 / UAV':'R/V / SHIP',labelX,labelY);
      c.font='9px "IBM Plex Mono",Consolas,monospace';c.fillStyle='#a9c1d1';c.fillText(`${speedOf(p).toFixed(1)} m/s`,labelX,labelY+14);
    }
  }
}
