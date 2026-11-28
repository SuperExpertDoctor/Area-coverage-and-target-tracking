/* Temporal layers stay independent from DOM, radar coverage and wall time. */
export const VISUAL_DEFAULTS = Object.freeze({
  visual_grid_cell_size:250, visual_grid_fade_seconds:600,
  visual_grid_min_alpha:.025, visual_grid_max_alpha:.55,
  visual_grid_min_spacing_px:12, visual_uav_length:9,
  visual_uav_wingspan:18, visual_ship_length:83, visual_ship_beam:16,
  visual_uav_min_pixels:12, visual_ship_min_pixels:8,
  visual_trail_seconds:60, visual_show_grid:true, visual_show_trail:true, visual_show_radar_age:true,
  visual_trail_style:'meteor',
  visual_show_potential:true,
  visual_show_field_vectors:true,
});

export function potentialCellBounds(field,row,col,roi=null) {
  const b=field.bounds,rows=field.values.length,cols=field.values[0].length;
  const dx=(b[2]-b[0])/Math.max(1,cols-1),dy=(b[3]-b[1])/Math.max(1,rows-1);
  const rect=[b[0]+(col-.5)*dx,b[1]+(row-.5)*dy,b[0]+(col+.5)*dx,b[1]+(row+.5)*dy];
  if(roi&&field.values[row][col]>0) {
    rect[0]=Math.max(rect[0],roi[0]);rect[1]=Math.max(rect[1],roi[1]);
    rect[2]=Math.min(rect[2],roi[2]);rect[3]=Math.min(rect[3],roi[3]);
  }
  return rect[2]>rect[0]&&rect[3]>rect[1]?rect:null;
}

export function potentialColor(value, valid=true) {
  if(!valid||!Number.isFinite(value))return [0,0,0,0];
  if(value<=-1.5)return [219,73,83,.65];
  const p=Math.max(-1,Math.min(1,value)),f=(p+1)/2;
  return [Math.round(205-142*f),Math.round(113+98*f),Math.round(174-13*f),.08+.24*Math.abs(p)];
}

export function visualConfig(scene={}, overrides={}) {
  const source={...(scene.config_snapshot||{}),...overrides}, result={...VISUAL_DEFAULTS};
  for(const [key,value] of Object.entries(source)) {
    if(!(key in VISUAL_DEFAULTS))continue;
    if(typeof VISUAL_DEFAULTS[key]==='number' && Number.isFinite(value) && value>=0)result[key]=value;
    else if(typeof VISUAL_DEFAULTS[key]==='boolean' && typeof value==='boolean')result[key]=value;
    else if(key==='visual_trail_style'&&['meteor','history','full'].includes(value))result[key]=value==='full'?'history':value;
  }
  for(const key of ['visual_grid_cell_size','visual_grid_fade_seconds','visual_grid_min_spacing_px','visual_uav_length','visual_uav_wingspan','visual_ship_length','visual_ship_beam','visual_trail_seconds'])result[key]=Math.max(.001,result[key]);
  result.visual_grid_min_alpha=Math.min(1,result.visual_grid_min_alpha);
  result.visual_grid_max_alpha=Math.max(result.visual_grid_min_alpha,Math.min(1,result.visual_grid_max_alpha));
  return result;
}

export function visitAlpha(lastVisit, now, config) {
  if(!Number.isFinite(lastVisit)||!Number.isFinite(now)||lastVisit>now)return 0;
  const remaining=Math.max(0,1-(now-lastVisit)/config.visual_grid_fade_seconds);
  return config.visual_grid_min_alpha+(config.visual_grid_max_alpha-config.visual_grid_min_alpha)*remaining**1.5;
}

export function adaptiveGridStep(cellSize, pixelsPerMetre, minSpacing) {
  const multiple=2**Math.max(0,Math.ceil(Math.log2(minSpacing/Math.max(cellSize*pixelsPerMetre,1e-12))));
  return cellSize*multiple;
}

export function modelSize(type, scale, config) {
  const length=config[type==='uav'?'visual_uav_length':'visual_ship_length'];
  const breadth=config[type==='uav'?'visual_uav_wingspan':'visual_ship_beam'];
  const minimum=config[type==='uav'?'visual_uav_min_pixels':'visual_ship_min_pixels'];
  const factor=Math.max(1,minimum/Math.max(length*scale,breadth*scale,1e-12));
  return {width:length*scale*factor,height:breadth*scale*factor,enlarged:factor>1};
}

const validPoint=p=>p&&Number.isFinite(p.x)&&Number.isFinite(p.y);
const interpolate=(a,b,f)=>({x:a.x+(b.x-a.x)*f,y:a.y+(b.y-a.y)*f});

function clipLine(a,b,bounds) {
  const dx=b.x-a.x,dy=b.y-a.y;
  let low=0,high=1;
  for(const [p,q] of [[-dx,a.x-bounds[0]],[dx,bounds[2]-a.x],[-dy,a.y-bounds[1]],[dy,bounds[3]-a.y]]) {
    if(p===0){if(q<0)return null;continue;}
    const ratio=q/p;
    if(p<0)low=Math.max(low,ratio);else high=Math.min(high,ratio);
    if(low>high)return null;
  }
  return [low,high];
}

export class SparseVisitGrid {
  constructor(bounds=[0,0,300000,300000],cellSize=250) {
    this.bounds=[...bounds];this.cellSize=cellSize;
    this.columns=Math.ceil((bounds[2]-bounds[0])/cellSize);
    this.rows=Math.ceil((bounds[3]-bounds[1])/cellSize);
    this.clear();
  }
  clear() {
    this.cells=new Map();this.frames=null;this.lastFrame=null;this.lastIndex=-1;this.time=-Infinity;
  }
  cell(p) {
    if(!validPoint(p)||p.x<this.bounds[0]||p.x>this.bounds[2]||p.y<this.bounds[1]||p.y>this.bounds[3])return null;
    return [Math.min(this.columns-1,Math.floor((p.x-this.bounds[0])/this.cellSize)),Math.min(this.rows-1,Math.floor((p.y-this.bounds[1])/this.cellSize))];
  }
  mark(x,y,t) {
    if(x<0||y<0||x>=this.columns||y>=this.rows)return;
    const key=`${x},${y}`,previous=this.cells.get(key);
    if(!previous||t>=previous.t)this.cells.set(key,{x,y,t});
  }
  segment(a,b,t0,t1) {
    const clip=clipLine(a,b,this.bounds);if(!clip)return;
    const p=interpolate(a,b,clip[0]),q=interpolate(a,b,clip[1]);
    for(const point of [p,q]){point.x=Math.max(this.bounds[0],Math.min(this.bounds[2],point.x));point.y=Math.max(this.bounds[1],Math.min(this.bounds[3],point.y));}
    const startTime=t0+(t1-t0)*clip[0],endTime=t0+(t1-t0)*clip[1];
    let [x,y]=this.cell(p),[endX,endY]=this.cell(q);
    const dx=q.x-p.x,dy=q.y-p.y,sx=Math.sign(dx),sy=Math.sign(dy);
    const nextX=this.bounds[0]+(x+(sx>0?1:0))*this.cellSize;
    const nextY=this.bounds[1]+(y+(sy>0?1:0))*this.cellSize;
    let tx=sx?(nextX-p.x)/dx:Infinity,ty=sy?(nextY-p.y)/dy:Infinity;
    const deltaX=sx?this.cellSize/Math.abs(dx):Infinity,deltaY=sy?this.cellSize/Math.abs(dy):Infinity;
    for(let safety=0;safety<this.columns+this.rows+3;safety++) {
      const exit=x===endX&&y===endY?1:Math.max(0,Math.min(1,tx,ty));
      this.mark(x,y,startTime+(endTime-startTime)*exit);
      if(x===endX&&y===endY)break;
      const crossX=tx<=ty,crossY=ty<=tx;
      if(crossX){x+=sx;tx+=deltaX;}if(crossY){y+=sy;ty+=deltaY;}
      if(x<0||y<0||x>=this.columns||y>=this.rows)break;
    }
  }
  update(frames,index) {
    if(!frames?.length||index<0){this.clear();return;}
    const end=Math.min(index,frames.length-1),now=frames[end].t;
    let start=0;
    if(this.frames===frames&&now>=this.time&&this.lastFrame) {
      const previousIndex=frames[this.lastIndex]===this.lastFrame?this.lastIndex:frames.indexOf(this.lastFrame);
      if(previousIndex>=0&&previousIndex<=end)start=previousIndex+1;else this.clear();
    }else this.clear();
    for(let i=start;i<=end;i++) {
      const frame=frames[i],previous=frames[i-1];
      if(!validPoint(frame?.uav)||!Number.isFinite(frame.t))continue;
      if(validPoint(previous?.uav)&&Number.isFinite(previous.t)&&frame.t>=previous.t)this.segment(previous.uav,frame.uav,previous.t,frame.t);
      else {const cell=this.cell(frame.uav);if(cell)this.mark(...cell,frame.t);}
    }
    this.frames=frames;this.lastFrame=frames[end];this.lastIndex=end;this.time=now;
  }
}

export function trailSegments(frames,index,platform,seconds=60,style='meteor') {
  const result=[],end=Math.min(index,frames.length-1);if(end<1)return result;
  const now=frames[end].t,cutoff=style==='history'?-Infinity:now-seconds;
  let start=end;
  while(start>0&&frames[start-1].t>=cutoff)start--;
  if(start>0)start--;
  for(let i=Math.max(1,start);i<=end;i++) {
    const a=frames[i-1],b=frames[i];
    if(!validPoint(a[platform])||!validPoint(b[platform])||b.t<cutoff||b.t<=a.t)continue;
    const t0=Math.max(a.t,cutoff),fraction=(t0-a.t)/(b.t-a.t);
    result.push({from:interpolate(a[platform],b[platform],fraction),to:b[platform],t0,t1:b.t,
      alpha0:style==='history'?1:Math.max(0,Math.min(1,(t0-cutoff)/seconds)),
      alpha1:style==='history'?1:Math.max(0,Math.min(1,(b.t-cutoff)/seconds)),
      breakBefore:i===start||!validPoint(frames[i-2]?.[platform])});
  }
  return result;
}
