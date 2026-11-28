import {MaritimeMap, finite, position, speedOf, headingDegrees} from './map.js';
import {validateReplay, frameAtTime, formatTime} from './replay.js';
import {MissionRecorder} from './recording.js';
import {fieldGeometry} from './field.js';
import {readTheme, applyTheme} from './theme.js';

const $ = id => document.getElementById(id);
const map = new MaritimeMap($('map'));
const state = {task:'coverage',actor:'expert',mode:'history',records:[],localReplays:new Map(),replay:null,index:0,playing:false,time:0,speed:60,live:new Map(),liveKey:null,loadToken:0,connected:false,catalogLoading:false,recording:false};
const actorName = (actor, metadata) => actor === 'expert' ? '专家策略' : metadata?.control_mode === 'forecast_constraint' ? '行为克隆网络 + 显式物理约束' : '行为克隆网络';
const localReplayRecord = (url, replay, name) => ({url,name:`本地导入 / ${name}`,task:replay.task,actor:replay.actor,frames:replay.frames.length,duration:replay.frames.at(-1)?.t||0,behavior:replay.metadata?.behavior,control_mode:replay.metadata?.control_mode});
const taskName = task => task === 'coverage' ? '区域覆盖' : '协同跟踪';
const numeric = (value, digits=0) => Number.isFinite(value) ? value.toFixed(digits) : '—';
const radians = value => Number.isFinite(value) ? value*180/Math.PI : NaN;
const keyFor = (task,actor) => `${task}/${actor === 'network' ? 'policy' : actor || 'expert'}`;
let renderQueued = false, trendDirty = true;

function status(message) { $('status').textContent = message; }
function scheduleRender() { renderQueued = true; trendDirty = true; }
function setTheme(theme) {
  const name=applyTheme(theme);map.setTheme(name);
  document.querySelectorAll('[data-theme-choice]').forEach(button=>{
    const active=button.dataset.themeChoice===name;
    button.classList.toggle('active',active);button.setAttribute('aria-pressed',active);
  });
  scheduleRender();
}
function initializeMenus() {
  const menus=[...document.querySelectorAll('.menu-bar>.menu')];
  for(const menu of menus)menu.addEventListener('toggle',()=>{
    if(menu.open)for(const other of menus)if(other!==menu)other.open=false;
  });
  document.addEventListener('pointerdown',event=>{
    if(!menus.some(menu=>menu.contains(event.target)))menus.forEach(menu=>menu.open=false);
  });
  document.addEventListener('keydown',event=>{
    if(event.key==='Escape')for(const menu of menus)if(menu.open){menu.open=false;menu.querySelector('summary').focus();}
  });
  document.querySelectorAll('[data-theme-choice]').forEach(button=>button.addEventListener('click',()=>setTheme(button.dataset.themeChoice)));
  setTheme(readTheme());
}
function updateFieldControls(frame) {
  const geometry=fieldGeometry(frame?.potential),enabled=map.showPotential||map.showFieldVectors;
  $('show-field-vectors').checked=map.showFieldVectors;
  $('potential-key').hidden=!enabled||!geometry;
  $('potential-scale').hidden=!map.showPotential;
  $('field-color-key').hidden=!map.showPotential;
  $('field-vector-key').hidden=!map.showFieldVectors;
  $('field-notice').hidden=!frame||!enabled||Boolean(geometry);
  if(geometry){
    $('field-resolution').textContent=`${geometry.rows} × ${geometry.cols} · R ${(geometry.radius/1000).toFixed(1)} km`;
  }
  document.querySelectorAll('[data-view="field"]').forEach(button=>button.disabled=!geometry);
}
function updateLayerControls() {
  $('show-grid').checked=map.showGrid;$('show-trail').checked=map.showTrail;
  $('show-age').checked=map.showAge;$('show-age').parentElement.hidden=state.task!=='coverage';
  $('show-potential').checked=map.showPotential;
  updateFieldControls(state.replay?.frames?.[state.index]);
  $('age-key').hidden=state.task!=='coverage'||!map.showAge;
  $('trail-style').value=map.trailStyle;$('trail-style').disabled=!map.showTrail;
  $('trail-style').querySelector('[value="meteor"]').textContent=`流星尾迹 · ${map.config.visual_trail_seconds}s`;
  $('grid-summary').hidden=!map.showGrid;$('visit-key').hidden=!map.showGrid;
  const cell=map.config.visual_grid_cell_size;
  $('grid-summary').textContent=`${cell>=1000?`${cell/1000} km`:`${cell} m`} · ${map.config.visual_grid_fade_seconds}s 渐淡`;
}
async function loadVisualConfig() {
  try {
    const response=await fetch('/api/visual-config',{cache:'no-store'});
    if(!response.ok)return; // Older standalone servers retain snapshot/defaults.
    const data=await response.json();map.setVisualConfig(data.visual_config||data.config||data);
    updateLayerControls();scheduleRender();
  }catch(_){ /* Local file / static-host mode uses the recorded configuration. */ }
}
function updateChoices() {
  document.querySelectorAll('[data-task]').forEach(button => {
    const active=button.dataset.task===state.task;
    button.classList.toggle('active',active);button.setAttribute('aria-pressed',active);
  });
  $('actor').value=state.actor;
  for(const mode of ['history','live']) {
    const active=state.mode===mode;
    $(`${mode}-mode`).classList.toggle('active',active);$(`${mode}-mode`).setAttribute('aria-pressed',active);
  }
  $('replay-select').disabled=state.mode==='live';
  $('mission-task').textContent=`${taskName(state.task)}任务`;
  $('primary-label').textContent=state.task==='coverage'?'累计覆盖率':'目标距离';
  $('primary-unit').textContent=state.task==='coverage'?'%':'km';
  $('secondary-label').textContent=state.task==='coverage'?'平均信息年龄':'视线方位误差';
  $('secondary-unit').textContent=state.task==='coverage'?'s':'°';
  $('tertiary-label').textContent=state.task==='coverage'?'最大信息年龄':'雷达扫描角';
  $('tertiary-unit').textContent=state.task==='coverage'?'s':'°';
  $('trend-title').textContent=state.task==='coverage'?'覆盖率历程':'目标距离历程';
  $('show-age').parentElement.hidden=state.task!=='coverage';
  $('age-key').hidden=state.task!=='coverage'||!map.showAge;
  updateLayerControls();
}
function setPlaying(playing) {
  state.playing=playing&&state.mode==='history'&&Boolean(state.replay?.frames.length);
  $('play').setAttribute('aria-label',state.playing?'暂停回放':'播放回放');
  $('play-icon').innerHTML=state.playing?'<path d="M7 5h4v14H7Zm7 0h4v14h-4Z"/>':'<path d="m9 5 11 7-11 7Z"/>';
}
function activateReplay(replay, name='实时仿真') {
  setPlaying(false);state.replay=replay;state.index=0;
  if(!replay?.frames?.length){$('seek').disabled=true;$('play').disabled=true;$('restart').disabled=true;}
  state.time=replay?.frames[0]?.t||0;
  map.setScene(replay?.scene||{});
  const initialView = state.task === 'tracking' ? 'ship' : 'mission';
  map.setView(initialView);
  document.querySelectorAll('[data-view]').forEach(button=>{
    const active=button.dataset.view===initialView;
    button.classList.toggle('active',active);button.setAttribute('aria-pressed',active);
    if(button.dataset.view==='mission')button.hidden=state.task!=='coverage';
  });
  if(state.mode==='live'&&replay?.frames.length){state.index=replay.frames.length-1;state.time=replay.frames[state.index].t;}
  $('seed-label').textContent=`SEED ${replay?.seed ?? replay?.scene?.seed ?? '—'}`;
  const b=replay?.scene?.world_bounds;
  $('map-extent').textContent=b?.length===4?`${((b[2]-b[0])/1000).toFixed(0)} × ${((b[3]-b[1])/1000).toFixed(0)} km 海域`:'300 × 300 km 海域';
  $('cloud-count').textContent=replay?.scene?.clouds?.length??'—';
  $('island-count').textContent=replay?.scene?.islands?.length??'—';
  if(replay){
    const training=replay.metadata?.behavior==='dagger_training'?` · DAgger训练混合控制 β=${replay.metadata.mixture_beta}`:'';
    const failure=replay.summary?.terminated?` · 失败终止：${replay.summary.termination_reason||'planning_failure'}`:'';
    status(`${state.mode==='live'?'实时':'回放'} · ${taskName(state.task)} / ${actorName(state.actor,replay.metadata)} · ${name}${training}${failure}`);
  }
  scheduleRender();updateChoices();
}
function renderCatalog() {
  const selected=$('replay-select').value;
  const records=state.records.filter(r=>r.task===state.task&&(r.actor==='network'?'policy':r.actor)===state.actor);
  $('replay-select').replaceChildren();
  if(!records.length){const option=document.createElement('option');option.value='';option.textContent='暂无该任务与策略的回放记录';$('replay-select').append(option);return records;}
  for(const record of records){
    const option=document.createElement('option');option.value=record.url;
    const constrained=record.actor!=='expert'&&record.control_mode==='forecast_constraint'?` [${actorName(record.actor,record)}]`:'';
    option.textContent=`${record.name}${constrained}${record.behavior==='dagger_training'?' [DAgger训练混合]':''} · ${formatTime(record.duration)} · ${record.frames} 帧`;
    option.title=record.name;$('replay-select').append(option);
  }
  if(records.some(r=>r.url===selected))$('replay-select').value=selected;
  return records;
}
async function refreshCatalog(autoLoad=false) {
  if(state.catalogLoading)return;
  state.catalogLoading=true;
  try {
    const response=await fetch('/api/replays',{cache:'no-store'});
    if(!response.ok)throw new Error(`HTTP ${response.status}`);
    const data=await response.json();
    const remote=Array.isArray(data)?data:data.replays||[];
    state.records=[...Array.from(state.localReplays,([url,entry])=>localReplayRecord(url,entry.replay,entry.name)),...remote];
    const records=renderCatalog();
    if(autoLoad&&state.mode==='history'&&!state.replay&&!state.deepLink){
      if(records.length)await loadReplay($('replay-select').value);
      else {status('暂无已生成回放；运行专家采集或策略评估后，记录会自动显示。');$('empty-message').textContent='运行仿真生成回放，或通过「导入 NPZ / JSON」打开本地记录。';}
    }
  } catch(error) {
    if(!state.replay)status(`无法读取回放列表：${error.message}。请通过本地可视化服务器打开本页。`);
  } finally {
    state.catalogLoading=false;
  }
}
async function loadReplay(url) {
  if(!url){activateReplay(null);status('暂无该任务与策略的回放，可导入 NPZ / JSON 文件。');return;}
  const token=++state.loadToken;setPlaying(false);status('正在加载回放…');
  activateReplay(null);status('正在加载回放…');
  try {
    if(state.localReplays.has(url)){
      const entry=state.localReplays.get(url);state.task=entry.replay.task;state.actor=entry.replay.actor;activateReplay(entry.replay,entry.name);return;
    }
    const response=await fetch(url,{cache:'no-store'});
    if(!response.ok)throw new Error(`HTTP ${response.status}`);
    const replay=validateReplay(await response.json());
    const replayHash=response.headers?.get('X-Replay-SHA256');
    if(replayHash)replay.metadata={...replay.metadata,replay_sha256:replayHash,replay_path:url};
    if(token!==state.loadToken)return;
    state.task=replay.task;state.actor=replay.actor;
    activateReplay(replay,decodeURIComponent(url.split('/').pop()));
  } catch(error) {if(token===state.loadToken){activateReplay(null);status(`回放加载失败：${error.message}`);$('empty-message').textContent=error.message;}}
}
function changeSelection() {
  ++state.loadToken;updateChoices();renderCatalog();
  if(state.mode==='history')loadReplay($('replay-select').value);
  else activateReplay(state.live.get(keyFor(state.task,state.actor))||null);
}
function switchMode(mode) {
  state.mode=mode;setPlaying(false);++state.loadToken;
  updateChoices();
  if(mode==='live'){
    const replay=state.live.get(keyFor(state.task,state.actor));
    activateReplay(replay||null);
    if(!replay)status(`等待 ${taskName(state.task)} / ${actorName(state.actor)} 的实时场景；选择与正在运行的仿真一致的任务与策略。`);
  }else {renderCatalog();loadReplay($('replay-select').value);}
}

function render() {
  renderQueued=false;
  const replay=state.replay,frames=replay?.frames||[],frame=frames[state.index],coverage=frame?.coverage||{},radar=frame?.radar||{},metrics=frame?.metrics||{};
  const hasFrame=Boolean(frame);
  const recorder=window.maritimeRecording?.recorder;
  if(recorder)$('record-toggle').disabled=!recorder.config||['loading','preparing','saving'].includes(recorder.status)||(!hasFrame&&!recorder.active);
  updateFieldControls(frame);
  $('empty-state').hidden=hasFrame;
  $('data-label').textContent=hasFrame?(state.mode==='live'?'LIVE':'REPLAY'):'无数据';
  $('data-label').style.color=hasFrame?'var(--cyan)':'';
  if(!hasFrame&&state.mode==='live')$('empty-message').textContent=`等待${taskName(state.task)} / ${actorName(state.actor)}的数据推送。`;
  const distance=Number.isFinite(metrics.distance)?metrics.distance:position(frame?.uav)&&position(frame?.ship)?Math.hypot(frame.uav.x-frame.ship.x,frame.uav.y-frame.ship.y):NaN;
  const ratio=coverage.ratio;
  $('primary-value').textContent=state.task==='coverage'?numeric(ratio*100,1):numeric(distance/1000,2);
  $('secondary-value').textContent=state.task==='coverage'?numeric(coverage.mean_age):numeric(radians(metrics.bearing_error),1);
  $('tertiary-value').textContent=state.task==='coverage'?numeric(coverage.max_age):numeric(radians(radar.pan),1);
  const progress=state.task==='coverage'?ratio:distance/radar.range;
  $('metric-progress').style.width=`${Number.isFinite(progress)?Math.max(0,Math.min(100,progress*100)):0}%`;
  $('metric-description').textContent=!hasFrame?'等待实际仿真帧':state.task==='coverage'?`持续重访 · 已重扫 ${numeric(coverage.rescan_ratio*100,1)}%`:`雷达范围 ${numeric(radar.range/1000,1)} km · 半视场角 ${numeric(radians(radar.half_angle))}°`;
  $('distance-value').textContent=numeric(distance/1000,2);
  $('visibility-value').textContent=typeof radar.visible==='boolean'?(radar.visible?'视场内':'视场外'):'—';
  $('visibility-value').style.fontSize='12px';$('visibility-value').style.color=radar.visible?'var(--cyan)':'var(--muted)';
  for(const key of ['uav','ship']) {
    const p=frame?.[key];
    $(`${key}-speed`).textContent=p?numeric(speedOf(p),1):'—';
    $(`${key}-heading`).textContent=p?numeric(headingDegrees(p)):'—';
  }
  $('uav-bank').textContent=numeric(radians(frame?.uav?.bank),1);
  $('ship-rudder').textContent=numeric(radians(frame?.ship?.rudder),1);
  $('radar-offset').textContent=Number.isFinite(radar.offset)?`${numeric(radians(radar.offset))}°`:'—';
  $('radar-range').textContent=Number.isFinite(radar.range)?`${numeric(radar.range/1000,1)} km`:'—';
  $('radar-key-label').textContent=Number.isFinite(radar.range)?`雷达 · ${numeric(radar.range/1000,1)} km`:'雷达视场';
  $('current-time').textContent=formatTime(frame?.t);
  $('total-time').textContent=state.mode==='live'?'LIVE':formatTime(frames.at(-1)?.t);
  $('timeline-middle').textContent=frames.length?formatTime((frames[0].t+frames.at(-1).t)/2):'—';
  $('timeline-end').textContent=frames.length?formatTime(frames.at(-1).t):'终点';
  $('seek').max=Math.max(0,frames.length-1);$('seek').value=state.index;
  $('seek').disabled=!hasFrame||state.mode==='live'||state.recording;
  $('seek').style.setProperty('--progress',`${frames.length>1?state.index/(frames.length-1)*100:0}%`);
  $('play').disabled=!hasFrame||state.mode==='live'||state.recording;$('restart').disabled=!hasFrame||state.mode==='live'||state.recording;$('speed').disabled=state.mode==='live'||state.recording;
  $('frame-count').textContent=hasFrame?`${state.index+1} / ${frames.length} 帧`:'0 帧';
  map.setFrame(frame,frames,state.index);
  if(trendDirty)drawTrend();
}
function drawTrend() {
  trendDirty=false;const canvas=$('trend'),rect=canvas.getBoundingClientRect(),dpr=Math.min(devicePixelRatio||1,2),w=Math.max(1,rect.width),h=Math.max(1,rect.height);
  canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr);const c=canvas.getContext('2d');c.scale(dpr,dpr);
  const frames=state.replay?.frames||[];
  c.strokeStyle=map.palette.grid;c.lineWidth=1;
  for(const yy of [h*.2,h*.6,h-6]){c.beginPath();c.moveTo(0,yy);c.lineTo(w,yy);c.stroke();}
  if(!frames.length){$('trend-range').textContent='—';return;}
  const getValue=f=>state.task==='coverage'?f.coverage?.ratio*100:f.metrics?.distance/1000;
  const points=frames.map((f,i)=>({t:f.t,v:getValue(f),i})).filter(p=>Number.isFinite(p.v));
  if(!points.length)return;
  const min=state.task==='coverage'?0:Math.max(0,Math.min(...points.map(p=>p.v))*.9),max=state.task==='coverage'?100:Math.max(.1,Math.max(...points.map(p=>p.v))*1.1),t0=frames[0].t,t1=frames.at(-1).t;
  $('trend-range').textContent=state.task==='coverage'?'0 — 100%':`${min.toFixed(1)} — ${max.toFixed(1)} km`;
  const xy=p=>[(p.t-t0)/Math.max(t1-t0,1)*(w-2)+1,h-7-(p.v-min)/(max-min)*(h-16)];
  const stride=Math.max(1,Math.floor(points.length/700));c.beginPath();
  points.forEach((p,i)=>{if(i%stride&&i!==points.length-1)return;const pos=xy(p);i?c.lineTo(...pos):c.moveTo(...pos);});
  const accent=map.palette.uav;
  c.lineWidth=1.4;c.strokeStyle=accent;c.stroke();
  const last=xy(points.at(-1)),first=xy(points[0]);c.lineTo(last[0],h-6);c.lineTo(first[0],h-6);c.closePath();
  const grad=c.createLinearGradient(0,5,0,h);grad.addColorStop(0,`${accent}22`);grad.addColorStop(1,`${accent}00`);c.fillStyle=grad;c.fill();
  const current=points.findLast(p=>p.i<=state.index);if(current){const p=xy(current);c.beginPath();c.moveTo(p[0],0);c.lineTo(p[0],h-5);c.setLineDash([2,3]);c.strokeStyle=map.palette.gridText;c.lineWidth=1;c.stroke();c.setLineDash([]);c.beginPath();c.arc(...p,3,0,Math.PI*2);c.fillStyle=accent;c.fill();}
}

function connectLive() {
  if(location.protocol==='file:'){status('请启动 Python 可视化服务器后打开 http://127.0.0.1:8787。');return;}
  const source=new EventSource('/events');
  let hasConnected=false;
  source.onopen=()=>{if(hasConnected)loadVisualConfig();hasConnected=true;state.connected=true;$('connection-dot').classList.add('connected');$('connection-text').textContent='实时通道已连接';};
  source.onerror=()=>{state.connected=false;$('connection-dot').classList.remove('connected');$('connection-text').textContent='实时通道重连中';};
  source.onmessage=event=>{
    try {
      const data=JSON.parse(event.data);
      if(data.type==='run_status'){displayRunStatus(typeof data.status==='object'?data.status:data.run_status||data);return;}
      if(data.type==='scene'){
        const task=data.task||data.scene?.task||'coverage',actor=data.actor==='network'?'policy':data.actor||'expert',key=keyFor(task,actor);
        const replay={task,actor,scene:data.scene||{},seed:data.seed??data.scene?.seed,metadata:data.metadata||{},frames:[]};
        state.liveKey=key;state.live.set(key,replay);
        if(state.mode==='live'&&key===keyFor(state.task,state.actor))activateReplay(replay);
      } else if(data.type==='frame') {
        const frame=data.frame||data;
        const key=data.task&&data.actor?keyFor(data.task,data.actor):state.liveKey;
        if(!key||!state.live.has(key))return;
        const replay=state.live.get(key);
        // Reconnection may send a tail already received. Avoid duplicate times.
        if(replay.frames.length&&frame.t<=replay.frames.at(-1).t)return;
        replay.frames.push(frame);if(replay.frames.length>12000)replay.frames.shift();
        if(state.mode==='live'&&key===keyFor(state.task,state.actor)){
          if(state.replay!==replay)activateReplay(replay);
          state.index=replay.frames.length-1;state.time=frame.t;scheduleRender();
        }
      }else if(data.type==='replay'||data.type==='complete')refreshCatalog(false);
    }catch(error){status(`实时消息格式错误：${error.message}`);}
  };
  window.addEventListener('beforeunload',()=>source.close());
}

document.querySelectorAll('[data-task]').forEach(button=>button.addEventListener('click',()=>{state.task=button.dataset.task;changeSelection();}));
$('actor').addEventListener('change',()=>{state.actor=$('actor').value;changeSelection();});
$('history-mode').addEventListener('click',()=>switchMode('history'));
$('live-mode').addEventListener('click',()=>switchMode('live'));
$('replay-select').addEventListener('change',()=>loadReplay($('replay-select').value));
$('refresh').addEventListener('click',()=>refreshCatalog(true));
$('replay-file').addEventListener('change',async event=>{
  const file=event.target.files[0];if(!file)return;
  try {
    let replay;
    if(file.name.toLowerCase().endsWith('.npz')){
      if(file.size>128*1024*1024)throw new Error('NPZ 导入上限为 128 MiB');
      const response=await fetch('/api/replays/convert',{method:'POST',headers:{'Content-Type':'application/octet-stream'},body:file});
      const converted=await response.json();
      if(!response.ok)throw new Error(converted.error||`HTTP ${response.status}`);
      replay=validateReplay(converted);
      const replayHash=response.headers?.get('X-Replay-SHA256');
      if(replayHash)replay.metadata={...replay.metadata,replay_sha256:replayHash,replay_path:file.name};
    }else replay=validateReplay(JSON.parse(await file.text()));
    ++state.loadToken;state.mode='history';state.task=replay.task;state.actor=replay.actor;
    const url=`local:${Date.now()}:${file.name}`;state.localReplays.set(url,{replay,name:file.name});
    state.records.unshift(localReplayRecord(url,replay,file.name));
    renderCatalog();$('replay-select').value=url;activateReplay(replay,file.name);
  }
  catch(error){status(`导入失败：${error.message}`);}
  event.target.value='';
});
$('play').addEventListener('click',()=>{
  if(!state.playing&&state.index>=state.replay.frames.length-1){state.index=0;state.time=state.replay.frames[0].t;scheduleRender();}
  setPlaying(!state.playing);
});
$('restart').addEventListener('click',()=>{setPlaying(false);state.index=0;state.time=state.replay?.frames[0]?.t||0;scheduleRender();});
$('seek').addEventListener('input',()=>{setPlaying(false);state.index=Number($('seek').value);state.time=state.replay?.frames[state.index]?.t||0;scheduleRender();});
$('speed').addEventListener('change',()=>state.speed=Number($('speed').value));
document.querySelectorAll('[data-view]').forEach(button=>button.addEventListener('click',()=>{
  map.setView(button.dataset.view);
  document.querySelectorAll('[data-view]').forEach(b=>{const active=b===button;b.classList.toggle('active',active);b.setAttribute('aria-pressed',active);});
}));
$('zoom-in').addEventListener('click',()=>map.adjustZoom(1.4));$('zoom-out').addEventListener('click',()=>map.adjustZoom(1/1.4));$('reset-view').addEventListener('click',()=>map.reset());
$('show-age').addEventListener('change',()=>{map.setLayerOptions({showAge:$('show-age').checked});updateLayerControls();});
$('show-potential').addEventListener('change',()=>{map.setLayerOptions({showPotential:$('show-potential').checked});updateLayerControls();});
$('show-field-vectors').addEventListener('change',()=>{map.setLayerOptions({showFieldVectors:$('show-field-vectors').checked});updateLayerControls();});
$('show-grid').addEventListener('change',()=>{map.setLayerOptions({showGrid:$('show-grid').checked});updateLayerControls();});
$('show-trail').addEventListener('change',()=>{map.setLayerOptions({showTrail:$('show-trail').checked});updateLayerControls();});
$('trail-style').addEventListener('change',()=>map.setLayerOptions({trailStyle:$('trail-style').value}));
new ResizeObserver(()=>{trendDirty=true;scheduleRender();}).observe($('trend'));
window.addEventListener('keydown',event=>{
  if(event.code==='Space'&&!['INPUT','SELECT','BUTTON','TEXTAREA','SUMMARY'].includes(document.activeElement?.tagName)){
    if(state.mode==='history'&&state.replay?.frames.length){event.preventDefault();$('play').click();}
  }
});
let previous=performance.now();
function tick(now) {
  const elapsed=Math.min((now-previous)/1000,.25);previous=now;
  if(!state.recording&&state.playing&&state.replay?.frames.length){
    const frames=state.replay.frames;state.time+=elapsed*state.speed;
    const index=frameAtTime(frames,state.time);
    if(index!==state.index){state.index=index;scheduleRender();}
    if(state.time>=frames.at(-1).t)setPlaying(false);
  }
  if(renderQueued)render();map.draw();requestAnimationFrame(tick);
}
updateChoices();activateReplay(null);loadVisualConfig();refreshCatalog(true);connectLive();requestAnimationFrame(tick);
if(typeof location!=='undefined'){initializeMenus();initializeRecording();}
setInterval(()=>refreshCatalog(state.mode==='history'&&!state.replay),15000);

function displayRunStatus(run) {
  if(!run?.phase)return;
  const current=run.tasks?.[run.task]||{},collection=current.collection||{},training=current.training||{};
  const details=[run.run_name,run.phase,run.task];
  if(collection.target)details.push(`采集 ${collection.accepted||0}/${collection.target}`);
  if(training.epoch!=null)details.push(`Epoch ${training.epoch}/${training.epochs??'—'}`);
  if(Number.isFinite(training.train_loss))details.push(`train ${training.train_loss.toFixed(5)}`);
  if(Number.isFinite(training.validation_loss))details.push(`val ${training.validation_loss.toFixed(5)}`);
  if(run.error)details.push(String(run.error));
  $('run-status').textContent=details.filter(Boolean).join(' · ');$('run-status').hidden=false;
  $('run-status').dataset.error=Boolean(run.error);
}

function initializeRecording() {
  const snapshot=()=>({replay:state.replay,frame:state.replay?.frames[state.index],mode:state.mode});
  const recorder=new MissionRecorder({canvas:$('map'),snapshot,
    seek:time=>{state.time=time;state.index=frameAtTime(state.replay.frames,time);render();map.draw();},
    freeze:active=>{
      state.recording=active;setPlaying(false);
      document.querySelectorAll('.source-bar button,.source-bar select,.source-bar input,.task-tab,.playback button,.playback input,.playback select').forEach(control=>control.disabled=active);
      if(!active){updateChoices();scheduleRender();}
    },
    onState:({status,message,active,result})=>{
      $('recording-status').textContent=message;$('recording-status').dataset.state=status;
      $('record-toggle').checked=active||status==='preparing';
      $('record-toggle').disabled=['loading','preparing','saving'].includes(status)||!recorder?.config||(!state.replay?.frames.length&&!active);
      $('recording-menu-label').textContent=active?'录制中':'录制';
      $('recording-indicator').hidden=!active;
      $('recording-download').hidden=!result;
      if(result){$('recording-download').href=result.url;$('recording-download').download=result.path.split(/[\\/]/).pop();$('recording-destination').textContent=result.path;}
    }});
  const reportError=error=>status(`录制操作失败：${error.message}`);
  $('record-toggle').addEventListener('change',()=>{
    if($('record-toggle').checked)recorder.start({fullReplay:state.mode==='history'}).catch(reportError);
    else recorder.stop()?.catch(reportError);
  });
  const load=async url=>{
    const resolved=new URL(url,location.href);
    if(resolved.origin!==location.origin||!resolved.pathname.startsWith('/outputs/'))throw new Error('Replay URL must be same-origin under /outputs/');
    state.deepLink=true;state.mode='history';await loadReplay(resolved.pathname+resolved.search);
    if(!state.replay?.frames?.length)throw new Error($('status').textContent);
    render();map.draw();return {task:state.replay.task,actor:state.replay.actor,frames:state.replay.frames.length};
  };
  window.maritimeRecording={recorder,ready:recorder.ready,load,
    info:()=>({status:recorder.status,error:recorder.error,result:recorder.result,...snapshot()}),
    recordReplay:async options=>{await recorder.start({fullReplay:true,...options});return recorder.completion;},
    stop:()=>recorder.stop()};
  recorder.ready.then(config=>{if(config){$('recording-settings').textContent=`${config.recording_width} × ${config.recording_height} · ${config.recording_fps} FPS · ${config.recording_duration_seconds}s 后自动保存`;
    $('recording-destination').textContent=config.output_directory||'当前实验 logs / recordings';}
    const replayUrl=new URLSearchParams(location.search).get('replay');if(replayUrl)load(replayUrl).catch(reportError);
  });
  if(new URLSearchParams(location.search).has('replay'))state.deepLink=true;
  fetch('/api/run-status',{cache:'no-store'}).then(r=>r.ok?r.json():{}).then(displayRunStatus).catch(()=>{});
}
