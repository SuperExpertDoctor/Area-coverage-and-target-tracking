/* Shared by the visible recording buttons and automated latest-policy capture. */
export function recordingMime(Recorder = globalThis.MediaRecorder) {
  for (const mime of ['video/mp4;codecs=avc1.42001E', 'video/mp4', 'video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm']) {
    if (Recorder?.isTypeSupported(mime)) return mime;
  }
  throw new Error('No supported canvas recording format in this browser.');
}

export function timelineTime(elapsed, duration, first, last, fps) {
  return first + (last - first) * Math.min(1, Math.max(0, elapsed / Math.max(1 / fps, duration - 1 / fps)));
}

export function replayProvenance(replay) {
  // The replay hash/config_hash identify the full config without putting a
  // potentially large training configuration into an HTTP request header.
  const {config, config_snapshot, ...metadata}=replay.metadata || {};
  return {...metadata, task:replay.task, actor:replay.actor,
    model_sha256:metadata.model_sha256 || metadata.policy_checkpoint?.sha256,
    model_checkpoint:metadata.model_checkpoint || metadata.policy_checkpoint?.path,
    control_mode:replay.metadata?.control_mode || (replay.actor==='expert'?'expert':'pure'),
    seed:replay.seed, simulation_start_seconds:replay.frames[0]?.t,
    simulation_end_seconds:replay.frames.at(-1)?.t, source_frame_count:replay.frames.length,
    outcome:replay.summary || {}};
}

export class MissionRecorder {
  constructor({canvas, snapshot, seek, freeze, onState}) {
    Object.assign(this, {canvas, snapshot, seek, freeze, onState});
    this.status = 'loading'; this.active = false; this.result = null; this.error = null;
    this.ready = this.configure();
  }
  async configure() {
    try {
      const response = await fetch('/api/recording-config', {cache:'no-store'});
      if (!response.ok) throw new Error(`Recording configuration: HTTP ${response.status}`);
      this.config = await response.json();
      recordingMime();
      if (!this.canvas.captureStream) throw new Error('Canvas captureStream is unavailable.');
      this.setStatus('ready', '准备录制 · MP4');
      return this.config;
    } catch (error) { this.config=null;this.fail(error); return null; }
  }
  setStatus(status, message) { this.status=status; this.onState?.({status,message,active:this.active,result:this.result}); }
  fail(error) {
    this.error = String(error.message || error);this.active=false;this.result=null;
    cancelAnimationFrame(this.animation);clearTimeout(this.timer);
    // Remove callbacks before stopping a failed capture. Its queued stop/data
    // events must never upload discarded bytes or affect the next recording.
    const media=this.media;this.media=null;
    if(media){
      media.ondataavailable=null;media.onerror=null;media.onstop=null;
      if(media.state!=='inactive')media.stop();
    }
    this.stream?.getTracks().forEach(track=>track.stop());
    this.stream=null;this.chunks=[];
    this.freeze?.(false);this.setStatus('error', `录制失败：${this.error}`);
    this.reject?.(error);this.resolve=null;this.reject=null;
  }
  async start({fullReplay=true, provenance={}}={}) {
    await this.ready;
    if (!this.config || this.status==='loading') throw new Error(this.error || 'Recording is unavailable');
    if (this.active || this.status==='saving' || this.status==='preparing') throw new Error('A recording is already in progress');
    const snapshot=this.snapshot();
    if (!snapshot.replay?.frames?.length) {
      const error=new Error('请先加载有实际仿真帧的回放或实时场景。');
      this.fail(error);throw error;
    }
    this.result=null;this.error=null;this.chunks=[];this.bytes=0;
    this.completion=new Promise((resolve,reject)=>{this.resolve=resolve;this.reject=reject;});
    // A visible button may start recording without awaiting completion.
    this.completion.catch(()=>{});
    this.setStatus('preparing','正在准备录制…');
    try {
      // Include the map's rendered CSS ocean backdrop beneath its transparent
      // canvas. Derive it from the page so the recording follows the visible UI.
      this.background=null;
      const background=getComputedStyle(this.canvas.parentElement).backgroundImage;
      const backgroundUrl=background.match(/^url\(["']?(.*?)["']?\)$/)?.[1];
      if(backgroundUrl){
        const image=new Image();image.src=backgroundUrl;await image.decode();this.background=image;
      }
      this.replay=snapshot.replay;this.fullReplay=fullReplay;
      this.provenance={...replayProvenance(this.replay),...provenance,source_mode:snapshot.mode,
        duration_seconds:this.config.recording_duration_seconds};
      this.active=true;
      this.freeze?.(true);
      this.output=document.createElement('canvas');
      this.output.width=this.config.recording_width;this.output.height=this.config.recording_height;
      this.context=this.output.getContext('2d');
      this.started=performance.now();this.paint(this.started);
      this.stream=this.output.captureStream(this.config.recording_fps);
      const media=new MediaRecorder(this.stream,{mimeType:recordingMime(),videoBitsPerSecond:this.config.recording_bitrate});
      this.media=media;
      media.ondataavailable=event=>{
        if(this.media!==media || !event.data.size)return;
        this.bytes+=event.data.size;
        if (this.bytes>this.config.recording_max_bytes) {
          this.fail(new Error('Recording exceeds configured upload size limit'));return;
        }
        this.chunks.push(event.data);
      };
      media.onerror=event=>{if(this.media===media)this.fail(event.error || new Error('MediaRecorder failed'));};
      media.onstop=()=>this.finish(media);
      media.start(1000);
      this.setStatus('recording','正在录制 · 00:00');
      const draw=now=>{
        if(!this.active)return;
        try {this.paint(now);this.animation=requestAnimationFrame(draw);}
        catch(error){this.fail(error);}
      };
      this.animation=requestAnimationFrame(draw);
      this.timer=setTimeout(()=>this.stop(),this.config.recording_duration_seconds*1000);
      return true;
    }catch(error){this.fail(error);throw error;}
  }
  paint(now) {
    const elapsed=(now-this.started)/1000,config=this.config;
    if (this.fullReplay) this.seek(timelineTime(elapsed,config.recording_duration_seconds,
      this.replay.frames[0].t,this.replay.frames.at(-1).t,config.recording_fps));
    const {frame}=this.snapshot(),c=this.context,w=this.output.width,h=this.output.height;
    const style=getComputedStyle(this.canvas.parentElement);
    const color=(name,fallback)=>style.getPropertyValue(`--record-${name}`).trim()||fallback;
    const colors={bg:color('bg','#091722'),text:color('text','#cfe0e8'),
      muted:color('muted','#9db2c1'),error:color('error','#f2a381')};
    c.fillStyle=colors.bg;c.fillRect(0,0,w,h);
    const header=Math.round(h*.105),footer=Math.round(h*.085),available=h-header-footer;
    const scale=Math.min(w/this.canvas.width,available/this.canvas.height);
    const dw=this.canvas.width*scale,dh=this.canvas.height*scale;
    // The source canvas is transparent. Preserve a solid light map background
    // as well as the dark theme's ocean image beneath the actual scene.
    c.fillStyle=style.backgroundColor||colors.bg;
    c.fillRect((w-dw)/2,header+(available-dh)/2,dw,dh);
    if(this.background && style.backgroundImage!=='none'){
      const image=this.background,cover=Math.max(dw/image.naturalWidth,dh/image.naturalHeight);
      const sw=dw/cover,sh=dh/cover;
      c.drawImage(image,(image.naturalWidth-sw)/2,(image.naturalHeight-sh)/2,sw,sh,(w-dw)/2,header+(available-dh)/2,dw,dh);
    }
    c.drawImage(this.canvas,(w-dw)/2,header+(available-dh)/2,dw,dh);
    const font=Math.max(12,Math.round(w/65));
    c.font=`600 ${font}px system-ui, sans-serif`;c.fillStyle=colors.text;
    const task=this.provenance.task==='coverage'?'区域覆盖 COVERAGE':'协同跟踪 TRACKING';
    const mode=this.provenance.control_mode==='forecast_constraint'?'网络 + 显式物理约束':this.provenance.actor==='expert'?'专家策略':'行为克隆网络 · PURE';
    c.fillText(`${task}  /  ${mode}`,w*.025,header*.43);
    c.fillStyle=colors.muted;c.font=`${Math.round(font*.75)}px monospace`;
    const sha=this.provenance.model_sha256 || this.provenance.checkpoint_sha256 || '';
    c.fillText(`SEED ${this.provenance.seed??'—'}  ·  MODEL ${sha?sha.slice(0,16):'unavailable'}  ·  ${this.provenance.model_checkpoint || this.provenance.checkpoint || ''}`,w*.025,header*.78,w*.95);
    c.fillStyle=colors.text;c.font=`${font}px monospace`;
    const first=this.replay.frames[0].t,last=this.replay.frames.at(-1).t;
    c.fillText(`SIM ${Number(frame?.t??first).toFixed(1)} / ${last.toFixed(1)} s   ·   REC ${Math.min(elapsed,config.recording_duration_seconds).toFixed(1)} s`,w*.025,h-footer*.57);
    const outcome=this.provenance.outcome;
    const objectiveFailed=outcome?.terminated || outcome?.task_success===false ||
      (this.provenance.task==='coverage' && Number.isFinite(outcome?.coverage_ratio) && outcome.coverage_ratio<1-1e-9) ||
      (this.provenance.task==='tracking' && ((Number.isFinite(outcome?.visible_fraction) && outcome.visible_fraction<1-1e-9) || outcome?.first_loss_time!=null));
    const metric=this.provenance.task==='coverage'?`COVERAGE ${Number((frame?.coverage?.ratio||0)*100).toFixed(1)}%`:`DISTANCE ${Number((frame?.metrics?.distance||0)/1000).toFixed(2)} km`;
    c.font=`${Math.round(font*.75)}px system-ui, sans-serif`;
    c.fillStyle=objectiveFailed?colors.error:colors.muted;
    c.fillText(`${metric}   ·   ${objectiveFailed?`FAILED: ${outcome.termination_reason || 'coverage incomplete'}`:'实际仿真回放'}   ·   ${first}—${last}s`,w*.025,h-footer*.18,w*.95);
    if (this.status==='recording') this.setStatus('recording',`正在录制 · ${elapsed.toFixed(1)} / ${config.recording_duration_seconds}s`);
  }
  stop() {
    if (!this.active) return this.completion;
    try {
      // Do not force a future endpoint for an intentionally stopped recording.
      this.paint(performance.now());this.active=false;
      cancelAnimationFrame(this.animation);clearTimeout(this.timer);
      this.setStatus('saving','正在保存 MP4…');
      if(this.media?.state!=='inactive')this.media.stop();
    }catch(error){this.fail(error);}
    return this.completion;
  }
  async finish(media) {
    if(this.media!==media)return;
    // Browsers may also stop MediaRecorder when the capture stream ends.
    this.active=false;cancelAnimationFrame(this.animation);clearTimeout(this.timer);
    this.stream?.getTracks().forEach(track=>track.stop());this.freeze?.(false);
    this.setStatus('saving','正在保存 MP4…');
    try {
      const blob=new Blob(this.chunks,{type:media.mimeType});this.chunks=[];
      const finalSnapshot=this.snapshot();
      this.provenance.recorded_simulation_end_seconds=finalSnapshot.frame?.t;
      this.provenance.recorded_wall_seconds=(performance.now()-this.started)/1000;
      const response=await fetch('/api/recordings',{method:'POST',headers:{
        'Content-Type':media.mimeType,
        'X-Recording-Metadata':encodeURIComponent(JSON.stringify(this.provenance))},body:blob});
      const result=await response.json();
      if(this.media!==media)return;
      if(!response.ok)throw new Error(result.error || `保存失败 HTTP ${response.status}`);
      this.result=result;this.setStatus('saved',`已保存 MP4 · ${result.duration_seconds.toFixed(1)}s`);this.resolve(result);
      this.media=null;this.stream=null;
      this.resolve=null;this.reject=null;
    }catch(error){if(this.media===media)this.fail(error);}
  }
}
