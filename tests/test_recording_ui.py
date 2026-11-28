"""Exercise the real recorder lifecycle with browser capture APIs at its boundary."""
from pathlib import Path
import shutil
import subprocess

import pytest


NODE = shutil.which('node')
ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(NODE is None, reason='Node required for recorder checks')


def run_recorder(script):
    uri = (ROOT / 'visual' / 'recording.js').as_uri()
    harness = r"""
      import assert from 'node:assert/strict';
      let now=0,nextTimer=0;
      const timers=new Map(),uploads=[],freezes=[],states=[],streams=[];
      const controls={constructorError:false,uploadError:false,decode:async()=>{}};
      const style={backgroundImage:'none',backgroundColor:'rgb(255, 255, 255)',
        getPropertyValue:name=>({'--record-bg':'#fafbfc','--record-text':'#14283a',
          '--record-muted':'#617184','--record-error':'#b45334'}[name]||'')};
      globalThis.performance={now:()=>now};
      globalThis.getComputedStyle=()=>style;
      globalThis.setTimeout=fn=>{const id=++nextTimer;timers.set(id,fn);return id;};
      globalThis.clearTimeout=id=>timers.delete(id);
      globalThis.requestAnimationFrame=()=>1;
      globalThis.cancelAnimationFrame=()=>{};
      globalThis.Image=class{
        naturalWidth=800;naturalHeight=600;
        decode(){return controls.decode();}
      };
      const config={recording_width:320,recording_height:240,recording_fps:10,
        recording_duration_seconds:1,recording_bitrate:1000000,
        recording_max_bytes:10000,recording_max_seconds:3,
        output_directory:'E:/experiment/recordings'};
      const saved={path:'E:/experiment/recordings/demo.mp4',url:'/outputs/recordings/demo.mp4',duration_seconds:1};
      globalThis.fetch=async(url,options)=>{
        if(url==='/api/recording-config')return {ok:true,json:async()=>config};
        assert.equal(url,'/api/recordings');uploads.push(options);
        return {ok:!controls.uploadError,status:controls.uploadError?422:201,
          json:async()=>controls.uploadError?{error:'Video conversion failed'}:{...saved}};
      };
      class FakeMediaRecorder {
        static instances=[];
        static isTypeSupported(mime){return mime==='video/webm';}
        constructor(stream,options){
          if(controls.constructorError)throw new Error('Capture setup failed');
          this.stream=stream;this.mimeType=options.mimeType;this.state='inactive';
          FakeMediaRecorder.instances.push(this);
        }
        start(){this.state='recording';}
        stop(){this.state='inactive';queueMicrotask(()=>{
          this.ondataavailable?.({data:new Blob(['actual video bytes'])});this.onstop?.();
        });}
      }
      globalThis.MediaRecorder=FakeMediaRecorder;
      const context={fills:[],texts:[],images:[],fillRect(...rect){this.fills.push({style:this.fillStyle,rect});},
        fillText(text){this.texts.push({style:this.fillStyle,text});},
        drawImage(...args){this.images.push(args);}};
      const output={getContext:()=>context,captureStream(){
        const track={stopped:false,stop(){this.stopped=true;}};
        const stream={getTracks:()=>[track]};streams.push(stream);return stream;
      }};
      globalThis.document={createElement:()=>output};
      const canvas={width:800,height:600,parentElement:{},captureStream(){}};
      const replay={task:'tracking',actor:'policy',seed:11,
        metadata:{model_sha256:'a'.repeat(64),model_checkpoint:'models/tracking/last.pt'},
        frames:[{t:0},{t:60},{t:120}],
        summary:{terminated:true,termination_reason:'actual_test_outcome'}};
      let frame=replay.frames[0];
      const makeRecorder=()=>new MissionRecorder({canvas,
        snapshot:()=>({replay,frame,mode:'history'}),
        seek:time=>{frame=replay.frames.findLast(item=>item.t<=time)||replay.frames[0];},
        freeze:value=>freezes.push(value),onState:value=>states.push(value)});
    """
    result = subprocess.run([NODE, '--input-type=module', '-e',
        f'import {{MissionRecorder}} from {uri!r};\n' + harness + script],
        cwd=ROOT, capture_output=True, text=True, encoding='utf-8', timeout=15)
    assert result.returncode == 0, result.stderr or result.stdout


def test_white_map_and_recording_colors_follow_visible_theme():
    run_recorder("""
      const recorder=makeRecorder();await recorder.start();
      assert.equal(context.fills[0].style,'#fafbfc');
      assert(context.fills.some(fill=>fill.style==='rgb(255, 255, 255)'));
      assert(context.texts.some(text=>text.style==='#14283a'));
      assert(context.texts.some(text=>text.style==='#617184'));
      assert(context.texts.some(text=>text.style==='#b45334'));
      recorder.stop();await recorder.completion;
    """)


def test_preparation_failure_cleans_state_and_allows_retry():
    run_recorder("""
      style.backgroundImage='url("ocean.png")';
      controls.decode=async()=>{throw new Error('Backdrop unavailable');};
      const recorder=makeRecorder();
      await assert.rejects(recorder.start(),/Backdrop unavailable/);
      assert.equal(recorder.status,'error');assert.equal(recorder.active,false);
      assert.equal(recorder.result,null);
      await assert.rejects(recorder.completion,/Backdrop unavailable/);
      controls.decode=async()=>{};await recorder.start();
      recorder.stop();await recorder.completion;assert.equal(recorder.status,'saved');
    """)


def test_capture_setup_failure_stops_tracks_and_unfreezes_scene():
    run_recorder("""
      controls.constructorError=true;const recorder=makeRecorder();
      await assert.rejects(recorder.start(),/Capture setup failed/);
      assert.equal(recorder.status,'error');assert.equal(recorder.active,false);
      assert.equal(freezes.at(-1),false);
      assert(streams.every(stream=>stream.getTracks().every(track=>track.stopped)));
      await assert.rejects(recorder.completion,/Capture setup failed/);
      controls.constructorError=false;await recorder.start();
      recorder.stop();await recorder.completion;assert.equal(uploads.length,1);
    """)


def test_second_start_is_rejected_while_first_backdrop_is_preparing():
    run_recorder("""
      style.backgroundImage='url("ocean.png")';let release;
      controls.decode=()=>new Promise(resolve=>{release=resolve;});
      const recorder=makeRecorder();await recorder.ready;
      const first=recorder.start();await Promise.resolve();
      await assert.rejects(recorder.start(),/already in progress/);
      assert.equal(recorder.status,'preparing');release();await first;
      assert.equal(FakeMediaRecorder.instances.length,1);
      recorder.stop();await recorder.completion;assert.equal(uploads.length,1);
    """)


def test_manual_stop_returns_saved_result_without_seeking_to_future_frame():
    run_recorder("""
      const recorder=makeRecorder();await recorder.start({fullReplay:false});
      now=350;frame=replay.frames[1];
      const result=await recorder.stop();
      assert.equal(result.path,saved.path);assert.equal(recorder.status,'saved');
      assert.equal(uploads.length,1);assert(uploads[0].body.size>0);
      const provenance=JSON.parse(decodeURIComponent(uploads[0].headers['X-Recording-Metadata']));
      assert.equal(provenance.recorded_simulation_end_seconds,60);
      assert.equal(provenance.recorded_wall_seconds,.35);
      assert.equal(provenance.model_sha256,'a'.repeat(64));
      assert.equal(provenance.outcome.termination_reason,'actual_test_outcome');
      assert.equal(timers.size,0);assert.equal(freezes.at(-1),false);
      assert(streams[0].getTracks()[0].stopped);
    """)


def test_duration_timer_finishes_real_replay_and_uploads_once():
    run_recorder("""
      const recorder=makeRecorder();await recorder.start();
      now=1000;[...timers.values()][0]();
      const result=await recorder.completion;
      assert.equal(result.path,saved.path);assert.equal(recorder.status,'saved');
      const provenance=JSON.parse(decodeURIComponent(uploads[0].headers['X-Recording-Metadata']));
      assert.equal(provenance.recorded_simulation_end_seconds,120);
      assert.equal(provenance.source_frame_count,3);
      recorder.stop();assert.equal(uploads.length,1);assert.equal(timers.size,0);
    """)


def test_failed_upload_clears_previous_result_and_can_record_again():
    run_recorder("""
      const recorder=makeRecorder();await recorder.start();
      recorder.stop();await recorder.completion;assert.equal(recorder.result.path,saved.path);
      controls.uploadError=true;await recorder.start();recorder.stop();
      await assert.rejects(recorder.completion,/Video conversion failed/);
      assert.equal(recorder.result,null);assert.equal(recorder.active,false);
      assert.equal(recorder.status,'error');assert.equal(freezes.at(-1),false);
      controls.uploadError=false;await recorder.start();recorder.stop();
      await recorder.completion;assert.equal(recorder.status,'saved');
    """)


def test_start_without_simulation_frames_does_not_keep_an_older_saved_result():
    run_recorder("""
      const recorder=makeRecorder();await recorder.start();
      recorder.stop();await recorder.completion;assert.equal(recorder.status,'saved');
      replay.frames=[];
      await assert.rejects(recorder.start(),/实际仿真帧/);
      assert.equal(recorder.status,'error');assert.equal(recorder.result,null);
      assert.equal(recorder.active,false);assert.equal(uploads.length,1);
    """)


def test_browser_ending_capture_releases_active_state_and_timer_before_upload():
    run_recorder("""
      const recorder=makeRecorder();await recorder.start();
      const media=FakeMediaRecorder.instances[0];
      media.ondataavailable({data:new Blob(['final browser frame'])});
      media.state='inactive';await media.onstop();await recorder.completion;
      assert.equal(recorder.active,false);assert.equal(timers.size,0);
      assert.equal(recorder.status,'saved');assert.equal(uploads.length,1);
      assert.equal(freezes.at(-1),false);
    """)


def test_failed_capture_callbacks_cannot_upload_or_mutate_next_recording():
    run_recorder("""
      const recorder=makeRecorder();await recorder.start();
      const failed=FakeMediaRecorder.instances[0];
      const oldStop=failed.onstop,oldData=failed.ondataavailable;
      failed.onerror({error:new Error('Capture failed')});
      await assert.rejects(recorder.completion,/Capture failed/);
      assert.equal(uploads.length,0);assert.equal(recorder.result,null);
      await recorder.start();
      oldData({data:new Blob(['stale bytes'])});await oldStop();
      assert.equal(recorder.status,'recording');assert.equal(uploads.length,0);
      recorder.stop();await recorder.completion;
      assert.equal(await uploads[0].body.text(),'actual video bytes');
      assert.equal(recorder.status,'saved');
    """)
