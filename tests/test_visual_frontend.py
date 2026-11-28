"""Exercise temporal browser layers through their actual ES modules."""
import shutil
import subprocess
from pathlib import Path

import pytest

NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="Node required for pure-JS layer checks")
ROOT = Path(__file__).resolve().parents[1]


def run_js(script):
    uri = (ROOT / "visual" / "layers.js").as_uri()
    result = subprocess.run([NODE, "--input-type=module", "-e", f"import assert from 'node:assert/strict'; import {{SparseVisitGrid, visualConfig, visitAlpha, adaptiveGridStep, modelSize, trailSegments}} from {uri!r};\n{script}"], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=15)
    assert result.returncode == 0, result.stderr or result.stdout


def test_crossed_cells_are_filled_between_sparse_frames():
    run_js("""
      const grid=new SparseVisitGrid([0,0,2000,2000],250);
      grid.update([{t:0,uav:{x:125,y:125}},{t:100,uav:{x:1125,y:125}}],1);
      assert.equal(grid.cells.size,5);
      for(let col=0;col<5;col++)assert(grid.cells.has(`${col},0`));
      assert.equal(grid.cells.get('4,0').t,100);
      assert(grid.cells.get('0,0').t<grid.cells.get('3,0').t);
      assert(grid.cells.get('3,0').t<100);
    """)


def test_rewind_removes_future_and_matches_fresh_replay():
    run_js("""
      const frames=[{t:0,uav:{x:125,y:125}},{t:10,uav:{x:625,y:125}},{t:20,uav:{x:1125,y:125}}];
      const grid=new SparseVisitGrid([0,0,2000,2000],250);
      grid.update(frames,2);assert(grid.cells.has('4,0'));
      grid.update(frames,0);assert.deepEqual([...grid.cells.keys()],['0,0']);
      grid.update(frames,1);
      const fresh=new SparseVisitGrid([0,0,2000,2000],250);fresh.update(frames,1);
      assert.deepEqual([...grid.cells],[...fresh.cells]);assert(!grid.cells.has('3,0'));
      grid.update([{t:0,uav:{x:125,y:875}}],0);
      assert.deepEqual([...grid.cells.keys()],['0,3']);
    """)


def test_live_append_and_trim_keep_sparse_visits():
    run_js("""
      const frames=[{t:0,uav:{x:125,y:125}},{t:10,uav:{x:625,y:125}}];
      const grid=new SparseVisitGrid([0,0,300000,300000],250);grid.update(frames,1);
      frames.shift();frames.push({t:20,uav:{x:1125,y:125}});grid.update(frames,1);
      assert.equal(grid.cells.size,5);assert(grid.cells.has('0,0'));assert(grid.cells.has('4,0'));
      assert(grid.cells instanceof Map);
    """)


def test_fade_uses_simulation_time_and_visual_overrides():
    run_js("""
      const c=visualConfig({});
      assert.equal(visitAlpha(100,100,c),.55);
      assert(visitAlpha(100,400,c)<visitAlpha(100,200,c));
      assert.equal(visitAlpha(100,700,c),.025);
      assert.equal(visitAlpha(200,100,c),0);
      const custom=visualConfig({config_snapshot:{visual_grid_fade_seconds:50,visual_grid_max_alpha:.8}},{visual_grid_fade_seconds:30});
      assert.equal(custom.visual_grid_fade_seconds,30);
      assert.equal(visitAlpha(0,0,custom),.8);
      const flags=visualConfig({config_snapshot:{visual_show_radar_age:true}},{visual_show_radar_age:false});
      assert.equal(flags.visual_show_radar_age,false);
      assert.equal(visualConfig({},{visual_trail_style:'full'}).visual_trail_style,'history');
    """)


def test_world_clipping_and_adaptive_spacing():
    run_js("""
      const grid=new SparseVisitGrid([-500,-500,500,500],250);
      grid.update([{t:0,uav:{x:-2000,y:0}},{t:10,uav:{x:2000,y:0}}],1);
      assert.equal(grid.cells.size,4);
      for(const cell of grid.cells.values())assert(cell.x>=0&&cell.x<4&&cell.y===2);
      const step=adaptiveGridStep(250,.002,12);
      assert.equal(step%250,0);assert(step*.002>=12);
      assert.equal(adaptiveGridStep(250,.1,12),250);
    """)


def test_physical_model_sizes_and_minimum_pixels():
    run_js("""
      const c=visualConfig({}),u=modelSize('uav',.001,c),s=modelSize('ship',.001,c);
      assert.equal(Math.max(u.width,u.height),12);
      assert.equal(Math.max(s.width,s.height),8);assert.equal(u.width/u.height,9/18);
      const real=modelSize('ship',2,c);assert.equal(real.width,166);assert.equal(real.height,32);
      const custom=visualConfig({config_snapshot:{visual_uav_length:10,visual_uav_wingspan:20,visual_uav_min_pixels:5}});
      assert.equal(modelSize('uav',1,custom).width,10);
      assert.equal(c.visual_show_trail,true);assert.equal(c.visual_trail_style,'meteor');
    """)


def test_meteor_window_has_no_future_segments():
    run_js("""
      const frames=[0,30,90,120,150].map(t=>({t,uav:{x:t,y:0},ship:{x:t,y:5}}));
      const segments=trailSegments(frames,3,'uav',60,'meteor');
      assert.equal(segments[0].from.x,60);assert.equal(segments[0].t0,60);
      assert.equal(segments.at(-1).to.x,120);assert.equal(segments.at(-1).alpha1,1);
      assert(segments.every(s=>s.t1<=120));
      assert.equal(trailSegments(frames,0,'uav',60,'meteor').length,0);
      const history=trailSegments(frames,2,'uav',60,'history');
      assert.equal(history[0].from.x,0);assert.equal(history.at(-1).to.x,90);
    """)


def test_missing_positions_do_not_invent_connecting_paths():
    run_js("""
      const frames=[{t:0,uav:{x:0,y:0}},{t:10},{t:20,uav:{x:20,y:0}}];
      assert.equal(trailSegments(frames,2,'uav',60,'meteor').length,0);
      const grid=new SparseVisitGrid([0,0,2000,2000],250);grid.update(frames,2);
      assert.equal(grid.cells.size,1);
    """)


def test_returned_cell_becomes_freshest_again_without_wall_time():
    run_js("""
      const frames=[{t:0,uav:{x:125,y:125}},{t:10,uav:{x:625,y:125}},{t:20,uav:{x:125,y:125}}];
      const grid=new SparseVisitGrid([0,0,2000,2000],250);grid.update(frames,2);
      assert.equal(grid.cells.get('0,0').t,20);
      assert(grid.cells.get('2,0').t<20);
      const before=JSON.stringify([...grid.cells]);grid.update(frames,2);
      assert.equal(JSON.stringify([...grid.cells]),before);
    """)


def test_server_visual_overrides_survive_scene_switches():
    run_js("""
      globalThis.ResizeObserver=class{observe(){}};
      globalThis.Image=class{};
      const {MaritimeMap}=await import('./visual/map.js');
      const map=new MaritimeMap({getContext:()=>({}),addEventListener(){},parentElement:{}});
      map.setVisualConfig({visual_grid_cell_size:500,visual_trail_seconds:30,visual_show_radar_age:false});
      for(const value of [100,250]){
        map.setScene({world_bounds:[0,0,300000,300000],config_snapshot:{visual_grid_cell_size:value,visual_trail_seconds:90}});
        assert.equal(map.config.visual_grid_cell_size,500);
        assert.equal(map.config.visual_trail_seconds,30);
        assert.equal(map.showAge,false);assert.equal(map.visits.cellSize,500);
      }
    """)


def test_actual_renderer_keeps_radar_physical_and_rotates_small_models():
    run_js("""
      globalThis.ResizeObserver=class{observe(){}};globalThis.Image=class{};
      const {MaritimeMap}=await import('./visual/map.js');
      const images=[],rotations=[],arcs=[];
      const context={save(){},restore(){},translate(){},rotate(a){rotations.push(a)},drawImage(...a){images.push(a)},fillText(){},createRadialGradient:()=>({addColorStop(){}}),beginPath(){},moveTo(){},arc(...a){arcs.push(a)},closePath(){},fill(){},stroke(){},lineTo(){},setLineDash(){}};
      const map=new MaritimeMap({getContext:()=>context,addEventListener(){},parentElement:{}});
      map.setScene({world_bounds:[0,0,300000,300000]});map.models.uav={};map.models.ship={};
      map.scale=.02;map.center=[0,0];map.width=400;map.height=300;
      const uav={x:0,y:0,vx:0,vy:61,heading:0};
      map.frame={uav,radar:{range:3000,offset:0,pan:0,half_angle:Math.PI/6}};
      map.drawVehicle(uav,'uav');map.drawVehicle({x:0,y:0,vx:6,vy:0},'ship');map.drawRadar();
      assert.equal(rotations[0],-Math.PI/2);
      assert.equal(images[0][7],6);assert.equal(images[0][8],12);
      assert.equal(images[1][7],8);assert.equal(arcs[0][2],60);
      map.setVisualConfig({visual_uav_min_pixels:120});map.drawRadar();
      assert.equal(arcs.at(-1)[2],60);
    """)


def run_app_js(script):
    """Exercise real app status/catalog code without starting a server or browser."""
    harness = """
      import assert from 'node:assert/strict';
      import fs from 'node:fs';
      import vm from 'node:vm';
      import {fieldGeometry} from './visual/field.js';
      class Element {
        constructor(){this.textContent='';this.value='';this.children=[];this.style={};this.classList={toggle(){}};this.listeners={};}
        get parentElement(){return this;}
        addEventListener(name,fn){this.listeners[name]=fn;}
        setAttribute(){}
        querySelector(){return new Element();}
        replaceChildren(){this.children=[];}
        append(node){this.children.push(node);}
      }
      const elements=new Map();
      const document={getElementById(id){if(!elements.has(id))elements.set(id,new Element());return elements.get(id);},querySelectorAll:()=>[],createElement:()=>new Element()};
      class MaritimeMap {
        constructor(){this.config={visual_trail_seconds:60,visual_grid_cell_size:250,visual_grid_fade_seconds:600};}
        setScene(){} setView(){}
      }
      const context=vm.createContext({document,MaritimeMap,fieldGeometry,Map,Number,Math,performance:{now:()=>0},window:{addEventListener(){}},ResizeObserver:class{observe(){}},setInterval(){},fetch:async()=>({ok:true,json:async()=>({replays:[]})}),formatTime:()=> '01:00'});
      const source=fs.readFileSync('visual/app.js','utf8').replace(/^import .*$/gm,'').replace('updateChoices();activateReplay(null);loadVisualConfig();refreshCatalog(true);connectLive();requestAnimationFrame(tick);','');
      vm.runInContext(source,context);
      const execute=code=>vm.runInContext(code,context);
    """
    result = subprocess.run([NODE, '--input-type=module', '-e', harness+script],
                            cwd=ROOT, capture_output=True, text=True, encoding='utf-8', timeout=15)
    assert result.returncode == 0, result.stderr or result.stdout


def test_constrained_policy_is_identified_in_history_and_live_status_and_catalog():
    run_app_js("""
      for(const mode of ['history','live']){
        execute(`state.mode='${mode}';state.actor='policy';activateReplay({metadata:{control_mode:'forecast_constraint'},frames:[{t:0}],scene:{}},'constraint.json')`);
        assert(elements.get('status').textContent.includes('行为克隆网络 + 显式物理约束'));
      }
      execute("state.mode='history';state.records=[{name:'constraint.json',url:'c',task:'coverage',actor:'policy',control_mode:'forecast_constraint',frames:2,duration:60}];renderCatalog()");
      assert(elements.get('replay-select').children[0].textContent.includes('行为克隆网络 + 显式物理约束'));
      execute("activateReplay({metadata:{},frames:[{t:0}],scene:{}},'pure.json')");
      assert(!elements.get('status').textContent.includes('显式物理约束'));
      execute("activateReplay({metadata:{behavior:'dagger_training',mixture_beta:0.5},frames:[{t:0}],scene:{}},'dagger.json')");
      assert(elements.get('status').textContent.includes('DAgger训练混合控制 β=0.5'));
      assert(!elements.get('status').textContent.includes('显式物理约束'));
      execute("state.records=[{name:'pure.json',url:'p',task:'coverage',actor:'policy',frames:2,duration:60},{name:'dagger.json',url:'d',task:'coverage',actor:'policy',behavior:'dagger_training',frames:2,duration:60}];renderCatalog()");
      assert.equal(elements.get('replay-select').children[0].textContent,'pure.json · 01:00 · 2 帧');
      assert.equal(elements.get('replay-select').children[1].textContent,'dagger.json [DAgger训练混合] · 01:00 · 2 帧');
      assert.equal(execute('state.actor'),'policy');
    """)


def test_field_controls_distinguish_missing_data_and_independent_layers():
    run_app_js("""
      execute('map.showPotential=true;map.showFieldVectors=true;updateFieldControls({t:0})');
      assert.equal(elements.get('potential-key').hidden,true);
      assert.equal(elements.get('field-notice').hidden,false);
      execute('map.showPotential=false;map.showFieldVectors=false;updateFieldControls({t:0})');
      assert.equal(elements.get('field-notice').hidden,true);
      execute('map.showFieldVectors=true;updateFieldControls({potential:{bounds:[0,0,12000,12000],values:[[0,1],[0,1]],range:[-2,1]}})');
      assert.equal(elements.get('potential-key').hidden,false);
      assert.equal(elements.get('potential-scale').hidden,true);
      assert.equal(elements.get('field-vector-key').hidden,false);
      assert(elements.get('field-resolution').textContent.includes('6.0 km'));
      execute('map.showPotential=true;map.showFieldVectors=false;updateFieldControls({potential:{bounds:[0,0,2,2],values:[[0,1],[0,1]]}})');
      assert.equal(elements.get('potential-scale').hidden,false);
      assert.equal(elements.get('field-vector-key').hidden,true);
      execute('updateFieldControls(null)');
      assert.equal(elements.get('potential-key').hidden,true);
      assert.equal(elements.get('field-notice').hidden,true);
    """)


def test_local_import_and_catalog_refresh_keep_constraint_disclosure():
    run_app_js("""
      context.validateReplay=data=>data;
      context.file={name:'local-constraint.json',text:async()=>JSON.stringify({task:'coverage',actor:'policy',metadata:{control_mode:'forecast_constraint'},scene:{},frames:[{t:0}]})};
      await execute("document.getElementById('replay-file').listeners.change({target:{files:[file]}})");
      assert(elements.get('replay-select').children[0].textContent.includes('行为克隆网络 + 显式物理约束'));
      await execute('refreshCatalog(false)');
      assert(elements.get('replay-select').children[0].textContent.includes('行为克隆网络 + 显式物理约束'));
    """)


def test_recording_mime_detection_and_real_timeline_endpoints():
    script = """
      import assert from 'node:assert/strict';
      import {recordingMime, timelineTime, replayProvenance} from './visual/recording.js';
      assert.equal(recordingMime({isTypeSupported:t=>t==='video/webm;codecs=vp8'}),'video/webm;codecs=vp8');
      assert.throws(()=>recordingMime({isTypeSupported:()=>false}),/recording format/i);
      assert.equal(timelineTime(0,30,0,10800,24),0);
      assert.equal(timelineTime(30-1/24,30,0,10800,24),10800);
      assert.equal(timelineTime(45,30,0,1800,24),1800);
      const provenance=replayProvenance({task:'coverage',actor:'policy',metadata:{model_sha256:'abc',control_mode:'forecast_constraint'},summary:{terminated:true,termination_reason:'collision'},frames:[{t:0},{t:123}]});
      assert.equal(provenance.control_mode,'forecast_constraint');
      assert.equal(provenance.simulation_end_seconds,123);
      assert.equal(provenance.outcome.termination_reason,'collision');
    """
    result = subprocess.run([NODE, '--input-type=module', '-e', script], cwd=ROOT,
                            capture_output=True, text=True, encoding='utf-8', timeout=15)
    assert result.returncode == 0, result.stderr
