"""Fixed physical preference scale must not change with frame min/max."""
import subprocess
import shutil
from pathlib import Path
import pytest
import numpy as np


def run_field_js(script):
    node = shutil.which('node')
    if node is None:
        pytest.skip('Node required')
    uri = (Path(__file__).parents[1] / 'visual' / 'field.js').as_uri()
    result = subprocess.run(
        [node, '--input-type=module', '-e',
         f"import assert from 'node:assert/strict'; import * as field from {uri!r};\n{script}"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True, encoding='utf-8', timeout=15)
    assert result.returncode == 0, result.stderr


def test_field_direction_uses_increasing_preference_and_physical_spacing():
    run_field_js("""
      const raster={bounds:[0,0,4,2],values:[[0,2,4],[1,3,5],[2,4,6]]};
      for(const [row,col] of [[1,1],[0,0],[2,2]]){
        const g=field.potentialGradient(raster,row,col);
        assert.equal(g.x,1);assert.equal(g.y,1);
        assert.equal(g.magnitude,Math.SQRT2);
      }
      // The project stores preference, so +gradient points out of a low-valued cloud.
      const cloud={bounds:[0,0,2,2],values:[[1,1,1],[-2,-1,1],[1,1,1]]};
      assert(field.potentialGradient(cloud,1,1).x>0);
    """)


def test_field_masks_and_missing_samples_never_create_artificial_force():
    run_field_js("""
      const raster={bounds:[0,0,2,2],values:[[0,0,0],[0,1,1],[0,1,1]],
        valid:[[0,0,0],[0,1,1],[0,1,1]]};
      assert.deepEqual(field.potentialGradient(raster,1,1),{x:0,y:0,magnitude:0});
      assert.equal(field.potentialGradient(raster,0,0),null);
      raster.valid[1][0]=true;raster.values[1][0]=-1;
      assert.equal(field.potentialGradient(raster,1,1).x,1);
      raster.values[1][0]=NaN;
      assert.equal(field.potentialGradient(raster,1,1).x,0);
      raster.valid[1][1]=false;
      assert.equal(field.potentialGradient(raster,1,1),null);
    """)


def test_field_vectors_handle_flat_empty_malformed_and_zoomed_out_data():
    run_field_js("""
      for(const raster of [null,{}, {bounds:[0,0,2,2],values:[]},
        {bounds:[0,0,0,2],values:[[0,1],[0,1]]},
        {bounds:[0,0,2,2],values:[[0,1],[0]]}]){
        assert.equal(field.fieldGeometry(raster),null);
        assert.deepEqual(field.potentialVectors(raster,{scale:1}),[]);
      }
      const flat={bounds:[0,0,3200,3200],values:Array.from({length:33},()=>Array(33).fill(1))};
      assert.deepEqual(field.potentialVectors(flat,{scale:1}),[]);
      const raster={...flat,values:Array.from({length:33},()=>Array.from({length:33},(_,i)=>i/32))};
      const close=field.potentialVectors(raster,{scale:1});
      const far=field.potentialVectors(raster,{scale:.03});
      assert(close.length>far.length);assert(far.length>0);
      assert(close.every(v=>v.gx>0&&v.gy===0&&v.magnitude>0));
      const visible=field.potentialVectors(raster,{scale:1,viewport:[1000,1000,2000,2000]});
      assert(visible.length<close.length);
      assert(visible.every(v=>v.x>=1000&&v.x<=2000&&v.y>=1000&&v.y<=2000));
    """)


def test_field_canvas_arrows_point_north_and_layer_preferences_survive_scene_changes():
    run_field_js("""
      globalThis.ResizeObserver=class{observe(){}};globalThis.Image=class{};
      const {MaritimeMap}=await import('./visual/map.js');
      const paths=[];let path=[];
      const c={save(){},restore(){},beginPath(){path=[]},moveTo(x,y){path.push([x,y])},
        lineTo(x,y){path.push([x,y])},stroke(){paths.push(path)},setLineDash(){}};
      const map=new MaritimeMap({getContext:()=>c,addEventListener(){},parentElement:{}});
      map.width=200;map.height=200;map.center=[100,100];map.scale=.5;
      map.frame={potential:{bounds:[0,0,200,200],values:[[0,0,0],[1,1,1],[2,2,2]]}};
      map.drawPotentialVectors();assert(paths.length>0);
      for(const p of paths){assert.equal(p[0][0],p[1][0]);assert(p[1][1]<p[0][1]);}
      map.setLayerOptions({showFieldVectors:false});
      map.setScene({config_snapshot:{visual_show_field_vectors:true}});
      assert.equal(map.showFieldVectors,false);
      map.setLayerOptions({showFieldVectors:true,showPotential:false});
      map.setScene({});assert.equal(map.showFieldVectors,true);assert.equal(map.showPotential,false);
      paths.length=0;map.frame={};map.drawPotentialVectors();assert.equal(paths.length,0);
      map.frame={uav:{x:100,y:200},potential:{bounds:[-5900,-5800,6100,6200],
        values:[[0,1],[0,1]],range:[-2,1]}};
      map.setView('field');map.camera();
      assert.deepEqual(map.center,[100,200]);
      assert(Math.abs(map.scale-110/16200)<1e-12);
      assert.equal(field.fieldGeometry(map.frame.potential).radius,6000);
    """)


def test_screen_sampling_preserves_thin_transition_bands():
    run_field_js("""
      const raster={bounds:[-6000,-6000,6000,6000],
        values:Array.from({length:33},()=>Array.from({length:33},(_,col)=>col<18?-1:1)),
        valid:Array.from({length:33},(_,row)=>Array.from({length:33},(_,col)=>
          (col-16)**2+(row-16)**2<=16**2))};
      for(const scale of [.02,.03]){
        const arrows=field.potentialVectors(raster,{scale});
        assert(arrows.length>0,'A sampled step boundary must retain its nonzero direction');
        assert(arrows.every(v=>v.gx>0&&v.gy===0));
        assert(arrows.every(v=>v.x===375||v.x===750),'Arrows stay at actual nonflat samples');
      }
    """)


def test_canvas_rewind_clears_the_entire_rounded_bitmap():
    run_field_js("""
      globalThis.ResizeObserver=class{observe(){}};globalThis.Image=class{};
      globalThis.document={getElementById:()=>({style:{}})};
      const {MaritimeMap}=await import('./visual/map.js');
      let cleared;
      const context={setTransform(){},save(){},restore(){},clearRect(...r){cleared=r},fillRect(){}};
      const map=new MaritimeMap({width:200,height:101,getContext:()=>context,addEventListener(){},parentElement:{}});
      map.width=200;map.height=100.5;map.dpr=1;map.scale=1;map.center=[0,0];
      map.camera=()=>[0,0,200,100.5];
      for(const name of ['drawGrid','drawCoverage','drawVisitGrid','drawPotential','drawObstacles',
        'drawPotentialVectors','drawFieldGuide','drawTrails','drawRadar'])map[name]=()=>{};
      map.draw();assert.deepEqual(cleared,[0,0,200,101]);
    """)


def test_matplotlib_field_uses_same_fixed_scale_and_mask():
    from rendering import potential_rgba
    values=np.array([[-2.,-1.,0.,1.]])
    colors=potential_rgba(values,np.array([[1,1,0,1]]))
    np.testing.assert_allclose(colors[0,0],[219/255,73/255,83/255,.65])
    np.testing.assert_allclose(colors[0,1],[205/255,113/255,174/255,.32])
    np.testing.assert_array_equal(colors[0,2],[0.,0.,0.,0.])
    np.testing.assert_allclose(colors[0,3],[63/255,211/255,161/255,.32])


def test_signed_field_color_scale_and_mask():
    node=shutil.which('node')
    if node is None: pytest.skip('Node required')
    uri=(Path(__file__).parents[1]/'visual'/'layers.js').as_uri()
    script=f"""
      import assert from 'node:assert/strict';
      import {{potentialColor}} from {uri!r};
      assert.deepEqual(potentialColor(-2),[219,73,83,0.65]);
      assert.deepEqual(potentialColor(-1),[205,113,174,0.32]);
      assert.deepEqual(potentialColor(1),[63,211,161,0.32]);
      assert.deepEqual(potentialColor(1,false),[0,0,0,0]);
      assert.deepEqual(potentialColor(NaN),[0,0,0,0]);
      assert(potentialColor(0)[3]<potentialColor(1)[3]);
    """
    result=subprocess.run([node,'--input-type=module','-e',script],capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def test_browser_attractive_cell_is_clipped_at_subtask_boundary():
    node=shutil.which('node')
    if node is None:pytest.skip('Node required')
    uri=(Path(__file__).parents[1]/'visual'/'layers.js').as_uri()
    script=f"""
      import assert from 'node:assert/strict';
      import {{potentialCellBounds}} from {uri!r};
      const f={{bounds:[0,0,100,100],values:[[1,1],[1,-2]]}};
      assert.deepEqual(potentialCellBounds(f,0,0,[0,0,100,100]),[0,0,50,50]);
      assert.deepEqual(potentialCellBounds(f,1,1,[0,0,100,100]),[50,50,150,150]);
      assert.equal(potentialCellBounds(f,0,0,[200,200,300,300]),null);
    """
    result=subprocess.run([node,'--input-type=module','-e',script],capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def test_movie_attraction_layer_has_roi_clip_without_clipping_clouds(tmp_path,monkeypatch):
    import rendering
    from matplotlib.figure import Figure
    roi=[10000,10000,20000,20000]
    frame={'t':0.,'uav':{'x':15000.,'y':15000.,'vx':61.,'vy':0.,'heading':0.},
           'ship':{'x':15000.,'y':15000.,'vx':6.,'vy':0.},'radar':{'range':3000.,'half_angle':.5},'metrics':{},
           'potential':{'values':[[1.,-2.],[-1.,1.]],'valid':[[1,1],[1,1]],'bounds':roi}}
    record={'task':'coverage','actor':'expert','seed':0,'summary':{},
            'scene':{'world_bounds':[0,0,300000,300000],'roi':roi},'frames':[frame,{**frame,'t':1.}]}
    class Writer:
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def append_data(self,*a):pass
    captured=[];close=rendering.plt.close
    monkeypatch.setattr(rendering.imageio,'get_writer',lambda *a,**k:Writer())
    monkeypatch.setattr(Figure,'savefig',lambda self,*a,**k:captured.append(self))
    monkeypatch.setattr(rendering.plt,'close',lambda *a:None)
    rendering.export_video(record,tmp_path/'field.mp4',max_frames=2,width=640,height=480,
                           visual_config={'visual_show_radar_age':False})
    fig=captured[0];ax=fig.axes[0]
    try:
        assert len(ax.images)==2,'Attraction must have a separate ROI-clipped artist'
        negative,positive=ax.images
        assert np.asarray(negative.get_array())[0,0,3]==0.
        assert np.asarray(negative.get_array())[0,1,3]>.5
        assert np.asarray(positive.get_array())[0,0,3]>0.
        clip=positive.get_clip_box().transformed(ax.transData.inverted())
        np.testing.assert_allclose(clip.extents,[10.,10.,20.,20.])
    finally:close(fig)
