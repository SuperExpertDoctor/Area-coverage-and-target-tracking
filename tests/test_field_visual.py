"""Fixed physical preference scale must not change with frame min/max."""
import subprocess
import shutil
from pathlib import Path
import pytest
import numpy as np


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
