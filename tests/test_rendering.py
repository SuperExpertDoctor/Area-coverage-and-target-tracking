"""Exported artists honor real simulation time and the shared layer switches."""
import numpy as np
import pytest


def test_meteor_clips_sparse_time_window_and_never_uses_future_points():
    from rendering import _meteor_segments
    positions=np.array([[0.,0.],[30.,0.],[120.,0.],[150.,0.]])
    segments,strength=_meteor_segments(positions,np.array([0.,30.,120.,150.]),2,60.)
    np.testing.assert_allclose(segments[0,0],[60.,0.])
    np.testing.assert_allclose(segments[-1,-1],[120.,0.])
    assert np.all(np.diff(strength)>=0) and strength[0]<strength[-1]
    assert _meteor_segments(positions,np.array([0.,30.,120.,150.]),0,60.)[0].size==0


@pytest.mark.parametrize('style,show,age',[('meteor',False,False),('meteor',True,True),('full',True,False)])
def test_movie_layer_controls_apply_to_both_vehicles_and_radar_age(tmp_path,monkeypatch,style,show,age):
    import rendering
    from matplotlib.collections import LineCollection
    from matplotlib.figure import Figure
    frames=[]
    for t in [0.,30.,120.]:
        frames.append({'t':t,'uav':{'x':11000+61*t,'y':10000,'vx':61,'vy':0,'heading':0},
            'ship':{'x':11000+6*t,'y':12000,'vx':6,'vy':0,'heading':0},
            'radar':{'range':3000,'half_angle':.5},
            'coverage':{'grid':[[0,1],[-1,2]],'seen':[[True,False],[False,True]],
                        'grid_bounds':[10000,10000,20000,20000]},'metrics':{}})
    record={'task':'coverage','actor':'expert','seed':7,'summary':{},'frames':frames,
            'scene':{'world_bounds':[0,0,300000,300000],'roi':[10000,10000,20000,20000]}}
    class Writer:
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def append_data(self,*args):pass
    captured=[];close=rendering.plt.close
    monkeypatch.setattr(rendering.imageio,'get_writer',lambda *a,**k:Writer())
    monkeypatch.setattr(Figure,'savefig',lambda self,*a,**k:captured.append(self))
    monkeypatch.setattr(rendering.plt,'close',lambda *a:None)
    rendering.export_video(record,tmp_path/'test.mp4',max_frames=3,width=640,height=480,
        visual_config={'visual_show_trail':show,'visual_trail_style':style,
                       'visual_show_radar_age':age,'visual_show_grid':False})
    figure=captured[0];ax=figure.axes[0]
    try:
        assert [len(line.get_xdata()) for line in ax.lines]==([3,3] if show and style=='full' else [0,0])
        collections=[c for c in ax.collections if isinstance(c,LineCollection)]
        assert all(bool(len(c.get_segments()))==(show and style=='meteor') for c in collections)
        assert len(ax.images)==int(age)
        if age:
            pixels=np.asarray(ax.images[0].get_array())
            assert pixels[1,0,3]==0 and pixels[0,1,3]>0
    finally:close(figure)
