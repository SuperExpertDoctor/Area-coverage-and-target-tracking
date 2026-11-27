"""Public local field, timestamp recovery and physical pan protocol."""
import math
import numpy as np
import pytest
from config import Config
from env_coverage import CoverageEnv
from env_tracking import TrackingEnv


def field_config(**kwargs):
    assert hasattr(Config(), 'potential_enabled'), 'Config must explicitly opt into field-v2'
    return Config(potential_enabled=True, **kwargs)


def test_preference_endpoints_monotonic_exponential():
    from potential_field import scan_preference
    values=scan_preference(np.linspace(0,3600,101),3600.,3.)
    assert values[0]==pytest.approx(-1)
    assert values[-1]==pytest.approx(1)
    assert np.all(np.diff(values)>0)
    assert np.diff(values)[0]>np.diff(values)[-1]
    assert scan_preference(7200.,3600.,3.)==pytest.approx(1)


def test_very_small_decay_retains_linear_limit_and_exact_endpoints():
    from potential_field import scan_preference
    values=scan_preference(np.array([0.,900.,1800.,2700.,3600.]),3600.,1e-16)
    np.testing.assert_allclose(values,[-1.,-.5,0.,.5,1.],atol=1e-14)


@pytest.mark.parametrize('time',[0.,1800.,3600.,36000.])
def test_attraction_never_leaks_outside_coverage_subtask_during_aging(time):
    from potential_field import LocalPotentialField
    c=field_config();roi=(90000.,90000.,110000.,110000.)
    cloud={'x':90000.,'y':100000.,'radius':750.}
    island={'x':100000.,'y':100000.,'radius':1000.}
    f=LocalPotentialField(c,roi,[island],[cloud])
    # Populate history, including cells beyond the ROI boundary: final domain
    # masking must dominate both history and normalized exponential recovery.
    for x in range(230,300):
        for y in range(230,300):f.last_seen[x,y]=0.
    for position in ((89000.,100000.),(100000.,100000.),(111000.,100000.)):
        p=f.snapshot(position,time);axis=np.linspace(-c.potential_radius,c.potential_radius,33)
        x,y=np.meshgrid(position[0]+axis,position[1]+axis)
        outside=(x<roi[0])|(x>roi[2])|(y<roi[1])|(y>roi[3])
        valid=p[3]>.5;hard=p[2]>.5
        assert np.all(p[0][outside&valid&~hard]==-1.)
        assert np.all(p[0][hard]==-2.)
        assert not np.any((p[0]>0)&(p[1]<.5))


def test_sparse_history_reentry_cloud_override_and_bounded_local_cache():
    from potential_field import LocalPotentialField
    c=field_config()
    field=LocalPotentialField(c,(90000.,90000.,110000.,110000.),[],[])
    center=(100312.5,100312.5)  # Center of the R/16 timestamp lattice cell.
    field.mark_scan(center,0.,3000.,math.pi/6,10.)
    assert field.snapshot(center,10.)[0,16,16]==pytest.approx(-1)
    field.snapshot((200000.,200000.),100.)
    returned=field.snapshot(center,1810.)
    assert -1<returned[0,16,16]<1
    assert isinstance(field.last_seen,dict)
    assert field.cache.shape==(4,33,33)
    assert all(v.shape[-2:]==(33,33) for v in vars(field).values() if isinstance(v,np.ndarray))
    field.clouds=[{'x':89000.,'y':100125.,'radius':1000.}]
    outside=field.snapshot((89000.,100125.),1810.)
    assert outside[0,16,16]==-2 and outside[2,16,16]==1
    assert outside[0,16,7]==-1
    assert not outside[3,0,0]


@pytest.mark.parametrize('cls,size',[(CoverageEnv,111),(TrackingEnv,134)])
def test_field_schema_context_and_vector_compatibility(cls,size):
    c=field_config(coverage_episode_seconds=5,tracking_episode_seconds=5)
    env=cls(c); obs,_=env.reset(seed=13)
    legacy=cls(Config(coverage_episode_seconds=5,tracking_episode_seconds=5))
    old,_=legacy.reset(options={'scene':env.scene})
    assert set(obs)=={'vector','potential','field_context'}
    assert obs['vector'].shape==(size,)
    np.testing.assert_array_equal(obs['vector'],old['vector'])
    assert obs['potential'].shape==(4,33,33)
    assert obs['field_context'].shape==(16,)
    assert env.observation_space.contains(obs)
    assert env.observation_schema['schema_version']=='obs-field-v2'
    context=obs['field_context']
    np.testing.assert_allclose(context[:7],[c.potential_radius/3000,c.potential_period/3600,c.potential_decay,
        env.radar_pan/math.pi,(c.coverage_offset if env.task=='coverage' else 0)/math.pi,
        (c.coverage_pan_limit if env.task=='coverage' else c.pan_limit)/math.pi,c.max_pan_rate])
    assert context[10]==1 and context[15]==int(env.task=='tracking')
    assert context[14]==pytest.approx(env.coverage_ratio if env.task=='coverage' else 0.)


@pytest.mark.parametrize('cls',[CoverageEnv,TrackingEnv])
def test_three_action_unit_xy_and_finite_pan(cls):
    env=cls(field_config(coverage_episode_seconds=5,tracking_episode_seconds=5))
    env.reset(seed=13); start=env.radar_pan
    xy=np.array([math.cos(env.uav.heading),math.sin(env.uav.heading)])*.25
    _,_,terminated,_,info=env.step([*xy,1.])
    assert not terminated
    limit=env.config.coverage_pan_limit if env.task=='coverage' else env.config.pan_limit
    assert env.radar_pan==pytest.approx(min(limit,start+env.config.max_pan_rate))
    assert np.linalg.norm(env.previous_action)==pytest.approx(1.)
    assert np.linalg.norm(info['executed_action'])==pytest.approx(1.)
    assert info['radar_pan']==env.radar_pan
    for action in ([1,0],[1,0,1.01],[1,0,float('nan')]):
        with pytest.raises(ValueError):env.step(action)


def test_coverage_actual_pan_scans_and_islands_do_not_block_uav():
    from dynamics import FixedWingState
    from geometry import sector_mask
    env=CoverageEnv(field_config());env.reset(seed=13)
    roi=env.scene['roi'];env.uav=FixedWingState((roi[0]+roi[2])/2,(roi[1]+roi[3])/2,0.)
    env.radar_pan=env.config.coverage_pan_limit;env.last_seen_tick[:]=-1;env._update_task()
    expected=sector_mask(env.grid_x,env.grid_y,(env.uav.x,env.uav.y),env.uav.heading+env.config.coverage_offset+env.radar_pan,env.config.coverage_range,env.config.coverage_half_angle)&env.sea_mask
    np.testing.assert_array_equal(env.last_seen_tick>=0,expected)
    env.scene['islands']=[{'x':env.uav.x,'y':env.uav.y,'radius':1000.}]
    env._safety(env.uav,env.ship,.1)
    assert not env.uav_collision


def test_tracking_has_explicit_pan_and_no_scan_repulsion():
    env=TrackingEnv(field_config());env.reset(seed=13)
    env.radar_pan=0.;env.desired_radar_pan=0.;env._update_radar(1.)
    assert env.radar_pan==0.
    before=env.potential_snapshot().copy()
    env.potential_field.mark_scan((env.uav.x,env.uav.y),env.uav.heading,3000.,math.pi,env.t)
    np.testing.assert_array_equal(env.potential_snapshot(),before)


def test_public_goal_holds_until_neighborhood_is_scanned():
    env=CoverageEnv(field_config());env.reset(seed=13);goal=env.public_goal.copy()
    env.physics_tick+=10;env.potential_snapshot()
    np.testing.assert_array_equal(env.public_goal,goal)
    near=((env.grid_x-goal[0])**2+(env.grid_y-goal[1])**2<=env.config.potential_goal_radius**2)&env.sea_mask
    env.last_seen_tick[near]=env.physics_tick;env.potential_snapshot()
    assert not np.array_equal(env.public_goal,goal)


def test_goal_expiry_reconsiders_distant_neglected_water_and_context_is_public():
    env=CoverageEnv(field_config());obs,_=env.reset(seed=13)
    travel=np.linalg.norm(env.public_goal-[env.uav.x,env.uav.y])/env.config.uav_speed
    env.physics_tick+=int((travel+env.config.potential_goal_hold_seconds)/env.config.physics_dt)+1
    env.last_seen_tick[env.sea_mask]=env.physics_tick
    distant=np.where(env.sea_mask,np.hypot(env.grid_x-env.uav.x,env.grid_y-env.uav.y),-1.)
    index=np.unravel_index(np.argmax(distant),distant.shape)
    env.last_seen_tick[index]=-1
    env.potential_snapshot();context=env.field_context()
    np.testing.assert_allclose(env.public_goal,[env.grid_x[index],env.grid_y[index]])
    np.testing.assert_allclose(context[7:9],(env.public_goal-[env.uav.x,env.uav.y])/3000)
    assert context[9]==pytest.approx(1.)


def test_goal_deadline_allows_initial_travel_before_hold_timer():
    env=CoverageEnv(field_config());env.reset(seed=13);goal=env.public_goal.copy()
    travel=np.linalg.norm(goal-[env.uav.x,env.uav.y])/env.config.uav_speed
    assert travel>env.config.potential_goal_hold_seconds
    env.physics_tick+=int(env.config.potential_goal_hold_seconds/env.config.physics_dt)+1
    env.last_seen_tick[env.sea_mask]=env.physics_tick
    near=((env.grid_x-goal[0])**2+(env.grid_y-goal[1])**2<=env.config.potential_goal_radius**2)&env.sea_mask
    env.last_seen_tick[near]=0
    candidates=np.where(env.sea_mask&~near,np.hypot(env.grid_x-env.uav.x,env.grid_y-env.uav.y),-1.)
    index=np.unravel_index(np.argmax(candidates),candidates.shape)
    env.last_seen_tick[index]=-1
    env.potential_snapshot()
    np.testing.assert_array_equal(env.public_goal,goal)


def test_field_orientation_world_boundary_and_island_are_distinct():
    from potential_field import LocalPotentialField
    field=LocalPotentialField(field_config(),(0.,0.,20000.,20000.),
        [{'x':4000.,'y':4000.,'radius':500.}],[])
    value=field.snapshot((4000.,4000.),0.)
    assert value[1,16,16]==0 and value[2,16,16]==0
    assert value[3,0,16]==1 and value[0,0,16]==-2  # South is below world y=0.
    assert value[0,32,16]==1  # North is inside ROI, never scanned.


def test_render_reuses_local_field_and_contains_explicit_pan_metadata():
    env=CoverageEnv(field_config());env.reset(seed=13)
    first=env.potential_snapshot()
    frame=env.render()
    assert env.potential_snapshot() is first
    assert len(frame['potential']['values'])==33
    assert frame['potential']['range']==[-2,1]
    assert frame['radar']['pan']==env.radar_pan
    np.testing.assert_allclose(frame['potential']['bounds'],[env.uav.x-env.config.potential_radius,
        env.uav.y-env.config.potential_radius,env.uav.x+env.config.potential_radius,env.uav.y+env.config.potential_radius])


def test_radar_scan_temporaries_stay_bounded_when_query_radius_is_small(monkeypatch):
    import potential_field
    field=potential_field.LocalPotentialField(field_config(potential_radius=1000.),(90000.,90000.,110000.,110000.),[],[])
    original=np.meshgrid
    materialized=[]
    def measured_meshgrid(*args,**kwargs):
        result=original(*args,**kwargs)
        materialized.extend(array.shape for array in result)
        return result
    monkeypatch.setattr(potential_field.np,'meshgrid',measured_meshgrid)
    field.mark_scan((100000.,100000.),0.,3000.,math.pi/6,0.)
    field.snapshot((100000.,100000.),0.)
    assert max(max(shape) for shape in materialized)<=33
    assert field.last_seen  # Real timestamp history, not a skipped physical scan.


def test_legacy_defaults_unchanged():
    assert hasattr(Config(),'potential_enabled')
    assert Config().potential_enabled is False
    env=CoverageEnv(Config());obs,_=env.reset(seed=13)
    assert set(obs)=={'vector','coverage_map'}
    assert env.action_space.shape==(2,)
    assert env.observation_schema['schema_version']=='obs-v1'


@pytest.mark.parametrize('pan',[math.pi/2,float('nan'),float('inf')])
def test_field_scene_rejects_invalid_coverage_initial_pan(pan):
    from copy import deepcopy
    env=CoverageEnv(field_config());env.reset(seed=13)
    scene=deepcopy(env.scene);scene['radar_pan_initial']=pan
    with pytest.raises(ValueError,match='radar pan'):env.reset(options={'scene':scene})
