"""Optional forecasting constraint wrapper; no task expert or scene access."""
import copy
import json
import math
import numpy as np
import pytest
from config import Config
from dynamics import FixedWingModel,FixedWingState
from geometry import in_bounds,segment_circle_collision
from observations import observation_metadata


class FixedPolicy:
    def __init__(self,action):
        self.action=np.asarray(action,dtype=np.float32);self.reset_count=0;self.calls=0
        self.checkpoint_metadata={'sha256':'test-checkpoint','task':'fixture'}
    def reset(self):self.reset_count+=1;self.calls=0
    def act(self,obs):self.calls+=1;return self.action.copy()


def observation(c,state,task='coverage',clouds=(),islands=()):
    vector=np.zeros(111 if task=='coverage' else 134,dtype=np.float32)
    vector[:10]=[state.x/300000,state.y/300000,math.cos(state.heading),math.sin(state.heading),
        math.cos(state.heading),math.sin(state.heading),state.bank/c.uav_config.bank_limit,
        math.tan(state.bank)*9.80665/c.uav_speed/c.uav_config.max_yaw_rate,math.cos(state.heading),math.sin(state.heading)]
    b=c.world_bounds;vector[10:14]=[(state.x-b[0])/300000,(b[2]-state.x)/300000,(state.y-b[1])/300000,(b[3]-state.y)/300000]
    for start,obstacles in [(14,clouds),(54,islands)]:
        for index,ob in enumerate(obstacles):
            dx,dy=ob['x']-state.x,ob['y']-state.y
            vector[start+5*index:start+5*index+5]=[dx/3000,dy/3000,ob['radius']/3000,(math.hypot(dx,dy)-ob['radius'])/3000,1.]
    meta=observation_metadata(task,True);start,end=meta['slices']['params']
    vector[start:end]=[getattr(c,name)/scale for name,scale in zip(meta['parameter_order'],meta['parameter_divisors'])]
    context=np.zeros(16,dtype=np.float32)
    context[:7]=[c.potential_radius/3000,c.potential_period/3600,c.potential_decay,0.,
        (c.coverage_offset if task=='coverage' else 0.)/math.pi,
        (c.coverage_pan_limit if task=='coverage' else c.pan_limit)/math.pi,c.max_pan_rate]
    context[15]=int(task=='tracking')
    return {'vector':vector,'potential':np.zeros((4,33,33),dtype=np.float32),'field_context':context}


def run_physical_forecast(state,action,clouds,c,seconds=None):
    model=FixedWingModel(c.uav_config);duration=c.expert_safety_horizon if seconds is None else seconds
    count=math.ceil(duration/.1);dt=duration/count
    for _ in range(count):
        nxt=model.step_velocity(state,action[:2],dt)
        assert in_bounds((nxt.x,nxt.y),c.world_bounds)
        for cloud in clouds:
            assert not segment_circle_collision((state.x,state.y),(nxt.x,nxt.y),
                (cloud['x'],cloud['y']),cloud['radius']+c.uav_cloud_margin)
        state=nxt
    return state


def test_safe_action_is_bitexact_and_does_not_call_safety_guidance():
    from policy_constraints import ForecastConstrainedPolicy
    c=Config(potential_enabled=True);raw=np.array([.6,.8,-.1234567],dtype=np.float32)
    base=FixedPolicy(raw);policy=ForecastConstrainedPolicy(base,c,'coverage')
    def forbidden(*args,**kwargs):raise AssertionError('Guidance called for a safe raw command')
    policy.safety.direction=forbidden
    action=policy.act(observation(c,FixedWingState(100000.,100000.,0.)))
    np.testing.assert_array_equal(action.view(np.uint32),raw.view(np.uint32))
    assert action.dtype==np.float32 and action.shape==(3,)
    assert not policy.last_intervened and base.calls==1
    assert policy.checkpoint_metadata is base.checkpoint_metadata
    meta=policy.execution_metadata;json.dumps(meta)
    assert meta['mode']=='forecast_constraint' and meta['heuristic_constraint'] is True
    assert meta['task_expert_fallback'] is False and meta['checked_actions']==1 and meta['interventions']==0


def test_safe_nearby_cloud_command_passes_full_forecast_bitexact_without_guidance():
    from policy_constraints import ForecastConstrainedPolicy
    c=Config(potential_enabled=True);state=FixedWingState(100000.,100000.,0.)
    clouds=[{'x':100000.,'y':102000.,'radius':700.}]
    raw=np.array([1.,0.,-.721],dtype=np.float32)
    policy=ForecastConstrainedPolicy(FixedPolicy(raw),c,'coverage')
    def forbidden(*args,**kwargs):raise AssertionError('Guidance called for a safe full forecast')
    policy.safety.direction=forbidden
    np.testing.assert_array_equal(policy.act(observation(c,state,clouds=clouds)).view(np.uint32),raw.view(np.uint32))
    assert policy.execution_metadata['reachable_disk_checks']==0
    assert policy.execution_metadata['forecast_checks']==1 and policy.safety.sides=={}
    run_physical_forecast(state,raw,clouds,c)


@pytest.mark.parametrize('task,sign',[('coverage',1),('tracking',-1)])
def test_unsafe_cloud_action_is_corrected_and_full_roll_execution_is_safe(task,sign):
    from policy_constraints import ForecastConstrainedPolicy
    c=Config(potential_enabled=True);state=FixedWingState(100000.,100000.,0.)
    clouds=[{'id':0,'x':102500.,'y':100000.,'radius':700.}]
    raw=np.array([1.,0.,.3712345],dtype=np.float32)
    policy=ForecastConstrainedPolicy(FixedPolicy(raw),c,task)
    action=policy.act(observation(c,state,task,clouds))
    assert policy.last_intervened and policy.execution_metadata['interventions']==1
    assert action[1]*sign>0
    assert action[2].view(np.uint32)==raw[2].view(np.uint32)
    assert not np.array_equal(action[:2],raw[:2])
    run_physical_forecast(state,action,clouds,c)
    np.testing.assert_array_equal(policy.last_raw_action,raw)
    np.testing.assert_array_equal(policy.last_applied_action,action)


def test_world_boundary_action_is_repaired_without_roi_constraints():
    from policy_constraints import ForecastConstrainedPolicy
    c=Config(potential_enabled=True);state=FixedWingState(1500.,100000.,math.pi)
    policy=ForecastConstrainedPolicy(FixedPolicy([-1.,0.,-.2]),c,'coverage')
    action=policy.act(observation(c,state))
    assert policy.last_intervened
    run_physical_forecast(state,action,[],c)


def test_islands_do_not_become_uav_obstacles():
    from policy_constraints import ForecastConstrainedPolicy
    c=Config(potential_enabled=True);state=FixedWingState(100000.,100000.,0.)
    raw=np.array([1.,0.,-.5],dtype=np.float32)
    obs=observation(c,state,islands=[{'x':100000.,'y':100000.,'radius':2000.}])
    policy=ForecastConstrainedPolicy(FixedPolicy(raw),c,'coverage')
    np.testing.assert_array_equal(policy.act(obs),raw)
    assert not policy.last_intervened


def test_unrecoverable_finite_roll_state_raises_explicit_failure():
    from policy_constraints import ForecastConstrainedPolicy,NoFeasibleAction
    c=Config(potential_enabled=True);state=FixedWingState(10.,100000.,math.pi)
    policy=ForecastConstrainedPolicy(FixedPolicy([-1.,0.,.3]),c,'coverage')
    with pytest.raises(NoFeasibleAction):policy.act(observation(c,state))
    assert policy.last_applied_action is None and policy.execution_metadata['failed_actions']==1


def test_correction_gets_independent_forecast_not_blind_guidance_trust():
    from policy_constraints import ForecastConstrainedPolicy,NoFeasibleAction
    c=Config(potential_enabled=True);state=FixedWingState(100000.,100000.,0.)
    clouds=[{'x':102500.,'y':100000.,'radius':700.}]
    policy=ForecastConstrainedPolicy(FixedPolicy([1.,0.,.2]),c,'coverage')
    policy.safety.direction=lambda *args,**kwargs:np.array([1.,0.],dtype=np.float32)
    with pytest.raises(NoFeasibleAction):policy.act(observation(c,state,clouds=clouds))


def test_reset_clears_base_recurrent_state_side_memory_and_all_counters():
    from policy_constraints import ForecastConstrainedPolicy
    c=Config(potential_enabled=True);state=FixedWingState(100000.,100000.,0.)
    clouds=[{'x':102500.,'y':100000.,'radius':700.}]
    base=FixedPolicy([1.,0.,.2]);policy=ForecastConstrainedPolicy(base,c,'coverage')
    policy.act(observation(c,state,clouds=clouds));assert policy.safety.sides
    policy.reset()
    assert base.reset_count==1 and base.calls==0 and policy.safety.sides=={}
    assert policy.last_raw_action is None and policy.last_applied_action is None and not policy.last_intervened
    assert policy.execution_metadata['checked_actions']==policy.execution_metadata['interventions']==0


def test_observation_remains_unchanged_and_return_history_is_not_an_alias():
    from policy_constraints import ForecastConstrainedPolicy
    c=Config(potential_enabled=True);state=FixedWingState(100000.,100000.,0.)
    clouds=[{'x':102500.,'y':100000.,'radius':700.}]
    obs=observation(c,state,clouds=clouds);before=copy.deepcopy(obs)
    for value in obs.values():value.flags.writeable=False
    policy=ForecastConstrainedPolicy(FixedPolicy([1.,0.,.2]),c,'coverage')
    action=policy.act(obs);saved=policy.last_applied_action.copy();action[:]=0
    for key in obs:np.testing.assert_array_equal(obs[key],before[key])
    np.testing.assert_array_equal(policy.last_applied_action,saved)


def test_nondefault_physics_forecast_uses_shared_config_and_obs_consistency():
    from policy_constraints import ForecastConstrainedPolicy
    c=Config(potential_enabled=True,uav_speed=50.,uav_min_turn_radius=1500.,uav_roll_time_constant=2.,
        uav_max_roll_rate=.1,uav_heading_time_constant=1.5,expert_safety_horizon=35.,expert_safety_projection_dt=1.)
    state=FixedWingState(100000.,100000.,0.,-.1)
    clouds=[{'x':102200.,'y':100000.,'radius':500.}]
    policy=ForecastConstrainedPolicy(FixedPolicy([1.,0.,.25]),c,'coverage')
    action=policy.act(observation(c,state,clouds=clouds))
    assert policy.last_intervened
    run_physical_forecast(state,action,clouds,c)
    assert policy.execution_metadata['horizon']==35. and policy.execution_metadata['projection_dt']==1.


def test_mismatched_observation_physics_is_rejected_before_policy_call():
    from policy_constraints import ForecastConstrainedPolicy
    c=Config(potential_enabled=True);base=FixedPolicy([1.,0.,0.])
    policy=ForecastConstrainedPolicy(base,c,'coverage')
    obs=observation(Config(potential_enabled=True,uav_speed=50.),FixedWingState(100000.,100000.,0.))
    with pytest.raises(ValueError,match='Config|physics'):policy.act(obs)
    assert base.calls==0


@pytest.mark.parametrize('key',['vector','potential','field_context'])
def test_nonfinite_shared_input_is_rejected_before_recurrent_state_changes(key):
    from policy_constraints import ForecastConstrainedPolicy
    c=Config(potential_enabled=True);base=FixedPolicy([1.,0.,0.])
    policy=ForecastConstrainedPolicy(base,c,'coverage')
    obs=observation(c,FixedWingState(100000.,100000.,0.))
    obs[key].flat[0]=float('nan')
    with pytest.raises(ValueError):policy.act(obs)
    assert base.calls==0


@pytest.mark.parametrize('action',[[1.,0.],[1.1,0.,0.],[1.,0.,float('nan')]])
def test_invalid_raw_three_action_is_rejected(action):
    from policy_constraints import ForecastConstrainedPolicy
    c=Config(potential_enabled=True);policy=ForecastConstrainedPolicy(FixedPolicy(action),c,'coverage')
    with pytest.raises(ValueError):policy.act(observation(c,FixedWingState(100000.,100000.,0.)))


def test_decision_interval_cannot_exceed_forecast_horizon():
    from policy_constraints import ForecastConstrainedPolicy
    c=Config(potential_enabled=True,decision_dt=40.,expert_safety_horizon=30.)
    with pytest.raises(ValueError,match='decision_dt|horizon'):
        ForecastConstrainedPolicy(FixedPolicy([1.,0.,0.]),c,'coverage')


@pytest.mark.parametrize('state,clouds',[
    (FixedWingState(-100.,100000.,0.),()),
    (FixedWingState(100000.,100000.,0.),({'x':100000.,'y':100000.,'radius':700.},)),
])
def test_initial_hard_violation_is_never_released_by_fast_path(state,clouds):
    from policy_constraints import ForecastConstrainedPolicy,NoFeasibleAction
    c=Config(potential_enabled=True);policy=ForecastConstrainedPolicy(FixedPolicy([1.,0.,0.]),c,'coverage')
    with pytest.raises(NoFeasibleAction):policy.act(observation(c,state,clouds=clouds))
    assert policy.execution_metadata['reachable_disk_checks']==0
