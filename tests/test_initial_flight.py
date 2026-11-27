"""Point clearance alone does not imply a fixed-wing aircraft can escape."""
import copy
import math
import numpy as np
import pytest


def test_initial_flight_certificate_rejects_inescapable_cloud_but_accepts_departure():
    from config import Config
    from dynamics import FixedWingState
    from scenario import initial_flight_is_provably_unviable
    c=Config(potential_enabled=True)
    state=FixedWingState(100000.,100000.,0.)
    cloud=dict(id=0,x=101023.657,y=99873.321,radius=751.110)
    assert math.hypot(state.x-cloud['x'],state.y-cloud['y'])>cloud['radius']+c.uav_cloud_margin
    assert initial_flight_is_provably_unviable(state,[cloud],c)
    away=FixedWingState(state.x,state.y,math.pi)
    assert not initial_flight_is_provably_unviable(away,[cloud],c)
    assert not initial_flight_is_provably_unviable(state,[],c)
    # Intersecting the enclosing reachable set is not a proof of unviability.
    farther=dict(cloud,x=103000.)
    assert not initial_flight_is_provably_unviable(state,[farther],c)


def test_supplied_scene_rejects_infeasible_start_and_seed_resampling_is_reproducible():
    from config import Config
    from scenario import sample_coverage_scene,validate_scene,SceneValidationError
    c=Config(potential_enabled=True)
    a=sample_coverage_scene(c,np.random.default_rng(13018),13018)
    b=sample_coverage_scene(c,np.random.default_rng(13018),13018)
    assert a==b
    assert any('initial flight' in reason for reason in a['sampling_rejections'])
    from scenario import SceneSpec
    assert SceneSpec.from_dict(a).to_dict()==a
    bad=copy.deepcopy(a);u=bad['uav_initial']
    u.update(x=151764.445,y=256642.793,heading=0.,bank=0.)
    bad['clouds']=[dict(id=0,x=u['x']+1023.657,y=u['y']-126.679,radius=751.110)]
    with pytest.raises(SceneValidationError,match='initial flight'):
        validate_scene(bad,c,'coverage')


def test_initial_flight_respects_world_boundaries_and_uses_no_island_restriction():
    from config import Config
    from dynamics import FixedWingState
    from scenario import initial_flight_is_provably_unviable
    c=Config()
    assert initial_flight_is_provably_unviable(FixedWingState(10.,150000.,math.pi),[],c)
    assert not initial_flight_is_provably_unviable(FixedWingState(10.,150000.,0.),[],c)
