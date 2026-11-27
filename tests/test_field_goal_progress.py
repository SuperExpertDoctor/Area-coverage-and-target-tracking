"""Goal timeouts must permit real detours yet expire without material progress."""
from dataclasses import replace
import math
import numpy as np

from config import Config
from env_coverage import CoverageEnv


def test_new_closer_field_cell_extends_deadline_before_forced_reversal():
    env=CoverageEnv(Config(potential_enabled=True,cloud_min_count=0,cloud_max_count=0,
                           island_min_count=0,island_max_count=0))
    env.reset(seed=3002)
    goal=env.public_goal.copy();selected=env._goal_selected_tick
    original_deadline=env._goal_deadline
    position=np.array([env.uav.x,env.uav.y]);direction=(goal-position)/np.linalg.norm(goal-position)
    position+=direction*env.config.potential_radius/4
    env.uav=replace(env.uav,x=position[0],y=position[1])
    env.physics_tick=int(math.ceil((original_deadline+1)/env.config.physics_dt))
    env._update_public_goal()
    np.testing.assert_array_equal(env.public_goal,goal)
    assert env._goal_selected_tick==selected
    assert env._goal_deadline>env.t+env.config.potential_goal_hold_seconds


def test_subcell_progress_cannot_extend_timeout_forever():
    env=CoverageEnv(Config(potential_enabled=True,cloud_min_count=0,cloud_max_count=0,
                           island_min_count=0,island_max_count=0))
    env.reset(seed=3002)
    goal=env.public_goal.copy();p=np.array([env.uav.x,env.uav.y]);d=(goal-p)/np.linalg.norm(goal-p)
    p+=d*env.config.potential_radius/4
    env.uav=replace(env.uav,x=p[0],y=p[1]);env.physics_tick=100
    env._update_public_goal();deadline=env._goal_deadline
    for i in range(1,10):
        env.uav=replace(env.uav,x=p[0]+d[0]*i*.001,y=p[1]+d[1]*i*.001)
        env.physics_tick+=1;env._update_public_goal()
        assert env._goal_deadline==deadline
    env.physics_tick=int(math.ceil((deadline+1)/env.config.physics_dt))
    env._update_public_goal()
    assert env._goal_selected_tick==env.physics_tick


def test_revisit_goal_distinguishes_ages_beyond_field_expiry():
    env=CoverageEnv(Config(potential_enabled=True,cloud_min_count=0,cloud_max_count=0,
                           island_min_count=0,island_max_count=0))
    env.reset(seed=3002)
    c=env.config;env.physics_tick=int(20*c.potential_period/c.physics_dt)
    distance=np.hypot(env.grid_x-env.uav.x,env.grid_y-env.uav.y)
    valid=np.flatnonzero(env.sea_mask)
    closest=valid[np.argmin(distance.ravel()[valid])];farthest=valid[np.argmax(distance.ravel()[valid])]
    env.sea_mask[:]=False;env.sea_mask.ravel()[[closest,farthest]]=True
    env.last_seen_tick[:]=env.physics_tick-int(1.1*c.potential_period/c.physics_dt)
    env.last_seen_tick.ravel()[closest]=env.physics_tick-int(10*c.potential_period/c.physics_dt)
    env.public_goal=None;env._update_public_goal()
    np.testing.assert_array_equal(env.public_goal,[env.grid_x.ravel()[closest],env.grid_y.ravel()[closest]])
    assert env.public_goal_priority==1./(1+c.potential_unseen_bonus)
