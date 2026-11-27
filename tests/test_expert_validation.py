import copy
import json
import math

import numpy as np


def test_validation_plan_is_fixed_before_case_results():
    from expert_validation import make_case_plan
    jobs = make_case_plan([2000, 3000], [4000], 8, [5000])
    assert len(jobs) == 11
    assert [j['seed'] for j in jobs[:2]] == [2000, 3000]
    starts = [j for j in jobs if j['kind'] == 'boundary']
    assert [j['start_index'] for j in starts] == list(range(8))
    assert len({j['case_id'] for j in jobs}) == len(jobs)


def test_boundary_starts_are_uniform_in_perimeter_distance_and_clockwise():
    from expert_validation import boundary_start
    expected = [(0,0),(0,75),(0,150),(25,200),(100,200),(100,125),(100,50),(75,0)]
    for index, point in enumerate(expected):
        state = boundary_start([0,0,100,200], index, 8)
        assert (state.x, state.y) == point
        assert state.bank == 0
        right = np.array([math.sin(state.heading), -math.cos(state.heading)])
        center = np.array([50-state.x,100-state.y])
        assert right @ center > 0


def test_algorithm_failure_is_saved_in_full_and_never_excluded(tmp_path):
    from config import Config
    from env_coverage import CoverageEnv
    from expert_validation import execute_case
    config = Config(coverage_episode_seconds=2)
    env = CoverageEnv(config)
    env.reset(seed=4)
    scene = copy.deepcopy(env.scene)
    scene['uav_initial']['heading'] = 0.
    initial = scene['uav_initial']
    scene['clouds'] = [{'id':0, 'x':initial['x']+450., 'y':initial['y'], 'radius':200.}]
    job = {'case_id':'coverage-random-test', 'task':'coverage', 'kind':'random', 'seed':4, 'scene':scene}
    result = execute_case(job, config.to_dict(), str(tmp_path), 1., 1.)
    assert result['status'] == 'expert_planning_failure'
    assert result['passed'] is False
    saved = json.loads((tmp_path/result['replay']).read_text(encoding='utf-8'))
    assert saved['scene'] == scene
    assert len(saved['frames']) == len(saved['actions'])+1
    assert saved['summary']['planning_failure']['type'] == 'ExpertPlanningError'


def test_boundary_hard_collision_is_excluded_with_explicit_reason(tmp_path):
    from config import Config
    from env_coverage import CoverageEnv
    from expert_validation import execute_case, boundary_start
    config = Config(coverage_episode_seconds=2)
    env = CoverageEnv(config)
    env.reset(seed=4)
    scene = copy.deepcopy(env.scene)
    start = boundary_start(scene['roi'], 0, 8)
    scene['clouds'] = [{'id':0, 'x':start.x, 'y':start.y, 'radius':500.}]
    job = {'case_id':'coverage-boundary-test', 'task':'coverage', 'kind':'boundary', 'seed':4,
           'start_index':0,'start_count':8,'scene':scene}
    result = execute_case(job, config.to_dict(), str(tmp_path), 1., 1.)
    assert result['status'] == 'excluded_initial_state'
    assert result['exclusion']['reason'] == 'initial_uav_cloud_collision'
    assert result['exclusion']['cloud_id'] == 0


def test_cache_identity_includes_physics_and_source_but_ignores_visuals():
    from expert_validation import _validation_id
    base = {'uav_speed':61.1,'visual_uav_size':10}
    assert _validation_id('source-a',base) == _validation_id('source-a',{**base,'visual_uav_size':20})
    assert _validation_id('source-a',base) != _validation_id('source-b',base)
    assert _validation_id('source-a',base) != _validation_id('source-a',{**base,'uav_speed':70})
