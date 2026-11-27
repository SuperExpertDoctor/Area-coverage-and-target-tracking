"""Round-trip the canonical JSON data, rather than testing its implementation."""
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


def test_json_training_arrays_round_trip(tmp_path):
    module = importlib.import_module('rollouts')
    arrays = {'vector': np.arange(20, dtype=np.float32).reshape(4, 5),
              'actions': np.ones((3, 2), dtype=np.float32)}
    packed = {key: module.encode_array(value) for key, value in arrays.items()}
    wire = json.loads(json.dumps(packed, allow_nan=False))
    for key, expected in arrays.items():
        actual = module.decode_array(wire[key])
        np.testing.assert_array_equal(actual, expected)
        assert actual.dtype == expected.dtype


def test_array_payload_validation():
    module = importlib.import_module('rollouts')
    import pytest
    with pytest.raises(ValueError):
        module.decode_array({'encoding': 'pretend', 'shape': [2], 'dtype': '<f4', 'data': ''})


def test_failed_rerun_cannot_leave_a_stale_training_cache(tmp_path):
    module = importlib.import_module('rollouts')
    from dataclasses import dataclass
    @dataclass
    class Configuration:
        decision_dt: float = 1.
    class PlanningError(Exception): pass
    class Actor:
        def reset(self): pass
        def act(self, obs): raise PlanningError('no safe candidate')
    class Env:
        task = 'tracking'
        config = Configuration()
        scene = {'world_bounds':[0,0,300000,300000],'roi':None,
                 'clouds':[],'islands':[],'ship_route':[]}
        def reset(self, **kwargs): return {'vector':np.ones(134,np.float32)},{}
        def render(self): return {'t':0,'uav':{'vx':1,'vy':0}}
        def summary(self): return {'time_seconds':0}
    cache=tmp_path/'repeat.npz'
    np.savez(cache,actions=np.ones((2,2)))
    record,path=module.run_episode(Env(),Actor(),seed=1,actor_name='expert',
        output_dir=tmp_path,name='repeat',record_observations=True)
    assert not cache.exists()
    assert record['summary']['terminated']
    assert record['summary']['planning_failure']['type']=='PlanningError'
    assert json.loads(path.read_text(encoding='utf-8'))['metadata']['expert_valid'] is False


def test_reset_failure_also_invalidates_the_old_training_cache(tmp_path):
    module=importlib.import_module('rollouts')
    import pytest
    class Env:
        def reset(self, **kwargs): raise ValueError('invalid replay scene')
    cache=tmp_path/'repeat.npz'
    np.savez(cache,actions=np.ones((2,2)))
    with pytest.raises(ValueError,match='invalid replay scene'):
        module.run_episode(Env(),None,seed=1,actor_name='expert',output_dir=tmp_path,
                           name='repeat',record_observations=True)
    assert not cache.exists()


def test_dagger_cache_restores_expert_labels_not_behavior_actions(tmp_path):
    module=importlib.import_module('rollouts')
    record={'observations':{'vector':module.encode_array(np.ones((3,134),np.float32))},
            'actions':[[1,0],[1,0]],'expert_actions':[[0,1],[0,1]],
            'executed_actions':[[.8,.6],[.6,.8]],'metadata':{'task':'tracking'}}
    path=tmp_path/'dagger.json';module.write_json(path,record)
    result=module.restore_training_cache(path)
    with np.load(result) as data:
        np.testing.assert_array_equal(data['actions'],np.array(record['expert_actions']))


def test_live_dagger_scene_identifies_training_mixture(tmp_path):
    module = importlib.import_module('rollouts')
    from dataclasses import dataclass

    @dataclass
    class Configuration:
        decision_dt: float = 1.

    class Actor:
        checkpoint_metadata = {'path':'archived.pt','sha256':'fixture-hash','epoch':3,'model_config':{}}
        def reset(self): pass
        def act(self, obs): return np.array([1., 0.], dtype=np.float32)

    class Env:
        task = 'tracking'
        config = Configuration()
        scene = {'world_bounds': [0, 0, 300000, 300000], 'roi': None,
                 'clouds': [], 'islands': [], 'ship_route': []}
        def reset(self, **kwargs):
            self.t = 0
            return {'vector': np.ones(134, np.float32)}, {}
        def step(self, action):
            self.t += 1
            return {'vector': np.ones(134, np.float32)}, 0., False, True, {}
        def render(self): return {'t': self.t, 'uav': {'vx': 1., 'vy': 0.}}
        def summary(self): return {'time_seconds': self.t}

    class Server:
        def __init__(self): self.events = []
        def publish(self, payload): self.events.append(payload)

    server = Server()
    record, _ = module.run_episode(Env(), Actor(), seed=7, actor_name='policy',
        output_dir=tmp_path, name='dagger', server=server, expert_labeler=Actor(),
        mixture_beta=.8, mixture_seed=24000)
    event = next(event for event in server.events if event['type'] == 'scene')
    expected = {'behavior': 'dagger_training', 'label_source': 'expert',
                'mixture_beta': .8, 'mixture_seed': 24000}
    assert event.get('metadata') == expected
    assert all(record['metadata'][key] == value for key, value in expected.items())
    assert record['metadata']['policy_checkpoint'] == Actor.checkpoint_metadata
    saved=json.loads((tmp_path/'dagger.json').read_text(encoding='utf-8'))
    assert saved['metadata']['policy_checkpoint'] == Actor.checkpoint_metadata

    deployment = Server()
    module.run_episode(Env(), Actor(), seed=7, actor_name='policy',
        output_dir=tmp_path, name='deployment', server=deployment)
    assert deployment.events[0].get('metadata', {}).get('behavior') != 'dagger_training'


def test_constraint_rollout_keeps_raw_admitted_and_actual_commands_distinct(tmp_path):
    module = importlib.import_module('rollouts')
    from dataclasses import dataclass

    @dataclass
    class Configuration:
        potential_enabled: bool = True
        coverage_pan_limit: float = 1.

    class Actor:
        def __init__(self): self.count = 0
        def reset(self): self.count = 0
        @property
        def execution_metadata(self):
            return {'mode': 'forecast_constraint', 'checked_actions': self.count,
                    'interventions': self.count, 'heuristic_constraint': True,
                    'task_expert_fallback': False}
        def act(self, obs):
            self.count += 1
            self.last_raw_action = np.array([1., 0., .2], np.float32)
            self.last_intervened = True
            return np.array([0., 1., .2], np.float32)

    class Env:
        task = 'coverage'
        config = Configuration()
        scene = {'world_bounds': [0, 0, 300000, 300000], 'roi': None,
                 'clouds': [], 'islands': [], 'ship_route': []}
        def reset(self, **kwargs): self.t = 0; return {'vector': np.zeros(111, np.float32)}, {}
        def step(self, action):
            np.testing.assert_array_equal(action, [0., 1., np.float32(.2)])
            self.t += 1
            return {'vector': np.zeros(111, np.float32)}, 0., False, True, {}
        def render(self):
            return {'t': self.t, 'uav': {'vx': .6, 'vy': .8}, 'radar': {'pan': .1}}
        def summary(self): return {'time_seconds': self.t}

    class Server:
        def __init__(self): self.events = []
        def publish(self, payload): self.events.append(payload)

    server = Server()
    record, path = module.run_episode(Env(), Actor(), seed=1, actor_name='policy',
        output_dir=tmp_path, name='constraints', server=server)
    assert record['metadata']['control_mode'] == 'forecast_constraint'
    assert record['metadata']['policy_constraints']['interventions'] == 1
    assert record['summary']['policy_constraints'] == record['metadata']['policy_constraints']
    np.testing.assert_array_equal(record['raw_policy_actions'], [[1., 0., np.float32(.2)]])
    np.testing.assert_array_equal(record['actions'], [[0., 1., np.float32(.2)]])
    np.testing.assert_allclose(record['executed_actions'], [[.6, .8, .1]])
    assert record['constraint_interventions'] == [True]
    assert record['metadata']['expert_valid'] is False
    assert not path.with_suffix('.npz').exists()
    scene = server.events[0]
    assert scene['metadata']['control_mode'] == 'forecast_constraint'
    assert scene['metadata']['policy_constraints']['interventions'] == 0  # immutable initial snapshot
    saved = json.loads(path.read_text(encoding='utf-8'))
    assert saved['raw_policy_actions'] == record['raw_policy_actions']
    index = json.loads(path.with_suffix('.replay-index.json').read_text(encoding='utf-8'))
    assert index['entry']['control_mode'] == 'forecast_constraint'


def test_constraint_layer_cannot_silently_enter_dagger_or_expert_collection(tmp_path):
    module = importlib.import_module('rollouts')
    class Actor:
        execution_metadata = {'mode': 'forecast_constraint'}
    class Env:
        def reset(self, **kwargs): pytest.fail('reject ambiguous behavior before reset')
    for kwargs in ({'actor_name': 'expert'}, {'actor_name': 'policy', 'expert_labeler': Actor()}):
        with pytest.raises(ValueError, match='constraint'):
            module.run_episode(Env(), Actor(), seed=1, output_dir=tmp_path, name='bad', **kwargs)


@pytest.mark.parametrize('failure,dagger,accepted', [
    ('collision', True, True),
    ('tracking_loss', True, True),
    ('teacher_planning', True, True),
    ('teacher_planning_empty', True, False),
    ('collision', False, False),
    ('tracking_loss', False, False),
    ('teacher_planning', False, False),
])
def test_only_dagger_keeps_valid_labels_before_a_failed_behavior(tmp_path, failure, dagger, accepted):
    module = importlib.import_module('rollouts')
    from dataclasses import dataclass
    from src.datasets import SequenceDataset

    @dataclass
    class Configuration:
        decision_dt: float = 1.

    class ExpertPlanningError(Exception): pass

    class Student:
        def reset(self): pass
        def act(self, obs): return np.array([1., 0.], dtype=np.float32)

    class Teacher:
        def reset(self): pass
        def act(self, obs):
            if failure == 'teacher_planning_empty' or (
                    failure == 'teacher_planning' and obs['vector'][0] >= 2):
                raise ExpertPlanningError('no feasible direction at this observation')
            return np.array([0., 1.], dtype=np.float32)

    class Env:
        task = 'tracking'
        config = Configuration()
        scene = {'world_bounds': [0, 0, 300000, 300000], 'roi': None,
                 'clouds': [], 'islands': [], 'ship_route': []}
        def observation(self):
            vector = np.ones(134, np.float32)
            vector[0] = self.t
            return {'vector': vector}
        def reset(self, **kwargs):
            self.t = 0
            return self.observation(), {}
        def step(self, action):
            self.t += 1
            collision = failure == 'collision' and self.t == 2
            done = self.t == (2 if failure == 'tracking_loss' else 3)
            return self.observation(), 0., collision, done and not collision, {}
        def render(self): return {'t': self.t, 'uav': {'vx': 1., 'vy': 0.}}
        def summary(self):
            return {'time_seconds': self.t,
                    'termination_reason': 'uav_cloud_collision' if failure == 'collision' else None,
                    'uav_collision': failure == 'collision',
                    'first_loss_time': 1. if failure == 'tracking_loss' else None}

    record, path = module.run_episode(Env(), Student() if dagger else Teacher(), seed=7,
        actor_name='policy' if dagger else 'expert', output_dir=tmp_path, name='prefix',
        record_observations=True, expert_labeler=Teacher() if dagger else None, mixture_beta=0.)
    assert record['metadata']['expert_valid'] is accepted
    assert path.with_suffix('.npz').exists() is accepted
    if failure == 'tracking_loss':
        assert record['summary']['first_loss_time'] == 1.
    else:
        assert record['summary']['terminated']
    if failure.startswith('teacher_planning'):
        assert record['summary']['planning_failure']['type'] == 'ExpertPlanningError'
    if dagger:
        assert record['metadata']['behavior_failed'] is True
        assert record['metadata']['label_valid'] is accepted
        assert record['metadata']['accepted_prefix'] is accepted
    if accepted:
        episode = SequenceDataset([path.with_suffix('.npz')], task='tracking')[0]
        np.testing.assert_array_equal(episode['vector'][:, 0].numpy(), [0., 1.])
        np.testing.assert_array_equal(episode['actions'].numpy(), [[0., 1.], [0., 1.]])
        np.testing.assert_array_equal(record['actions'], [[1., 0.], [1., 0.]])
        restored = module.restore_training_cache(path, tmp_path / 'restored.npz')
        with np.load(path.with_suffix('.npz'), allow_pickle=False) as before, np.load(restored, allow_pickle=False) as after:
            assert before.files == after.files
            assert all(np.array_equal(before[key], after[key]) for key in before.files)


class _SparseEnv:
    from dataclasses import make_dataclass
    task = 'coverage'
    config = make_dataclass('SparseConfig', [('decision_dt', float, 1.),
        ('potential_enabled', bool, True), ('coverage_pan_limit', float, 1.)])()
    scene = {'world_bounds': [0, 0, 300000, 300000], 'roi': None,
             'clouds': [], 'islands': [], 'ship_route': []}

    def __init__(self):
        self.render_calls = 0
        self.uav_model = self
        self.uav = self

    def velocity(self, state): return (.6, .8)
    def observation(self): return {'vector': np.full(111, self.t, np.float32)}
    def reset(self, **kwargs): self.t = 0; self.radar_pan = 0.; return self.observation(), {}
    def step(self, action):
        self.t += 1
        self.radar_pan = self.t / 10
        return self.observation(), 0., False, self.t == 7, {'time_seconds': self.t}
    def render(self):
        self.render_calls += 1
        return {'t': self.t, 'uav': {'vx': .6, 'vy': .8},
                'radar': {'pan': self.radar_pan}}
    def summary(self): return {'time_seconds': self.t}


class _SparseActor:
    def reset(self): pass
    def act(self, obs): return np.array([1., 0., obs['vector'][0] / 10], np.float32)


def test_sparse_replay_preserves_all_training_labels_and_actual_actions(tmp_path):
    module = importlib.import_module('rollouts')
    full_env, sparse_env = _SparseEnv(), _SparseEnv()
    full, _ = module.run_episode(full_env, _SparseActor(), seed=7, actor_name='expert',
        output_dir=tmp_path, name='full', record_observations=True)
    sparse, sparse_path = module.run_episode(sparse_env, _SparseActor(), seed=7, actor_name='expert',
        output_dir=tmp_path, name='sparse', record_observations=True, replay_every=3)
    assert sparse['frame_action_indices'] == [0, 3, 6, 7]
    assert [frame['t'] for frame in sparse['frames']] == [0, 3, 6, 7]
    assert sparse_env.render_calls == 4
    assert full_env.render_calls == 8
    assert sparse['actions'] == full['actions']
    assert sparse['executed_actions'] == full['executed_actions']
    assert sparse['observations'] == full['observations']
    assert len(sparse['actions']) == 7
    assert module.decode_array(sparse['observations']['vector']).shape == (8, 111)
    restored = module.restore_training_cache(sparse_path, tmp_path / 'restored.npz')
    with np.load(sparse_path.with_suffix('.npz')) as before, np.load(restored) as after:
        for key in before.files:
            np.testing.assert_array_equal(before[key], after[key])


def test_sparse_replay_keeps_last_state_on_off_cadence_planning_failure(tmp_path):
    module = importlib.import_module('rollouts')
    class PlanningError(Exception): pass
    class Actor(_SparseActor):
        def act(self, obs):
            if obs['vector'][0] == 2: raise PlanningError('no continuation')
            return super().act(obs)
    record, _ = module.run_episode(_SparseEnv(), Actor(), seed=1, actor_name='expert',
        output_dir=tmp_path, name='failed-sparse', replay_every=5, record_observations=True)
    assert record['frame_action_indices'] == [0, 2]
    assert record['frames'][-1]['t'] == 2
    assert record['summary']['planning_failure']['t'] == 2
    assert record['metadata']['expert_valid'] is False


@pytest.mark.parametrize('cadence', [1.5, True])
def test_replay_cadence_requires_a_positive_integer(tmp_path, cadence):
    module = importlib.import_module('rollouts')
    with pytest.raises(ValueError, match='cadence'):
        module.run_episode(_SparseEnv(), _SparseActor(), seed=1, actor_name='expert',
            output_dir=tmp_path, name='invalid-cadence', replay_every=cadence)
