"""Public rollout protocol must retain explicit radar commands and actual pan."""
from dataclasses import dataclass
import numpy as np
import pytest
from gymnasium import spaces

from rollouts import run_episode, restore_training_cache
from episode_io import load_episode


def test_three_command_rollout_and_dagger_roundtrip(tmp_path):
    @dataclass
    class C:
        decision_dt: float = 1.
        potential_enabled: bool = True
        pan_limit: float = 2.

    class E:
        task = 'tracking'
        config = C()
        action_space = spaces.Box(-1., 1., (3,), dtype=np.float32)
        observation_schema = {'schema_version':'obs-field-v2'}
        scene = {'world_bounds':[0,0,300000,300000], 'roi':None,
                 'clouds':[], 'islands':[], 'ship_route':[]}
        radar_pan = .25
        def reset(self, **kwargs):
            self.t = 0
            return {'vector':np.ones(134,np.float32)}, {}
        def step(self, action):
            np.testing.assert_array_equal(action, [1.,0.,.5])
            self.t += 1
            return {'vector':np.ones(134,np.float32)}, 0., False, True, {}
        def render(self):
            return {'t':self.t, 'uav':{'vx':1.,'vy':0.}, 'radar':{'pan':self.radar_pan}}
        def summary(self):
            return {'time_seconds':self.t, 'visible_fraction':1.}

    class A:
        def reset(self): pass
        def act(self, obs): return np.array([1.,0.,.5],np.float32)

    record, path = run_episode(E(), A(), seed=1, actor_name='policy',
        expert_labeler=A(), mixture_beta=0., output_dir=tmp_path, name='field',
        record_observations=True)
    assert record['metadata']['obs_schema'] == 'obs-field-v2'
    assert record['metadata']['action_schema'] == 'action-field-v2'
    assert record['executed_actions'] == [[1.,0.,.125]]
    cache = restore_training_cache(path)
    with np.load(cache) as data:
        np.testing.assert_array_equal(data['actions'], [[1.,0.,.5]])
        np.testing.assert_array_equal(data['executed_actions'], [[1.,0.,.125]])
    saved = load_episode(path)
    assert saved['expert_actions'] == saved['actions']


def test_cli_field_warm_start_rejects_legacy_before_training(tmp_path,monkeypatch):
    import main
    import torch
    import src.train as trainer
    monkeypatch.setattr(torch,'load',lambda *a,**k:{'model_config':{},'observation_schema':'obs-v1'})
    def unexpected(*args,**kwargs): pytest.fail('mismatched checkpoint must not reach trainer')
    monkeypatch.setattr(trainer,'train_policy',unexpected)
    checkpoint=tmp_path/'old.pt';checkpoint.write_bytes(b'fixture')
    with pytest.raises(ValueError,match='protocol'):
        main.main(['--mode','train','--task','tracking','--potential-field','--warm-start',
            '--checkpoint-path',str(checkpoint),'--output-dir',str(tmp_path),
            '--tf-log-dir',str(tmp_path/'tf-logs'),'--device','cpu','--no-visualize','--no-save-video'])


def test_cli_default_field_output_preserves_legacy_artifacts(tmp_path,monkeypatch):
    import main
    import expert_validation
    monkeypatch.setattr(main,'PROJECT_ROOT',tmp_path)
    legacy=tmp_path/'outputs'/'models'/'tracking'/'best.pt'
    legacy.parent.mkdir(parents=True);legacy.write_bytes(b'preserve legacy checkpoint')
    calls=[]
    def validate(config,**kwargs):
        calls.append(kwargs['output_dir'])
        return {'all_eligible_passed':True,'results':[]}
    monkeypatch.setattr(expert_validation,'run_validation',validate)
    main.main(['--mode','expert-validate','--task','tracking','--potential-field',
        '--tf-log-dir',str(tmp_path/'tf-logs'),'--no-visualize','--no-save-video'])
    assert calls==[tmp_path/'outputs'/'field-v2'/'logs'/'diagnostics'/'expert-randomized']
    assert legacy.read_bytes()==b'preserve legacy checkpoint'
