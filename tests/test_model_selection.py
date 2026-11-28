"""Offline MSE must not promote a model that fails closed-loop validation."""
import json
from pathlib import Path
import numpy as np
import torch


def test_selection_metadata_is_loaded_from_exact_snapshot_bytes(tmp_path,monkeypatch):
    from src import model_selection
    model_dir=tmp_path/'models';model_dir.mkdir()
    paths=[]
    for name in ('train','val'):
        path=tmp_path/f'{name}.npz';np.savez(path,metadata=np.asarray(json.dumps({'scene_group_id':name})));paths.append(str(path))
    original={'task':'coverage','observation_schema':'obs-field-v2','dataset_split':{'train_paths':paths[:1],'validation_paths':paths[1:]}}
    for name in ('best','last'):torch.save(original,model_dir/f'{name}.pt')
    real_load=torch.load
    def ensure_snapshot_loaded(source,*args,**kwargs):
        assert hasattr(source,'read'),'Metadata must be parsed from frozen payload, not re-open a mutable path'
        return real_load(source,*args,**kwargs)
    monkeypatch.setattr(torch,'load',ensure_snapshot_loaded)
    monkeypatch.setattr(model_selection,'_evaluate_candidate',lambda job:{'scene_group_id':'val','path':'x',
        'summary':{'termination_reason':'time_limit','terminated':False,'coverage_ratio':1.}})
    assert model_selection.select_checkpoint('coverage',model_dir,tmp_path/'out')['accepted']


def test_selection_uses_validation_only_and_preserves_offline_best(tmp_path,monkeypatch):
    from src import model_selection
    model_dir=tmp_path/'models';model_dir.mkdir()
    paths=[]
    for name,group in [('train','train-layout'),('val','val-layout')]:
        path=tmp_path/f'{name}.npz'
        np.savez(path,metadata=np.asarray(json.dumps({'task':'tracking','scene_group_id':group})))
        paths.append(str(path))
    split={'train_paths':paths[:1],'validation_paths':paths[1:]}
    for name in ('best','last'):
        torch.save({'task':'tracking','checkpoint_version':'bc-field-v2','observation_schema':'obs-field-v2',
                    'dataset_split':split,'epoch':1 if name=='best' else 2},model_dir/f'{name}.pt')
    original=(model_dir/'best.pt').read_bytes();calls=[]
    def evaluate(job):
        calls.append(job)
        tag=job['tag']
        summary={'termination_reason':'time_limit','terminated':False,'time_seconds':1800.,
                 'visible_fraction':.8 if tag=='best' else 1.,'first_loss_time':10. if tag=='best' else None}
        return {'summary':summary,'scene_group_id':'val-layout','path':'validation.json'}
    monkeypatch.setattr(model_selection,'_evaluate_candidate',evaluate)
    result=model_selection.select_checkpoint('tracking',model_dir,tmp_path/'selection',workers=1)
    assert result['accepted']
    assert result['selected_candidate']=='last'
    assert all(job['episode_path']==paths[1] for job in calls)
    assert all(Path(job['output']).suffix=='.npz' for job in calls)
    assert all(Path(job['checkpoint']).parent==model_dir/'selection-snapshots' for job in calls)
    assert not (model_dir/'selection.json').exists()
    assert {round(job['pan_offset'],6) for job in calls}=={round(-np.pi/9,6),0.,round(np.pi/9,6)}
    assert (model_dir/'best.pt').read_bytes()==original
    assert (model_dir/'selected.pt').read_bytes()==(model_dir/'last.pt').read_bytes()


def test_failed_selection_cannot_leave_a_stale_selected_model(tmp_path,monkeypatch):
    from src import model_selection
    model_dir=tmp_path/'models';model_dir.mkdir()
    val=tmp_path/'val.npz';np.savez(val,metadata=np.asarray(json.dumps({'task':'coverage','scene_group_id':'val'})))
    for name in ('best','last'):
        torch.save({'task':'coverage','checkpoint_version':'bc-field-v2','observation_schema':'obs-field-v2',
                    'dataset_split':{'train_paths':[],'validation_paths':[str(val)]}},model_dir/f'{name}.pt')
    (model_dir/'selected.pt').write_bytes(b'stale model from previous experiment')
    monkeypatch.setattr(model_selection,'_evaluate_candidate',lambda job:{'scene_group_id':'val','path':'x',
        'summary':{'termination_reason':'time_limit','terminated':False,'coverage_ratio':.5,'mean_age':100.,'time_seconds':10800.}})
    result=model_selection.select_checkpoint('coverage',model_dir,tmp_path/'selection',workers=1)
    assert not result['accepted'] and result['selected_checkpoint'] is None
    assert not (model_dir/'selected.pt').exists()


def test_tracking_validation_pan_stress_changes_only_feasible_pan():
    from src.model_selection import _tracking_pan_stress
    from config import Config
    from scenario import sample_tracking_scene, validate_scene
    cfg=Config(potential_enabled=True)
    scene=sample_tracking_scene(cfg,np.random.default_rng(22004),seed=22004)
    original=json.loads(json.dumps(scene))
    stressed,details=_tracking_pan_stress(scene,cfg,np.pi/9)
    validate_scene(stressed,cfg,'tracking')
    assert scene==original
    assert stressed['radar_pan_initial']!=scene['radar_pan_initial']
    assert {k:v for k,v in stressed.items() if k!='radar_pan_initial'}=={k:v for k,v in original.items() if k!='radar_pan_initial'}
    assert details['requested_offset_radians']==np.pi/9


def test_validation_worker_replays_npz_source_and_writes_only_npz(tmp_path):
    from src.model_selection import _evaluate_candidate
    from config import Config
    from env_tracking import TrackingEnv
    from expert_field import FieldTrackingExpert
    from rollouts import run_episode, load_replay
    from src.train import create_policy
    cfg=Config(potential_enabled=True,tracking_episode_seconds=2.)
    record,source=run_episode(TrackingEnv(cfg),FieldTrackingExpert(cfg),seed=22000,
        actor_name='expert',output_dir=tmp_path/'datasets',name='source',
        record_observations=True,max_steps=1)
    model=create_policy('tracking',{'potential_field':True,'field_frame':'body','pan_residual':True})
    checkpoint=tmp_path/'candidate.pt'
    torch.save(dict(checkpoint_version='bc-field-v2',task='tracking',observation_schema='obs-field-v2',
        action_schema='action-field-v2',model_config=model.model_config,model_state_dict=model.state_dict()),checkpoint)
    target=tmp_path/'eval'/'candidate.npz'
    result=_evaluate_candidate({'task':'tracking','checkpoint':str(checkpoint),'episode_path':str(source),
        'scene_group_id':record['metadata']['scene_group_id'],'duration':2.,'pan_offset':0.,
        'num_threads':1,'render_every':1,'output':str(target)})
    assert result['path']==str(target.resolve()) and target.is_file()
    assert not target.with_suffix('.json').exists() and not source.with_suffix('.json').exists()
    replay=load_replay(target)
    assert replay['summary']['steps']==2 and len(replay['actions'])==2
    assert replay['metadata']['selection_subset']=='validation'
