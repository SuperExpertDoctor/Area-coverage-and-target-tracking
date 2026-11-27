"""Short recovery rollouts need exposure without invented temporal transitions."""
import json
from pathlib import Path
import numpy as np
import pytest
import torch


def test_training_sampler_repeats_only_complete_copies_of_recovery_prefixes(tmp_path):
    from src.train import _training_sample_order
    paths=[]
    for name,length,behavior,prefix in [('expert',8,None,False),('recovery',2,'dagger_training',True),
                                         ('full-dagger',8,'dagger_training',False)]:
        path=tmp_path/f'{name}.npz'
        np.savez(path,actions=np.zeros((length,3)),metadata=np.asarray(json.dumps(
            {'task':'coverage','behavior':behavior,'accepted_prefix':prefix})))
        paths.append(path)
    class Dataset:
        def __len__(self):return len(self.paths)
    data=Dataset();data.paths=paths
    order,report=_training_sample_order(data,8)
    assert np.bincount(order).tolist()==[1,4,1]
    assert report['unique_episodes']==3 and report['episode_draws']==6
    assert report['expected_action_steps']==24
    assert report['repeated_episodes'][0]['path']==str(paths[1])
    untouched,_=_training_sample_order(data,0)
    np.testing.assert_array_equal(untouched,[0,1,2])
    for invalid in (-1,1.5,float('nan'),True):
        with pytest.raises(ValueError):_training_sample_order(data,invalid)


def test_repeated_rollout_resets_hidden_and_keeps_contiguous_chunks():
    from src.train import _run_epoch
    class Counter(torch.nn.Module):
        def __init__(self):super().__init__();self.bias=torch.nn.Parameter(torch.zeros(()))
        def forward(self,obs,hidden=None):
            x=obs['vector'][...,:1];prior=0 if hidden is None else hidden.transpose(0,1)
            sequence=x.cumsum(1)+prior+self.bias*0
            return torch.cat((sequence,sequence*0),-1),sequence[:,-1:].transpose(0,1)
    rollout={'vector':torch.ones(3,1),'actions':torch.tensor([[1.,0.],[2.,0.],[3.,0.]])}
    model=Counter();optimizer=torch.optim.SGD(model.parameters(),lr=0.)
    result=_run_epoch(model,[rollout],1,2,'cpu',optimizer=optimizer,order=[0,0,0])
    assert result['steps']==9 and result['action_mse']==0.


def test_training_applies_budget_without_resampling_validation_or_normalization(tmp_path):
    from src.datasets import split_episodes
    from src.train import train_policy
    folder=tmp_path/'data';folder.mkdir()
    vector=np.zeros((4,134),np.float32);vector[:,2]=1
    actions=np.tile([1.,0.],(3,1)).astype(np.float32)
    for i in range(4):
        np.savez(folder/f'{i}.npz',vector=vector,actions=actions,
                 metadata=json.dumps(dict(task='tracking',scene_id=str(i),schema_version='obs-v1')))
    original=split_episodes(folder,'tracking',.25,42)
    recovery=Path(original['train_paths'][0])
    with np.load(recovery,allow_pickle=False) as arrays:
        metadata=json.loads(arrays['metadata'].item())
    metadata.update(behavior='dagger_training',accepted_prefix=True)
    np.savez(recovery,vector=vector,actions=actions,metadata=json.dumps(metadata))
    result=train_policy('tracking',folder,tmp_path/'model',tmp_path/'tf',epochs=1,batch_size=2,
                        sequence_length=2,hidden_size=8,validation_fraction=.25,seed=42,
                        dagger_recovery_min_steps=9)
    report=result['train_config']['training_sampling']
    assert report['unique_episodes']==3 and report['episode_draws']==5
    assert report['expected_action_steps']==15
    assert result['initial_train']['steps']==9
    assert result['initial_validation']['steps']==3
    assert result['history'][0]['Train']['steps']==15
    assert result['history'][0]['Validation']['steps']==3
    assert result['split']['train_paths']==original['train_paths']
    checkpoint=torch.load(result['last_checkpoint'],weights_only=False)
    assert checkpoint['normalization']['fit_subset']=='train'
    assert checkpoint['train_config']['training_sampling']==report


def test_training_only_selection_evaluates_unique_episodes_after_updates(tmp_path,monkeypatch):
    import src.train as training
    folder=tmp_path/'data';folder.mkdir()
    vector=np.zeros((4,134),np.float32);vector[:,2]=1
    np.savez(folder/'recovery.npz',vector=vector,actions=np.tile([1.,0.],(3,1)),
             metadata=json.dumps(dict(task='tracking',scene_id='a',schema_version='obs-v1',
                                      behavior='dagger_training',accepted_prefix=True)))
    original=training._run_epoch;calls=[]
    def observed(*args,**kwargs):
        metrics=original(*args,**kwargs)
        calls.append((kwargs.get('optimizer') is not None,metrics))
        return metrics
    monkeypatch.setattr(training,'_run_epoch',observed)
    result=training.train_policy('tracking',folder,tmp_path/'model',tmp_path/'tf',epochs=1,
                                batch_size=1,sequence_length=2,hidden_size=8,validation_fraction=0,
                                dagger_recovery_min_steps=9)
    evaluations=[metrics for updating,metrics in calls if not updating]
    assert len(evaluations)==2  # both candidates scored on the same unique data
    assert all(metrics['steps']==3 for metrics in evaluations)
    assert result['history'][0]['Train']['steps']==9
    assert result['best_selection_mse']==min(metrics['action_mse'] for metrics in evaluations)
    checkpoint=torch.load(result['last_checkpoint'],weights_only=False)
    assert checkpoint['selection_metrics']==evaluations[-1]
