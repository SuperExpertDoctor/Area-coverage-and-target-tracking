"""Data roles, latest-weight provenance and task success remain explicit."""
import hashlib
import json
from pathlib import Path

import pytest
import torch


def test_task_success_requires_full_duration_and_physical_safety():
    from src.formal_run import task_success
    good=dict(time_seconds=10800.,terminated=False,coverage_ratio=1.,uav_collision=False,
              ship_collision=False,out_of_bounds=False)
    assert task_success('coverage',good,10800.)
    for changes in ({'coverage_ratio':.999},{'terminated':True},{'time_seconds':10799.},
                    {'uav_collision':True},{'out_of_bounds':True}):
        assert not task_success('coverage',dict(good,**changes),10800.)
    track=dict(time_seconds=1800.,terminated=False,visible_fraction=1.,first_loss_time=None)
    assert task_success('tracking',track,1800.)
    assert not task_success('tracking',dict(track,first_loss_time=900.),1800.)
    assert not task_success('tracking',dict(track,visible_fraction=.999),1800.)


def test_freeze_latest_uses_last_bytes_and_never_selects_offline_best(tmp_path):
    from src.formal_run import freeze_latest
    models=tmp_path/'models';models.mkdir()
    torch.save(dict(task='coverage',epoch=3,checkpoint_version='bc-field-v2'),models/'last.pt')
    torch.save(dict(task='coverage',epoch=1,checkpoint_version='bc-field-v2'),models/'best.pt')
    frozen=freeze_latest('coverage',models,tmp_path/'frozen')
    assert frozen['epoch']==3
    assert frozen['sha256']==hashlib.sha256((models/'last.pt').read_bytes()).hexdigest()
    assert Path(frozen['path']).read_bytes()==(models/'last.pt').read_bytes()
    (models/'last.pt').unlink()
    with pytest.raises(FileNotFoundError,match='last.pt'):
        freeze_latest('coverage',models,tmp_path/'other')


def test_formal_protocol_declares_fresh_seeds_and_modes_before_outcomes():
    from main import parser
    from config import Config
    from src.formal_run import formal_protocol
    args=parser().parse_args(['--mode','formal','--episodes','1000','--validation-fraction','.2'])
    protocol=formal_protocol(args,Config(potential_enabled=True),['coverage','tracking'])
    assert protocol['episodes_per_task']==1000
    assert protocol['validation_fraction']==.2
    assert protocol['latest_checkpoint_name']=='last.pt'
    assert protocol['tasks']['coverage']['modes']==['none','forecast']
    assert protocol['tasks']['tracking']['modes']==['none']
    assert protocol['tasks']['coverage']['fresh_seeds']==list(range(940000,940008))
    assert protocol['tasks']['tracking']['fresh_seeds']==list(range(950000,950008))
    assert protocol['test_outcomes_used_for_weight_selection'] is False


def test_formal_status_atomic_updates_keep_other_task_state(tmp_path):
    from src.formal_run import FormalStatus
    status=FormalStatus(tmp_path,'test',['coverage','tracking'])
    status.task_update('coverage','collection',{'accepted':12,'target':1000})
    status.task_update('tracking','training',{'epoch':2,'epochs':20,'validation_loss':.1})
    value=json.loads((tmp_path/'formal-status.json').read_text(encoding='utf-8'))
    assert value['tasks']['coverage']['collection']['accepted']==12
    assert value['tasks']['tracking']['training']['epoch']==2
    assert not list(tmp_path.glob('*.tmp'))


def test_formal_protocol_rejects_collection_and_test_overlap():
    from main import parser
    from config import Config
    from src.formal_run import formal_protocol
    args=parser().parse_args(['--mode','formal','--formal-test-seed','1000'])
    with pytest.raises(ValueError,match='overlap'):
        formal_protocol(args,Config(potential_enabled=True),['coverage'])


def test_failed_independent_case_retains_error_and_allows_other_cases(tmp_path,monkeypatch):
    from src import formal_run
    def broken(job):
        raise ValueError('infeasible initialization')
    monkeypatch.setattr(formal_run,'_rollout',broken)
    result=formal_run._test_rollout(dict(task='coverage',seed=11,mode='none',
                                       output_dir=str(tmp_path),name='failed'))
    assert result['success'] is False
    assert result['summary']['error']['message']=='infeasible initialization'
    assert json.loads(Path(result['path']).read_text(encoding='utf-8'))==result
