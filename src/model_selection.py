"""Choose field deployment weights on validation layouts, never test layouts."""
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
import numpy as np
import torch
from .datasets import read_episode_metadata


def _tracking_pan_stress(scene,config,offset):
    """Deterministic validation perturbation; layout and dynamics stay fixed."""
    from geometry import wrap
    from scenario import validate_scene
    result=deepcopy(scene);u,s=result['uav_initial'],result['ship_initial']
    center=wrap(math.atan2(s['y']-u['y'],s['x']-u['x'])-u['heading'])
    guard=config.tracking_half_angle-config.tracking_angle_guard
    lower=max(-config.pan_limit,center-guard);upper=min(config.pan_limit,center+guard)
    if lower>upper:raise ValueError('No feasible initial tracking pan interval')
    result['radar_pan_initial']=float(np.clip(center+offset,lower,upper))
    validate_scene(result,config,'tracking')
    return result,{'requested_offset_radians':float(offset),'actual_offset_radians':wrap(result['radar_pan_initial']-center)}


def _evaluate_candidate(job):
    from config import Config
    from env_coverage import CoverageEnv
    from env_tracking import TrackingEnv
    from rollouts import json_value, write_episode, write_replay_index, load_replay
    from .policy_tracking import load_policy
    torch.set_num_threads(job['num_threads'])
    source=load_replay(job['episode_path'])
    scene=deepcopy(source['scene']);task=job['task'];duration=job['duration']
    values=scene['config_snapshot'];values[f'{task}_episode_seconds']=duration
    config=Config.from_dict(values);scene['episode_seconds']=duration;scene['config_snapshot']=config.to_dict()
    if not config.potential_enabled or scene['task']!=task:raise ValueError('Selection requires matching field-v2 scenes')
    layout={key:scene.get(key) for key in ('task','world_bounds','roi','clouds','islands','ship_route')}
    group=hashlib.sha256(json.dumps(layout,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    if group!=job['scene_group_id']:raise ValueError('Validation replay geometry disagrees with dataset group')
    stress=None
    if task=='tracking' and job['pan_offset']:
        scene,stress=_tracking_pan_stress(scene,config,job['pan_offset'])
    env=(CoverageEnv if task=='coverage' else TrackingEnv)(config)
    actor=load_policy(job['checkpoint'],device='cpu');actor.reset()
    obs,_=env.reset(seed=source['seed'],options={'scene':scene})
    frames=[json_value(env.render())];actions=[];executed=[];failure=None
    try:
        terminated=truncated=False
        while not(terminated or truncated):
            action=actor.act(obs);obs,_,terminated,truncated,info=env.step(action);actions.append(action.tolist())
            limit=config.coverage_pan_limit if task=='coverage' else config.pan_limit
            executed.append([*np.asarray(info['executed_action'],dtype=float),env.radar_pan/limit])
            if len(actions)%job['render_every']==0 or terminated or truncated:frames.append(json_value(env.render()))
    except Exception as error:
        failure={'type':type(error).__name__,'message':str(error),'step':len(actions)};terminated=True;truncated=False
        if frames[-1]['t']!=env.t:frames.append(json_value(env.render()))
    summary=json_value(env.summary());summary.update(terminated=terminated,truncated=truncated,steps=len(actions))
    if failure:summary.update(termination_reason='runtime_failure',failure=failure)
    metadata={'task':task,'actor':'policy','obs_schema':'obs-field-v2','action_schema':'action-field-v2',
        'scene_group_id':group,'policy_checkpoint':actor.checkpoint_metadata,'selection_subset':'validation',
        'expert_fallback':False,'frame_stride_steps':job['render_every']}
    metadata['validation_pan_stress']=stress
    record={'schema_version':'maritime-replay-v1','task':task,'actor':'policy','seed':source['seed'],'scene':scene,
        'metadata':metadata,'frames':frames,'actions':actions,'executed_actions':executed,'summary':summary}
    path=write_episode(job['output'],record);write_replay_index(path,record);env.close()
    return {'path':str(path.resolve()),'scene_group_id':group,'summary':summary}


def _rank(task,results):
    summaries=[row['summary'] for row in results]
    healthy=sum(not s['terminated'] and s['termination_reason']=='time_limit' for s in summaries)
    key='visible_fraction' if task=='tracking' else 'coverage_ratio';quality=[s.get(key,0.) for s in summaries]
    full=sum(q>=1.-1e-9 and (task!='tracking' or s.get('first_loss_time') is None) for q,s in zip(quality,summaries))
    return healthy,full,min(quality),sum(quality)/len(quality)


def select_checkpoint(task,model_dir,output_dir,*,coverage_seconds=10800.,tracking_seconds=1800.,
                      render_every=60,num_threads=1,workers=1,tracking_pan_stress=math.pi/9):
    from rollouts import write_json
    if task not in ('coverage','tracking') or min(workers,render_every,num_threads)<1 or min(coverage_seconds,tracking_seconds)<=0 or not math.isfinite(tracking_pan_stress) or tracking_pan_stress<0:
        raise ValueError('Invalid closed-loop selection settings')
    model_dir=Path(model_dir);output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    selected=model_dir/'selected.pt'
    if selected.exists():
        archive=model_dir/'selection-archive';archive.mkdir(exist_ok=True)
        digest=hashlib.sha256(selected.read_bytes()).hexdigest()
        target=archive/f'{digest}.pt'
        if not target.exists():shutil.copyfile(selected,target)
        selected.unlink()
    candidates={};jobs=[];reference_split=None
    for tag in ('best','last'):
        path=(model_dir/f'{tag}.pt').resolve();payload=path.read_bytes();checkpoint=torch.load(io.BytesIO(payload),map_location='cpu',weights_only=False)
        if checkpoint.get('task')!=task or checkpoint.get('observation_schema')!='obs-field-v2':raise ValueError('Checkpoint selection protocol/task mismatch')
        split=checkpoint['dataset_split']
        if reference_split is not None and split!=reference_split:raise ValueError('Candidates must use identical validation split')
        reference_split=split
        train_groups={read_episode_metadata(p).get('scene_group_id') for p in split['train_paths']}
        if not split['validation_paths']:raise ValueError('Selection requires validation scenes; disable explicitly for training-only runs')
        digest=hashlib.sha256(payload).hexdigest();snapshot_dir=model_dir/'selection-snapshots';snapshot_dir.mkdir(exist_ok=True)
        snapshot=snapshot_dir/f'{tag}-{digest}.pt'
        if not snapshot.exists():snapshot.write_bytes(payload)
        if hashlib.sha256(snapshot.read_bytes()).hexdigest()!=digest:raise ValueError('Candidate snapshot hash mismatch')
        candidates[tag]={'checkpoint':str(path),'snapshot_checkpoint':str(snapshot.resolve()),'sha256':digest,'results':[]};groups=set()
        for index,episode in enumerate(split['validation_paths']):
            group=read_episode_metadata(episode).get('scene_group_id')
            if group is None or group in train_groups:raise ValueError('Validation group missing or leaks into training')
            if group in groups:continue
            groups.add(group)
            offsets=(0.,-tracking_pan_stress,tracking_pan_stress) if task=='tracking' and tracking_pan_stress else (0.,)
            for stress_index,offset in enumerate(offsets):
                jobs.append({'task':task,'tag':tag,'checkpoint':str(snapshot.resolve()),'episode_path':str(episode),'scene_group_id':group,
                    'duration':coverage_seconds if task=='coverage' else tracking_seconds,'render_every':render_every,
                    'pan_offset':offset,'num_threads':num_threads,
                    'output':str(output_dir/f'{tag}-validation-{index}-stress-{stress_index}.npz')})
    if workers==1:results=list(map(_evaluate_candidate,jobs))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:results=list(pool.map(_evaluate_candidate,jobs))
    for job,result in zip(jobs,results):
        if result['scene_group_id']!=job['scene_group_id']:raise ValueError('Selection worker returned a different scene group')
        candidates[job['tag']]['results'].append(result)
    for candidate in candidates.values():candidate['rank']=_rank(task,candidate['results'])
    chosen=max(candidates,key=lambda name:candidates[name]['rank']);rank=candidates[chosen]['rank'];count=len(candidates[chosen]['results'])
    accepted=rank[0]==count and rank[1]==count
    if accepted:
        temp=selected.with_suffix('.pt.tmp');shutil.copyfile(candidates[chosen]['snapshot_checkpoint'],temp);temp.replace(selected)
    result={'schema_version':'field-model-selection-v1','task':task,'subset':'validation','test_scenes_used':False,
        'accepted':accepted,'selected_candidate':chosen,'selected_checkpoint':str(selected.resolve()) if accepted else None,
        'selected_sha256':candidates[chosen]['sha256'] if accepted else None,'candidates':candidates,
        'criterion':'safe complete episodes, full coverage/visibility count, worst quality, mean quality',
        'tracking_pan_stress_radians':tracking_pan_stress,'durations':{'coverage':coverage_seconds,'tracking':tracking_seconds}}
    write_json(output_dir/'selection.json',result)
    return result
