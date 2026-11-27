"""Training-only DAgger: deployment never calls an expert or mixes controls."""
from __future__ import annotations
from copy import deepcopy
import json
import math
from pathlib import Path
import numpy as np
from config import Config
from env_coverage import CoverageEnv
from env_tracking import TrackingEnv
from expert_coverage import CoverageExpert
from expert_tracking import TrackingExpert
from src.policy_tracking import NeuralPolicy
from rollouts import run_episode,write_json,write_replay_index,restore_training_cache
from geometry import wrap
from scenario import validate_scene


def jitter_tracking_initial_pan(scene,config,jitter,rng):
    """Clone a training scene and perturb only its physically valid actual pan.

    ``jitter`` is radians. No route/layout or held-out scene is modified. The
    initial target stays inside the same guarded radar envelope used by scene
    validation, rather than starting from an already failed physical state.
    """
    if not math.isfinite(jitter) or jitter<0:
        raise ValueError('initial pan jitter must be finite and nonnegative')
    result=deepcopy(scene)
    if result.get('task')!='tracking':
        raise ValueError('initial pan jitter is supported for tracking training scenes only')
    u,s=result['uav_initial'],result['ship_initial']
    center=wrap(math.atan2(s['y']-u['y'],s['x']-u['x'])-u['heading'])
    guard=config.tracking_half_angle-config.tracking_angle_guard
    lower=max(-config.pan_limit,center-guard)
    upper=min(config.pan_limit,center+guard)
    if lower>upper:
        raise ValueError('initial pan jitter has no feasible guarded radar interval')
    sampled=float(rng.uniform(-jitter,jitter))
    actual=float(np.clip(center+sampled,lower,upper))
    original=float(result['radar_pan_initial'])
    result['radar_pan_initial']=actual
    validate_scene(result,config,'tracking')
    return result,dict(requested_radians=float(jitter),sampled_offset_radians=sampled,
                       target_bearing_pan=center,original_initial_pan=original,
                       actual_initial_pan=actual,actual_offset_radians=wrap(actual-center),
                       guarded_interval=[float(lower),float(upper)])


def collect_dagger(task,checkpoint,manifest_path,output_dir,*,iteration=1,
                   episodes=4,beta=.8,seed=24000,max_steps=None,
                   render_every=5,progress_every=300,server=None,num_threads=1,device='cpu',
                   initial_pan_jitter=0.):
    """Visit training layouts only and preserve the held-out scene groups."""
    if not math.isfinite(initial_pan_jitter) or initial_pan_jitter<0:
        raise ValueError('initial pan jitter must be finite and nonnegative')
    if initial_pan_jitter and task!='tracking':
        raise ValueError('initial pan jitter is supported for tracking only')
    import torch
    torch.set_num_threads(num_threads)
    from src.datasets import read_episode_metadata
    ckpt=torch.load(checkpoint,map_location='cpu',weights_only=False)
    unique={}
    for path in ckpt['dataset_split']['train_paths']:
        meta=read_episode_metadata(path)
        unique.setdefault(meta.get('scene_group_id',meta['scene_id']),Path(path))
    paths=sorted(unique.values())[:episodes]
    manifest_path=Path(manifest_path).resolve()
    original=json.loads(manifest_path.read_text(encoding='utf-8'))
    if not original.get('complete'):raise ValueError('Cannot augment an incomplete expert dataset')
    accepted=[];rejected=[]
    for index,path in enumerate(paths):
        source=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
        scene=source['scene'];cfg=Config.from_dict(scene['config_snapshot'])
        augmentation=None
        if initial_pan_jitter:
            augmentation_seed=seed+index
            scene,augmentation=jitter_tracking_initial_pan(scene,cfg,initial_pan_jitter,
                                                            np.random.default_rng(augmentation_seed))
            augmentation['seed']=augmentation_seed
        env=(CoverageEnv if task=='coverage' else TrackingEnv)(cfg,render_mode='json')
        teacher=(CoverageExpert if task=='coverage' else TrackingExpert)(cfg)
        if cfg.potential_enabled:
            from expert_field import FieldCoverageExpert, FieldTrackingExpert
            teacher=(FieldCoverageExpert if task=='coverage' else FieldTrackingExpert)(cfg)
        student=NeuralPolicy.from_checkpoint(checkpoint,device=device)
        try:
            record,target=run_episode(env,student,seed=source['seed'],actor_name='policy',
                output_dir=output_dir,name=f'dagger-{iteration}-{source["seed"]}',
                record_observations=True,max_steps=max_steps,scene=scene,
                expert_labeler=teacher,mixture_beta=beta,mixture_seed=seed+index,
                server=server,render_every=render_every,progress_every=progress_every)
        finally:env.close()
        if augmentation is not None:
            record['metadata']['training_augmentation']={'initial_pan_jitter':augmentation}
            write_json(target,record)
            write_replay_index(target,record)
            if record['metadata']['expert_valid']:
                # Canonical JSON and derived NPZ carry the same augmentation
                # provenance; original teacher actions remain unchanged labels.
                restore_training_cache(target)
        (accepted if record['metadata']['expert_valid'] else rejected).append(str(target.resolve()))
    merged=dict(original)
    original_paths=[]
    for entry in original['paths']:
        path=Path(entry)
        original_paths.append(str((path if path.is_absolute() else manifest_path.parent/path).resolve()))
    merged['paths']=list(dict.fromkeys(original_paths+[str(Path(p).with_suffix('.npz')) for p in accepted]))
    merged['dagger']={'iteration':iteration,'beta':beta,'seed':seed,
                       'initial_pan_jitter':float(initial_pan_jitter),
                       'accepted':accepted,'rejected':rejected}
    target=Path(output_dir)/'accepted-manifest.json'
    write_json(target,merged)
    return target,merged['dagger']
