"""Training-only DAgger: deployment never calls an expert or mixes controls."""
from __future__ import annotations
import json
from pathlib import Path
from config import Config
from env_coverage import CoverageEnv
from env_tracking import TrackingEnv
from expert_coverage import CoverageExpert
from expert_tracking import TrackingExpert
from src.policy_tracking import NeuralPolicy
from rollouts import run_episode,write_json


def collect_dagger(task,checkpoint,manifest_path,output_dir,*,iteration=1,
                   episodes=4,beta=.8,seed=24000,max_steps=None,
                   render_every=5,progress_every=300,server=None,num_threads=1,device='cpu'):
    """Visit training layouts only and preserve the held-out scene groups."""
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
        env=(CoverageEnv if task=='coverage' else TrackingEnv)(cfg,render_mode='json')
        teacher=(CoverageExpert if task=='coverage' else TrackingExpert)(cfg)
        student=NeuralPolicy.from_checkpoint(checkpoint,device=device)
        try:
            record,target=run_episode(env,student,seed=source['seed'],actor_name='policy',
                output_dir=output_dir,name=f'dagger-{iteration}-{source["seed"]}',
                record_observations=True,max_steps=max_steps,scene=scene,
                expert_labeler=teacher,mixture_beta=beta,mixture_seed=seed+index,
                server=server,render_every=render_every,progress_every=progress_every)
        finally:env.close()
        (accepted if record['metadata']['expert_valid'] else rejected).append(str(target.resolve()))
    merged=dict(original)
    original_paths=[]
    for entry in original['paths']:
        path=Path(entry)
        original_paths.append(str((path if path.is_absolute() else manifest_path.parent/path).resolve()))
    merged['paths']=list(dict.fromkeys(original_paths+[str(Path(p).with_suffix('.npz')) for p in accepted]))
    merged['dagger']={'iteration':iteration,'beta':beta,'accepted':accepted,'rejected':rejected}
    target=Path(output_dir)/'accepted-manifest.json'
    write_json(target,merged)
    return target,merged['dagger']
