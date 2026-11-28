"""Scene-reproducible rollouts persisted as authoritative NPZ trajectories."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

try:
    from .episode_io import json_value, encode_array, decode_array, write_episode, load_episode, load_replay
except ImportError:
    from episode_io import json_value, encode_array, decode_array, write_episode, load_episode, load_replay

REPLAY_SCHEMA = 'maritime-replay-v1'


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(json_value(value), ensure_ascii=False,
                              allow_nan=False, separators=(',', ':')), encoding='utf-8')
    for attempt in range(8):
        try:
            tmp.replace(path)
            break
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(.01*2**attempt)


def restore_training_cache(json_path, npz_path=None):
    """Convert a legacy JSON record, or copy a canonical NPZ episode."""
    path=Path(json_path)
    return write_episode(Path(npz_path) if npz_path else path.with_suffix('.npz'),
                         load_episode(path))


def write_replay_index(path,record):
    """Small disposable catalog sidecar; browsers need not parse training tensors."""
    path=Path(path);stat=path.stat();frames=record['frames']
    entry={'task':record['task'],'actor':record['actor'],'seed':record.get('seed'),
           'frames':len(frames),'duration':frames[-1]['t'] if frames else 0,
           'behavior':record.get('metadata',{}).get('behavior')}
    if record.get('metadata',{}).get('control_mode'):
        entry['control_mode']=record['metadata']['control_mode']
    write_json(path.with_suffix('.replay-index.json'),
        {'schema_version':'maritime-replay-index-v1','fingerprint':[stat.st_mtime_ns,stat.st_size],'entry':entry})


def run_episode(env, actor, *, seed, actor_name, output_dir, name,
                record_observations=False, max_steps=None, server=None,
                render_every=1, realtime_factor=0., scene=None,
                progress_every=300, expert_labeler=None, mixture_beta=0., mixture_seed=None,
                replay_every=1):
    """Run the same closed loop for experts, deployment and DAgger collection.

    Original expert demonstrations must finish without physical/tracking failure.
    DAgger may retain a nonempty successfully labeled prefix after behavior or
    known teacher-planning failure. Labels describe the pre-step observations;
    the final collision or unlabeled observation is stored for replay but is
    excluded by SequenceDataset's T+1-to-T input slicing. Unexpected errors still
    propagate and never produce an accepted training cache.
    """
    if (isinstance(replay_every, bool) or not isinstance(replay_every, int) or
            render_every < 1 or replay_every < 1 or progress_every < 1 or
            realtime_factor < 0 or not 0<=mixture_beta<=1):
        raise ValueError('Invalid rendering cadence or playback speed')
    constrained=bool(getattr(actor,'execution_metadata',None))
    if constrained and (actor_name!='policy' or expert_labeler is not None):
        raise ValueError('An explicit policy constraint layer cannot enter expert or DAgger collection')
    path = Path(output_dir) / f'{name}.npz'
    obs, info = env.reset(seed=int(seed), options={'scene': scene} if scene else None)
    actor.reset()
    if expert_labeler is not None: expert_labeler.reset()
    rng=np.random.default_rng(seed if mixture_seed is None else mixture_seed)
    scene_spec = json_value(env.scene)
    task = env.task
    field_mode = bool(getattr(env.config, 'potential_enabled', False))
    action_shape = (3,) if field_mode else (2,)
    scene_id = hashlib.sha256(json.dumps(scene_spec, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    layout = {key:scene_spec.get(key) for key in
              ['task','world_bounds','roi','clouds','islands','ship_route']}
    scene_group_id = hashlib.sha256(json.dumps(layout, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    behavior_metadata = ({'behavior': 'dagger_training', 'label_source': 'expert',
                          'mixture_beta': mixture_beta, 'mixture_seed': mixture_seed}
                         if expert_labeler is not None else {})
    if constrained:
        constraints=json_value(actor.execution_metadata)
        behavior_metadata.update(control_mode=constraints['mode'],policy_constraints=constraints)
    if server:
        server.publish({'type': 'scene', 'scene': scene_spec, 'task': task, 'actor': actor_name,
                        'metadata': behavior_metadata})
    frames, actions, executed = [json_value(env.render())], [], []
    frame_action_indices = [0]
    state = frames[0]
    expert_actions=[];policy_actions=[];expert_controls=0
    raw_policy_actions=[];constraint_interventions=[]
    sequences = {k: [np.array(v, copy=True)] for k, v in obs.items()} if record_observations else {}
    terminated = truncated = False
    start = time.perf_counter()
    step_index = 0
    planning_failure = None
    unexpected_error = None
    runtime_failure = None
    while not (terminated or truncated):
        if max_steps is not None and step_index >= max_steps:
            truncated=True
            break
        stage='actor'
        expert_control=False
        try:
            action = np.asarray(actor.act(obs), dtype=np.float32)
            if expert_labeler is not None:
                policy_action=action.copy()
                expert_action=np.asarray(expert_labeler.act(obs),dtype=np.float32)
                if expert_action.shape!=action_shape or not np.isfinite(expert_action).all():
                    raise ValueError('Expert label must be a finite direction')
                if rng.random()<mixture_beta:
                    action=expert_action.copy();expert_control=True
            if action.shape != action_shape or not np.isfinite(action).all():
                raise ValueError(f'{actor_name} returned an invalid action at step {step_index}')
            if constrained:
                raw_action=np.asarray(actor.last_raw_action,dtype=np.float32)
                if raw_action.shape!=action_shape or not np.isfinite(raw_action).all():
                    raise ValueError('Policy constraint layer did not retain a valid raw action')
            stage='environment'
            obs, _, terminated, truncated, info = env.step(action)
        except Exception as error:
            failure = {'type':type(error).__name__,'message':str(error),
                       'step':step_index,'t':state['t']}
            if stage=='environment':
                runtime_failure=dict(failure,attempted_action=action.tolist())
            else:
                planning_failure=failure
                if constrained and getattr(actor,'last_raw_action',None) is not None:
                    candidate=np.asarray(actor.last_raw_action)
                    if candidate.dtype.kind in 'biuf' and np.isfinite(candidate).all():
                        planning_failure['raw_policy_action']=json_value(candidate)
            terminated = True
            if stage=='environment' or type(error).__name__ not in {'ExpertPlanningError','PlanningError','NoFeasibleAction'}:
                unexpected_error = error
            break
        if constrained:
            raw_policy_actions.append(raw_action.tolist())
            constraint_interventions.append(bool(actor.last_intervened))
        if expert_control:expert_controls+=1
        save_frame = (step_index + 1) % replay_every == 0 or terminated or truncated
        live_frame = server and (step_index + 1) % render_every == 0
        if save_frame or live_frame or not hasattr(env, 'uav_model'):
            state = json_value(env.render())
        else:
            # Rendering field/grid tensors is expensive and unnecessary for
            # training. Read the identical physical velocity used by render().
            vx, vy = env.uav_model.velocity(env.uav)
            state = {'t': env.t, 'uav': {'vx': vx, 'vy': vy},
                     'radar': {'pan': getattr(env, 'radar_pan', 0.)},
                     'coverage': {'ratio': info.get('coverage_ratio', 0.)},
                     'metrics': {'distance': info.get('distance', 0.)}}
        if save_frame:
            frames.append(state)
            frame_action_indices.append(step_index + 1)
        actions.append(action.tolist())
        if expert_labeler is not None:
            expert_actions.append(expert_action.tolist());policy_actions.append(policy_action.tolist())
        u = state['uav']
        speed = math.hypot(u['vx'], u['vy'])
        actual = [u['vx'] / speed, u['vy'] / speed]
        if field_mode:
            limit = env.config.coverage_pan_limit if task == 'coverage' else env.config.pan_limit
            actual.append(float(state['radar']['pan']) / limit)
        executed.append(actual)
        if record_observations:
            for k, value in obs.items():
                sequences[k].append(np.array(value, copy=True))
        step_index += 1
        if server and step_index % render_every == 0:
            server.publish({'type': 'frame', 'frame': state, 'action_index': step_index})
        if step_index % progress_every == 0:
            c = state.get('coverage', {}).get('ratio', 0.)
            d = state.get('metrics', {}).get('distance', 0.)
            print(f'[{task}/{actor_name}] t={state["t"]:.0f}s coverage={c:.3f} distance={d:.0f}m', flush=True)
            if server:
                server.publish({'type': 'progress', 'task': task, 'actor': actor_name,
                                'steps': step_index, 'time_seconds': state['t'],
                                'coverage_ratio': c, 'distance': d})
        if realtime_factor:
            time.sleep(max(0., state['t'] / realtime_factor - (time.perf_counter() - start)))
    if frame_action_indices[-1] != step_index or (runtime_failure is not None and
            getattr(env,'t',frames[-1]['t'])!=frames[-1]['t']):
        frames.append(json_value(env.render()))
        frame_action_indices.append(step_index)
    summary = json_value(env.summary())
    summary.update(steps=step_index, wall_seconds=time.perf_counter() - start,
                   terminated=bool(terminated), truncated=bool(truncated))
    if truncated and summary.get('termination_reason') is None:
        summary['termination_reason']='collection_step_limit'
    if planning_failure:
        summary['planning_failure'] = planning_failure
        summary['termination_reason'] = 'policy_constraint_failure' if constrained else 'expert_planning_failure'
    if runtime_failure:
        summary.update(runtime_failure=runtime_failure,termination_reason='runtime_failure')
    behavior_failed=bool(terminated or (task=='tracking' and summary.get('first_loss_time') is not None))
    label_valid=step_index>0 and len(expert_actions)==len(actions)==step_index
    expert_valid=(label_valid and unexpected_error is None if expert_labeler is not None
                  else actor_name=='expert' and step_index>0 and not behavior_failed)
    metadata = {'scene_id': scene_id, 'scene_group_id':scene_group_id,
                'expert_valid':expert_valid,
                'task': task, 'seed': int(seed),
                'actor': actor_name, 'obs_schema': 'obs-field-v2' if field_mode else 'obs-v1', 'schema_version': REPLAY_SCHEMA,
                'config': json_value(env.config),
                'config_hash':getattr(env.config,'config_hash',None)}
    if field_mode:
        metadata.update(dataset_version='bc-dataset-field-v2', observation_schema='obs-field-v2',
                        action_schema='action-field-v2')
    if getattr(actor,'checkpoint_metadata',None):
        metadata['policy_checkpoint']=json_value(actor.checkpoint_metadata)
    if constrained:
        metadata.update(control_mode=actor.execution_metadata['mode'],
                        policy_constraints=json_value(actor.execution_metadata))
        summary['policy_constraints']=metadata['policy_constraints']
    record = {'schema_version': REPLAY_SCHEMA, 'task': task, 'actor': actor_name,
              'seed': int(seed), 'scene': scene_spec, 'metadata': metadata,
              'frames': frames, 'actions': actions, 'executed_actions': executed,
              'frame_action_indices': frame_action_indices,
              'summary': summary}
    if constrained:
        record.update(raw_policy_actions=raw_policy_actions,
                      constraint_interventions=constraint_interventions)
    if expert_labeler is not None:
        record['expert_actions']=expert_actions;record['policy_actions']=policy_actions
        metadata.update(behavior_metadata)
        metadata.update(label_valid=label_valid,behavior_failed=behavior_failed,
                        accepted_prefix=bool(expert_valid and (behavior_failed or
                            summary.get('termination_reason')=='collection_step_limit')))
        summary['behavior_failed']=behavior_failed
        summary['expert_control_fraction']=expert_controls/max(1,step_index)
    arrays = {k: np.stack(v) for k, v in sequences.items()}
    if record_observations:
        record['observations'] = {k: encode_array(v) for k, v in arrays.items()}
    # Every completed attempt remains replayable. Only accepted-manifest.json
    # selects valid expert demonstrations for training.
    path=write_episode(path,record)
    write_replay_index(path,record)
    print(f'[{task}/{actor_name}] saved {path} {json.dumps(summary, ensure_ascii=False)}', flush=True)
    if unexpected_error is not None:
        raise unexpected_error
    return record, path
