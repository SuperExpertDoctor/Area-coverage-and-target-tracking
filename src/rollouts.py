"""Scene-reproducible rollouts, complete JSON records and derived NPZ caches."""
from __future__ import annotations

import base64
from dataclasses import asdict, is_dataclass
import hashlib
import json
import math
from pathlib import Path
import time
import zlib

import numpy as np

REPLAY_SCHEMA = 'maritime-replay-v1'


def json_value(value):
    if is_dataclass(value):
        return json_value(asdict(value))
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def encode_array(array):
    """Lossless numeric payload inside JSON; no pickle or external binary required."""
    a = np.ascontiguousarray(array)
    if a.dtype.hasobject or not np.isfinite(a).all():
        raise ValueError('Training arrays must be finite numeric values')
    return {'encoding': 'zlib+base64', 'dtype': a.dtype.str,
            'shape': list(a.shape),
            'data': base64.b64encode(zlib.compress(a.tobytes(), level=6)).decode('ascii')}


def decode_array(payload):
    if payload.get('encoding') != 'zlib+base64':
        raise ValueError('Unsupported array encoding')
    dtype = np.dtype(payload['dtype'])
    if dtype.hasobject or dtype.kind not in 'biuf':
        raise ValueError('Unsupported array dtype')
    shape = tuple(int(n) for n in payload['shape'])
    if any(n < 0 for n in shape):
        raise ValueError('Invalid array shape')
    raw = zlib.decompress(base64.b64decode(payload['data'], validate=True))
    if len(raw) != math.prod(shape) * dtype.itemsize:
        raise ValueError('Array shape does not match its contents')
    result = np.frombuffer(raw, dtype=dtype).reshape(shape).copy()
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite decoded training data')
    return result


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(json_value(value), ensure_ascii=False,
                              allow_nan=False, separators=(',', ':')), encoding='utf-8')
    tmp.replace(path)


def restore_training_cache(json_path, npz_path=None):
    """Regenerate the optional training cache from canonical JSON alone."""
    path = Path(json_path)
    record = json.loads(path.read_text(encoding='utf-8'))
    arrays = {k: decode_array(v) for k, v in record['observations'].items()}
    arrays['actions'] = np.asarray(record.get('expert_actions',record['actions']), dtype=np.float32)
    arrays['executed_actions'] = np.asarray(record['executed_actions'], dtype=np.float32)
    arrays['metadata'] = np.asarray(json.dumps(record['metadata']))
    target = Path(npz_path) if npz_path else path.with_suffix('.npz')
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, **arrays)
    return target


def write_replay_index(path,record):
    """Small disposable catalog sidecar; browsers need not parse training tensors."""
    path=Path(path);stat=path.stat();frames=record['frames']
    entry={'task':record['task'],'actor':record['actor'],'seed':record.get('seed'),
           'frames':len(frames),'duration':frames[-1]['t'] if frames else 0,
           'behavior':record.get('metadata',{}).get('behavior')}
    write_json(path.with_suffix('.replay-index.json'),
        {'schema_version':'maritime-replay-index-v1','fingerprint':[stat.st_mtime_ns,stat.st_size],'entry':entry})


def run_episode(env, actor, *, seed, actor_name, output_dir, name,
                record_observations=False, max_steps=None, server=None,
                render_every=1, realtime_factor=0., scene=None,
                progress_every=300, expert_labeler=None, mixture_beta=0., mixture_seed=None):
    """Run the same closed loop for experts, deployment and DAgger collection.

    Original expert demonstrations must finish without physical/tracking failure.
    DAgger may retain a nonempty successfully labeled prefix after behavior or
    known teacher-planning failure. Labels describe the pre-step observations;
    the final collision or unlabeled observation is stored for replay but is
    excluded by SequenceDataset's T+1-to-T input slicing. Unexpected errors still
    propagate and never produce an accepted training cache.
    """
    if render_every < 1 or progress_every < 1 or realtime_factor < 0 or not 0<=mixture_beta<=1:
        raise ValueError('Invalid rendering cadence or playback speed')
    path = Path(output_dir) / f'{name}.json'
    if record_observations:
        path.with_suffix('.npz').unlink(missing_ok=True)
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
    if server:
        server.publish({'type': 'scene', 'scene': scene_spec, 'task': task, 'actor': actor_name,
                        'metadata': behavior_metadata})
    frames, actions, executed = [json_value(env.render())], [], []
    expert_actions=[];policy_actions=[];expert_controls=0
    sequences = {k: [np.array(v, copy=True)] for k, v in obs.items()} if record_observations else {}
    terminated = truncated = False
    start = time.perf_counter()
    step_index = 0
    planning_failure = None
    unexpected_error = None
    while not (terminated or truncated):
        if max_steps is not None and step_index >= max_steps:
            truncated=True
            break
        try:
            action = np.asarray(actor.act(obs), dtype=np.float32)
            if expert_labeler is not None:
                policy_action=action.copy()
                expert_action=np.asarray(expert_labeler.act(obs),dtype=np.float32)
                if expert_action.shape!=action_shape or not np.isfinite(expert_action).all():
                    raise ValueError('Expert label must be a finite direction')
                if rng.random()<mixture_beta:
                    action=expert_action.copy();expert_controls+=1
        except Exception as error:
            planning_failure = {'type':type(error).__name__,'message':str(error),
                                'step':step_index,'t':frames[-1]['t']}
            terminated = True
            if type(error).__name__ not in {'ExpertPlanningError','PlanningError','NoFeasibleAction'}:
                unexpected_error = error
            break
        if action.shape != action_shape or not np.isfinite(action).all():
            raise ValueError(f'{actor_name} returned an invalid action at step {step_index}')
        obs, _, terminated, truncated, info = env.step(action)
        frame = json_value(env.render())
        frames.append(frame)
        actions.append(action.tolist())
        if expert_labeler is not None:
            expert_actions.append(expert_action.tolist());policy_actions.append(policy_action.tolist())
        u = frame['uav']
        speed = math.hypot(u['vx'], u['vy'])
        actual = [u['vx'] / speed, u['vy'] / speed]
        if field_mode:
            limit = env.config.coverage_pan_limit if task == 'coverage' else env.config.pan_limit
            actual.append(float(frame['radar']['pan']) / limit)
        executed.append(actual)
        if record_observations:
            for k, value in obs.items():
                sequences[k].append(np.array(value, copy=True))
        step_index += 1
        if server and step_index % render_every == 0:
            server.publish({'type': 'frame', 'frame': frame})
        if step_index % progress_every == 0:
            c = frame.get('coverage', {}).get('ratio', 0.)
            d = frame.get('metrics', {}).get('distance', 0.)
            print(f'[{task}/{actor_name}] t={frame["t"]:.0f}s coverage={c:.3f} distance={d:.0f}m', flush=True)
        if realtime_factor:
            time.sleep(max(0., frame['t'] / realtime_factor - (time.perf_counter() - start)))
    summary = json_value(env.summary())
    summary.update(steps=step_index, wall_seconds=time.perf_counter() - start,
                   terminated=bool(terminated), truncated=bool(truncated))
    if truncated and summary.get('termination_reason') is None:
        summary['termination_reason']='collection_step_limit'
    if planning_failure:
        summary['planning_failure'] = planning_failure
        summary['termination_reason'] = 'expert_planning_failure'
    behavior_failed=bool(terminated or (task=='tracking' and summary.get('first_loss_time') is not None))
    label_valid=step_index>0 and len(expert_actions)==len(actions)==step_index
    expert_valid=(label_valid and unexpected_error is None if expert_labeler is not None
                  else actor_name=='expert' and not behavior_failed)
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
    record = {'schema_version': REPLAY_SCHEMA, 'task': task, 'actor': actor_name,
              'seed': int(seed), 'scene': scene_spec, 'metadata': metadata,
              'frames': frames, 'actions': actions, 'executed_actions': executed,
              'summary': summary}
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
    write_json(path, record)
    write_replay_index(path,record)
    # Ordinary failed expert demonstrations are excluded; DAgger prefixes carry
    # explicit validity/failure metadata and use expert labels, never behavior.
    if record_observations and expert_valid:
        arrays.update(actions=np.asarray(expert_actions if expert_labeler is not None else actions, dtype=np.float32),
                      executed_actions=np.asarray(executed, dtype=np.float32),
                      metadata=np.asarray(json.dumps(metadata)))
        np.savez_compressed(path.with_suffix('.npz'), **arrays)
    print(f'[{task}/{actor_name}] saved {path} {json.dumps(summary, ensure_ascii=False)}', flush=True)
    if unexpected_error is not None:
        raise unexpected_error
    return record, path
