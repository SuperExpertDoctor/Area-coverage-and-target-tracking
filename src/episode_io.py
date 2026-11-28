"""Atomic numeric NPZ episode archives with lazy replay-only and legacy readers.

Observation keys remain at the NPZ root for SequenceDataset. Replay JSON contains
no training observations; NumPy always reads with allow_pickle=False.
"""
from __future__ import annotations

import base64
from dataclasses import asdict, is_dataclass
import io
import json
import math
from pathlib import Path
import tempfile
import time
import zipfile
import zlib

import numpy as np

FORMAT = 'maritime-episode-npz-v1'
_TRAJECTORY_FIELDS = ('actions', 'executed_actions', 'expert_actions', 'policy_actions',
                      'raw_policy_actions', 'constraint_interventions')
_RESERVED = set(_TRAJECTORY_FIELDS) | {'metadata', 'metadata_json', 'replay_json',
                                     'episode_manifest', 'behavior_actions'}


def archive_fingerprint(path):
    """Identify an atomic archive replacement without reading training tensors."""
    try:
        stat=Path(path).stat()
    except FileNotFoundError:
        return None
    return stat.st_ino,stat.st_size,stat.st_mtime_ns


def json_value(value):
    if is_dataclass(value): return json_value(asdict(value))
    if isinstance(value, dict): return {str(k):json_value(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [json_value(v) for v in value]
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, Path): return str(value)
    return value


def _numeric(value):
    array=np.asarray(value)
    if array.dtype.kind not in 'biuf' or not np.isfinite(array).all():
        raise ValueError('Episode arrays must contain finite numeric values')
    return np.ascontiguousarray(array)


def encode_array(array):
    array=_numeric(array)
    return {'encoding':'zlib+base64','dtype':array.dtype.str,'shape':list(array.shape),
            'data':base64.b64encode(zlib.compress(array.tobytes(),level=6)).decode('ascii')}


def decode_array(payload):
    if not isinstance(payload,dict) or payload.get('encoding')!='zlib+base64':
        raise ValueError('Unsupported array encoding')
    dtype=np.dtype(payload['dtype'])
    if dtype.kind not in 'biuf': raise ValueError('Unsupported array dtype')
    shape=tuple(int(n) for n in payload['shape'])
    if any(n<0 for n in shape): raise ValueError('Invalid array shape')
    raw=zlib.decompress(base64.b64decode(payload['data'],validate=True))
    if len(raw)!=math.prod(shape)*dtype.itemsize:
        raise ValueError('Array shape does not match its contents')
    return _numeric(np.frombuffer(raw,dtype=dtype).reshape(shape)).copy()


def _json_text(value):
    return json.dumps(json_value(value),ensure_ascii=False,allow_nan=False,separators=(',',':'))


def _replace(temporary,path):
    for attempt in range(8):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt==7: raise
            time.sleep(.01*2**attempt)


def write_episode(path,record):
    """Save one authoritative archive and return its canonical .npz path."""
    path=Path(path).with_suffix('.npz')
    if not isinstance(record,dict): raise ValueError('Episode must be a dictionary')
    metadata=record.get('metadata',{})
    if not isinstance(metadata,dict): raise ValueError('Episode metadata must be an object')
    task=record.get('task',metadata.get('task'))
    if metadata.get('task') is not None and record.get('task') is not None and metadata['task']!=record['task']:
        raise ValueError('Replay and metadata task disagree')
    schema=metadata.get('obs_schema',metadata.get('observation_schema'))
    if schema not in (None,'obs-v1','obs-field-v2'):
        raise ValueError('Unsupported observation protocol')
    if metadata.get('obs_schema') and metadata.get('observation_schema') and metadata['obs_schema']!=metadata['observation_schema']:
        raise ValueError('Observation protocols disagree')
    action_dim=3 if schema=='obs-field-v2' else 2
    if metadata.get('action_schema') not in (None,'action-field-v2' if action_dim==3 else 'action-v1'):
        raise ValueError('Action and observation protocols disagree')
    arrays={};mapping={}
    length=None
    for field in _TRAJECTORY_FIELDS:
        if field not in record: continue
        array=_numeric(record[field])
        if field=='constraint_interventions':
            if array.ndim!=1: raise ValueError('constraint_interventions must have shape (T,)')
        else:
            if array.size==0: array=array.reshape(0,action_dim)
            allowed=(2,3) if schema is None or (field=='executed_actions' and action_dim==3) else (action_dim,)
            if array.ndim!=2 or array.shape[1] not in allowed:
                raise ValueError(f'{field} shape disagrees with the episode action protocol')
        if length is None: length=len(array)
        if len(array)!=length: raise ValueError('Episode action lengths disagree')
        key='behavior_actions' if field=='actions' and 'expert_actions' in record else (
            'actions' if field=='expert_actions' else field)
        arrays[key]=array;mapping[field]=key
    length=0 if length is None else length
    observation_keys=[]
    for key,value in record.get('observations',{}).items():
        if not isinstance(key,str) or not key.isidentifier() or key in _RESERVED:
            raise ValueError('Reserved or invalid observation key')
        array=decode_array(value) if isinstance(value,dict) else _numeric(value)
        if array.ndim<1 or len(array)!=length+1:
            raise ValueError('Observations must have T+1 entries')
        expected={'potential':(4,33,33),'field_context':(16,),'coverage_map':(3,64,64)}
        if task in ('coverage','tracking'):expected['vector']=(111 if task=='coverage' else 134,)
        if key in expected and array.shape[1:]!=expected[key]:
            raise ValueError(f'{key} shape disagrees with the observation protocol')
        arrays[key]=array;observation_keys.append(key)
    replay={key:value for key,value in record.items()
            if key not in (*_TRAJECTORY_FIELDS,'observations','metadata')}
    arrays['metadata']=np.asarray(_json_text(metadata))
    arrays['replay_json']=np.frombuffer(_json_text(replay).encode('utf-8'),dtype=np.uint8)
    arrays['episode_manifest']=np.asarray(_json_text({'format':FORMAT,
        'observations':observation_keys,'trajectory_fields':mapping,
        'has_observations':'observations' in record,'has_metadata':'metadata' in record}))
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(prefix='.'+path.name+'-',suffix='.tmp',dir=path.parent,delete=False) as stream:
            temporary=Path(stream.name)
            np.savez_compressed(stream,**arrays)
            stream.flush()
        _replace(temporary,path)
    finally:
        if temporary is not None: temporary.unlink(missing_ok=True)
    return path


def _validate_npz(source,max_uncompressed_bytes=None):
    """Check NPY headers before NumPy can allocate from a forged shape."""
    position=source.tell() if hasattr(source,'tell') else None
    try:
        with zipfile.ZipFile(source) as archive:
            entries=archive.infolist()
            if len(entries)>64 or (max_uncompressed_bytes is not None and
                    sum(item.file_size for item in entries)>max_uncompressed_bytes):
                raise ValueError('NPZ expanded size exceeds import limit')
            if len({item.filename for item in entries})!=len(entries):
                raise ValueError('Duplicate NPZ entries')
            for item in entries:
                if not item.filename.endswith('.npy') or not item.filename[:-4].isidentifier():
                    raise ValueError('Invalid NPZ array name')
                with archive.open(item) as stream:
                    version=np.lib.format.read_magic(stream)
                    if version==(1,0):shape,_,dtype=np.lib.format.read_array_header_1_0(stream)
                    elif version==(2,0):shape,_,dtype=np.lib.format.read_array_header_2_0(stream)
                    else:raise ValueError('Unsupported NPY header version')
                    if dtype.hasobject:raise ValueError('NPZ object arrays and pickle are forbidden')
                    if len(shape)>8 or stream.tell()+math.prod(shape)*dtype.itemsize!=item.file_size:
                        raise ValueError('NPZ array shape does not match its stored size')
    finally:
        if position is not None:source.seek(position)


def _read_archive(source,*,observations):
    _validate_npz(source)
    with np.load(source,allow_pickle=False) as data:
        if 'episode_manifest' not in data:
            raise ValueError('NPZ is a legacy training cache without a replay')
        manifest=json.loads(data['episode_manifest'].item())
        if not isinstance(manifest,dict) or manifest.get('format')!=FORMAT:
            raise ValueError('Unsupported episode archive format')
        payload=data['replay_json']
        if payload.dtype!=np.uint8 or payload.ndim!=1:
            raise ValueError('Invalid replay JSON payload')
        replay=json.loads(payload.tobytes().decode('utf-8'))
        if not isinstance(replay,dict): raise ValueError('Replay JSON must be an object')
        if manifest.get('has_metadata'):
            replay['metadata']=json.loads(data['metadata'].item())
        mapping=manifest.get('trajectory_fields',{})
        if not isinstance(mapping,dict): raise ValueError('Invalid trajectory mapping')
        for field,key in mapping.items():
            if field not in _TRAJECTORY_FIELDS or key not in _RESERVED:
                raise ValueError('Invalid trajectory field')
            replay[field]=_numeric(data[key]).tolist()
        if observations and manifest.get('has_observations'):
            keys=manifest.get('observations',[])
            if not isinstance(keys,list) or any(not isinstance(k,str) or not k.isidentifier() or k in _RESERVED for k in keys):
                raise ValueError('Invalid observation fields')
            replay['observations']={key:encode_array(data[key]) for key in keys}
        return replay


def _load(path,*,observations):
    path=Path(path)
    if path.suffix.lower()=='.npz':
        # Old training caches used an adjacent canonical JSON; preserve them.
        with np.load(path,allow_pickle=False) as data:
            canonical='episode_manifest' in data
        if canonical: return _read_archive(path,observations=observations)
        path=path.with_suffix('.json')
    with path.open('r',encoding='utf-8') as stream: record=json.load(stream)
    if not isinstance(record,dict): raise ValueError('Episode must be an object')
    if not observations: record.pop('observations',None)
    return record


def load_episode(path):
    """Return the historical record shape, including encoded training arrays."""
    return _load(path,observations=True)


def load_replay(path):
    """Read replay data without decompressing or serializing observations."""
    return _load(path,observations=False)


def load_replay_bytes(payload,*,max_uncompressed_bytes=512*1024*1024):
    """Bound uploaded NPZ expansion before NumPy reads any archive entries."""
    try:
        _validate_npz(io.BytesIO(payload),max_uncompressed_bytes=max_uncompressed_bytes)
        return _read_archive(io.BytesIO(payload),observations=False)
    except (zipfile.BadZipFile,KeyError,TypeError,UnicodeError) as error:
        raise ValueError('Invalid NPZ replay archive') from error
