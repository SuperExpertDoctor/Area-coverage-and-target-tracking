"""Canonical NPZ trajectories and backward-compatible JSON reading."""
import io
import json
import zipfile

import numpy as np
import pytest


def record_fixture():
    from rollouts import encode_array
    return {'schema_version':'maritime-replay-v1','task':'tracking','actor':'policy','seed':7,
            'scene':{},'metadata':{'task':'tracking','obs_schema':'obs-v1','expert_valid':True,
                                  'scene_id':'scene','behavior':'dagger_training'},
            'frames':[{'t':0},{'t':2}], 'frame_action_indices':[0,2], 'summary':{'steps':2},
            'actions':[[1.,0.],[1.,0.]],'expert_actions':[[0.,1.],[0.,1.]],
            'executed_actions':[[.8,.6],[.6,.8]],
            'observations':{'vector':encode_array(np.arange(3*134,dtype=np.float32).reshape(3,134))}}


def test_npz_round_trip_preserves_dataset_and_omits_duplicate_observation_payloads(tmp_path):
    from episode_io import write_episode, load_episode, load_replay
    from datasets import SequenceDataset
    from rollouts import decode_array
    record=record_fixture()
    path=write_episode(tmp_path/'episode.json',record)
    assert path==tmp_path/'episode.npz'
    assert not path.with_suffix('.json').exists()
    actual=load_episode(path)
    assert actual==record
    replay=load_replay(path)
    assert replay=={k:v for k,v in record.items() if k!='observations'}
    with np.load(path,allow_pickle=False) as data:
        np.testing.assert_array_equal(data['vector'],decode_array(record['observations']['vector']))
        np.testing.assert_array_equal(data['actions'],record['expert_actions'])
        assert all(not data[key].dtype.hasobject for key in data.files)
        contents=data['replay_json'].tobytes()
        assert b'zlib+base64' not in contents and b'observations' not in contents
    item=SequenceDataset([path],task='tracking')[0]
    np.testing.assert_array_equal(item['actions'],record['expert_actions'])
    assert item['vector'].shape==(2,134)


def test_legacy_json_and_cache_sidecar_still_load(tmp_path):
    from episode_io import load_episode, load_replay
    record=record_fixture();path=tmp_path/'legacy.json'
    path.write_text(json.dumps(record),encoding='utf-8')
    np.savez(path.with_suffix('.npz'),metadata=np.asarray(json.dumps(record['metadata'])))
    assert load_episode(path)==record
    assert load_episode(path.with_suffix('.npz'))==record
    assert 'observations' not in load_replay(path)


@pytest.mark.parametrize('kind',['object','shape','nonfinite','reserved'])
def test_invalid_arrays_do_not_replace_existing_episode(tmp_path,kind):
    from episode_io import write_episode
    record=record_fixture();path=write_episode(tmp_path/'episode.npz',record);before=path.read_bytes()
    if kind=='object': record['observations']['vector']=np.array([object()],dtype=object)
    if kind=='shape': record['observations']['vector']=np.ones((2,134),np.float32)
    if kind=='nonfinite': record['actions'][0][0]=float('nan')
    if kind=='reserved': record['observations']['metadata']=np.ones((3,1),np.float32)
    with pytest.raises(ValueError):write_episode(path,record)
    assert path.read_bytes()==before
    assert not list(tmp_path.glob('*.tmp'))


def test_untrusted_npz_conversion_rejects_pickle_and_expanded_archive_limit(tmp_path):
    from episode_io import load_replay_bytes
    stream=io.BytesIO();np.savez(stream,replay_json=np.asarray([object()],dtype=object))
    with pytest.raises(ValueError):load_replay_bytes(stream.getvalue())
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('payload.npy',b'x'*10000)
    with pytest.raises(ValueError,match='limit|large'):load_replay_bytes(stream.getvalue(),max_uncompressed_bytes=100)


@pytest.mark.parametrize('field',['actions','vector','metadata'])
def test_writer_rejects_schema_shape_or_task_mismatch(tmp_path,field):
    from episode_io import write_episode
    record=record_fixture()
    if field=='actions':
        record['metadata']['obs_schema']='obs-field-v2'
    elif field=='vector':record['observations']['vector']=np.zeros((3,133),np.float32)
    else:record['metadata']['task']='coverage'
    with pytest.raises(ValueError):write_episode(tmp_path/'invalid.npz',record)


def test_untrusted_import_checks_numpy_shapes_before_allocating(tmp_path):
    from episode_io import write_episode, load_replay_bytes
    path=write_episode(tmp_path/'episode.npz',record_fixture())
    original=zipfile.ZipFile(path);stream=io.BytesIO()
    forged=io.BytesIO()
    np.lib.format.write_array_header_1_0(forged,{'descr':'<f8','fortran_order':False,'shape':(10**12,2)})
    with zipfile.ZipFile(stream,'w') as archive:
        for entry in original.infolist():
            archive.writestr(entry.filename,forged.getvalue() if entry.filename=='actions.npy' else original.read(entry))
    original.close()
    with pytest.raises(ValueError,match='shape|size'):load_replay_bytes(stream.getvalue())


def test_untrusted_import_rejects_pickled_observations_even_when_only_replaying(tmp_path):
    from episode_io import write_episode, load_replay_bytes
    path=write_episode(tmp_path/'episode.npz',record_fixture())
    with np.load(path,allow_pickle=False) as data:arrays={key:data[key] for key in data.files}
    arrays['vector']=np.array([object()],dtype=object)
    stream=io.BytesIO();np.savez(stream,**arrays)
    with pytest.raises(ValueError,match='pickle|object'):load_replay_bytes(stream.getvalue())
