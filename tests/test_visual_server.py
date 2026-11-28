"""Integration tests for the read-only local visualization service."""
import http.client
import json
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest


@pytest.fixture
def visual_server(tmp_path):
    from src.visual_server import VisualServer

    (tmp_path / "visual").mkdir()
    (tmp_path / "visual" / "index.html").write_text("<h1>Maritime</h1>", encoding="utf-8")
    (tmp_path / "figs").mkdir()
    (tmp_path / "figs" / "sample.txt").write_text("asset", encoding="utf-8")
    output = tmp_path / "outputs"
    (output / "run").mkdir(parents=True)
    replay = {"schema_version": "maritime-replay-v1", "task": "coverage", "actor": "expert", "seed": 7, "scene": {}, "frames": [{"t": 0}], "summary": {}}
    (output / "run" / "coverage_expert.json").write_text(json.dumps(replay), encoding="utf-8")
    (output / "metrics.json").write_text('{"loss": 0.1}', encoding="utf-8")
    (tmp_path / "private.txt").write_text("private", encoding="utf-8")
    service = VisualServer(tmp_path, output, port=0).start()
    yield service
    service.close()


def test_static_files_and_replay_catalog(visual_server):
    with urlopen(visual_server.url) as response:
        assert response.status == 200
        assert b"Maritime" in response.read()
    with urlopen(visual_server.url + "api/replays") as response:
        result = json.load(response)
    assert len(result["replays"]) == 1
    item = result["replays"][0]
    assert item["task"] == "coverage"
    assert item["actor"] == "expert"
    with urlopen(visual_server.url.rstrip("/") + item["url"]) as response:
        assert json.load(response)["seed"] == 7


def test_current_visual_configuration_is_available_without_changing_replay_physics(tmp_path):
    from src.visual_server import VisualServer
    service=VisualServer(tmp_path,tmp_path/'outputs',port=0,
        visual_config={'visual_ship_length':83.,'visual_show_trail':False,'tracking_range':99999.}).start()
    try:
        with urlopen(service.url+'api/visual-config') as response:
            configuration=json.load(response)
        assert configuration=={'visual_ship_length':83.,'visual_show_trail':False}
    finally:service.close()


def test_busy_port_cannot_be_silently_taken_over(visual_server):
    from src.visual_server import VisualServer
    second=VisualServer(visual_server.project_root,visual_server.output_dir,port=visual_server.port)
    try:
        with pytest.raises(OSError):second.start()
    finally:second.close()


@pytest.mark.parametrize("path", ["/private.txt", "/figs/../private.txt", "/figs/%2e%2e/private.txt", "/outputs/%2e%2e/private.txt", "/figs/%5c..%5cprivate.txt"])
def test_path_traversal_is_not_served(visual_server, path):
    with pytest.raises(HTTPError) as error:
        urlopen(visual_server.url.rstrip("/") + path)
    assert error.value.code in (403, 404)


def test_sse_delivers_scene_and_frame_and_caches_latest(visual_server):
    visual_server.publish({"type": "scene", "task": "tracking", "actor": "policy", "scene": {"world_bounds": [0, 0, 300000, 300000]}})
    visual_server.publish({"type": "frame", "frame": {"t": 1, "uav": {"x": 4, "y": 5}}})
    connection = http.client.HTTPConnection(visual_server.host, visual_server.port, timeout=3)
    connection.request("GET", "/events")
    response = connection.getresponse()
    assert response.status == 200
    assert response.getheader("Content-Type").startswith("text/event-stream")
    events = []
    while len(events) < 2:
        line = response.readline().decode("utf-8")
        if line.startswith("data: "):
            events.append(json.loads(line[6:]))
    assert events[0]["type"] == "scene"
    assert events[1]["frame"]["t"] == 1
    visual_server.publish({"t": 2, "uav": {"x": 6, "y": 7}})
    while True:
        line = response.readline().decode("utf-8")
        if line.startswith("data: "):
            assert json.loads(line[6:])["frame"]["t"] == 2
            break
    response.close()
    connection.close()


def test_read_only_methods(visual_server):
    connection = http.client.HTTPConnection(visual_server.host, visual_server.port, timeout=3)
    connection.request("POST", "/outputs/new.json", body=b"{}")
    response = connection.getresponse()
    assert response.status in (405, 501)
    response.close()
    connection.close()


def test_replay_catalog_excludes_cache_and_test_artifacts(visual_server):
    fake = {"task": "coverage", "actor": "expert", "scene": {}, "frames": [{"t": 0}]}
    for folder in ("cache/pytest-tmp", ".pytest_cache", "logs"):
        directory = visual_server.output_dir / folder
        directory.mkdir(parents=True)
        (directory / "synthetic-replay.json").write_text(json.dumps(fake), encoding="utf-8")
    with urlopen(visual_server.url + "api/replays") as response:
        catalog = json.load(response)["replays"]
    assert len(catalog) == 1
    assert catalog[0]["name"] == "run/coverage_expert.json"


def test_large_legacy_replays_are_indexed_once_without_blocking_http(visual_server,monkeypatch):
    import threading
    from time import monotonic
    output=visual_server.output_dir
    replay={'schema_version':'maritime-replay-v1','task':'tracking','actor':'policy',
            'scene':{},'frames':[{'t':17}],'padding':'x'*(3*1024*1024)}
    path=output/'large.json'
    # Valid legacy JSON can place schema after large payloads; order is not a schema.
    path.write_text(json.dumps(replay,sort_keys=True),encoding='utf-8')
    original=visual_server._read_replay_metadata
    started=threading.Event();release=threading.Event();calls=[]
    def blocked(file,stat):
        if file==path:
            calls.append(file);started.set();release.wait(5)
        return original(file,stat)
    monkeypatch.setattr(visual_server,'_read_replay_metadata',blocked)
    try:
        start=monotonic()
        with urlopen(visual_server.url+'api/replays',timeout=2) as response:
            assert len(json.load(response)['replays'])==1
        assert monotonic()-start<2
        assert started.wait(2)
        with urlopen(visual_server.url+'api/replays',timeout=2) as response:
            assert len(json.load(response)['replays'])==1
        assert len(calls)==1
    finally:release.set()
    visual_server._catalog_thread.join(5)
    assert path.with_suffix('.replay-index.json').is_file()
    visual_server._catalog_cache.clear()
    with urlopen(visual_server.url+'api/replays',timeout=2) as response:
        entries=json.load(response)['replays']
    assert len(entries)==2 and len(calls)==1


@pytest.mark.parametrize('sidecar_state',['missing','expired'])
@pytest.mark.parametrize('metadata',[
    pytest.param({'control_mode':'forecast_constraint'},id='constrained'),
    pytest.param({},id='pure'),
    pytest.param({'behavior':'dagger_training','mixture_beta':.5},id='dagger'),
])
def test_rebuilt_replay_index_preserves_control_mode_and_existing_entry_fields(visual_server,sidecar_state,metadata):
    path=visual_server.output_dir/'copied-policy.json'
    replay={'task':'coverage','actor':'policy','seed':19,'scene':{},
            'metadata':metadata,'frames':[{'t':0},{'t':60}]}
    path.write_text(json.dumps(replay),encoding='utf-8')
    index=path.with_suffix('.replay-index.json')
    if sidecar_state=='expired':
        index.write_text(json.dumps({'schema_version':'maritime-replay-index-v1',
            'fingerprint':[0,0],'entry':{'task':'coverage','actor':'expert','frames':999}}),encoding='utf-8')
    with urlopen(visual_server.url+'api/replays') as response:
        item=next(row for row in json.load(response)['replays'] if row['name']=='copied-policy.json')
    expected={'task':'coverage','actor':'policy','seed':19,'frames':2,'duration':60,
              'behavior':metadata.get('behavior')}
    if 'control_mode' in metadata:expected['control_mode']=metadata['control_mode']
    rebuilt=json.loads(index.read_text(encoding='utf-8'))
    assert rebuilt['entry']==expected
    stat=path.stat()
    assert rebuilt['fingerprint']==[stat.st_mtime_ns,stat.st_size]
    assert item==dict(expected,name='copied-policy.json',url='/outputs/copied-policy.json',
                      modified=stat.st_mtime,bytes=stat.st_size)


def test_npz_catalog_and_http_convert_replay_without_training_arrays(visual_server):
    import numpy as np
    from episode_io import write_episode
    record={'task':'tracking','actor':'policy','scene':{},'frames':[{'t':0},{'t':1}],
            'metadata':{'task':'tracking'},'actions':[[1,0]],'executed_actions':[[1,0]],
            'observations':{'vector':np.ones((2,134),np.float32)}}
    path=write_episode(visual_server.output_dir/'eval'/'policy.npz',record)
    with urlopen(visual_server.url+'api/replays') as response:
        item=next(row for row in json.load(response)['replays'] if row['name']=='eval/policy.npz')
    with urlopen(visual_server.url.rstrip('/')+item['url']) as response:
        assert response.headers['Content-Type'].startswith('application/json')
        import hashlib
        assert response.headers['X-Replay-SHA256']==hashlib.sha256(path.read_bytes()).hexdigest()
        replay=json.load(response)
    assert replay['actions']==[[1,0]] and 'observations' not in replay
    assert not path.with_suffix('.json').exists()


def test_local_npz_import_is_bounded_same_origin_and_does_not_persist(visual_server):
    from episode_io import write_episode
    path=write_episode(visual_server.output_dir/'source.npz',
        {'task':'coverage','actor':'expert','scene':{},'frames':[{'t':0}]})
    def upload(body,**headers):
        connection=http.client.HTTPConnection(visual_server.host,visual_server.port,timeout=3)
        connection.request('POST','/api/replays/convert',body=body,
            headers={'Content-Type':'application/octet-stream',**headers})
        response=connection.getresponse();result=response.status,json.loads(response.read());connection.close()
        return result
    before=set(visual_server.output_dir.rglob('*'))
    status,replay=upload(path.read_bytes())
    assert status==200 and replay['actor']=='expert' and replay['frames']==[{'t':0}]
    assert set(visual_server.output_dir.rglob('*'))==before
    assert upload(b'not zip')[0]==400
    assert upload(b'x',Origin='https://other.invalid')[0]==403
    assert upload(b'',**{'Content-Length':str(128*1024*1024+1)})[0]==413


def test_import_concurrency_is_bounded_before_reading_request_body(visual_server):
    assert visual_server._replay_import_slot.acquire(blocking=False)
    try:
        connection=http.client.HTTPConnection(visual_server.host,visual_server.port,timeout=3)
        connection.request('POST','/api/replays/convert',body=b'x',headers={'Content-Type':'application/octet-stream'})
        response=connection.getresponse()
        assert response.status==429 and 'conversion' in json.loads(response.read())['error'].lower()
        connection.close()
    finally:visual_server._replay_import_slot.release()
