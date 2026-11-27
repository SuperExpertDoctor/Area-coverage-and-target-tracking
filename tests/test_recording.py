"""Bounded recording uploads and actual MP4 authenticity."""
import http.client
import json
from pathlib import Path
import subprocess
from urllib.parse import quote
from urllib.request import urlopen

import imageio_ffmpeg
import pytest

from src.visual_server import VisualServer


@pytest.fixture
def service(tmp_path):
    server = VisualServer(Path(__file__).resolve().parents[1], tmp_path / 'outputs', port=0,
        recording_config={'recording_width': 320, 'recording_height': 240,
                          'recording_fps': 10, 'recording_duration_seconds': 1,
                          'recording_max_bytes': 1000000, 'recording_max_seconds': 3}).start()
    yield server
    server.close()


def upload(service, body, mime='video/webm', metadata=None, **headers):
    connection = http.client.HTTPConnection(service.host, service.port, timeout=30)
    values = {'Content-Type': mime, 'X-Recording-Metadata': quote(json.dumps(metadata or {}))}
    values.update(headers)
    connection.request('POST', '/api/recordings', body=body, headers=values)
    response = connection.getresponse()
    result = response.status, json.loads(response.read())
    connection.close()
    return result


def test_recording_configuration_and_bounds(service):
    with urlopen(service.url + 'api/recording-config') as response:
        config = json.load(response)
    assert config['recording_width'] == 320
    assert config['recording_max_bytes'] == 1000000
    assert upload(service, b'x' * 1000001)[0] == 413
    assert upload(service, b'x', mime='text/plain')[0] == 415
    assert upload(service, b'x', Origin='https://unrelated.invalid')[0] == 403
    assert upload(service, b'x', metadata={'duration_seconds': 9})[0] == 400
    assert not list((service.output_dir / 'recordings').glob('*.mp4'))


def test_corrupt_video_has_meaningful_error_and_no_fake_mp4(service):
    status, result = upload(service, b'not a real webm')
    assert status == 422
    assert 'video' in result['error'].lower() or 'ffmpeg' in result['error'].lower()
    assert not list((service.output_dir / 'recordings').glob('*.mp4'))


def test_failed_ffmpeg_upload_returns_structured_error(service, monkeypatch):
    import src.browser_recording as recording
    monkeypatch.setattr(recording.subprocess, 'run', lambda *a, **kw: (_ for _ in ()).throw(
        subprocess.TimeoutExpired('ffmpeg', 1)))
    status, result = upload(service, b'not video')
    assert status == 422 and 'timed out' in result['error'].lower()


def test_run_status_missing_then_persisted_then_published(service):
    with urlopen(service.url + 'api/run-status') as response:
        assert json.load(response) == {}
    service.output_dir.mkdir(exist_ok=True)
    status = {'phase': 'training', 'task': 'tracking', 'tasks': {
        'tracking': {'training': {'epoch': 7, 'epochs': 20, 'validation_loss': .4}}}}
    (service.output_dir / 'formal-status.json').write_text(json.dumps(status), encoding='utf-8')
    with urlopen(service.url + 'api/run-status') as response:
        assert json.load(response) == status


def test_real_browser_policy_capture_decodes_and_preserves_failure(service):
    from src.browser_recording import _playwright_runtime, record_policy_replay
    try:
        _playwright_runtime()
    except RuntimeError as error:
        pytest.skip(str(error))
    service.output_dir.mkdir(exist_ok=True)
    replay = {'task': 'tracking', 'actor': 'policy', 'seed': 7,
        'metadata': {'model_sha256': 'a' * 64, 'model_checkpoint': 'models/tracking/last.pt',
                     'control_mode': 'forecast_constraint'},
        'scene': {'world_bounds': [0, 0, 300000, 300000], 'clouds': [], 'islands': []},
        'frames': [{'t': t, 'uav': {'x': 60000 + t, 'y': 70000, 'vx': 61, 'vy': 0},
                    'ship': {'x': 62000, 'y': 70000, 'vx': 6, 'vy': 0},
                    'metrics': {'distance': 2000}, 'radar': {'range': 3000, 'half_angle': .5}}
                   for t in (0, 600, 1200, 1800)],
        'summary': {'terminated': True, 'termination_reason': 'test_failure'}}
    source = service.output_dir / 'tracking-latest.json'
    source.write_text(json.dumps(replay), encoding='utf-8')
    result = record_policy_replay(service.url, source, service.output_dir,
        fps=10, width=320, height=240, duration_seconds=1)
    assert result['provenance']['simulation_end_seconds'] == 1800
    assert result['provenance']['recorded_simulation_end_seconds'] == 1800
    assert result['provenance']['outcome']['termination_reason'] == 'test_failure'
    assert result['provenance']['model_sha256'] == 'a' * 64
    assert result['provenance']['control_mode'] == 'forecast_constraint'


def test_automatic_capture_rejects_expert_and_dagger_substitution(tmp_path):
    from src.browser_recording import record_policy_replay
    for actor, metadata in [('expert', {}), ('policy', {'behavior': 'dagger_training'})]:
        source = tmp_path / 'invalid.json'
        source.write_text(json.dumps({'actor': actor, 'task': 'coverage',
            'metadata': metadata, 'frames': [{'t': 0}]}), encoding='utf-8')
        with pytest.raises(ValueError, match='expert|mixture'):
            record_policy_replay('http://127.0.0.1:1/', source, tmp_path,
                fps=24, width=1440, height=900, duration_seconds=30)


def test_webm_is_decoded_transcoded_and_paths_cannot_escape(service, tmp_path):
    source = tmp_path / 'source.webm'
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-v', 'error', '-f', 'lavfi', '-i',
        'testsrc2=size=320x240:rate=10', '-t', '1', '-c:v', 'libvpx-vp9', str(source)],
        check=True, capture_output=True, timeout=30)
    status, result = upload(service, source.read_bytes(), metadata={
        'name': '../../escape', 'task': 'coverage', 'actor': 'policy',
        'model_sha256': 'f' * 64, 'duration_seconds': 1})
    assert status == 201, result
    saved = Path(result['path'])
    assert saved.resolve().parent == (service.output_dir / 'recordings').resolve()
    assert saved.suffix == '.mp4' and saved.read_bytes()[4:8] == b'ftyp'
    assert result['container'] == 'mp4' and result['width'] == 320 and result['height'] == 240
    assert result['fps'] == 10 and .8 < result['duration_seconds'] < 1.2
    assert result['provenance']['model_sha256'] == 'f' * 64
    assert Path(result['metadata_path']).is_file()
    decoded = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-v', 'error', '-i',
        str(saved), '-f', 'null', '-'], capture_output=True, timeout=30)
    assert decoded.returncode == 0, decoded.stderr
    with urlopen(service.url.rstrip('/') + result['url']) as response:
        assert response.read(12)[4:8] == b'ftyp'


def test_recording_directory_symlink_is_rejected(service, tmp_path):
    directory = service.output_dir / 'recordings'
    directory.parent.mkdir(parents=True, exist_ok=True)
    target = tmp_path / 'elsewhere'
    target.mkdir()
    try:
        directory.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip('Creating symlinks requires Windows developer mode')
    assert upload(service, b'invalid')[0] == 400
    assert list(target.iterdir()) == []
