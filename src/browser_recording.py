"""Real frontend canvas capture and bounded, decoded MP4 publication.

The browser helper uses the same MediaRecorder bridge as the visible controls.
FFmpeg is supplied by the project's existing imageio-ffmpeg dependency.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from urllib.parse import quote, urljoin, urlsplit
from urllib.request import urlopen
import uuid

import imageio_ffmpeg

try:
    from .episode_io import load_replay
except ImportError:
    from episode_io import load_replay


RECORDING_DEFAULTS = {
    'recording_fps': 24, 'recording_width': 1440, 'recording_height': 900,
    'recording_duration_seconds': 30., 'recording_bitrate': 8000000,
    'recording_max_bytes': 268435456, 'recording_max_seconds': 600.,
}


def recording_configuration(values=None):
    result = dict(RECORDING_DEFAULTS)
    for key, value in (values or {}).items():
        name = key if key.startswith('recording_') else 'recording_' + key
        if name in result:
            result[name] = value
    for key in result:
        value = float(result[key])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f'{key} must be positive and finite')
        result[key] = value if key.endswith('seconds') else int(value)
    if result['recording_width'] % 2 or result['recording_height'] % 2:
        raise ValueError('recording width and height must be even for H.264')
    if result['recording_duration_seconds'] > result['recording_max_seconds']:
        raise ValueError('recording duration exceeds recording_max_seconds')
    return result


def inspect_mp4(path):
    """Read container/stream metadata and decode every video frame."""
    path = Path(path)
    with path.open('rb') as stream:
        if stream.read(12)[4:8] != b'ftyp':
            raise ValueError('FFmpeg did not produce an actual MP4 container')
    frames = imageio_ffmpeg.read_frames(str(path), pix_fmt='rgb24')
    try:
        metadata = next(frames)
        count = sum(1 for _ in frames)
    finally:
        frames.close()
    if count < 1:
        raise ValueError('Video contains no decodable frames')
    width, height = metadata['size']
    fps = float(metadata['fps'])
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return {'container': 'mp4', 'codec': metadata.get('codec'), 'width': width,
            'height': height, 'fps': fps, 'frames': count,
            'duration_seconds': count / fps, 'bytes': path.stat().st_size,
            'sha256': digest.hexdigest()}


def transcode_recording(source, directory, config, provenance):
    """Publish only successfully decoded H.264 MP4s, never renamed WebM."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if directory.is_symlink() or directory.resolve().parent != directory.parent.resolve():
        raise ValueError('Unsafe recordings directory')
    name = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:12]}"
    final = directory / (name + '.mp4')
    temporary = directory / (name + '.partial.mp4')
    metadata_path = final.with_suffix('.json')
    maximum = config['recording_max_seconds']
    command = [imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-v', 'error', '-nostdin',
        '-y', '-i', str(source), '-map', '0:v:0', '-an', '-t', str(maximum + 1),
        '-vf', f"scale={config['recording_width']}:{config['recording_height']},fps={config['recording_fps']}",
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-b:v', str(config['recording_bitrate']),
        '-movflags', '+faststart', str(temporary)]
    try:
        result = subprocess.run(command, capture_output=True, timeout=max(30, maximum * 2))
        if result.returncode:
            detail = result.stderr.decode('utf-8', errors='replace')[-1500:]
            raise ValueError(f'FFmpeg could not decode/convert the uploaded video: {detail}')
        details = inspect_mp4(temporary)
        if details['duration_seconds'] > maximum + 1 / config['recording_fps']:
            raise ValueError('Uploaded video exceeds recording_max_seconds')
        if details['bytes'] > config['recording_max_bytes']:
            raise ValueError('Converted MP4 exceeds recording_max_bytes')
        temporary.replace(final)
        record = dict(details, path=str(final.resolve()), metadata_path=str(metadata_path.resolve()),
                      url='/outputs/logs/recordings/' + final.name, provenance=provenance,
                      created_at=datetime.now(timezone.utc).isoformat())
        metadata_path.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        return record
    except Exception:
        temporary.unlink(missing_ok=True)
        final.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)
        raise


def _playwright_runtime():
    """Prefer project/portable installations, then discover existing npm caches."""
    node = shutil.which('node')
    if not node:
        raise RuntimeError('Frontend recording requires Node.js and Playwright. Install Node.js, then run npm install playwright and npx playwright install chromium.')
    explicit = os.environ.get('PLAYWRIGHT_MODULE_PATH')
    if explicit and Path(explicit).exists():
        return node, str(Path(explicit).resolve())
    for module in ('playwright', 'playwright-core'):
        resolved = subprocess.run([node, '-p', f'require.resolve({json.dumps(module)})'],
            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=10)
        if resolved.returncode == 0:
            return node, resolved.stdout.strip()
    roots = [Path.home() / '.npm' / '_npx', Path.home() / 'AppData' / 'Local' / 'npm-cache' / '_npx']
    if os.environ.get('npm_config_cache'):
        roots.insert(0, Path(os.environ['npm_config_cache']) / '_npx')
    if os.environ.get('LOCALAPPDATA'):
        roots.insert(0, Path(os.environ['LOCALAPPDATA']) / 'npm-cache' / '_npx')
    for root in dict.fromkeys(roots):
        for package in ('playwright', 'playwright-core'):
            matches = sorted(root.glob(f'*/node_modules/{package}/package.json'),
                             key=lambda item: item.stat().st_mtime, reverse=True)
            if matches:
                return node, str(matches[0].parent)
    raise RuntimeError('Frontend recording requires Playwright. Run npm install playwright; use existing Chrome/Edge or run npx playwright install chromium. PLAYWRIGHT_MODULE_PATH can select a portable installation.')


_CAPTURE_SCRIPT = r"""
const fs=require('fs');
const job=JSON.parse(fs.readFileSync(process.argv[1],'utf8'));
const {chromium}=require(job.module);
(async()=>{
  let browser, failures=[];
  for(const channel of ['chrome','msedge',null]) {
    try {browser=await chromium.launch({headless:job.headless,...(channel?{channel}:{}),args:['--disable-background-timer-throttling','--disable-renderer-backgrounding']});break;}
    catch(error){failures.push(String(error.message));}
  }
  if(!browser)throw new Error('Cannot launch installed Chrome/Edge/Chromium: '+failures.join('\n'));
  try {
    const page=await browser.newPage({viewport:{width:job.width,height:job.height},deviceScaleFactor:1});
    const errors=[];page.on('pageerror',error=>errors.push(String(error.message)));
    await page.goto(job.server_url,{waitUntil:'domcontentloaded',timeout:60000});
    await page.waitForFunction(()=>Boolean(window.maritimeRecording),null,{timeout:60000});
    const config=await page.evaluate(()=>window.maritimeRecording.ready);
    for(const [key,value] of Object.entries({recording_fps:job.fps,recording_width:job.width,recording_height:job.height,recording_duration_seconds:job.duration_seconds})){
      if(config?.[key]!==value)throw new Error('Recording configuration mismatch for '+key+': server='+config?.[key]+', requested='+value);
    }
    await page.evaluate(url=>window.maritimeRecording.load(url),job.replay_url);
    await page.evaluate(async()=>{await document.fonts.ready;await Promise.all([...document.images].map(image=>image.decode().catch(()=>{})));});
    const result=await page.evaluate(async provenance=>window.maritimeRecording.recordReplay({provenance}),job.provenance);
    if(errors.length)throw new Error('Frontend runtime error: '+errors.join('; '));
    process.stdout.write(JSON.stringify(result));
  } finally {await browser.close();}
})().catch(error=>{process.stderr.write(String(error.stack||error));process.exitCode=1;});
"""


def record_policy_replay(server_url, replay_path, output_dir, *, fps, width, height,
                         duration_seconds, headless=True):
    """Record a genuine policy replay through the visible UI facility.

    ``output_dir`` must be the serving VisualServer's outputs root. The replay
    must be inside it; the returned MP4 is under ``output_dir/logs/recordings``.
    Parameters must match ``VisualServer(recording_config=...)`` so automated
    capture cannot silently diverge from the UI's exposed settings. Install
    Playwright with ``npm install playwright`` if no portable/cached runtime is
    available; installed Chrome/Edge are reused before bundled Chromium.
    """
    output_dir = Path(output_dir).resolve()
    replay_path = Path(replay_path).resolve()
    if not replay_path.is_relative_to(output_dir):
        raise ValueError('Replay must be under the VisualServer output_dir')
    if urlsplit(server_url).scheme not in ('http', 'https'):
        raise ValueError('server_url must be an HTTP URL')
    replay_bytes = replay_path.read_bytes()
    replay = load_replay(replay_path)
    if replay.get('actor') not in ('policy', 'network'):
        raise ValueError('Automatic latest-policy recording requires a policy replay; expert substitution is forbidden')
    if ((replay.get('metadata') or {}).get('behavior') == 'dagger_training'
            or (replay.get('summary') or {}).get('expert_control_fraction', 0) > 0):
        raise ValueError('Automatic policy recording forbids an expert/policy training mixture')
    frames = replay.get('frames')
    if not frames:
        raise ValueError('Policy replay contains no actual simulation frames')
    times = [frame.get('t') for frame in frames]
    if any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in times) or times != sorted(times):
        raise ValueError('Policy replay times must be finite and nondecreasing')
    metadata = {key: value for key, value in (replay.get('metadata') or {}).items()
                if key not in ('config', 'config_snapshot')}
    checkpoint = metadata.get('policy_checkpoint') or {}
    provenance = dict(metadata, task=replay['task'], actor='policy',
        model_sha256=metadata.get('model_sha256') or checkpoint.get('sha256'),
        model_checkpoint=metadata.get('model_checkpoint') or checkpoint.get('path'),
        replay_path=str(replay_path), replay_sha256=hashlib.sha256(replay_bytes).hexdigest(),
        simulation_start_seconds=times[0], simulation_end_seconds=times[-1],
        source_frame_count=len(frames), outcome=replay.get('summary') or {})
    node, module = _playwright_runtime()
    config_url = urljoin(server_url.rstrip('/') + '/', 'api/recording-config')
    with urlopen(config_url, timeout=15) as response:
        config = json.load(response)
    expected = {'recording_fps': fps, 'recording_width': width, 'recording_height': height,
                'recording_duration_seconds': duration_seconds}
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError(f'Automated recording settings must match server recording_config: {expected}')
    job = dict(module=module, server_url=server_url,
        replay_url='/outputs/' + quote(replay_path.relative_to(output_dir).as_posix(), safe='/'),
        fps=fps, width=width, height=height, duration_seconds=duration_seconds,
        headless=bool(headless), provenance=provenance)
    cache = output_dir / 'logs' / 'cache'
    if cache.resolve()!=cache or cache.is_symlink():
        raise ValueError('Unsafe recording cache directory')
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='browser-recording-', dir=cache) as temporary:
        job_path = Path(temporary) / 'job.json'
        job_path.write_text(json.dumps(job, ensure_ascii=False, allow_nan=False), encoding='utf-8')
        process = subprocess.run([node, '-e', _CAPTURE_SCRIPT, str(job_path)],
            capture_output=True, text=True, encoding='utf-8',
            timeout=duration_seconds + config['recording_max_seconds'] * 2 + 90)
    if process.returncode:
        raise RuntimeError(f'Frontend policy recording failed: {process.stderr[-6000:]}')
    try:
        record = json.loads(process.stdout)
    except ValueError as error:
        raise RuntimeError(f'Frontend recording returned invalid metadata: {process.stdout[-1000:]}') from error
    path = Path(record['path']).resolve()
    if path.parent != output_dir / 'logs' / 'recordings':
        raise RuntimeError('Frontend recording returned a path outside logs/recordings')
    decoded = inspect_mp4(path)
    if decoded['width'] != width or decoded['height'] != height or abs(decoded['fps'] - fps) > .01:
        raise RuntimeError('Decoded MP4 dimensions/fps disagree with requested recording settings')
    if abs(decoded['duration_seconds'] - duration_seconds) > max(.25, 2 / fps):
        raise RuntimeError(f"Decoded MP4 duration {decoded['duration_seconds']} disagrees with requested {duration_seconds}")
    recorded_end = record['provenance'].get('recorded_simulation_end_seconds')
    if recorded_end != times[-1]:
        raise RuntimeError('Frontend recording did not reach the final real simulation frame')
    return record
