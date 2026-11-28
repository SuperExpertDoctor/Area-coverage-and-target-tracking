"""Local static server, bounded recording uploads and SSE broadcast.

The simulation owns all numerical state. This module only transports published
scene/frame dictionaries and serves JSON views of recorded NPZ and legacy JSON replays and frontend assets.
It uses the Python standard library and is safe to leave running during training.
"""
from __future__ import annotations

from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hashlib
import json
import mimetypes
import os
from pathlib import Path
import queue
import socket
import subprocess
import threading
import tempfile
from urllib.parse import quote, unquote, urlsplit

try:
    from .browser_recording import recording_configuration, transcode_recording
    from .episode_io import load_replay, load_replay_bytes
except ImportError:  # main.py also supports the existing top-level src imports.
    from browser_recording import recording_configuration, transcode_recording
    from episode_io import load_replay, load_replay_bytes


class _ExclusiveHTTPServer(ThreadingHTTPServer):
    allow_reuse_address=False

    def server_bind(self):
        if hasattr(socket,'SO_EXCLUSIVEADDRUSE'):
            self.socket.setsockopt(socket.SOL_SOCKET,socket.SO_EXCLUSIVEADDRUSE,1)
        super().server_bind()


def _json_default(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


class VisualServer:
    """Threaded localhost visualizer; ``publish`` never waits for browsers.

    ``VisualServer(root, outputs, port=8787).start()`` starts a background
    listener. Publish ``{'type': 'scene', 'scene': ..., 'task': ..., 'actor': ...}``
    at episode boundaries, then ``{'type': 'frame', 'frame': ...}`` per render.
    A bare frame dictionary is also accepted. ``close`` is idempotent.
    """

    def __init__(self, project_root, output_dir, host="127.0.0.1", port=8787, visual_config=None, *, recording_config=None):
        self.project_root = Path(project_root).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.host = host
        self.port = int(port)
        self.visual_config = {key:value for key,value in (visual_config or {}).items()
                              if key.startswith('visual_')}
        self.recording_config = recording_configuration(recording_config)
        self._recording_slot = threading.BoundedSemaphore(1)
        self._replay_import_slot = threading.BoundedSemaphore(1)
        self._lock = threading.RLock()
        self._clients = set()
        self._scene = None
        self._recent = deque(maxlen=48)
        self._http = None
        self._thread = None
        self._catalog_cache = {}
        self._catalog_lock = threading.RLock()
        self._catalog_thread = None

    @property
    def url(self):
        host = "127.0.0.1" if self.host in ("0.0.0.0", "::") else self.host
        return f"http://{host}:{self.port}/"

    def start(self):
        if self._http is not None:
            return self
        service = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_):
                pass

            def _headers(self, status, content_type, length=None, replay_sha256=None):
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Cache-Control", "no-cache")
                if length is not None:
                    self.send_header("Content-Length", str(length))
                if replay_sha256:
                    self.send_header("X-Replay-SHA256", replay_sha256)
                self.end_headers()

            def do_GET(self):
                self._serve(False)

            def do_HEAD(self):
                self._serve(True)

            def _json(self, status, value):
                data = json.dumps(value, ensure_ascii=False, allow_nan=False).encode('utf-8')
                self._headers(status, 'application/json; charset=utf-8', len(data))
                self.wfile.write(data)

            def do_POST(self):
                if urlsplit(self.path).path == '/api/replays/convert':
                    self._convert_replay()
                    return
                if urlsplit(self.path).path != '/api/recordings':
                    self.close_connection = True
                    self._json(405, {'error': 'Only POST /api/recordings is supported'})
                    return
                # This local application accepts uploads only from its own UI.
                origin = self.headers.get('Origin')
                if origin and urlsplit(origin).netloc != self.headers.get('Host'):
                    self.close_connection = True
                    self._json(403, {'error': 'Recording uploads must be same-origin'})
                    return
                try:
                    if self.headers.get('Transfer-Encoding'):
                        raise ValueError('Content-Length is required; chunked upload is unsupported')
                    length = int(self.headers.get('Content-Length', '0'))
                    if length <= 0:
                        raise ValueError('Recording upload is empty or Content-Length is missing')
                    if length > service.recording_config['recording_max_bytes']:
                        self.close_connection = True
                        self._json(413, {'error': 'Recording exceeds recording_max_bytes'})
                        return
                    mime = self.headers.get('Content-Type', '').split(';')[0].strip().lower()
                    if mime not in ('video/mp4', 'video/webm'):
                        self.close_connection = True
                        self._json(415, {'error': 'Upload an actual video/mp4 or video/webm recording'})
                        return
                    raw = self.headers.get('X-Recording-Metadata', '{}')
                    if len(raw) > 16384:
                        raise ValueError('Recording metadata is too large')
                    metadata = json.loads(unquote(raw))
                    if not isinstance(metadata, dict):
                        raise ValueError('Recording metadata must be a JSON object')
                    json.dumps(metadata, allow_nan=False)
                    if 'duration_seconds' in metadata:
                        duration = float(metadata['duration_seconds'])
                        if not 0 < duration <= service.recording_config['recording_max_seconds']:
                            raise ValueError('Recording duration exceeds recording_max_seconds')
                    directory = service.output_dir / 'logs' / 'recordings'
                    if directory.resolve() != service.output_dir / 'logs' / 'recordings' or directory.is_symlink():
                        raise ValueError('Unsafe recordings directory')
                except (ValueError, TypeError) as error:
                    self.close_connection = True
                    self._json(400, {'error': str(error)})
                    return
                if not service._recording_slot.acquire(blocking=False):
                    self.close_connection = True
                    self._json(429, {'error': 'A recording is already being saved; try again shortly'})
                    return
                try:
                    directory.mkdir(parents=True, exist_ok=True)
                    self.connection.settimeout(30)
                    with tempfile.TemporaryDirectory(prefix='.upload-', dir=directory) as temporary:
                        source = Path(temporary) / ('capture.mp4' if mime == 'video/mp4' else 'capture.webm')
                        remaining = length
                        with source.open('wb') as stream:
                            while remaining:
                                chunk = self.rfile.read(min(65536, remaining))
                                if not chunk:
                                    raise ValueError('Video upload ended before Content-Length bytes arrived')
                                stream.write(chunk)
                                remaining -= len(chunk)
                        record = transcode_recording(source, directory, service.recording_config, metadata)
                    self._json(201, record)
                except (OSError, ValueError, RuntimeError, TimeoutError, subprocess.SubprocessError) as error:
                    self.close_connection = True
                    self._json(422, {'error': str(error)})
                finally:
                    service._recording_slot.release()

            def _convert_replay(self):
                # Conversion is memory-only, same-origin and bounded both before
                # upload and before compressed NPZ entries are expanded.
                origin=self.headers.get('Origin')
                if origin and urlsplit(origin).netloc!=self.headers.get('Host'):
                    self.close_connection=True
                    self._json(403,{'error':'Replay imports must be same-origin'})
                    return
                if not service._replay_import_slot.acquire(blocking=False):
                    self.close_connection=True
                    self._json(429,{'error':'A replay conversion is already in progress'})
                    return
                try:
                    if self.headers.get('Transfer-Encoding'):
                        raise ValueError('Content-Length is required')
                    length=int(self.headers.get('Content-Length','0'))
                    if length<=0: raise ValueError('Replay upload is empty')
                    if length>128*1024*1024:
                        self.close_connection=True
                        self._json(413,{'error':'Replay exceeds the 128 MiB import limit'})
                        return
                    if self.headers.get('Content-Type','').split(';')[0].strip().lower() not in (
                            'application/octet-stream','application/zip','application/x-npz'):
                        raise ValueError('Upload an NPZ archive')
                    self.connection.settimeout(30)
                    payload=self.rfile.read(length)
                    if len(payload)!=length: raise ValueError('Replay upload ended early')
                    replay=load_replay_bytes(payload)
                    if not isinstance(replay.get('frames'),list) or 'scene' not in replay:
                        raise ValueError('Archive does not contain a replay')
                    data=json.dumps(replay,ensure_ascii=False,allow_nan=False).encode('utf-8')
                    self._headers(200,'application/json; charset=utf-8',len(data),
                                  hashlib.sha256(payload).hexdigest())
                    self.wfile.write(data)
                except (OSError,ValueError,TypeError,KeyError) as error:
                    self.close_connection=True
                    self._json(400,{'error':str(error)})
                finally:
                    service._replay_import_slot.release()

            def _serve(self, head):
                path = unquote(urlsplit(self.path).path)
                if path == "/events" and not head:
                    self._events()
                    return
                if path == "/api/visual-config":
                    data=json.dumps(service.visual_config,ensure_ascii=False,allow_nan=False).encode('utf-8')
                    self._headers(200,'application/json; charset=utf-8',len(data))
                    if not head:self.wfile.write(data)
                    return
                if path == '/api/recording-config':
                    config = dict(service.recording_config,
                                  output_directory=str(service.output_dir / 'logs' / 'recordings'))
                    data = json.dumps(config, ensure_ascii=False, allow_nan=False).encode('utf-8')
                    self._headers(200, 'application/json; charset=utf-8', len(data))
                    if not head:self.wfile.write(data)
                    return
                if path == '/api/run-status':
                    try:
                        status_file = service.output_dir / 'logs' / 'formal-status.json'
                        if not status_file.is_file():
                            status_file = service.output_dir / 'formal-status.json'
                        run_status = json.loads(status_file.read_text(encoding='utf-8')) if status_file.is_file() else {}
                    except (OSError, ValueError):
                        run_status = {}
                    data = json.dumps(run_status, ensure_ascii=False, allow_nan=False).encode('utf-8')
                    self._headers(200, 'application/json; charset=utf-8', len(data))
                    if not head:self.wfile.write(data)
                    return
                if path == "/api/replays":
                    data = json.dumps({"replays": service._replays()}, ensure_ascii=False).encode("utf-8")
                    self._headers(200, "application/json; charset=utf-8", len(data))
                    if not head:
                        self.wfile.write(data)
                    return
                file = service._static_path(path)
                if file is None or not file.is_file():
                    self.send_error(404)
                    return
                if file.suffix.lower()=='.npz':
                    try:
                        replay=load_replay(file)
                        if not isinstance(replay.get('frames'),list) or 'scene' not in replay:
                            raise ValueError('NPZ does not contain a replay')
                        data=json.dumps(replay,ensure_ascii=False,allow_nan=False).encode('utf-8')
                        with file.open('rb') as stream:
                            checksum=hashlib.sha256()
                            for chunk in iter(lambda:stream.read(1024*1024),b''): checksum.update(chunk)
                        self._headers(200,'application/json; charset=utf-8',len(data),checksum.hexdigest())
                        if not head:self.wfile.write(data)
                    except (OSError,ValueError,KeyError,TypeError):self.send_error(422,'Invalid NPZ replay')
                    return
                try:
                    with file.open("rb") as stream:
                        mime = mimetypes.guess_type(file.name)[0] or "application/octet-stream"
                        if mime.startswith("text/") or mime in ("application/json", "application/javascript"):
                            mime += "; charset=utf-8"
                        self._headers(200, mime, file.stat().st_size)
                        if not head:
                            while chunk := stream.read(65536):
                                self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass

            def _events(self):
                client = queue.Queue(maxsize=64)
                with service._lock:
                    if len(service._clients) >= 32:
                        self.send_error(503, "Too many event streams")
                        return
                    if service._scene is not None:
                        client.put_nowait(service._scene)
                    for event in service._recent:
                        client.put_nowait(event)
                    service._clients.add(client)
                self.connection.settimeout(3)
                self._headers(200, "text/event-stream; charset=utf-8")
                try:
                    self.wfile.write(b": maritime stream\nretry: 2000\n\n")
                    self.wfile.flush()
                    while service._http is not None:
                        try:
                            item = client.get(timeout=1)
                        except queue.Empty:
                            self.wfile.write(b": heartbeat\n\n")
                        else:
                            if item is None:
                                break
                            self.wfile.write(b"data: " + item + b"\n\n")
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, TimeoutError, OSError):
                    pass
                finally:
                    with service._lock:
                        service._clients.discard(client)
                    self.close_connection = True

        self._http = _ExclusiveHTTPServer((self.host, self.port), Handler)
        self._http.daemon_threads = True
        self._http.block_on_close = False
        self.port = self._http.server_address[1]
        self._thread = threading.Thread(target=self._http.serve_forever, name="maritime-visual-http", daemon=True)
        self._thread.start()
        return self

    def _static_path(self, path):
        # Decode before checking; both separators must be rejected on Windows.
        if "\\" in path or "\x00" in path or any(part == ".." for part in path.split("/")):
            return None
        parts = path.strip("/").split("/")
        if path in ("/", "/index.html"):
            root, relative = self.project_root / "visual", "index.html"
        elif parts[0] in ("figs", "outputs", "visual"):
            root = self.output_dir if parts[0] == "outputs" else self.project_root / parts[0]
            relative = "/".join(parts[1:])
        else:
            root, relative = self.project_root / "visual", path.lstrip("/")
        root = root.resolve()
        candidate = (root / relative).resolve()
        return candidate if candidate.is_relative_to(root) else None

    def _catalog_item(self,file,stat,entry):
        relative=file.relative_to(self.output_dir).as_posix()
        return dict(entry,name=relative,url='/outputs/'+quote(relative,safe='/'),
                    modified=stat.st_mtime,bytes=stat.st_size)

    def _read_replay_metadata(self,file,stat):
        replay=load_replay(file)
        if not isinstance(replay,dict) or not isinstance(replay.get('frames'),list) or 'scene' not in replay:return None
        frames=replay['frames'];metadata=replay.get('metadata') or {}
        entry={'task':replay.get('task','coverage'),'actor':replay.get('actor','expert'),
               'seed':replay.get('seed'),'frames':len(frames),
               'duration':frames[-1].get('t',0) if frames else 0,'behavior':metadata.get('behavior')}
        if 'control_mode' in metadata:entry['control_mode']=metadata['control_mode']
        if (file.stat().st_mtime_ns,file.stat().st_size)!=(stat.st_mtime_ns,stat.st_size):return None
        index=file.with_suffix('.replay-index.json');temporary=index.with_suffix(index.suffix+'.tmp')
        if index.resolve().is_relative_to(self.output_dir) and temporary.resolve().is_relative_to(self.output_dir):
            try:
                temporary.write_text(json.dumps({'schema_version':'maritime-replay-index-v1',
                    'fingerprint':[stat.st_mtime_ns,stat.st_size],'entry':entry},allow_nan=False),encoding='utf-8')
                temporary.replace(index)
            except OSError:pass
        return self._catalog_item(file,stat,entry)

    def _replays(self):
        if not self.output_dir.exists():return []
        ignored={'cache','__pycache__','pytest-tmp','logs','tf-logs'}
        candidates=[];pending=[];records=[]
        for directory,directories,files in os.walk(self.output_dir,followlinks=False):
            directories[:]=[name for name in directories if not name.startswith('.') and name.lower() not in ignored]
            for name in files:
                if not name.lower().endswith(('.json','.npz')) or name.endswith('.replay-index.json'):continue
                file=Path(directory)/name
                try:
                    if file.resolve().is_relative_to(self.output_dir):candidates.append((file,file.stat()))
                except OSError:pass
        candidates.sort(key=lambda pair:(pair[0].relative_to(self.output_dir).parts[0]=='replays',pair[1].st_mtime_ns),reverse=True)
        for file,stat in candidates:
            fingerprint=(stat.st_mtime_ns,stat.st_size);key=str(file)
            with self._catalog_lock:cached=self._catalog_cache.get(key)
            if cached and cached[0]==fingerprint:
                if cached[1] is not None:records.append(cached[1])
                continue
            try:
                sidecar=file.with_suffix('.replay-index.json')
                index=json.loads(sidecar.read_text(encoding='utf-8')) if sidecar.is_file() else None
                if index and index.get('schema_version')=='maritime-replay-index-v1' and index.get('fingerprint')==list(fingerprint):
                    item=self._catalog_item(file,stat,index['entry'])
                elif stat.st_size>2*1024*1024:
                    pending.append((file,stat));continue
                else:item=self._read_replay_metadata(file,stat)
                with self._catalog_lock:self._catalog_cache[key]=(fingerprint,item)
                if item is not None:records.append(item)
            except (OSError,ValueError,TypeError,AttributeError,KeyError):continue
        # One scanner handles legacy large files. HTTP/SSE threads never parse
        # their training tensors, and concurrent catalog requests cannot multiply
        # memory use. New rollouts have small sidecars and appear immediately.
        with self._catalog_lock:
            if pending and (self._catalog_thread is None or not self._catalog_thread.is_alive()):
                def scan():
                    for file,stat in pending:
                        if self._http is None:break
                        try:item=self._read_replay_metadata(file,stat)
                        except (OSError,ValueError,TypeError,AttributeError,KeyError):continue
                        with self._catalog_lock:self._catalog_cache[str(file)]=((stat.st_mtime_ns,stat.st_size),item)
                self._catalog_thread=threading.Thread(target=scan,name='maritime-replay-catalog',daemon=True)
                self._catalog_thread.start()
        return sorted(records,key=lambda item:(item['name'].startswith('replays/'),item['modified']),reverse=True)

    def publish(self, payload):
        if not isinstance(payload, dict):
            raise TypeError("publish expects a frame or event dictionary")
        if "type" not in payload:
            payload = {"type": "frame", "frame": payload}
        data = json.dumps(payload, ensure_ascii=False, default=_json_default, separators=(",", ":"), allow_nan=False).encode("utf-8")
        with self._lock:
            if payload.get("type") == "scene":
                self._scene = data
                self._recent.clear()
            elif payload.get("type") == "frame":
                self._recent.append(data)
            for client in tuple(self._clients):
                try:
                    client.put_nowait(data)
                except queue.Full:
                    # Slow browsers skip stale frames, while retaining scene context.
                    while True:
                        try:
                            client.get_nowait()
                        except queue.Empty:
                            break
                    if self._scene is not None and self._scene != data:
                        client.put_nowait(self._scene)
                    client.put_nowait(data)

    def close(self):
        http, self._http = self._http, None
        if http is None:
            return
        with self._lock:
            for client in tuple(self._clients):
                try:
                    client.put_nowait(None)
                except queue.Full:
                    pass
        http.shutdown()
        http.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2)
        self._thread = None


if __name__ == "__main__":
    import argparse
    import time
    parser = argparse.ArgumentParser(description="Serve maritime replay visualizer")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    server = VisualServer(root, root / args.output_dir, args.host, args.port).start()
    print(f"Maritime visualization: {server.url}", flush=True)
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        server.close()
