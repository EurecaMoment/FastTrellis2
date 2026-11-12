"""OpenAI-style 3D extension. Transport, storage and admission never load models.

OpenAI has no official object-generation endpoint. This service deliberately
uses a separate /v1/objects namespace instead of claiming images/chat support.
"""
import base64
import binascii
from copy import deepcopy
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import ipaddress
import json
import logging
import math
import os
from pathlib import Path
import queue
import re
import threading
import time
from urllib.parse import urlsplit
import uuid


MODEL_ID = 'fasttrellis2'
MAX_BODY = 16 * 1024 * 1024
MAX_IMAGE = 6 * 1024 * 1024
TERMINAL = {'completed', 'failed', 'cancelled'}
JOB_ID = re.compile(r'objgen_[0-9a-f]{32}')
FILE_ID = re.compile(r'file_[0-9a-f]{32}')
LOG = logging.getLogger('fasttrellis2.api')


class ApiError(Exception):
    def __init__(self, message, status=400, code='invalid_request', param=None):
        super().__init__(message)
        self.status, self.code, self.param = status, code, param

    def body(self):
        kind = ('authentication_error' if self.status == 401 else
                'rate_limit_error' if self.status == 429 else
                'server_error' if self.status >= 500 else 'invalid_request_error')
        return {'error': {'message': str(self), 'type': kind,
                          'param': self.param, 'code': self.code}}


def _atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('w', encoding='utf8') as stream:
            json.dump(value, stream, ensure_ascii=False, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _image(value, field, *, mask=False):
    from PIL import Image, UnidentifiedImageError
    if not isinstance(value, str) or not value:
        raise ApiError('Provide a base64 image or an image data URL.', param=field)
    if value.startswith('data:'):
        match = re.fullmatch(r'data:image/(?:png|jpeg|webp);base64,([A-Za-z0-9+/=\r\n]+)', value)
        if not match:
            raise ApiError('Expected a PNG, JPEG or WebP base64 data URL.', param=field)
        value = match[1]
    if len(value) > (MAX_IMAGE + 2) // 3 * 4:
        raise ApiError('Decoded image exceeds 6 MiB.', status=413, code='image_too_large', param=field)
    try:
        data = base64.b64decode(value.replace('\r', '').replace('\n', ''), validate=True)
        if len(data) > MAX_IMAGE:
            raise ApiError('Decoded image exceeds 6 MiB.', 413, 'image_too_large', field)
        with Image.open(io.BytesIO(data)) as source:
            if source.format not in {'PNG', 'JPEG', 'WEBP'}:
                raise ApiError('Only PNG, JPEG and WebP images are accepted.', param=field)
            if source.size != (1024, 1024):
                raise ApiError('Input image and mask must be exactly 1024x1024.', param=field)
            source.load()
            image = source.convert('L' if mask else 'RGBA')
    except (binascii.Error, ValueError, OSError, UnidentifiedImageError) as exc:
        raise ApiError('Image must contain valid base64 PNG, JPEG or WebP data.', param=field) from exc
    if mask and image.getextrema()[1] < 128:
        raise ApiError('Mask must contain foreground pixels (values at least 128).', param=field)
    return image


class Generations:
    def __init__(self, resident, output_root, *, max_history=256):
        self.resident = resident
        self.root = Path(output_root).resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.capacity = resident.config.get('max_requests', 64)
        self.slots = threading.BoundedSemaphore(self.capacity)
        self.max_history = max_history
        self.lock = threading.RLock()
        self.records, self.items = {}, {}
        self.stopped = threading.Event()
        self.created = int(time.time())
        self._restore()
        self.monitor = threading.Thread(target=self._monitor, daemon=True, name='object-generation-state')
        self.monitor.start()

    def _restore(self):
        # Each receipt is independently recoverable; no global writable registry
        # can make successful artifacts disappear on a process restart.
        paths = sorted(self.root.glob('objgen_*/job.json'), key=lambda p: p.stat().st_mtime)
        for path in paths:
            if not JOB_ID.fullmatch(path.parent.name) or path.is_symlink() or path.parent.is_symlink():
                continue
            try:
                value = json.loads(path.read_text(encoding='utf8'))
                if value.get('id') != path.parent.name:
                    continue
                if value.get('status') not in TERMINAL:
                    value.update(status='failed', stage='failed', completed_at=int(time.time()),
                                 error={'message': 'Service restarted before generation completed.',
                                        'type': 'server_error', 'param': None, 'code': 'service_restarted'})
                    _atomic_json(path, value)
                self.records[value['id']] = value
            except (OSError, ValueError):
                LOG.warning('Cannot read generation receipt %s', path.parent.name)
        self._trim()

    def _trim(self):
        terminal = [key for key, value in self.records.items() if value['status'] in TERMINAL]
        for key in terminal[:max(0, len(self.records) - self.max_history)]:
            self.records.pop(key, None)

    def _save(self, value):
        _atomic_json(self.root / value['id'] / 'job.json', value)

    @staticmethod
    def _validate(payload):
        if not isinstance(payload, dict):
            raise ApiError('Request body must be a JSON object.')
        extra = set(payload) - {'model', 'image', 'mask', 'seed', 'wait', 'timeout_seconds', 'response_format'}
        if extra:
            raise ApiError('Unknown request field.', param=sorted(extra)[0])
        if payload.get('model') != MODEL_ID:
            raise ApiError('Use model="fasttrellis2".', 404, 'model_not_found', 'model')
        seed = payload.get('seed', 42)
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ApiError('seed must be an integer from 0 to 4294967295.', param='seed')
        timeout = payload.get('timeout_seconds', 3600)
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or not .01 <= timeout <= 3600:
            raise ApiError('timeout_seconds must be finite and between 0.01 and 3600.', param='timeout_seconds')
        if type(payload.get('wait', False)) is not bool:
            raise ApiError('wait must be a boolean.', param='wait')
        if not isinstance(payload.get('response_format', 'url'), str) or payload.get('response_format', 'url') not in {'url', 'b64_json'}:
            raise ApiError('response_format must be url or b64_json.', param='response_format')
        return seed, timeout

    def submit(self, payload):
        seed, timeout = self._validate(payload)
        if self.stopped.is_set() or self.resident.state != 'ready':
            raise ApiError('Resident model is not ready.', 503, 'model_not_ready')
        if not self.slots.acquire(blocking=False):
            raise ApiError(f'All {self.capacity} generation slots are occupied; query existing jobs before retrying.',
                           429, 'generation_capacity_exceeded')
        handed_off = False
        directory = None
        try:
            image = _image(payload.get('image'), 'image')
            mask = _image(payload['mask'], 'mask', mask=True) if payload.get('mask') is not None else None
            if mask is None and image.getchannel('A').getextrema()[0] < 255:
                mask = image.getchannel('A')
                if mask.getextrema()[1] < 128:
                    raise ApiError('Input alpha contains no foreground.', param='image')
            identity = 'objgen_' + uuid.uuid4().hex
            directory = self.root / identity
            directory.mkdir(mode=0o700)
            image.save(directory / 'image.png')
            row = {'object_id': 'object', 'image': str(directory / 'image.png'), 'seed': seed}
            if mask is not None:
                mask.save(directory / 'mask.png')
                row['mask'] = str(directory / 'mask.png')
            request = {'profile': 'native512', 'generation_id': identity,
                       'output_dir': str(directory / 'output'), 'objects': [row]}
            value = {'id': identity, 'object': 'object.generation', 'created': int(time.time()),
                     'model': MODEL_ID, 'status': 'queued', 'stage': 'queued', 'completed_at': None,
                     'data': [], 'error': None, 'response_format': payload.get('response_format', 'url'),
                     'seed': seed, 'timeout_seconds': timeout}
            self._save(value)
            item = self.resident.submit(request, timeout)
            with self.lock:
                self.records[identity], self.items[identity] = value, item
                self._trim()
                handed_off = True
            return identity
        except queue.Full as exc:
            raise ApiError('Resident queue is full.', 429, 'generation_capacity_exceeded') from exc
        finally:
            if not handed_off:
                self.slots.release()
                if directory is not None and (directory / 'job.json').is_file():
                    value.update(status='failed', completed_at=int(time.time()),
                                 error={'message': 'Generation could not be dispatched.', 'type': 'server_error',
                                        'param': None, 'code': 'dispatch_failed'})
                    self._save(value)

    def _refresh(self, identity):
        item = self.items.get(identity)
        if item is None:
            return
        value = self.records[identity]
        done = item.done.is_set()
        before = value['status']
        previous_stage = value.get('stage')
        value['stage'] = item.stage
        if not done:
            value['status'] = ('cancelling' if item.canceled.is_set() else
                               'in_progress' if item.started_at is not None else 'queued')
        else:
            value['completed_at'] = int(time.time())
            if item.error_code is not None:
                cancelled = item.error_code == 'request_cancelled'
                value.update(status='cancelled' if cancelled else 'failed',
                             error={'message': 'Generation cancelled.' if cancelled else
                                    'Generation deadline exceeded.' if item.error_code == 'inference_timeout' else
                                    'Generation failed; inspect the service log with this generation ID.',
                                    'type': 'server_error', 'param': None,
                                    'code': item.error_code or 'inference_failed'})
            else:
                try:
                    asset = item.result['assets'][0]
                    file = self.root / identity / 'output' / 'object.glb'
                    if not file.is_file() or file.stat().st_size < 12:
                        raise ValueError('Incomplete GLB')
                    value.update(status='completed', data=[{
                        'file_id': 'file_' + identity.removeprefix('objgen_'), 'format': 'glb',
                        'url': '/v1/files/file_' + identity.removeprefix('objgen_') + '/content',
                        'bytes': file.stat().st_size, 'representation': 'mesh_pbr', 'source_frame': 'y_up',
                        'texture_size': asset.get('base_color_texture_size', [4096, 4096]),
                        'seed': value['seed'], 'vertices': asset.get('vertices'), 'triangles': asset.get('triangles')}],
                        usage={'timing': item.result.get('timing', {}),
                               'pipeline': item.result.get('pipeline', {}),
                               'peak_allocated_gib': item.result.get('peak_allocated_gib'),
                               'peak_reserved_gib': item.result.get('peak_reserved_gib'), 'cache_hit': False,
                               'model': 'microsoft/TRELLIS.2-4B', 'profile': 'native512'})
                except (KeyError, IndexError, TypeError, ValueError, OSError):
                    value.update(status='failed', error={'message': 'No complete mesh artifact was produced.',
                        'type': 'server_error', 'param': None, 'code': 'incomplete_artifact'})
        if before != value['status'] or previous_stage != value['stage'] or done:
            try:
                self._save(value)
            except OSError:
                # Keep the item and slot until its terminal receipt is durable;
                # the monitor retries on the next pass.
                value['status'] = before
                value['stage'] = previous_stage
                raise
        if done:
            self.items.pop(identity)
            self.slots.release()

    def _monitor(self):
        while not self.stopped.wait(.1):
            with self.lock:
                for identity in list(self.items):
                    try:
                        self._refresh(identity)
                    except OSError:
                        LOG.exception('Cannot persist generation state %s', identity)

    def get(self, identity, *, encode=True):
        if not JOB_ID.fullmatch(identity):
            raise ApiError('Generation not found.', 404, 'generation_not_found')
        with self.lock:
            if identity not in self.records:
                path = self.root / identity / 'job.json'
                if not path.is_file() or path.is_symlink() or path.parent.is_symlink():
                    raise ApiError('Generation not found.', 404, 'generation_not_found')
                self.records[identity] = json.loads(path.read_text(encoding='utf8'))
                self._trim()
            self._refresh(identity)
            value = deepcopy(self.records[identity])
        if encode and value['status'] == 'completed' and value['response_format'] == 'b64_json':
            value['data'][0]['b64_json'] = base64.b64encode(self.file(value['data'][0]['file_id'])[0].read_bytes()).decode()
            value['data'][0].pop('url', None)
        return value

    def wait(self, identity, seconds=30):
        with self.lock:
            item = self.items.get(identity)
        if item is not None:
            item.done.wait(min(30, seconds))
        return self.get(identity)

    def cancel(self, identity):
        self.get(identity, encode=False)
        with self.lock:
            item = self.items.get(identity)
            if item is not None:
                item.canceled.set()
                self._refresh(identity)
        return self.get(identity)

    def file(self, identity):
        if not FILE_ID.fullmatch(identity):
            raise ApiError('File not found.', 404, 'file_not_found')
        generation = self.get('objgen_' + identity.removeprefix('file_'), encode=False)
        if generation['status'] != 'completed':
            raise ApiError('The generation has no completed artifact.', 409, 'artifact_not_ready')
        file = self.root / generation['id'] / 'output' / 'object.glb'
        if file.is_symlink() or not file.is_file():
            raise ApiError('File not found.', 404, 'file_not_found')
        return file, {'id': identity, 'object': 'file', 'bytes': file.stat().st_size,
                      'created_at': generation['completed_at'], 'filename': 'object.glb',
                      'purpose': 'object-generation'}

    def close(self):
        self.stopped.set()
        self.monitor.join(timeout=2)
        with self.lock:
            for identity in list(self.items):
                try:
                    self._refresh(identity)
                except OSError:
                    LOG.exception('Cannot persist final state %s', identity)


def make_server(address, resident, *, api_key=None, output_root=None):
    try:
        loopback = ipaddress.ip_address(address[0]).is_loopback
    except ValueError:
        loopback = address[0].lower() == 'localhost'
    if not loopback and not api_key:
        raise ValueError('A non-loopback API listener requires an API key.')
    generations = Generations(resident, output_root or Path(resident.config['root']) / 'outputs/api')

    class Handler(BaseHTTPRequestHandler):
        server_version = 'FastTrellis2/1'

        def log_message(self, *_):
            pass

        def reply(self, value, status=200):
            body = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('x-request-id', self.request_id)
            if status == 429:
                self.send_header('Retry-After', '1')
            self.end_headers()
            self.wfile.write(body)

        def read_json(self):
            if self.headers.get_content_type() != 'application/json':
                raise ApiError('Use Content-Type: application/json.', 415, 'unsupported_media_type')
            try:
                size = int(self.headers.get('Content-Length', '-1'))
            except ValueError as exc:
                raise ApiError('Invalid Content-Length.') from exc
            if not 0 < size <= MAX_BODY:
                raise ApiError('JSON body must be between 1 byte and 16 MiB.', 413, 'request_too_large')
            body = self.rfile.read(size)
            if len(body) != size:
                raise ApiError('Incomplete JSON body.')
            try:
                return json.loads(body)
            except (ValueError, UnicodeError) as exc:
                raise ApiError('Invalid JSON body.', code='invalid_json') from exc

        def model(self):
            return {'id': MODEL_ID, 'object': 'model', 'created': generations.created, 'owned_by': 'microsoft'}

        def dispatch(self):
            self.request_id = 'req_' + uuid.uuid4().hex
            self.connection.settimeout(30)
            try:
                if api_key and not hmac.compare_digest(self.headers.get('Authorization', '').encode('utf8'),
                                                      ('Bearer ' + api_key).encode('utf8')):
                    raise ApiError('Incorrect or missing API key.', 401, 'invalid_api_key')
                path = urlsplit(self.path).path
                method = self.command
                if method == 'GET' and path == '/openapi.json':
                    self.reply(json.loads((Path(__file__).resolve().parents[1] / 'openapi.json').read_text(encoding='utf8')))
                    return
                if method == 'GET' and path in {'/health', '/ready', '/metrics'}:
                    value = resident.status()
                    value.update(api_style='openai-objects-v1', generation_capacity=generations.capacity)
                    self.reply(value, 503 if path == '/ready' and resident.state != 'ready' else 200)
                    return
                if method == 'GET' and path == '/v1/models':
                    self.reply({'object': 'list', 'data': [self.model()]})
                    return
                if method == 'GET' and path.startswith('/v1/models/'):
                    if path != '/v1/models/' + MODEL_ID:
                        raise ApiError('Model not found.', 404, 'model_not_found', 'model')
                    self.reply(self.model())
                    return
                if method == 'POST' and path == '/v1/objects/generations':
                    payload = self.read_json()
                    identity = generations.submit(payload)
                    result = generations.wait(identity) if payload.get('wait') else generations.get(identity)
                    self.reply(result, 200 if result['status'] in TERMINAL else 202)
                    return
                match = re.fullmatch(r'/v1/objects/generations/(objgen_[0-9a-f]{32})(/cancel)?', path)
                if match:
                    if method == 'GET' and match[2] is None:
                        self.reply(generations.get(match[1]))
                        return
                    if (method == 'DELETE' and match[2] is None) or (method == 'POST' and match[2] == '/cancel'):
                        self.reply(generations.cancel(match[1]))
                        return
                match = re.fullmatch(r'/v1/files/(file_[0-9a-f]{32})(/content)?', path)
                if method == 'GET' and match:
                    file, metadata = generations.file(match[1])
                    if match[2] is None:
                        self.reply(metadata)
                        return
                    self.send_response(200)
                    self.send_header('Content-Type', 'model/gltf-binary')
                    self.send_header('Content-Length', str(metadata['bytes']))
                    self.send_header('Content-Disposition', 'attachment; filename="object.glb"')
                    self.send_header('x-request-id', self.request_id)
                    self.end_headers()
                    with file.open('rb') as stream:
                        while chunk := stream.read(64 * 1024):
                            self.wfile.write(chunk)
                    return
                raise ApiError('Endpoint not found.', 404, 'invalid_endpoint')
            except ApiError as exc:
                self.reply(exc.body(), exc.status)
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                pass  # Disconnects never resubmit or cancel an admitted generation.
            except Exception:
                LOG.exception('API failure %s %s %s', self.request_id, self.command, urlsplit(self.path).path)
                self.reply(ApiError('Internal server error.', 500, 'internal_error').body(), 500)

        do_GET = do_POST = do_DELETE = dispatch

    class Server(ThreadingHTTPServer):
        request_queue_size = 256
        daemon_threads = True

        def server_close(self):
            try:
                super().server_close()
            finally:
                generations.close()

    try:
        server = Server(address, Handler)
    except BaseException:
        generations.close()
        raise
    server.generations = generations
    return server
