"""HTTP/admission regression tests; the fake model does not load a GPU."""
import base64
from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, ProxyHandler, build_opener

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'runtime'))
from trellis_service import Resident
from openai_api import Generations, make_server


def encoded_image(size=(1024, 1024), color=(200, 100, 50, 255)):
    stream = io.BytesIO()
    Image.new('RGBA', size, color).save(stream, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(stream.getvalue()).decode()


class FakeModel:
    def __init__(self, config):
        self.gate = config['test_gate']
        self.calls = self.active = self.peak = 0
        self.export_active = self.export_peak = 0
        self.export_gate = config['test_export_gate']
        self.export_started = threading.Event()
        self.lock = threading.Lock()

    def prepare(self, item):
        with self.lock:
            self.calls += 1
            self.active += 1
            self.peak = max(self.peak, self.active)
        try:
            while not self.gate.wait(.005):
                item.check()
            item.check()
            return {'seed': item.request['objects'][0]['seed']}
        finally:
            with self.lock:
                self.active -= 1

    def finish(self, item, prepared):
        with self.lock:
            self.export_active += 1
            self.export_peak = max(self.export_peak, self.export_active)
        self.export_started.set()
        try:
            while not self.export_gate.wait(.005):
                item.check()
            item.check()
            output = Path(item.request['output_dir'])
            output.mkdir(parents=True, exist_ok=True)
            # A minimal valid GLB container, used only for byte-transport tests.
            payload = json.dumps({'asset': {'version': '2.0'}, 'extras': prepared}).encode()
            payload += b' ' * (-len(payload) % 4)
            data = struct.pack('<4sII', b'glTF', 2, 20 + len(payload))
            data += struct.pack('<I4s', len(payload), b'JSON') + payload
            (output / 'object.glb').write_bytes(data)
            return {'status': 'ok', 'assets': [{'asset': 'object.glb', 'object_id': 'object',
                'base_color_texture_size': [4096, 4096], 'vertices': 3, 'triangles': 1}],
                'timing': {'total_seconds': .001}}
        finally:
            with self.lock:
                self.export_active -= 1


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.gate = threading.Event()
        self.gate.set()
        self.export_gate = threading.Event()
        self.export_gate.set()
        self.patch = patch.object(Resident, 'status', lambda resident: {
            'state': resident.state, 'max_requests': resident.config['max_requests']})
        self.patch.start()
        self.resident = Resident({'root': str(self.root), 'max_requests': 64,
                                  'test_gate': self.gate, 'test_export_gate': self.export_gate},
                                 model_factory=FakeModel)
        deadline = time.monotonic() + 3
        while self.resident.state != 'ready' and time.monotonic() < deadline:
            time.sleep(.005)
        self.server = make_server(('127.0.0.1', 0), self.resident, api_key='test-key',
                                  output_root=self.root / 'api')
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = 'http://127.0.0.1:' + str(self.server.server_port)
        self.image = encoded_image()

    def tearDown(self):
        self.gate.set()
        self.export_gate.set()
        self.server.shutdown()
        self.resident.close(timeout=3)
        self.server.server_close()
        self.thread.join(timeout=2)
        self.patch.stop()
        self.temp.cleanup()

    def rpc(self, path, payload=None, *, method=None, token='test-key', raw=None):
        body = raw if raw is not None else None if payload is None else json.dumps(payload).encode()
        request = Request(self.url + path, data=body, method=method,
            headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + token})
        try:
            response = build_opener(ProxyHandler({})).open(request, timeout=30)
        except HTTPError as exc:
            response = exc
        with response:
            data = response.read()
            value = json.loads(data) if response.headers.get_content_type() == 'application/json' else data
            return response.status, value, response.headers

    def submit(self, **extra):
        return self.rpc('/v1/objects/generations', {'model': 'fasttrellis2', 'image': self.image, **extra})

    def terminal(self, identity, timeout=10):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            code, value, _ = self.rpc('/v1/objects/generations/' + identity)
            self.assertEqual(code, 200)
            if value['status'] in {'completed', 'failed', 'cancelled'}:
                return value
            time.sleep(.01)
        self.fail('Generation did not reach a terminal state')

    def test_auth_models_and_removed_native_endpoint(self):
        code, error, headers = self.rpc('/v1/models', token='wrong')
        self.assertEqual(code, 401)
        self.assertEqual(error['error']['code'], 'invalid_api_key')
        self.assertTrue(headers['x-request-id'].startswith('req_'))
        code, value, _ = self.rpc('/v1/models')
        self.assertEqual((code, value['object'], value['data'][0]['id']), (200, 'list', 'fasttrellis2'))
        self.assertEqual(self.rpc('/v1/models/fasttrellis2')[0], 200)
        self.assertEqual(self.rpc('/v1/models/missing')[0], 404)
        code, value, _ = self.rpc('/infer', {})
        self.assertEqual((code, value['error']['code']), (404, 'invalid_endpoint'))
        self.assertEqual(self.rpc('/v1/images/generations', {})[0], 404)

    def test_sync_async_file_download_and_base64(self):
        code, value, _ = self.submit(wait=True)
        self.assertEqual((code, value['status']), (200, 'completed'))
        artifact = value['data'][0]
        self.assertEqual(artifact['representation'], 'mesh_pbr')
        self.assertEqual(artifact['source_frame'], 'y_up')
        code, metadata, _ = self.rpc('/v1/files/' + artifact['file_id'])
        self.assertEqual((code, metadata['object']), (200, 'file'))
        code, binary, headers = self.rpc(artifact['url'])
        self.assertEqual(code, 200)
        self.assertEqual(binary[:4], b'glTF')
        self.assertEqual(len(binary), metadata['bytes'])
        self.assertEqual(headers.get_content_type(), 'model/gltf-binary')
        code, value, _ = self.submit(wait=True, response_format='b64_json')
        self.assertEqual(code, 200)
        self.assertEqual(base64.b64decode(value['data'][0]['b64_json'])[:4], b'glTF')
        self.assertNotIn('url', value['data'][0])

    def test_rejects_invalid_inputs_without_dispatch(self):
        cases = [{'image': encoded_image((32, 32))}, {'image': '/etc/passwd'},
                 {'image': 'https://example.com/image.png'}, {'mask': encoded_image(color=(0, 0, 0, 255))},
                 {'image': encoded_image(color=(0, 0, 0, 0))}, {'seed': True}, {'seed': -1},
                 {'wait': 'false'}, {'timeout_seconds': float('nan')}, {'response_format': []},
                 {'output_dir': '/tmp/client-chosen-output'}]
        for extra in cases:
            with self.subTest(fields=list(extra)):
                code, value, _ = self.submit(**extra)
                self.assertEqual(code, 400)
                self.assertIn('param', value['error'])
        self.assertEqual(self.resident.model.calls, 0)
        self.assertEqual(self.rpc('/v1/objects/generations', raw=b'{')[0], 400)
        self.assertEqual(self.submit(wait=True)[1]['status'], 'completed')

    def test_64_concurrent_admissions_and_65th_backpressure(self):
        self.gate.clear()
        with ThreadPoolExecutor(max_workers=64) as pool:
            responses = list(pool.map(lambda _: self.submit(), range(64)))
        self.assertEqual([row[0] for row in responses], [202] * 64)
        identities = {row[1]['id'] for row in responses}
        self.assertEqual(len(identities), 64)
        code, value, headers = self.submit()
        self.assertEqual(code, 429)
        self.assertEqual(value['error']['code'], 'generation_capacity_exceeded')
        self.assertEqual(headers['Retry-After'], '1')
        self.assertEqual(self.resident.model.calls, 1)
        self.gate.set()
        completed = [self.terminal(identity) for identity in identities]
        self.assertEqual({row['status'] for row in completed}, {'completed'})
        self.assertEqual(len({row['data'][0]['file_id'] for row in completed}), 64)
        self.assertEqual(self.resident.model.peak, 1)
        self.assertEqual(self.submit(wait=True)[1]['status'], 'completed')

    def test_cross_request_export_pipeline_is_bounded_and_isolates_payloads(self):
        self.export_gate.clear()
        first = self.submit(seed=11)[1]['id']
        self.assertTrue(self.resident.model.export_started.wait(3))
        others = [self.submit(seed=seed)[1]['id'] for seed in (22, 33, 44)]
        deadline = time.monotonic() + 3
        while (self.resident.model.calls < 3 or self.resident.model.export_peak < 2) and time.monotonic() < deadline:
            time.sleep(.005)
        self.assertEqual(self.resident.model.calls, 3)
        self.assertEqual(self.resident.active, 3)
        self.assertEqual(self.resident.model.export_peak, 2)
        self.assertEqual(self.rpc('/v1/objects/generations/' + first)[1]['stage'], 'cpu_export')
        self.assertEqual(self.rpc('/v1/objects/generations/' + others[-1])[1]['status'], 'queued')
        # The fourth GPU stage cannot allocate another 4K payload until one of
        # the three in-flight stages completes. A cancelled export frees it.
        self.rpc('/v1/objects/generations/' + first, method='DELETE')
        self.assertEqual(self.terminal(first)['status'], 'cancelled')
        deadline = time.monotonic() + 3
        while self.resident.model.calls < 4 and time.monotonic() < deadline:
            time.sleep(.005)
        self.assertEqual(self.resident.model.calls, 4)
        self.export_gate.set()
        completed = [self.terminal(identity) for identity in others]
        for value, seed in zip(completed, (22, 33, 44)):
            self.assertEqual(value['status'], 'completed')
            binary = self.rpc(value['data'][0]['url'])[1]
            document_length = struct.unpack_from('<I', binary, 12)[0]
            self.assertEqual(json.loads(binary[20:20 + document_length])['extras']['seed'], seed)
        self.assertEqual(self.resident.model.peak, 1)
        self.assertLess(completed[0]['usage']['pipeline']['gpu_started_at'],
                        completed[1]['usage']['pipeline']['cpu_export_finished_at'])
        self.assertEqual(self.resident.active, 0)

    def test_export_timeout_and_failure_release_pipeline_capacity(self):
        self.export_gate.clear()
        identity = self.submit(timeout_seconds=.03)[1]['id']
        self.assertEqual(self.terminal(identity)['error']['code'], 'inference_timeout')
        self.export_gate.set()
        with patch.object(self.resident.model, 'finish', side_effect=OSError('disk failure')):
            result = self.submit(wait=True)[1]
            self.assertEqual((result['status'], result['error']['code']), ('failed', 'inference_failed'))
        self.assertEqual(self.submit(wait=True)[1]['status'], 'completed')
        self.assertEqual(self.resident.active, 0)

    def test_terminal_persistence_failure_is_retried(self):
        generations = self.server.generations
        with generations.lock:
            identity = generations.submit({'model': 'fasttrellis2', 'image': self.image})
            self.assertTrue(generations.items[identity].done.wait(3))
            with patch.object(generations, '_save', side_effect=OSError('temporary disk failure')):
                with self.assertRaises(OSError):
                    generations._refresh(identity)
            self.assertIn(identity, generations.items)
            generations._refresh(identity)
            self.assertNotIn(identity, generations.items)
        self.assertEqual(self.terminal(identity)['status'], 'completed')

    def test_queued_and_running_cancellation_do_not_publish_partial_files(self):
        self.gate.clear()
        first = self.submit()[1]['id']
        second = self.submit()[1]['id']
        for identity in (second, first):
            code, value, _ = self.rpc('/v1/objects/generations/' + identity, method='DELETE')
            self.assertEqual(code, 200)
            self.assertIn(value['status'], {'cancelling', 'cancelled'})
        self.gate.set()
        for identity in (first, second):
            result = self.terminal(identity)
            self.assertEqual(result['status'], 'cancelled')
            self.assertEqual(result['data'], [])
            self.assertEqual(result['error']['code'], 'request_cancelled')
        self.assertEqual(self.resident.model.calls, 1)

    def test_deadline_is_distinct_from_cancellation(self):
        self.gate.clear()
        identity = self.submit(timeout_seconds=.02)[1]['id']
        result = self.terminal(identity)
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['error']['code'], 'inference_timeout')
        self.assertEqual(result['data'], [])

    def test_restart_preserves_completed_artifacts_and_fails_interrupted_jobs(self):
        result = self.submit(wait=True)[1]
        interrupted = 'objgen_' + '1' * 32
        directory = self.root / 'api' / interrupted
        directory.mkdir()
        (directory / 'job.json').write_text(json.dumps({'id': interrupted, 'status': 'queued'}))
        restored = Generations(self.resident, self.root / 'api')
        try:
            self.assertEqual(restored.get(result['id'])['status'], 'completed')
            self.assertEqual(restored.file(result['data'][0]['file_id'])[0].read_bytes()[:4], b'glTF')
            self.assertEqual(restored.get(interrupted)['error']['code'], 'service_restarted')
        finally:
            restored.close()

    def test_not_ready_and_non_loopback_auth_configuration(self):
        self.resident.state = 'loading'
        self.assertEqual(self.submit()[0], 503)
        self.resident.state = 'ready'
        with self.assertRaises(ValueError):
            make_server(('0.0.0.0', 0), self.resident)


if __name__ == '__main__':
    unittest.main()
