"""Reproducible real-model load test for the OpenAI-style 3D endpoint.

Sends one generation per worker, never retries a submission, and verifies each
GLB container plus embedded 4K PBR images. Run against an otherwise idle service.
"""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import os
import statistics
import struct
import subprocess
import threading
import time
from urllib.error import HTTPError
from urllib.request import Request, ProxyHandler, HTTPRedirectHandler, build_opener


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def verify_glb(path):
    """Check container lengths, geometry accessors and lossless PBR payloads."""
    from PIL import Image
    with path.open('rb') as stream:
        magic, version, size = struct.unpack('<4sII', stream.read(12))
        assert magic == b'glTF' and version == 2 and size == path.stat().st_size
        length, kind = struct.unpack('<I4s', stream.read(8))
        assert kind == b'JSON' and length < 32 * 1024**2
        document = json.loads(stream.read(length))
        binary_length, kind = struct.unpack('<I4s', stream.read(8))
        binary_start = stream.tell()
        assert kind == b'BIN\0' and binary_start + binary_length == size
        pbr = document['materials'][0]['pbrMetallicRoughness']
        images = []
        for field in ('baseColorTexture', 'metallicRoughnessTexture'):
            source = document['textures'][pbr[field]['index']]['source']
            image = document['images'][source]
            assert image['mimeType'] == 'image/png'
            view = document['bufferViews'][image['bufferView']]
            stream.seek(binary_start + view.get('byteOffset', 0))
            payload = stream.read(view['byteLength'])
            with Image.open(io.BytesIO(payload)) as pixels:
                assert pixels.size == (4096, 4096)
                pixels.load()
                images.append({'size': list(pixels.size),
                               'pixel_sha256': hashlib.sha256(pixels.tobytes()).hexdigest()})
        for accessor in document['accessors']:
            if 'bufferView' in accessor:
                view = document['bufferViews'][accessor['bufferView']]
                assert view.get('byteOffset', 0) + view['byteLength'] <= binary_length
        primitive = document['meshes'][0]['primitives'][0]
        position = document['accessors'][primitive['attributes']['POSITION']]
        vertices = position['count']
        index = document['accessors'][primitive['indices']]
        indices = index['count']
        assert vertices > 0 and indices > 0 and indices % 3 == 0
        geometry = hashlib.sha256()
        for accessor in (position, index):
            view = document['bufferViews'][accessor['bufferView']]
            stream.seek(binary_start + view.get('byteOffset', 0))
            data = stream.read(view['byteLength'])
            geometry.update(data)
            import numpy as np
            array = np.frombuffer(data, dtype='<f4' if accessor is position else '<u4')
            assert np.isfinite(array).all()
            if accessor is index:
                assert array.max() < vertices
        return {'bytes': size, 'vertices': vertices, 'triangles': indices // 3,
                'images': images, 'geometry_sha256': geometry.hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:3956')
    parser.add_argument('--image', required=True)
    parser.add_argument('--mask')
    parser.add_argument('--count', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', required=True)
    parser.add_argument('--server-output', help='Shared API output root; otherwise download and verify each file')
    parser.add_argument('--api-key-env', default='FASTTRELLIS2_API_KEY')
    parser.add_argument('--timeout', type=int, default=3600)
    parser.add_argument('--server-pid', type=int, help='Optional Linux service PID for memory monitoring')
    args = parser.parse_args()
    if not 1 <= args.count <= 64:
        parser.error('count must be between 1 and 64')
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    token = os.environ.get(args.api_key_env)

    def exchange(route, payload=None, target=None):
        headers = {'Content-Type': 'application/json'}
        if token:
            headers['Authorization'] = 'Bearer ' + token
        request = Request(args.url.rstrip('/') + route,
                          data=None if payload is None else json.dumps(payload).encode(), headers=headers)
        try:
            response = build_opener(ProxyHandler({}), NoRedirect()).open(request, timeout=60)
        except HTTPError as exc:
            response = exc
        with response:
            if target is not None:
                assert response.status == 200
                import shutil
                with target.open('wb') as stream:
                    shutil.copyfileobj(response, stream, length=1024 * 1024)
                return response.status, None
            return response.status, json.load(response)

    def encoded(filename):
        return base64.b64encode(Path(filename).read_bytes()).decode()

    template = {'model': 'fasttrellis2', 'image': encoded(args.image),
                'wait': False, 'timeout_seconds': args.timeout}
    if args.mask:
        template['mask'] = encoded(args.mask)
    before = exchange('/health')[1]
    assert before['state'] == 'ready' and before.get('active_requests', 0) == 0 and before['queue'] == 0
    (output / 'baseline.json').write_text(json.dumps(before, indent=2))
    barrier = threading.Barrier(args.count)
    started = time.perf_counter()

    def process_memory():
        if not args.server_pid:
            return {}
        try:
            text = Path('/proc/' + str(args.server_pid) + '/status').read_text()
            rss = next(int(line.split()[1])*1024 for line in text.splitlines() if line.startswith('VmRSS:'))
            result = subprocess.check_output(['nvidia-smi', '--id=' + str(before['gpu']),
                '--query-compute-apps=pid,used_gpu_memory', '--format=csv,noheader,nounits'], text=True, timeout=5)
            gpu_mib = next((int(line.split(',')[1]) for line in result.splitlines()
                           if int(line.split(',')[0]) == args.server_pid), 0)
            return {'server_rss_bytes': rss, 'server_gpu_mib': gpu_mib}
        except (OSError, ValueError, StopIteration, subprocess.SubprocessError):
            return {}

    def submit(index):
        barrier.wait(timeout=30)
        began = time.perf_counter()
        status, value = exchange('/v1/objects/generations', {**template, 'seed': args.seed + index})
        return {'index': index, 'http': status, 'job': value,
                'submission_seconds': time.perf_counter()-began}

    with ThreadPoolExecutor(max_workers=args.count) as executor:
        rows = list(executor.map(submit, range(args.count)))
    (output / 'admissions.json').write_text(json.dumps(rows, indent=2))
    assert all(row['http'] == 202 for row in rows), 'Some requests were not admitted'
    assert len({row['job']['id'] for row in rows}) == args.count
    overflow = None
    if args.count == 64:
        status, value = exchange('/v1/objects/generations', {**template, 'seed': args.seed + 1000})
        overflow = {'http': status, 'error': value.get('error')}
        # If generation completed during submissions, a new request may fit.
        # Record that honestly instead of inventing a saturated-capacity claim.
        if status == 202:
            rows.append({'index': 64, 'http': status, 'job': value, 'extra_probe': True})
        else:
            assert status == 429 and value['error']['code'] == 'generation_capacity_exceeded'
    pending = {row['job']['id']: row for row in rows}
    samples, completed = [], []
    verification = ThreadPoolExecutor(max_workers=1)
    futures = []
    try:
        while pending:
            assert time.perf_counter()-started <= args.timeout + 60, 'Load test deadline exceeded'
            for identity, row in list(pending.items()):
                status, job = exchange('/v1/objects/generations/' + identity)
                assert status == 200
                if job['status'] not in {'completed', 'failed', 'cancelled'}:
                    continue
                row['job'] = job
                completed.append(row)
                del pending[identity]
                (output / (identity + '.json')).write_text(json.dumps(job, indent=2))
                if job['status'] == 'completed':
                    if args.server_output:
                        path = Path(args.server_output) / identity / 'output/object.glb'
                    else:
                        path = output / (identity + '.glb')
                        exchange(job['data'][0]['url'], target=path)
                    futures.append((identity, verification.submit(verify_glb, path)))
            health = exchange('/metrics')[1]
            sample = {'at': time.time(), 'pending': len(pending), **process_memory(),
                      **{key: health.get(key) for key in
                         ('queue', 'active_requests', 'gpu_active', 'cpu_exports_active', 'last_gpu_memory')}}
            samples.append(sample)
            progress = {'finished': not pending, 'elapsed_seconds': time.perf_counter()-started,
                        'completed': sum(row['job']['status'] == 'completed' for row in completed),
                        'failed': sum(row['job']['status'] != 'completed' for row in completed),
                        'pending': len(pending), 'sample': sample}
            temporary = output / 'progress.tmp'
            temporary.write_text(json.dumps(progress, indent=2))
            temporary.replace(output / 'progress.json')
            print(json.dumps(progress), flush=True)
            if pending:
                time.sleep(2)
        generation_wall = time.perf_counter()-started
        verified = {identity: future.result() for identity, future in futures}
    finally:
        verification.shutdown(wait=True)
    intervals = [(row['job']['id'], row['job'].get('usage', {}).get('pipeline', {})) for row in completed]
    overlaps = []
    for left_id, left in intervals:
        for right_id, right in intervals:
            if left_id == right_id:
                continue
            if all(key in left for key in ('cpu_export_started_at', 'cpu_export_finished_at')) and all(
                    key in right for key in ('gpu_started_at', 'gpu_finished_at')):
                seconds = min(left['cpu_export_finished_at'], right['gpu_finished_at']) - max(
                    left['cpu_export_started_at'], right['gpu_started_at'])
                if seconds > 0:
                    overlaps.append({'export_job': left_id, 'gpu_job': right_id, 'seconds': seconds})
    timings = [row['job']['usage']['timing'] for row in completed if row['job']['status'] == 'completed']
    summary = {'requested': args.count, 'actual_generations': len(completed),
               'completed': sum(row['job']['status'] == 'completed' for row in completed),
               'failed': sum(row['job']['status'] != 'completed' for row in completed),
               'generation_wall_seconds': generation_wall,
               'generations_per_minute': len(verified) / generation_wall * 60,
               'cpu_export_mean_seconds': statistics.mean(row['cpu_export_seconds'] for row in timings) if timings else None,
               'gpu_stage_mean_seconds': statistics.mean(row['gpu_stage_seconds'] for row in timings) if timings else None,
               'overflow_probe': overflow, 'cross_request_overlaps': overlaps,
               'verified': verified, 'health_before': before, 'health_after': exchange('/health')[1],
               'metrics': samples, 'rows': completed}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({key: summary[key] for key in ('requested', 'completed', 'failed', 'generation_wall_seconds',
                     'generations_per_minute', 'cpu_export_mean_seconds', 'gpu_stage_mean_seconds', 'overflow_probe')}, indent=2), flush=True)
    assert summary['failed'] == 0 and len(verified) == summary['actual_generations']
    if args.count > 1 and before.get('pipeline_depth', 1) > 1:
        assert overlaps, 'No real cross-request stage overlap was observed'


if __name__ == '__main__':
    main()
