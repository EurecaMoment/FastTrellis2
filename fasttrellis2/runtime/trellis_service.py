"""Resident TRELLIS.2 mesh/PBR service with bounded requests and real deadlines."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import os
from pathlib import Path
import queue
import re
import socket
import signal
import subprocess
import threading
import time
import traceback
import uuid


PIPELINE_TYPE = '512'


def native_export_profile(config):
    return config.get('native512_release_profile', 'fasttrellis2-v1')


def resolve_profile(request):
    profile = request.get('profile', 'native512')
    if profile != 'native512':
        raise ValueError('FastTrellis2 supports only the native512 profile')
    return profile, PIPELINE_TYPE


class RequestExpired(RuntimeError):
    pass


class RequestCancelled(RequestExpired):
    pass


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf8')
    temporary.replace(path)


class Item:
    def __init__(self, request, timeout):
        self.id = request.get('generation_id') or uuid.uuid4().hex
        self.request = request
        self.arrived = time.perf_counter()
        self.deadline = self.arrived + timeout
        self.canceled = threading.Event()
        self.done = threading.Event()
        self.result = None
        self.error = None
        self.error_code = None
        self.started_at = None
        self.stage = 'queued'
        self.pipeline_timing = {}

    def check(self, *_args):
        if self.canceled.is_set():
            raise RequestCancelled('Generation cancelled before a complete mesh/PBR artifact was produced')
        if time.perf_counter() >= self.deadline:
            raise RequestExpired('Request deadline exceeded before a complete mesh/PBR artifact was produced')


class TrellisModel:
    def __init__(self, config):
        import sys
        self.root = Path(config['root'])
        self.config = config
        cache_home = Path(config['cache_home'])
        cache_home.mkdir(parents=True, exist_ok=True)
        Path.home = classmethod(lambda _cls: cache_home)
        source_dir = Path(config.get('source_dir', self.root))
        sys.path.insert(0, str(source_dir))
        import numpy as np
        import torch
        import torch.nn.functional as functional
        import timm
        from safetensors.torch import load_file
        from trellis2.modules import image_feature_extractor

        class PublicDinoV3FeatureExtractor(image_feature_extractor.DinoV3FeatureExtractor):
            def __init__(self, model_name, image_size=512):
                from torchvision import transforms
                self.model_name = model_name
                self.model = timm.create_model('vit_large_patch16_dinov3.lvd1689m', pretrained=False,
                    num_classes=0, dynamic_img_size=True)
                self.model.load_state_dict(load_file(model_name), strict=True)
                self.model.norm = torch.nn.Identity()
                self.model.eval()
                self.image_size = image_size
                self.transform = transforms.Compose([transforms.Normalize(
                    mean=[.485, .456, .406], std=[.229, .224, .225])])

            def extract_features(self, image):
                image = image.to(self.model.patch_embed.proj.weight.dtype)
                hidden = self.model.forward_features(image)
                return functional.layer_norm(hidden, hidden.shape[-1:])

        image_feature_extractor.PublicDinoV3FeatureExtractor = PublicDinoV3FeatureExtractor
        from trellis2.pipelines import Trellis2ImageTo3DPipeline
        import o_voxel
        self.np, self.torch, self.voxel = np, torch, o_voxel
        if __package__:
            from .trellis_postprocess import to_glb
        else:
            from trellis_postprocess import to_glb
        self.postprocess = to_glb
        self.parallel_inpaint = config.get('parallel_inpaint', True)
        torch.set_num_threads(config.get('cpu_threads', 8))
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(config['memory_budget_gib']*1024**3/total)
        torch.backends.cuda.matmul.allow_tf32 = True
        self.pipeline = Trellis2ImageTo3DPipeline.from_pretrained(config['model_dir'])
        self.pipeline.low_vram = config.get('low_vram', False)
        self.pipeline.cuda()
        self.current = None
        for model in self.pipeline.models.values():
            if hasattr(model, 'register_forward_pre_hook'):
                model.register_forward_pre_hook(self.check)
        self.pipeline.image_cond_model.model.register_forward_pre_hook(self.check)
        torch.cuda.synchronize()
        from trellis_memory import DeviceMemory
        self.memory_monitor=DeviceMemory(os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0])
        self.startup_warmup = None
        from trellis_native512_runtime import Native512Runtime
        self.native_runtime = Native512Runtime(self)
        # This function only reads the detached CPU mesh and material pixels.
        # Export threads must never touch the shared native runtime or CUDA.
        self.export_glb = self.native_runtime.export_glb
        if config.get('startup_warmup_image'):
            warmup_image = Path(config['startup_warmup_image'])
            warmup = Item({'profile': 'native512',
                'output_dir': str(self.root / 'outputs/service-startup-native512'),
                'objects': [{'object_id': 'startup', 'image': str(warmup_image), 'seed': 42}]}, 1800)
            self.startup_warmup = self.infer(warmup)['timing']
            print('STARTUP_WARMUP ' + json.dumps(self.startup_warmup), flush=True)

    def check(self, *_args):
        if self.current is not None:
            self.current.check()

    def infer(self, item):
        return self.finish(item, self.prepare(item))

    def prepare(self, item):
        with self.torch.inference_mode():
            return self._prepare(item)

    def _prepare(self, item):
        from PIL import Image
        self.memory_monitor.admit(self.torch.cuda.memory_reserved(),
            self.config['memory_budget_gib']*1024**3)
        request = item.request
        profile, pipeline_type = resolve_profile(request)
        steps = 12
        texture_size = 4096
        output = Path(request['output_dir'])
        output.mkdir(parents=True, exist_ok=True)
        prepared = []
        self.current = item
        self.torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        try:
            for index, row in enumerate(request['objects']):
                item.check()
                object_started = previous = time.perf_counter()
                outer_timing = {}
                def mark_outer(name):
                    nonlocal previous
                    now = time.perf_counter()
                    outer_timing[name] = now - previous
                    previous = now
                diagnostic_dir = output / (row['object_id'] + '.diagnostic') if request.get('capture_optimization_inputs') is True else None
                uv_capture_seconds = request.get('capture_uv_slow_seconds')
                uv_capture_dir = output / (row['object_id'] + '.slow_uv') if uv_capture_seconds is not None else None
                self.native_runtime.begin(diagnostic_dir=diagnostic_dir,
                    uv_capture_dir=uv_capture_dir,
                    uv_capture_slow_seconds=max(0.0, float(uv_capture_seconds or 0.0)))
                image_path = Path(row['image'])
                with Image.open(image_path) as source:
                    image = source.convert('RGBA' if row.get('mask') else 'RGB')
                if image.size != (1024, 1024):
                    raise ValueError('TRELLIS service requires a 1024x1024 source image')
                inputs = {'image': {'path': str(image_path), 'bytes': image_path.stat().st_size,
                                    'mtime_ns': image_path.stat().st_mtime_ns}}
                if row.get('mask'):
                    mask_path = Path(row['mask'])
                    with Image.open(mask_path) as source:
                        mask = source.convert('L')
                    if mask.size != image.size or mask.getextrema()[1] < 128:
                        raise ValueError('Object mask must match the image and contain foreground pixels')
                    image.putalpha(mask)
                    inputs['mask'] = {'path': str(mask_path), 'bytes': mask_path.stat().st_size,
                                      'mtime_ns': mask_path.stat().st_mtime_ns}
                seed = int(row.get('seed', request.get('seed', 42) + index))
                mark_outer('image_read_seconds')
                generation_start = time.perf_counter()
                generation_timing = {}
                remesh = bool(request.get('remesh', False))
                with self.torch.inference_mode():
                    mesh, generation_timing = self.native_runtime.generate(image, seed, item.check)
                self.torch.cuda.synchronize()
                generation_seconds = time.perf_counter() - generation_start
                mark_outer('shape_generation_seconds')
                item.check()
                mesh.simplify(16777216)
                mark_outer('pre_export_simplify_seconds')
                export_start = time.perf_counter()
                export_timing = {}
                material_arguments = self.native_runtime.postprocess_arguments()
                glb = self.postprocess(vertices=mesh.vertices, faces=mesh.faces,
                    **material_arguments, aabb=[[-.5, -.5, -.5], [.5, .5, .5]],
                    decimation_target=1000000,
                    texture_size=texture_size,
                    remesh=remesh, remesh_band=1, remesh_project=0,
                    verbose=self.config.get('native512_verbose', False),
                    timings=export_timing, parallel_inpaint=self.parallel_inpaint,
                    verify_inpaint=request.get('verify_inpaint') is True, check=item.check,
                    fast_cleanup=False, project_texture_to_source=True,
                    texture_padding='telea', uv_mode='xatlas',
                    simplify_options=None, texture_sampling='sparse',
                    uv_group_face_cap=self.config.get('uv_group_face_cap', 4096))
                mark_outer('postprocess_call_seconds')
                item.check()
                material = glb.visual.material
                destination = output / (row['object_id'] + '.glb')
                self.torch.cuda.synchronize()
                mark_outer('gpu_final_sync_seconds')
                asset = {'object_id': row['object_id'], 'asset': destination.name,
                    'model': 'microsoft/TRELLIS.2-4B', 'source_frame': 'y_up',
                    'representation': 'mesh_pbr', 'metric_reconstruction': False,
                    'camera_pose_estimated': False, 'seed': seed,
                    'pipeline': pipeline_type,
                    'sampling_steps': {'sparse': steps, 'shape': steps, 'texture': steps},
                    'vertices': len(glb.vertices), 'triangles': len(glb.faces),
                    'base_color_texture_size': list(material.baseColorTexture.size),
                    'metallic_roughness_texture_size': list(material.metallicRoughnessTexture.size),
                    'inputs': inputs,
                    'generation_seconds': generation_seconds,
                    'export_profile': native_export_profile(self.config),
                    'profile': profile, 'remesh': remesh, 'generation_timing': generation_timing,
                    'export_timing': export_timing,
                    'outer_timing': outer_timing,
                    'diagnostic_capture': request.get('capture_optimization_inputs') is True,
                    'uv_mode': 'xatlas', 'texture_sampling': 'sparse'}
                asset.update(deepcopy(self.native_runtime.details()))
                # to_glb has joined every GPU copy. The CPU mesh retains the
                # per-request NumPy arrays (including raw material pixels),
                # independent of the next request's scratch arena and caches.
                self.native_runtime.end()
                prepared.append((glb, destination, asset, object_started, export_start))
                del mesh
            self.torch.cuda.synchronize()
            item.check()
            result = {'status': 'ok', 'provider': 'trellis2', 'model': 'microsoft/TRELLIS.2-4B',
                'assets': [], 'cache_hit': False, 'asset_cache_hit': False,
                'cache_hit_scope': 'generated assets only; inference caches are reported per object',
                'representations': ['mesh_pbr'],
                'timing': {'queue_seconds': started-item.arrived,
                    'gpu_stage_seconds': time.perf_counter()-started},
                'peak_allocated_gib': self.torch.cuda.max_memory_allocated()/1024**3,
                'peak_reserved_gib': self.torch.cuda.max_memory_reserved()/1024**3,
                'memory_budget_gib': self.config['memory_budget_gib'],
                'process_memory_limit_bytes':50_000_000_000,'gpu_headroom_gib':2}
            return prepared, result, started
        finally:
            try:
                self.native_runtime.end()
            finally:
                self.current = None

    def finish(self, item, prepared):
        """Validate, encode and write only CPU-owned per-request data."""
        objects, result, started = prepared
        output = Path(item.request['output_dir'])
        export_started = time.perf_counter()
        for glb, destination, asset, object_started, postprocess_started in objects:
            item.check()
            if not len(glb.faces) or not self.np.isfinite(glb.vertices).all():
                raise ValueError('TRELLIS produced empty or nonfinite mesh geometry')
            material = glb.visual.material
            if (getattr(material, 'baseColorTexture', None) is None or
                    getattr(material, 'metallicRoughnessTexture', None) is None):
                raise ValueError('TRELLIS mesh is missing PBR texture payloads')
            temporary = destination.with_name(destination.stem + '.' + item.id + '.partial.glb')
            encode_started = time.perf_counter()
            try:
                asset['export_timing'].update(self.export_glb(glb, temporary, 0))
                item.check()
                temporary.replace(destination)
                asset.update(bytes=destination.stat().st_size,
                    export_seconds=time.perf_counter()-postprocess_started,
                    image_to_glb_seconds=time.perf_counter()-object_started)
                asset['outer_timing']['validation_encode_write_seconds'] = time.perf_counter()-encode_started
                write_json(destination.with_suffix('.complete.json'), {'asset': asset,
                    'provider': 'trellis2', 'request_id': item.id, 'status': 'complete'})
                result['assets'].append(asset)
            finally:
                temporary.unlink(missing_ok=True)
        item.check()
        result['timing'].update(cpu_export_seconds=time.perf_counter()-export_started,
            compute_and_export_seconds=time.perf_counter()-started,
            total_seconds=time.perf_counter()-item.arrived)
        write_json(output / 'trellis_result.json', result)
        return result


class Resident:
    def __init__(self, config, model_factory=TrellisModel):
        self.config = config
        self.queue = queue.Queue(maxsize=config.get('max_requests', 64))
        self.slots = threading.BoundedSemaphore(config.get('max_requests', 64))
        self.state, self.error = 'loading', None
        self.completed = self.failed = self.expired = self.cancelled = self.active = 0
        self.gpu_active = self.export_active = 0
        self.lock = threading.RLock()
        self.pending = {}
        self.stopping = threading.Event()
        self.export_workers = config.get('export_workers', 2)
        self.pipeline_depth = config.get('pipeline_depth', self.export_workers + 1)
        if (type(self.export_workers) is not int or not 1 <= self.export_workers <= 4 or
                type(self.pipeline_depth) is not int or not 1 <= self.pipeline_depth <= self.export_workers + 1):
            raise ValueError('export_workers must be 1..4; pipeline_depth must be 1..export_workers+1')
        # Reserve before GPU preparation, so even a stalled filesystem cannot
        # accumulate 64 detached 4K material payloads in host memory.
        self.stage_slots = threading.BoundedSemaphore(self.pipeline_depth)
        self.exporter = ThreadPoolExecutor(max_workers=self.export_workers,
                                         thread_name_prefix='trellis-cpu-export')
        self.history = []
        self.factory = model_factory
        self.worker = threading.Thread(target=self.run, daemon=True, name='trellis-gpu')
        self.worker.start()

    def status(self):
        _, pipeline_type = resolve_profile({})
        try:
            row = subprocess.check_output(['nvidia-smi', '--id='+str(self.config['gpu']),
                '--query-gpu=memory.total,memory.used,memory.free', '--format=csv,noheader,nounits'], text=True)
            total, used, free = map(int, row.strip().split(','))
            memory = {'total_mib':total,'used_mib':used,'free_mib':free}
        except Exception:
            memory = None
        return {'model': 'trellis2', 'model_id': 'microsoft/TRELLIS.2-4B',
            'state': self.state, 'error': self.error, 'host': socket.gethostname(),
            'gpu': os.environ.get('CUDA_VISIBLE_DEVICES'), 'queue': self.queue.qsize(),
            'active_batch': self.gpu_active, 'active_requests': self.active,
            'gpu_active': self.gpu_active, 'cpu_exports_active': self.export_active,
            'pipeline_depth': self.pipeline_depth, 'export_workers': self.export_workers,
            'execution': 'gpu-with-bounded-cpu-export-pipeline',
            'max_requests': self.config.get('max_requests', 64),
            'max_batch': 1, 'completed': self.completed, 'failed': self.failed,
            'expired': self.expired, 'latency_recent': self.history[-64:],
            'cancelled': self.cancelled,
            'representations': ['mesh_pbr'], 'gaussian_supported': False,
            'pipeline': pipeline_type, 'texture_size': 4096, 'resident': True,
            'sampling_steps': 12, 'uv_mode': 'xatlas',
            'uv_strategy': 'whole_bounded',
            'uv_group_face_cap': 4096,
            'models_resident_on_gpu': not self.config.get('low_vram', True),
            'export_profile': native_export_profile(self.config),
            'native512_resident_runtime': True,
            'native512_parallel_dense_cfg': self.config.get('native512_parallel_dense_cfg', True),
            'native512_zero_negative_cross_attention': self.config.get('native512_zero_negative_cross_attention', True),
            'fused_pointwise': True,
            'remesh': False, 'default_profile': 'native512',
            'profiles': ['native512'], 'last_gpu_memory':memory,
            'memory_budget_gib':self.config.get('memory_budget_gib',32),
            'process_memory_limit_bytes':50_000_000_000,'gpu_headroom_gib':2}

    def submit(self, request, timeout):
        if self.stopping.is_set() or self.state != 'ready':
            raise RuntimeError('Resident model is not ready')
        resolve_profile(request)
        rows = request.get('objects')
        if not isinstance(rows, list) or len(rows) != 1 or not Path(request.get('output_dir', '')).is_absolute():
            raise ValueError('Exactly one object and an absolute output_dir are required')
        seen = set()
        for row in rows:
            identity = row.get('object_id')
            if not isinstance(identity, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,100}', identity) or identity in seen:
                raise ValueError('object_id must be a unique safe filename component')
            if not isinstance(row.get('image'), str) or not Path(row['image']).is_file():
                raise ValueError('Object image does not exist')
            seen.add(identity)
        if not self.slots.acquire(blocking=False):
            raise queue.Full()
        item = Item(request, timeout)
        try:
            with self.lock:
                if self.stopping.is_set():
                    raise RuntimeError('Resident model is stopping')
                self.pending[item.id] = item
                self.queue.put_nowait(item)
        except BaseException:
            with self.lock:
                self.pending.pop(item.id, None)
            self.slots.release()
            raise
        return item

    def _complete(self, item, *, result=None, error=None, admitted=False):
        with self.lock:
            if error is None:
                item.result = result
                item.stage = 'completed'
                self.completed += 1
                self.history.append(time.perf_counter()-item.arrived)
                self.history = self.history[-64:]
            else:
                item.error = str(error)
                if isinstance(error, RequestCancelled):
                    item.error_code, item.stage = 'request_cancelled', 'cancelled'
                    self.cancelled += 1
                elif isinstance(error, RequestExpired):
                    item.error_code, item.stage = 'inference_timeout', 'failed'
                    self.expired += 1
                else:
                    item.error_code, item.stage = 'inference_failed', 'failed'
                    self.failed += 1
                    print(f'Generation {item.id} failed: {error}', flush=True)
            if admitted:
                self.active -= 1
                self.stage_slots.release()
            self.pending.pop(item.id, None)
            self.slots.release()
            self.queue.task_done()
            item.done.set()

    def _export(self, item, prepared):
        with self.lock:
            self.export_active += 1
            item.stage = 'cpu_export'
            item.pipeline_timing['cpu_export_started_at'] = time.time()
        result = error = None
        try:
            item.check()
            result = self.model.finish(item, prepared)
            item.check()
            item.pipeline_timing['cpu_export_finished_at'] = time.time()
            result['pipeline'] = dict(item.pipeline_timing)
            # Persist stage times before publishing the successful receipt.
            write_json(Path(item.request['output_dir']) / 'trellis_result.json', result)
        except Exception as exc:
            error = exc
            if not isinstance(exc, RequestExpired):
                print(traceback.format_exc(), flush=True)
        finally:
            with self.lock:
                self.export_active -= 1
            self._complete(item, result=result, error=error, admitted=True)

    def run(self):
        try:
            self.model = self.factory(self.config)
            self.state = 'ready'
            print(json.dumps(self.status()), flush=True)
        except Exception:
            self.state, self.error = 'failed', traceback.format_exc()
            print(self.error, flush=True)
            return
        while not self.stopping.is_set() or not self.queue.empty():
            try:
                item = self.queue.get(timeout=.1)
            except queue.Empty:
                continue
            admitted = False
            try:
                item.check()
                while not self.stage_slots.acquire(timeout=.05):
                    item.check()
                admitted = True
                with self.lock:
                    self.active += 1
                    self.gpu_active = 1
                item.check()
                item.started_at = time.time()
                item.stage = 'gpu'
                item.pipeline_timing['gpu_started_at'] = time.time()
                prepared = self.model.prepare(item)
                item.pipeline_timing['gpu_finished_at'] = time.time()
                item.check()
                item.stage = 'waiting_for_cpu_export'
                self.exporter.submit(self._export, item, prepared)
            except Exception as error:
                if not isinstance(error, RequestExpired):
                    print(traceback.format_exc(), flush=True)
                self._complete(item, error=error, admitted=admitted)
            finally:
                self.gpu_active = 0
                # Do not retain the previous mesh while waiting for new work.
                prepared = None

    def close(self, timeout=30):
        self.stopping.set()
        self.state = 'stopping'
        with self.lock:
            for item in self.pending.values():
                item.canceled.set()
        self.worker.join(timeout=timeout)
        self.exporter.shutdown(wait=not self.worker.is_alive())


if __package__:
    from .openai_api import make_server
else:
    from openai_api import make_server


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Optional JSON configuration; relative paths use its directory')
    parser.add_argument('--model-dir', help='Local TRELLIS.2 model folder with pipeline.json')
    parser.add_argument('--gpu', help='GPU index or UUID; sets CUDA_VISIBLE_DEVICES')
    parser.add_argument('--port', type=int)
    parser.add_argument('--max-requests', type=int, help='Maximum admitted generations, up to 64 (default: 64)')
    parser.add_argument('--export-workers', type=int, help='CPU export threads, 1..4 (default: 2)')
    parser.add_argument('--pipeline-depth', type=int, help='GPU/CPU payload limit, 1..export-workers+1 (default: 3)')
    parser.add_argument('--api-key-env', help='Environment variable containing the API Bearer token')
    parser.add_argument('--api-output-dir', help='Server-managed input, receipt and GLB directory')
    parser.add_argument('--warmup-image', help='Optional 1024x1024 startup image')
    parser.add_argument('--memory-budget-gib', type=float)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    config = dict(root=str(repo), source_dir=str(repo),
        model_dir=str(repo / 'models/trellis2'), cache_home=str(repo / '.cache'),
        gpu=os.environ.get('CUDA_VISIBLE_DEVICES', '0'), host='127.0.0.1', port=3956,
        memory_budget_gib=32, cpu_threads=8, low_vram=False,
        native512_release_profile='fasttrellis2-v1',
        uv_group_face_cap=4096, native512_parallel_dense_cfg=True,
        native512_zero_negative_cross_attention=True, fused_pointwise=True,
        parallel_inpaint=True, max_requests=64, export_workers=2,
        api_key_env='FASTTRELLIS2_API_KEY')
    config_base = repo
    if args.config:
        config_file = Path(args.config).expanduser().resolve()
        config.update(json.loads(config_file.read_text(encoding='utf8')))
        config_base = config_file.parent
    overrides = dict(model_dir=args.model_dir, gpu=args.gpu, port=args.port,
        startup_warmup_image=args.warmup_image,
        memory_budget_gib=args.memory_budget_gib, max_requests=args.max_requests,
        export_workers=args.export_workers, pipeline_depth=args.pipeline_depth,
        api_key_env=args.api_key_env, api_output_dir=args.api_output_dir)
    for key, value in overrides.items():
        if value is not None:
            config[key] = value
    required = {'uv_group_face_cap': 4096, 'native512_parallel_dense_cfg': True,
                'native512_zero_negative_cross_attention': True, 'fused_pointwise': True,
                'low_vram': False}
    for key, value in required.items():
        if config.get(key) != value:
            parser.error(f'{key} must be {value!r} in the measured fast 512 profile')
    if config.get('native512_uv_strategy', 'whole_bounded') != 'whole_bounded':
        parser.error('Only the whole_bounded UV route is supported')
    if config.get('quality_profile_enabled', False):
        parser.error('Only the native512 profile is supported')
    if config.get('native512_resident_runtime', True) is not True:
        parser.error('The measured fast 512 profile requires the resident runtime')
    if type(config.get('max_requests')) is not int or not 1 <= config['max_requests'] <= 64:
        parser.error('max_requests must be an integer between 1 and 64')
    workers = config['export_workers']
    depth = config.setdefault('pipeline_depth', workers + 1 if type(workers) is int else 3)
    if type(workers) is not int or not 1 <= workers <= 4:
        parser.error('export_workers must be an integer between 1 and 4')
    if type(depth) is not int or not 1 <= depth <= workers + 1:
        parser.error('pipeline_depth must be an integer between 1 and export_workers+1')
    for key in ('root', 'source_dir', 'model_dir', 'cache_home', 'startup_warmup_image', 'api_output_dir'):
        if config.get(key):
            value = Path(config[key]).expanduser()
            base = Path.cwd() if key in ('model_dir', 'startup_warmup_image', 'api_output_dir') and overrides.get(key) else config_base
            config[key] = str(value if value.is_absolute() else (base / value).resolve())
    if not (Path(config['model_dir']) / 'pipeline.json').is_file():
        parser.error(f"Missing pipeline.json in model directory: {config['model_dir']}")
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config['gpu'])
    api_key = os.environ.get(config['api_key_env']) or None
    # Check listener credentials before allocating the resident GPU model.
    import ipaddress
    host = config.get('host', '127.0.0.1')
    try:
        loopback = ipaddress.ip_address(host).is_loopback
    except ValueError:
        loopback = host.lower() == 'localhost'
    if not loopback and not api_key:
        parser.error('A non-loopback API listener requires the configured API key environment variable')
    resident = Resident(config)
    server = make_server((host, config.get('port', 3956)), resident, api_key=api_key,
                         output_root=config.get('api_output_dir'))
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *_: threading.Thread(target=server.shutdown, daemon=True).start())
    try:
        server.serve_forever()
    finally:
        resident.close()
        server.server_close()


if __name__ == '__main__':
    main()
