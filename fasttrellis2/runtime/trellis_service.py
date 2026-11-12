"""Resident TRELLIS.2 mesh/PBR service with bounded requests and real deadlines."""
import argparse
import json
import os
from pathlib import Path
import queue
import re
import socket
import subprocess
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


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


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf8')
    temporary.replace(path)


class Item:
    def __init__(self, request, timeout):
        self.id = uuid.uuid4().hex
        self.request = request
        self.arrived = time.perf_counter()
        self.deadline = self.arrived + timeout
        self.canceled = threading.Event()
        self.done = threading.Event()
        self.result = None
        self.error = None

    def check(self, *_args):
        if self.canceled.is_set() or time.perf_counter() >= self.deadline:
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
        with self.torch.inference_mode():
            return self._infer(item)

    def _infer(self, item):
        from PIL import Image
        self.memory_monitor.admit(self.torch.cuda.memory_reserved(),
            self.config['memory_budget_gib']*1024**3)
        request = item.request
        profile, pipeline_type = resolve_profile(request)
        steps = 12
        texture_size = 4096
        output = Path(request['output_dir'])
        output.mkdir(parents=True, exist_ok=True)
        assets = []
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
                if not len(glb.faces) or not self.np.isfinite(glb.vertices).all():
                    raise ValueError('TRELLIS produced empty or nonfinite mesh geometry')
                material = glb.visual.material
                if getattr(material, 'baseColorTexture', None) is None or getattr(material, 'metallicRoughnessTexture', None) is None:
                    raise ValueError('TRELLIS mesh is missing PBR texture payloads')
                destination = output / (row['object_id'] + '.glb')
                temporary = output / (row['object_id'] + '.' + item.id + '.partial.glb')
                export_timing.update(self.native_runtime.export_glb(glb, temporary, 0))
                mark_outer('validation_encode_write_seconds')
                item.check()
                temporary.replace(destination)
                self.torch.cuda.synchronize()
                mark_outer('rename_and_final_sync_seconds')
                image_to_glb_seconds = time.perf_counter() - object_started
                asset = {'object_id': row['object_id'], 'asset': destination.name,
                    'model': 'microsoft/TRELLIS.2-4B', 'source_frame': 'y_up',
                    'representation': 'mesh_pbr', 'metric_reconstruction': False,
                    'camera_pose_estimated': False, 'seed': seed,
                    'pipeline': pipeline_type,
                    'sampling_steps': {'sparse': steps, 'shape': steps, 'texture': steps},
                    'vertices': len(glb.vertices), 'triangles': len(glb.faces),
                    'base_color_texture_size': list(material.baseColorTexture.size),
                    'metallic_roughness_texture_size': list(material.metallicRoughnessTexture.size),
                    'bytes': destination.stat().st_size, 'inputs': inputs,
                    'generation_seconds': generation_seconds,
                    'export_seconds': time.perf_counter() - export_start,
                    'export_profile': native_export_profile(self.config),
                    'profile': profile, 'remesh': remesh, 'generation_timing': generation_timing,
                    'export_timing': export_timing,
                    'image_to_glb_seconds': image_to_glb_seconds, 'outer_timing': outer_timing,
                    'diagnostic_capture': request.get('capture_optimization_inputs') is True,
                    'uv_mode': 'xatlas', 'texture_sampling': 'sparse'}
                asset.update(self.native_runtime.details())
                write_json(destination.with_suffix('.complete.json'), {'asset': asset,
                    'provider': 'trellis2', 'request_id': item.id, 'status': 'complete'})
                assets.append(asset)
                self.native_runtime.end()
                del mesh, glb
            self.torch.cuda.synchronize()
            item.check()
            result = {'status': 'ok', 'provider': 'trellis2', 'model': 'microsoft/TRELLIS.2-4B',
                'assets': assets, 'cache_hit': False, 'asset_cache_hit': False,
                'cache_hit_scope': 'generated assets only; inference caches are reported per object',
                'representations': ['mesh_pbr'],
                'timing': {'queue_seconds': started-item.arrived,
                    'compute_and_export_seconds': time.perf_counter()-started,
                    'total_seconds': time.perf_counter()-item.arrived},
                'peak_allocated_gib': self.torch.cuda.max_memory_allocated()/1024**3,
                'peak_reserved_gib': self.torch.cuda.max_memory_reserved()/1024**3,
                'memory_budget_gib': self.config['memory_budget_gib'],
                'process_memory_limit_bytes':50_000_000_000,'gpu_headroom_gib':2}
            write_json(output / 'trellis_result.json', result)
            return result
        finally:
            try:
                self.native_runtime.end()
            finally:
                self.current = None


class Resident:
    def __init__(self, config, model_factory=TrellisModel):
        self.config = config
        self.queue = queue.Queue(maxsize=config.get('max_requests', 64))
        self.slots = threading.BoundedSemaphore(config.get('max_requests', 64))
        self.state, self.error = 'loading', None
        self.completed = self.failed = self.expired = self.active = 0
        self.history = []
        self.factory = model_factory
        threading.Thread(target=self.run, daemon=True).start()

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
            'active_batch': self.active, 'max_requests': self.config.get('max_requests', 64),
            'max_batch': 1, 'completed': self.completed, 'failed': self.failed,
            'expired': self.expired, 'latency_recent': self.history[-64:],
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
        resolve_profile(request)
        rows = request.get('objects')
        if not isinstance(rows, list) or not rows or not Path(request.get('output_dir', '')).is_absolute():
            raise ValueError('objects and an absolute output_dir are required')
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
            self.queue.put_nowait(item)
        except BaseException:
            self.slots.release()
            raise
        return item

    def run(self):
        try:
            self.model = self.factory(self.config)
            self.state = 'ready'
            print(json.dumps(self.status()), flush=True)
        except Exception:
            self.state, self.error = 'failed', traceback.format_exc()
            print(self.error, flush=True)
            return
        while True:
            item = self.queue.get()
            self.active = 1
            try:
                item.check()
                item.result = self.model.infer(item)
                self.completed += 1
                self.history.append(time.perf_counter()-item.arrived)
            except RequestExpired as error:
                item.error = str(error)
                self.expired += 1
            except Exception as error:
                item.error = str(error)
                self.failed += 1
                print(traceback.format_exc(), flush=True)
            finally:
                item.done.set()
                self.slots.release()
                self.active = 0
                self.queue.task_done()


class Server(ThreadingHTTPServer):
    request_queue_size = 256
    daemon_threads = True


def make_server(address, resident):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def reply(self, status, value):
            data = json.dumps(value).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            if self.path in ('/health', '/ready', '/metrics'):
                self.reply(503 if self.path == '/ready' and resident.state != 'ready' else 200, resident.status())
            else:
                self.reply(404, {'error': 'Unknown endpoint'})

        def do_POST(self):
            if self.path != '/infer':
                self.reply(404, {'error': 'Unknown endpoint'})
                return
            if resident.state != 'ready':
                self.reply(503, resident.status())
                return
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 4*1024**2:
                    raise ValueError('Request JSON size must be between 1 byte and 4 MiB')
                request = json.loads(self.rfile.read(size))
                try:
                    from fit_paths import FitPaths
                except ModuleNotFoundError:
                    transport = None
                else:
                    transport = FitPaths(request)
                timeout = float(request.get('deadline_seconds', 3600))
                if not .01 <= timeout <= 3600:
                    raise ValueError('deadline_seconds must be between 0.01 and 3600')
                item = resident.submit(request, timeout)
                if not item.done.wait(max(0, item.deadline-time.perf_counter())):
                    item.canceled.set()
                    self.reply(504, {'status': 'deadline_exceeded', 'request_id': item.id,
                        'completed_artifact': False, 'provider': 'trellis2'})
                elif item.error:
                    self.reply(504 if 'deadline' in item.error.lower() else 500,
                        {'status': 'failed', 'error': item.error, 'provider': 'trellis2'})
                else:
                    self.reply(200, transport.finish(item.result) if transport else item.result)
            except queue.Full:
                self.reply(429, {'error': '64-request capacity reached; no extra GPU work was dispatched'})
            except (ValueError, KeyError, TypeError) as error:
                self.reply(400, {'error': str(error)})

    return Server(address, Handler)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Optional JSON configuration; relative paths use its directory')
    parser.add_argument('--model-dir', help='Local TRELLIS.2 model folder with pipeline.json')
    parser.add_argument('--gpu', help='GPU index or UUID; sets CUDA_VISIBLE_DEVICES')
    parser.add_argument('--port', type=int)
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
        parallel_inpaint=True, max_requests=32)
    config_base = repo
    if args.config:
        config_file = Path(args.config).expanduser().resolve()
        config.update(json.loads(config_file.read_text(encoding='utf8')))
        config_base = config_file.parent
    overrides = dict(model_dir=args.model_dir, gpu=args.gpu, port=args.port,
        startup_warmup_image=args.warmup_image,
        memory_budget_gib=args.memory_budget_gib)
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
    for key in ('root', 'source_dir', 'model_dir', 'cache_home', 'startup_warmup_image'):
        if config.get(key):
            value = Path(config[key]).expanduser()
            base = Path.cwd() if key in ('model_dir', 'startup_warmup_image') and overrides.get(key) else config_base
            config[key] = str(value if value.is_absolute() else (base / value).resolve())
    if not (Path(config['model_dir']) / 'pipeline.json').is_file():
        parser.error(f"Missing pipeline.json in model directory: {config['model_dir']}")
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config['gpu'])
    resident = Resident(config)
    make_server((config.get('host', '127.0.0.1'), config.get('port', 3956)), resident).serve_forever()


if __name__ == '__main__':
    main()
