"""Overlap a single object's material inference with its native CPU UV work."""
from concurrent.futures import ThreadPoolExecutor
import time

import torch


class DeferredMaterial:
    """Retain the native sampler arithmetic and the request's original noise."""

    def __init__(self, pipeline, cond, coords, shape, noise, subs, started):
        self.pipeline, self.cond = pipeline, cond
        self.coords, self.shape, self.noise, self.subs = coords, shape, noise, subs
        self.started = started
        self.timing = {}
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='trellis-material')
        self.future = None
        self.texture = self.voxels = None

    def start(self):
        if self.future is not None:
            raise RuntimeError('Material branch already started')
        self.timing['uv_cpu_start_offset'] = time.perf_counter() - self.started
        self.future = self.executor.submit(self._run)

    def _run(self):
        p = self.pipeline
        with torch.inference_mode(), torch.cuda.device(self.shape.device):
            self.timing['material_start_offset'] = time.perf_counter() - self.started
            flow = p.models['tex_slat_flow_model_512']
            # Same order of arithmetic as sample_tex_slat; noise was drawn on
            # CPU immediately after shape sampling, before reordering decode.
            std = torch.tensor(p.shape_slat_normalization['std'])[None].to(self.shape.device)
            mean = torch.tensor(p.shape_slat_normalization['mean'])[None].to(self.shape.device)
            normalized = (self.shape - mean) / std
            noise = normalized.replace(feats=self.noise.to(p.device))
            texture = p.tex_slat_sampler.sample(flow, noise, concat_cond=normalized,
                **self.cond, **{**p.tex_slat_sampler_params, 'steps': 12},
                verbose=True, tqdm_desc='Sampling texture SLat').samples
            std = torch.tensor(p.tex_slat_normalization['std'])[None].to(texture.device)
            mean = torch.tensor(p.tex_slat_normalization['mean'])[None].to(texture.device)
            self.texture = texture * std + mean
            torch.cuda.synchronize()
            self.timing['material_sample_end_offset'] = time.perf_counter() - self.started
            self.voxels = p.decode_tex_slat(self.texture, self.subs)
            torch.cuda.synchronize()
            self.timing['material_end_offset'] = time.perf_counter() - self.started
            voxel = self.voxels[0]
            return {'attrs': voxel.feats, 'coords': voxel.coords[:, 1:], 'layout': p.pbr_attr_layout}

    def result(self):
        if self.future is None:
            raise RuntimeError('UV work did not start the material branch')
        self.timing['uv_cpu_end_offset'] = time.perf_counter() - self.started
        result = self.future.result()
        self.timing['join_end_offset'] = time.perf_counter() - self.started
        left = max(self.timing['uv_cpu_start_offset'], self.timing['material_start_offset'])
        right = min(self.timing['uv_cpu_end_offset'], self.timing['material_end_offset'])
        self.timing['measured_material_uv_overlap_seconds'] = max(0.0, right - left)
        return result

    def latents(self):
        return {'coords': self.coords.cpu(), 'shape': self.shape.feats.cpu(),
                'texture': self.texture.feats.cpu()}

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)


def generate_shape_first(pipeline, image, seed, check=None, capture=None):
    """Generate original geometry and defer material until the UV CPU window."""
    timing = {}
    started = previous = time.perf_counter()

    def mark(name):
        nonlocal previous
        torch.cuda.synchronize()
        now = time.perf_counter()
        timing[name] = now - previous
        previous = now
        if check is not None:
            check()

    part_started = time.perf_counter()
    image = pipeline.preprocess_image(image)
    timing['image_preprocess_seconds'] = time.perf_counter() - part_started
    part_started = time.perf_counter()
    torch.manual_seed(seed)
    timing['seed_setup_seconds'] = time.perf_counter() - part_started
    part_started = time.perf_counter()
    cond = pipeline.get_cond([image], 512)
    timing['feature_submission_seconds'] = time.perf_counter() - part_started
    mark('condition')
    coords = pipeline.sample_sparse_structure(cond, 32, 1, {'steps': 12})
    mark('structure')
    shape = pipeline.sample_shape_slat(cond, pipeline.models['shape_slat_flow_model_512'],
        coords, {'steps': 12})
    mark('shape')
    flow = pipeline.models['tex_slat_flow_model_512']
    noise = torch.randn(shape.coords.shape[0], flow.in_channels - shape.feats.shape[1])
    meshes, subs = pipeline.decode_shape_slat(shape, 512)
    mark('shape_decode_seconds')
    mesh = meshes[0]
    if capture is not None:
        capture(shape, mesh)
        mark('diagnostic_capture_seconds')
    mesh.fill_holes()
    mark('initial_hole_fill_seconds')
    timing['shape_decode_and_holes'] = timing['shape_decode_seconds'] + timing['initial_hole_fill_seconds']
    deferred = DeferredMaterial(pipeline, cond, coords, shape, noise, subs, started)
    timing['total_before_geometry'] = time.perf_counter() - started
    return mesh, timing, deferred


class Native512PipelineRequest:
    """Keep request-local model patches alive until deferred material finishes."""
    def __init__(self,pipeline,fused_pointwise=True,check=None):
        from trellis_inference_exact import ExactMetadataReuse,ExactConditionCache,SamplesOnly
        self.pipeline=pipeline
        self.check=check
        self.deferred=None
        self.modules=[]
        try:
            self.modules.append(ExactMetadataReuse(pipeline))
            self.modules.append(ExactConditionCache(pipeline))
            self.modules.append(SamplesOnly(pipeline))
            if fused_pointwise:
                from trellis_fused_exact import FusedBlocks
                self.modules.append(FusedBlocks(pipeline))
        except BaseException:
            self.close()
            raise

    def generate(self,image,seed):
        mesh,timing,self.deferred=generate_shape_first(self.pipeline,image,seed,check=self.check)
        return mesh,timing

    def material_arguments(self):
        return dict(attr_volume=None,coords=self.deferred.shape.coords[:,1:],
            attr_layout=self.pipeline.pbr_attr_layout,voxel_size=1/512,
            on_uv_cpu_start=self.deferred.start,deferred_material=self.deferred.result)

    def close(self):
        if self.deferred is not None:
            self.deferred.close()
        for module in reversed(self.modules):
            module.close()
        self.modules.clear()
