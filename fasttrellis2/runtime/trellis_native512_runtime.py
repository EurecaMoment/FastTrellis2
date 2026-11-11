"""Resident integration of the measured native 512, 12-step PBR pipeline.

Only weights, execution graphs, time constants and allocator capacity persist.
Image conditions, sparse metadata, deferred work and pixels are cleared per object.
"""
import ctypes
import importlib
import os
from pathlib import Path
import time

import torch


class Native512Runtime:
    export_profile = 'fasttrellis2-v1'

    def __init__(self, model):
        if model.pipeline.low_vram:
            raise ValueError('Resident native512 graphs require stationary GPU weights')
        self.model = model
        self.export_profile = model.config.get('native512_release_profile', self.export_profile)
        self.pipeline = model.pipeline
        self.modules = []
        self.active = False
        self.deferred = None
        self.geometry_futures = []
        self.bvhs = []
        self.diagnostic_dir = None
        self.uv_capture_dir = None
        self.uv_capture_slow_seconds = 0.0
        self.captured = {}
        self.original_atlas = None
        self.uv_group_face_cap = int(model.config.get('uv_group_face_cap', 4096))
        if self.uv_group_face_cap < 1:
            raise ValueError('uv_group_face_cap must be positive')
        self.atlas_module = 'trellis_xatlas_bounded_whole'
        os.environ.update(TRELLIS_XATLAS_THREADS='16', TRELLIS_XATLAS_RNG_MODE='1',
            TRELLIS_BOUNDED_XATLAS_MODULE=self.atlas_module,
            TRELLIS_XATLAS_PRE_GROUP_CAP=str(self.uv_group_face_cap))
        if os.environ.get('TRELLIS_UV_ASYNC_NORMALS') == '1':
            raise ValueError('The final UV route does not use the older normal hook')

        from trellis_inference_exact import ExactMetadataReuse, ExactConditionCache, SamplesOnly
        from trellis_fused_exact import FusedBlocks
        from fused_rms_pointwise import FusedRMSPointwise
        from invariant_inference_cache import FixedTimestepCache
        from fused_rope_exact import FusedRotary
        from dense_graph_static_kv import DenseGraphStaticKV
        from PIL import Image

        self.metadata = ExactMetadataReuse(self.pipeline)
        self.modules.append(self.metadata)
        self.conditions = ExactConditionCache(self.pipeline)
        self.modules.append(self.conditions)
        self.samples_only = SamplesOnly(self.pipeline)
        self.modules.append(self.samples_only)
        self.modules.append(FusedBlocks(self.pipeline))
        self.modules.append(FusedRMSPointwise(self.pipeline))
        self.time_cache = FixedTimestepCache(self.pipeline)
        self.modules.append(self.time_cache)
        self.modules.append(FusedRotary(self.pipeline, 2))
        with torch.inference_mode():
            shape = tuple(self.pipeline.get_cond([Image.new('RGB', (512, 512), (127, 127, 127))], 512)['cond'].shape)
            graph_class = DenseGraphStaticKV
            if model.config.get('native512_zero_negative_cross_attention', False):
                from zero_condition_dense_graph import ZeroConditionDenseGraph
                graph_class = ZeroConditionDenseGraph
            self.dense_graph = graph_class(self.pipeline.models['sparse_structure_flow_model'], shape)
        self.negative_shape_cache = None
        if model.config.get('native512_zero_negative_cross_attention', False):
            from zero_condition_sparse_cache import ZeroConditionSparseCache
            self.negative_shape_cache = ZeroConditionSparseCache(self.pipeline.models['shape_slat_flow_model_512'])
            self.negative_shape_cache.enabled = True
            self.modules.append(self.negative_shape_cache)
        self.parallel_cfg = None
        if model.config.get('native512_parallel_dense_cfg', False):
            from concurrent_dense_cfg import ConcurrentDenseCFG
            self.parallel_cfg = ConcurrentDenseCFG(self.pipeline.models['sparse_structure_flow_model'])
            self.modules.append(self.parallel_cfg)

        atlas = importlib.import_module(self.atlas_module)
        atlas.prepare_rng()
        from trellis_async_bvh import AsyncBVHFactory
        from trellis_geometry_pool import GeometryScratchPool
        from trellis_texture_exact import ExactTextureLaunch
        from trellis_inpaint_stream import StreamingTelea
        from trellis_geometry_export import PipelinedGeometryExport
        from trellis_pbr_channels import extract_pbr_channels
        import trellis_glb_direct
        from trellis_png_fast import encode_png

        self.bvh_factory = AsyncBVHFactory()
        self.geometry_scope = GeometryScratchPool()
        self.texture_launch = ExactTextureLaunch()
        self.texture_launch.warm()
        self.telea = StreamingTelea(blocks=432, module_name='trellis_telea_stream_seeded',
            async_copy=True, pack_material=True)
        self.geometry_export = PipelinedGeometryExport()
        self.extract_channels = extract_pbr_channels
        trellis_glb_direct.encode_png = encode_png
        self.export_glb = trellis_glb_direct.export_glb_direct

        arena = ctypes.CDLL(None)
        arena.trellis_pool_prepare_arena.argtypes = [ctypes.c_size_t]
        self.arena_bytes = 3072 * 1024**2
        result = arena.trellis_pool_prepare_arena(self.arena_bytes)
        if result:
            raise RuntimeError(f'Cannot prepare bounded CUDA arena: {result}')
        self.clear_conditions()
        from trellis_kernel_cache_stats import KernelTuningStats
        self.kernel_tuning = KernelTuningStats()

    def clear_conditions(self):
        self.conditions.clear()
        self.metadata.clear()
        self.dense_graph.clear()
        if self.negative_shape_cache is not None:
            self.negative_shape_cache.clear()

    def begin(self, diagnostic_dir=None, uv_capture_dir=None, uv_capture_slow_seconds=0.0):
        if self.active:
            raise RuntimeError('Only one object may use the native512 runtime at a time')
        self.clear_conditions()
        self.active = True
        self.cache_baseline = (self.conditions.hits, self.conditions.misses,
            self.dense_graph.refreshes, self.dense_graph.replays)
        self.parallel_cfg_baseline = self.parallel_cfg.paired_evaluations if self.parallel_cfg else 0
        self.negative_dense_baseline = getattr(self.dense_graph, 'constant_graph_replays', 0)
        self.kernel_tuning.clear()
        if diagnostic_dir is not None:
            self.diagnostic_dir = Path(diagnostic_dir)
            self.diagnostic_dir.mkdir(parents=True, exist_ok=False)
        if uv_capture_dir is not None:
            self.uv_capture_dir = Path(uv_capture_dir)
            self.uv_capture_slow_seconds = float(uv_capture_slow_seconds)
        if self.diagnostic_dir is not None or self.uv_capture_dir is not None:
            atlas_module = importlib.import_module(self.atlas_module)
            self.original_atlas = atlas_module.Atlas
            original, captured = self.original_atlas, self.captured
            class CapturedAtlas:
                def __init__(self):
                    self.value = original()
                def add_meshes(self, vertices, faces, vo, fo):
                    self.started = time.perf_counter()
                    captured.update(chart_vertices=vertices, chart_faces=faces, vo=vo, fo=fo)
                    return self.value.add_meshes(vertices, faces, vo, fo)
                def get_meshes(self, mapping, vo):
                    values = self.value.get_meshes(mapping, vo)
                    captured.update(mapping=mapping, outputs=values,
                        cpu_uv_wall_seconds=time.perf_counter()-self.started)
                    return values
                def release(self):
                    started = time.perf_counter()
                    try:
                        return self.value.release()
                    finally:
                        captured['cpu_uv_release_seconds'] = time.perf_counter() - started
                def __getattr__(self, name):
                    return getattr(self.value, name)
            atlas_module.Atlas = CapturedAtlas
        self.telea.last_timing = {}
        self.telea.last_material_pixels = None

    def generate(self, image, seed, check):
        if not self.active:
            raise RuntimeError('Native generation requires an active native512 request')
        from trellis_pipeline_exact import generate_shape_first
        def capture(shape, mesh):
            torch.save(dict(shape_feats=shape.feats.cpu(), shape_coords=shape.coords.cpu(),
                vertices=mesh.vertices.cpu(), faces=mesh.faces.cpu()), self.diagnostic_dir / 'decoded_source.pt')
        mesh, timing, self.deferred = generate_shape_first(self.pipeline, image, seed, check=check,
            capture=capture if self.diagnostic_dir is not None else None)
        return mesh, timing

    def _prepare_geometry(self, *args):
        future = self.geometry_export(*args)
        self.geometry_futures.append(future)
        return future

    def _build_bvh(self, *args):
        value = self.bvh_factory(*args)
        self.bvhs.append(value)
        return value

    def postprocess_arguments(self):
        if self.deferred is None:
            raise RuntimeError('Shape generation must precede material export')
        return dict(attr_volume=None, coords=self.deferred.shape.coords[:, 1:],
            attr_layout=self.pipeline.pbr_attr_layout, voxel_size=1/512,
            on_uv_cpu_start=self.deferred.start, deferred_material=self.deferred.result,
            bvh_factory=self._build_bvh, exact_inpaint=self.telea,
            on_texture_mask_ready=self.telea.start_plan, geometry_scope=self.geometry_scope,
            exact_channel_extract=self.extract_channels,
            geometry_export_prepare=self._prepare_geometry,
            material_pixels=lambda: self.telea.last_material_pixels)

    def details(self):
        return dict(single_object_pipeline=dict(self.deferred.timing),
            telea=dict(self.telea.last_timing), geometry_scope=dict(self.geometry_scope.last_timing),
            async_bvh=dict(self.bvh_factory.last_timing), startup_scratch_arena_bytes=self.arena_bytes,
            xatlas_implementation=self.atlas_module, uv_strategy='whole_bounded',
            uv_group_face_cap=self.uv_group_face_cap,
            uv_cpu_release_seconds=self.captured.get('cpu_uv_release_seconds'),
            inference_execution=dict(dense_cfg_parallel=self.parallel_cfg is not None,
                dense_cfg_parallel_pairs=(self.parallel_cfg.paired_evaluations-self.parallel_cfg_baseline
                    if self.parallel_cfg else 0)),
            slow_uv_capture=(str(self.uv_capture_dir / 'xatlas_input.pt')
                if self.uv_capture_dir is not None
                and self.captured.get('cpu_uv_wall_seconds', 0) >= self.uv_capture_slow_seconds else None),
            inference_caches=dict(
                image_kv_projection_hits=self.conditions.hits-self.cache_baseline[0],
                image_kv_projection_misses=self.conditions.misses-self.cache_baseline[1],
                dense_graph_condition_refreshes=self.dense_graph.refreshes-self.cache_baseline[2],
                dense_graph_replays=self.dense_graph.replays-self.cache_baseline[3],
                dense_graph_condition_hits=(self.dense_graph.replays-self.cache_baseline[3])
                    -(self.dense_graph.refreshes-self.cache_baseline[2]),
                fixed_time_conditioning=True, stale_self_attention_kv_reuse=False,
                image_cache_scope='one object; cleared before the next object',
                zero_negative_cross_attention=dict(
                    enabled=self.negative_shape_cache is not None,
                    structure_constant_replays=getattr(self.dense_graph, 'constant_graph_replays', 0)-self.negative_dense_baseline,
                    structure_constant_bytes=getattr(self.dense_graph, 'constant_output_bytes', 0),
                    shape=self.negative_shape_cache.details() if self.negative_shape_cache is not None else None,
                    texture='not used; the native material schedule has no negative branch'),
                kernel_tuning=self.kernel_tuning.summary()))

    def end(self):
        if not self.active:
            return
        try:
            if self.parallel_cfg is not None:
                self.parallel_cfg.stream.synchronize()
            if self.deferred is not None:
                self.deferred.close()
            self.telea.cancel_plan()
            for future in self.geometry_futures:
                future.result()
            for bvh in self.bvhs:
                bvh.impl  # Join CPU/GPU construction even after a failed export.
        finally:
            if self.original_atlas is not None:
                importlib.import_module(self.atlas_module).Atlas = self.original_atlas
                self.original_atlas = None
            if self.diagnostic_dir is not None:
                if 'outputs' in self.captured:
                    torch.save(self.captured, self.diagnostic_dir / 'xatlas_input.pt')
            elif (self.uv_capture_dir is not None and 'outputs' in self.captured
                    and self.captured.get('cpu_uv_wall_seconds', 0) >= self.uv_capture_slow_seconds):
                # These tensors are already on CPU. Persist after the complete
                # GLB has been written; HTTP latency includes this diagnostic.
                self.uv_capture_dir.mkdir(parents=True, exist_ok=False)
                torch.save(self.captured, self.uv_capture_dir / 'xatlas_input.pt')
            self.captured.clear()
            self.diagnostic_dir = None
            self.uv_capture_dir = None
            self.uv_capture_slow_seconds = 0.0
            self.deferred = None
            self.geometry_futures.clear()
            self.bvhs.clear()
            self.telea.last_material_pixels = None
            self.clear_conditions()
            self.active = False
