"""Final bounded whole-mesh UV route with xatlas parameterization and packing."""
import ctypes
import importlib
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
import os
import time

import torch

_atlas_cleanup = ThreadPoolExecutor(max_workers=1, thread_name_prefix='trellis-atlas-cleanup')


@contextmanager
def scratch_cache(timings=None):
    """Pool only synchronous CuMesh work on the calling worker thread."""
    handle = ctypes.CDLL(None)
    if not hasattr(handle, 'trellis_pool_begin'):
        yield
        return
    handle.trellis_pool_begin.argtypes = [ctypes.c_int]
    torch.cuda.synchronize()
    mode = 3 if os.environ.get('TRELLIS_UV_ASYNC_NORMALS') == '1' else 1
    if mode == 3 and not hasattr(handle, 'trellis_pool_uv_stats'):
        raise RuntimeError('Asynchronous normal cones require the verified UV allocator')
    handle.trellis_pool_begin(mode)
    try:
        yield
    finally:
        torch.cuda.synchronize()
        if timings is not None and hasattr(handle, 'trellis_pool_stats'):
            handle.trellis_pool_stats.argtypes = [ctypes.POINTER(ctypes.c_double)]
            values = (ctypes.c_double * 6)()
            handle.trellis_pool_stats(values)
            timings['uv_allocator'] = dict(zip(
                ('malloc_calls', 'free_calls', 'reused_allocations',
                 'malloc_seconds', 'free_seconds', 'cached_bytes'), values))
            if hasattr(handle, 'trellis_pool_uv_stats'):
                handle.trellis_pool_uv_stats.argtypes = [ctypes.POINTER(ctypes.c_double)]
                barriers = (ctypes.c_double * 3)()
                handle.trellis_pool_uv_stats(barriers)
                timings['uv_barriers'] = dict(zip(
                    ('skipped_normal_barriers', 'retained_stream_barriers', 'stream_barrier_seconds'), barriers))
        handle.trellis_pool_end()


def unwrap_bounded(mesh, compute_charts_kwargs, return_vmaps=True, verbose=False,
                   timings=None, on_cpu_start=None, cleanup_futures=None,
                   face_cap=4096):
    """Use unchanged xatlas parameterization on bounded connected face groups.

    Keep the final degenerate-face cleanup, xatlas parameterization, and packing.
    """
    if face_cap < 1:
        raise ValueError('face_cap must be positive')
    extension = importlib.import_module(os.environ.get(
        'TRELLIS_BOUNDED_XATLAS_MODULE', 'trellis_xatlas_bounded_whole'))
    os.environ['TRELLIS_XATLAS_PRE_GROUP_CAP'] = str(face_cap)
    entered = time.perf_counter()
    with scratch_cache(timings):
        mesh.remove_degenerate_faces()
    cleanup_seconds = time.perf_counter() - entered
    vertices, faces = mesh.read()
    cpu_vertices = vertices.cpu().contiguous()
    cpu_faces = faces.to(dtype=torch.int32, device='cpu').contiguous()
    mapping = torch.arange(len(cpu_vertices), dtype=torch.int64)
    vertex_offsets = torch.tensor([0, len(cpu_vertices)], dtype=torch.int64)
    face_offsets = torch.tensor([0, len(cpu_faces)], dtype=torch.int64)
    if timings is not None:
        timings.update(uv_gpu_charts_seconds=0.0,
                       uv_final_cleanup_seconds=cleanup_seconds,
                       uv_chart_transfer_seconds=time.perf_counter()-entered-cleanup_seconds,
                       uv_prepartition='bounded connected face groups before xatlas',
                       uv_group_face_cap=face_cap)
    if on_cpu_start is not None:
        on_cpu_start()
    started = time.perf_counter()
    atlas = extension.Atlas()
    try:
        atlas.add_meshes(cpu_vertices, cpu_faces, vertex_offsets, face_offsets)
        after_add = time.perf_counter()
        options = extension.ChartOptions()
        options.max_iterations = 1
        atlas.compute_charts(options, None)
        after_compute = time.perf_counter()
        atlas.pack_charts(extension.PackOptions(), None)
        after_pack = time.perf_counter()
        vmaps, out_faces, uvs = atlas.get_meshes(mapping, vertex_offsets)
        out_vertices = vertices[vmaps.to(vertices.device)]
        after_gather = time.perf_counter()
        if timings is not None:
            timings.update(uv_cpu_add_seconds=after_add-started,
                           uv_cpu_parameterize_seconds=after_compute-after_add,
                           uv_cpu_pack_seconds=after_pack-after_compute,
                           uv_cpu_gather_seconds=after_gather-after_pack,
                           uv_cpu_atlas_seconds=after_gather-started)
        if verbose:
            print(f'XATLAS_BOUNDED faces={len(cpu_faces)} cap={face_cap} '
                  f'seconds={after_gather-started:.6f}', flush=True)
        if cleanup_futures is not None:
            cleanup_futures.append(_atlas_cleanup.submit(atlas.release))
        else:
            atlas.release()
        result = (out_vertices, out_faces, uvs)
        return result + (vmaps,) if return_vmaps else result
    except BaseException:
        atlas.release()
        raise
