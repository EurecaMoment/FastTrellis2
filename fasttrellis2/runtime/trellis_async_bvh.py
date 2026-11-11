"""Overlap the installed CPU BVH builder with independent native mesh cleanup."""
from concurrent.futures import ThreadPoolExecutor
import time

import numpy as np
import torch
import cumesh


class AsyncBVHFactory:
    def __init__(self):
        import trellis_bvh_async
        self.extension = trellis_bvh_async
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.original = cumesh.cuBVH
        self.last_timing = {}

    def __call__(self, vertices, triangles):
        started = time.perf_counter()
        vertices = vertices.detach().cpu().numpy() if torch.is_tensor(vertices) else np.asarray(vertices)
        triangles = triangles.detach().cpu().numpy() if torch.is_tensor(triangles) else np.asarray(triangles)
        triangles = np.ascontiguousarray(triangles, dtype=np.uint32)
        snapshot_seconds = time.perf_counter() - started
        original, extension = self.original, self.extension
        device = torch.cuda.current_device()
        timing = {'input_snapshot_seconds': snapshot_seconds}
        self.last_timing = timing

        def build():
            torch.cuda.set_device(device)
            started = time.perf_counter()
            result = original.__new__(original)
            result.impl = extension.create(vertices, triangles)
            timing['build_seconds'] = time.perf_counter() - started
            return result

        future = self.executor.submit(build)

        class PendingBVH:
            def __getattr__(self, name):
                started = time.perf_counter()
                result = future.result()
                timing['wait_seconds'] = timing.get('wait_seconds', 0) + time.perf_counter() - started
                return getattr(result, name)

        return PendingBVH()

    def install(self):
        cumesh.cuBVH = self

    def close(self):
        cumesh.cuBVH = self.original
        self.executor.shutdown(wait=True)
