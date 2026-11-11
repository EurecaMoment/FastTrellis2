"""Reuse scratch over unchanged cleanup, retaining every cudaFree barrier."""
from contextlib import contextmanager
import ctypes
import time

import torch


class GeometryScratchPool:
    def __init__(self):
        self.handle=ctypes.CDLL(None)
        if not hasattr(self.handle,'trellis_pool_sync_supported') or not self.handle.trellis_pool_sync_supported():
            raise RuntimeError('Geometry pooling requires the synchronization-preserving allocator')
        self.handle.trellis_pool_begin.argtypes=[ctypes.c_int]
        self.handle.trellis_pool_stats.argtypes=[ctypes.POINTER(ctypes.c_double)]
        self.last_timing={}

    @contextmanager
    def __call__(self):
        torch.cuda.synchronize()
        started=time.perf_counter()
        self.handle.trellis_pool_begin(2)
        try:
            yield
        finally:
            torch.cuda.synchronize()
            values=(ctypes.c_double*6)()
            self.handle.trellis_pool_stats(values)
            self.handle.trellis_pool_end()
            self.last_timing=dict(seconds=time.perf_counter()-started,malloc_calls=values[0],
                free_calls=values[1],reused_allocations=values[2],malloc_seconds=values[3],
                free_seconds=values[4],cached_bytes_before_release=values[5])
