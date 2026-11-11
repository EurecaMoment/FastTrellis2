"""Observe actual cache-miss tuning work without changing its selected kernels."""
from functools import wraps
import threading
import time


class KernelTuningStats:
    def __init__(self):
        from flex_gemm.utils.autotuner import PersistentCacheAutoTuner, TritonPersistentCacheAutotuner
        self.events = []
        self.local = threading.local()
        self._observe(PersistentCacheAutoTuner, '_benchmark', 'algorithm_selection')
        self._observe(TritonPersistentCacheAutotuner, '_bench', 'kernel_configuration')

    def _observe(self, cls, name, kind):
        original = getattr(cls, name)

        @wraps(original)
        def measured(instance, *args, **kwargs):
            depth = getattr(self.local, 'depth', 0)
            self.local.depth = depth + 1
            started = time.perf_counter()
            try:
                return original(instance, *args, **kwargs)
            finally:
                seconds = time.perf_counter() - started
                self.local.depth = depth
                self.events.append(dict(kind=kind, seconds=seconds, outermost=depth == 0))

        setattr(cls, name, measured)

    def clear(self):
        self.events.clear()

    def summary(self):
        return dict(
            algorithm_selection_runs=sum(x['kind'] == 'algorithm_selection' for x in self.events),
            kernel_configurations_benchmarked=sum(x['kind'] == 'kernel_configuration' for x in self.events),
            tuning_wall_seconds=sum(x['seconds'] for x in self.events if x['outermost']),
            scope='actual tuning calls on cache misses; ordinary compilation is not measured')
