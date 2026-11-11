"""Fast device/process accounting without starting nvidia-smi for every request."""
import ctypes
import os


class MemoryInfo(ctypes.Structure):
    _fields_=[('total',ctypes.c_ulonglong),('free',ctypes.c_ulonglong),('used',ctypes.c_ulonglong)]


class ProcessInfo(ctypes.Structure):
    _fields_=[('pid',ctypes.c_uint),('used',ctypes.c_ulonglong),
              ('gpu_instance',ctypes.c_uint),('compute_instance',ctypes.c_uint)]


class DeviceMemory:
    def __init__(self,device):
        self.nvml=ctypes.CDLL('libnvidia-ml.so.1')
        self._check(self.nvml.nvmlInit_v2())
        self.handle=ctypes.c_void_p()
        if str(device).isdigit():
            self._check(self.nvml.nvmlDeviceGetHandleByIndex_v2(
                int(device), ctypes.byref(self.handle)))
        else:
            self._check(self.nvml.nvmlDeviceGetHandleByUUID(
                str(device).encode(), ctypes.byref(self.handle)))

    @staticmethod
    def _check(status):
        if status:
            raise RuntimeError(f'NVML memory query failed with status {status}')

    def snapshot(self):
        memory=MemoryInfo()
        self._check(self.nvml.nvmlDeviceGetMemoryInfo(self.handle,ctypes.byref(memory)))
        count=ctypes.c_uint(64)
        processes=(ProcessInfo*64)()
        self._check(self.nvml.nvmlDeviceGetComputeRunningProcesses_v2(self.handle,ctypes.byref(count),processes))
        own=sum(processes[i].used for i in range(count.value) if processes[i].pid==os.getpid())
        return {'total_bytes':memory.total,'free_bytes':memory.free,'process_bytes':own}

    def admit(self,reserved_bytes,budget_bytes,limit_bytes=50_000_000_000):
        current=self.snapshot()
        additional=max(0,budget_bytes-reserved_bytes)
        native_reserve=2*1024**3
        headroom=2*1024**3
        if current['process_bytes']+additional+native_reserve>limit_bytes:
            raise RuntimeError('Request would exceed the 50 GB process memory budget')
        if current['free_bytes']<additional+native_reserve+headroom:
            raise RuntimeError('Not enough currently free GPU memory for the bounded request')
        return current
