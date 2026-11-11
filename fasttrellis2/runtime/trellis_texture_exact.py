"""Use the native trilinear kernel without request-time launch autotuning."""
import importlib
import torch
import triton


class ExactTextureLaunch:
    def __init__(self):
        module=importlib.import_module('flex_gemm.kernels.triton.grid_sample.indice_weighed_sum_fwd')
        self.module=module
        self.kernel=module.indice_weighed_sum_fwd_kernel
        self.original_configs=self.kernel.configs
        # This is one of the library's native choices. It changes only thread
        # placement; all eight weights and FMA operations are the same kernel.
        self.kernel.configs=[triton.Config({'BM':32,'BK':8},num_warps=2)]

    def warm(self):
        with torch.inference_mode():
            values=torch.zeros((1024,6),dtype=torch.float32,device='cuda')
            indices=torch.zeros((257,8),dtype=torch.int32,device='cuda')
            weights=torch.ones((257,8),dtype=torch.float32,device='cuda')
            self.module.indice_weighed_sum_fwd(values,indices,weights)
            torch.cuda.synchronize()

    def close(self):
        if self.original_configs is not None:
            self.kernel.configs=self.original_configs
            self.original_configs=None
