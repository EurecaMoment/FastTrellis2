"""Retain native FP32 norm reduction and fuse its subsequent elementwise work."""
import torch
import triton
import triton.language as tl
from triton.language.math import div_rn


@triton.jit
def _normalize_scale(X,DEN,GAMMA,OUT,N,WIDTH:tl.constexpr,HEAD_WIDTH:tl.constexpr,
                     SCALE:tl.constexpr,BLOCK:tl.constexpr):
    index=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    value=tl.load(X+index,index<N,other=0)
    denominator=tl.load(DEN+index//HEAD_WIDTH,index<N,other=1)
    denominator=tl.where(denominator<1.0e-12,1.0e-12,denominator)
    gamma=tl.load(GAMMA+index%WIDTH).to(tl.float32)
    normalized=div_rn(value,denominator)
    scaled=(normalized*gamma)*SCALE
    tl.store(OUT+index,scaled,index<N)


def normalize_scale(value,gamma,scale):
    promoted=value.float()
    if not promoted.is_contiguous():promoted=promoted.contiguous()
    # This is the exact reduction invoked by F.normalize in the original code.
    denominator=promoted.norm(2,dim=-1,keepdim=True)
    result=torch.empty_like(promoted,dtype=value.dtype)
    _normalize_scale[(triton.cdiv(value.numel(),1024),)](
        promoted,denominator,gamma,result,value.numel(),gamma.numel(),value.shape[-1],scale,1024,
        enable_fp_fusion=False)
    return result


class FusedRMSPointwise:
    def __init__(self,pipeline):
        from trellis_inference_exact import FLOW_NAMES
        from trellis2.modules.sparse.basic import VarLenTensor
        self.restores=[]
        for name in FLOW_NAMES:
            for module in pipeline.models[name].modules():
                if type(module).__name__ not in ('MultiHeadRMSNorm','SparseMultiHeadRMSNorm'):
                    continue
                original=module.forward
                def forward(x,_module=module,_original=original):
                    sparse=isinstance(x,VarLenTensor)
                    features=x.feats if sparse else x
                    if features.dtype!=torch.bfloat16 or not _module.gamma.is_contiguous():
                        return _original(x)
                    result=normalize_scale(features,_module.gamma,_module.scale)
                    return x.replace(result) if sparse else result
                self.restores.append((module,original))
                module.forward=forward

    def close(self):
        for module,original in self.restores:module.forward=original
