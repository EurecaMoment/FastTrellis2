"""Fuse BF16 pointwise operators with an explicit round at every native boundary."""
import torch
import triton
import triton.language as tl


@triton.jit
def _affine(X, SCALE, SHIFT, OUT, N, CHANNELS: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X+index, index<N, other=0)
    scale = tl.load(SCALE+index%CHANNELS).to(tl.float32)
    shift = tl.load(SHIFT+index%CHANNELS).to(tl.float32)
    factor = (scale+1.0).to(x.dtype).to(tl.float32)
    product = (x.to(tl.float32)*factor).to(x.dtype).to(tl.float32)
    result = (product+shift).to(x.dtype)
    tl.store(OUT+index, result, index<N)


@triton.jit
def _residual(X, H, GATE, OUT, N, CHANNELS: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X+index, index<N, other=0)
    h = tl.load(H+index, index<N, other=0)
    gate = tl.load(GATE+index%CHANNELS).to(tl.float32)
    product = (h.to(tl.float32)*gate).to(h.dtype).to(tl.float32)
    result = (x.to(tl.float32)+product).to(x.dtype)
    tl.store(OUT+index, result, index<N)


def affine(x, scale, shift):
    if not x.is_contiguous() or x.dtype != torch.bfloat16:
        return x*(1+scale)+shift
    output = torch.empty_like(x)
    _affine[(triton.cdiv(x.numel(),1024),)](x,scale,shift,output,x.numel(),x.shape[-1],1024,
                                          enable_fp_fusion=False)
    return output


def residual(x, h, gate):
    if not x.is_contiguous() or not h.is_contiguous() or x.dtype != torch.bfloat16:
        return x+h*gate
    output = torch.empty_like(x)
    _residual[(triton.cdiv(x.numel(),1024),)](x,h,gate,output,x.numel(),x.shape[-1],1024,
                                            enable_fp_fusion=False)
    return output


class FusedBlocks:
    def __init__(self, pipeline):
        from trellis_inference_exact import FLOW_NAMES
        self.restores = []
        for name in FLOW_NAMES:
            sparse = name != 'sparse_structure_flow_model'
            for block in pipeline.models[name].blocks:
                original = block._forward

                def forward(x, mod, context, phases=None, _block=block, _sparse=sparse,
                            _original=original):
                    if x.shape[0] != 1:
                        return _original(x, mod, context) if _sparse else _original(x,mod,context,phases)
                    parameter_cache = getattr(_block, '_trellis_step_parameters', None)
                    parameter_entry = None if parameter_cache is None else parameter_cache.get(id(mod))
                    if parameter_entry is not None and parameter_entry[0] is mod:
                        parameters = parameter_entry[1]
                    elif _block.share_mod:
                        parameters = (_block.modulation+mod).type(mod.dtype).chunk(6,dim=1)
                    else:
                        parameters = _block.adaLN_modulation(mod).chunk(6,dim=1)
                    shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = parameters
                    if _sparse:
                        h = x.replace(affine(_block.norm1(x.feats),scale_a,shift_a))
                        h = _block.self_attn(h)
                        x = x.replace(residual(x.feats,h.feats,gate_a))
                        h = _block.cross_attn(x.replace(_block.norm2(x.feats)),context)
                        x = x+h
                        h = x.replace(affine(_block.norm3(x.feats),scale_m,shift_m))
                        h = _block.mlp(h)
                        return x.replace(residual(x.feats,h.feats,gate_m))
                    h = affine(_block.norm1(x),scale_a,shift_a)
                    h = _block.self_attn(h,phases=phases)
                    x = residual(x,h,gate_a)
                    x = x+_block.cross_attn(_block.norm2(x),context)
                    h = affine(_block.norm3(x),scale_m,shift_m)
                    h = _block.mlp(h)
                    return residual(x,h,gate_m)

                self.restores.append((block,original))
                block._forward = forward

    def close(self):
        for block, original in self.restores:
            block._forward = original
