"""Fuse complex RoPE conversion/multiplication, retaining explicit FP32 arithmetic."""
import torch
import triton
import triton.language as tl


@triton.jit
def _rope(X, PHASE, OUT, N, WIDTH: tl.constexpr, HEADS: tl.constexpr,
          SX: tl.constexpr, SH: tl.constexpr, SD: tl.constexpr, MODE: tl.constexpr,
          BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    pair = index % (WIDTH // 2)
    head = (index // (WIDTH // 2)) % HEADS
    token = index // ((WIDTH // 2) * HEADS)
    address = token * SX + head * SH + pair * 2 * SD
    ar = tl.load(X + address, index < N, other=0).to(tl.float32)
    ai = tl.load(X + address + SD, index < N, other=0).to(tl.float32)
    pr = tl.load(PHASE + token * WIDTH + pair * 2, index < N, other=0)
    pi = tl.load(PHASE + token * WIDTH + pair * 2 + 1, index < N, other=0)
    if MODE == 0:
        real, imag = ar * pr - ai * pi, ar * pi + ai * pr
    elif MODE == 1:
        real, imag = tl.fma(-ai, pi, ar * pr), tl.fma(ai, pr, ar * pi)
    else:
        real, imag = tl.fma(ar, pr, -ai * pi), tl.fma(ar, pi, ai * pr)
    tl.store(OUT + index * 2, real, index < N)
    tl.store(OUT + index * 2 + 1, imag, index < N)


def rotary(value, phases, mode=0):
    if value.ndim == 4:
        if value.shape[0] != 1:
            raise ValueError('Only batch-one flow is supported')
        shape = value.shape
        value = value[0]
    else:
        shape = value.shape
    assert value.ndim == 3 and phases.is_contiguous()
    tokens, heads, width = value.shape
    phase = torch.view_as_real(phases).reshape(tokens, width)
    result = torch.empty(shape, dtype=value.dtype, device=value.device)
    count = value.numel() // 2
    _rope[(triton.cdiv(count, 256),)](value, phase, result, count, width, heads,
        *value.stride(), mode, 256, enable_fp_fusion=False)
    return result


class FusedRotary:
    def __init__(self, pipeline, mode):
        from trellis_inference_exact import FLOW_NAMES
        from trellis2.modules.attention.rope import RotaryPositionEmbedder
        self.restores = []
        descriptor = RotaryPositionEmbedder.__dict__['apply_rotary_embedding']
        self.restores.append((RotaryPositionEmbedder, 'apply_rotary_embedding', descriptor))
        RotaryPositionEmbedder.apply_rotary_embedding = staticmethod(lambda x, p: rotary(x, p, mode))
        for name in FLOW_NAMES[1:]:
            for block in pipeline.models[name].blocks:
                rope = block.self_attn.rope
                original = rope._rotary_embedding
                self.restores.append((rope, '_rotary_embedding', original))
                rope._rotary_embedding = lambda x, p: rotary(x, p, mode)

    def close(self):
        for obj, name, original in reversed(self.restores):
            setattr(obj, name, original)
