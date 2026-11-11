"""Apply the native float32 multiply, clip, and truncation before transfer."""
import torch


def extract_pbr_channels(attrs, layout):
    packed = (attrs * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
    return tuple(packed[..., layout[name]] for name in ('base_color', 'metallic', 'roughness', 'alpha'))
