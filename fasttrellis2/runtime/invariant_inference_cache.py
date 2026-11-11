"""Exact caching of fixed time conditioning and image-only attention operands."""
import numpy as np
import torch

from trellis_inference_exact import FLOW_NAMES


class FixedTimestepCache:
    """Precompute constants with the original single-row operators, not batched GEMM."""
    def __init__(self, pipeline):
        self.restores = []
        self.blocks = []
        self.active = {}
        self.storage = []
        for name, prefix in zip(FLOW_NAMES, ('sparse_structure', 'shape_slat', 'tex_slat')):
            model = pipeline.models[name]
            if not model.share_mod:
                continue
            options = getattr(pipeline, prefix + '_sampler_params')
            steps = options.get('steps', 50)
            rescale = options.get('rescale_t', 1.0)
            sequence = np.linspace(1, 0, steps + 1)
            sequence = (rescale * sequence / (1 + (rescale - 1) * sequence)).tolist()[:-1]
            embeds, modulations = {}, {}
            parameters = [{} for block in model.blocks]
            with torch.inference_mode():
                for t in sequence:
                    tensor = torch.tensor([1000 * t], device=model.device, dtype=torch.float32)
                    embedding = model.t_embedder(tensor)
                    modulation = model.adaLN_modulation(embedding).to(model.dtype)
                    embeds[t] = embedding
                    modulations[id(embedding)] = (embedding, modulation)
                    for block, values in zip(model.blocks, parameters):
                        result = (block.modulation + modulation).type(modulation.dtype).chunk(6, dim=1)
                        values[id(modulation)] = (modulation, result)
            self.storage.extend((embeds, modulations, parameters))
            sampler = getattr(pipeline, prefix + '_sampler')
            original_infer = sampler._inference_model

            def inference(m, x, t, *args, _original=original_infer, **kwargs):
                key = id(m)
                previous = self.active.get(key)
                self.active[key] = t if x.shape[0] == 1 else None
                try:
                    return _original(m, x, t, *args, **kwargs)
                finally:
                    if previous is None:
                        self.active.pop(key, None)
                    else:
                        self.active[key] = previous

            self.restores.append((sampler, '_inference_model', original_infer))
            sampler._inference_model = inference
            original_embed = model.t_embedder.forward

            def embed(t, _model=model, _values=embeds, _original=original_embed):
                value = _values.get(self.active.get(id(_model)))
                return _original(t) if value is None else value

            self.restores.append((model.t_embedder, 'forward', original_embed))
            model.t_embedder.forward = embed
            original_mod = model.adaLN_modulation.forward

            def modulate(value, _values=modulations, _original=original_mod):
                cached = _values.get(id(value))
                return _original(value) if cached is None or cached[0] is not value else cached[1]

            self.restores.append((model.adaLN_modulation, 'forward', original_mod))
            model.adaLN_modulation.forward = modulate
            for block, values in zip(model.blocks, parameters):
                assert not hasattr(block, '_trellis_step_parameters')
                block._trellis_step_parameters = values
                self.blocks.append(block)

    def close(self):
        for obj, name, original in reversed(self.restores):
            setattr(obj, name, original)
        for block in self.blocks:
            del block._trellis_step_parameters
        self.storage.clear()
        self.active.clear()


class NormalizedImageKVCache:
    """Request-local K/V reuse through projection, layout and K normalization."""
    def __init__(self, pipeline):
        from trellis2.modules.attention.full_attn import scaled_dot_product_attention
        from trellis2.modules.sparse.attention.full_attn import sparse_scaled_dot_product_attention
        self.restores, self.caches = [], []
        self.hits = self.misses = 0
        for name in FLOW_NAMES:
            for module in pipeline.models[name].modules():
                if getattr(module, '_type', None) != 'cross' or not hasattr(module, 'to_kv'):
                    continue
                if not module.qk_rms_norm:
                    continue
                sparse = hasattr(module, '_linear')
                original = module.forward
                values = {}
                self.caches.append(values)

                def forward(x, context=None, phases=None, _module=module, _sparse=sparse,
                            _values=values, _original=original):
                    if not isinstance(context, torch.Tensor):
                        return _original(x, context) if _sparse else _original(x, context, phases)
                    key = id(context)
                    entry = _values.get(key)
                    if entry is None:
                        projected = _module.to_kv(context)
                        projected = projected.reshape(*context.shape[:2], 2, _module.num_heads, -1)
                        k, v = projected.unbind(dim=2)
                        k = _module.k_rms_norm(k)
                        entry = (context, k, v)
                        _values[key] = entry
                        self.misses += 1
                    else:
                        self.hits += 1
                    if _sparse:
                        q = _module._linear(_module.to_q, x)
                        q = _module._reshape_chs(q, (_module.num_heads, -1))
                        q = _module.q_rms_norm(q)
                        h = sparse_scaled_dot_product_attention(q, entry[1], entry[2])
                        h = _module._reshape_chs(h, (-1,))
                        return _module._linear(_module.to_out, h)
                    batch, length, _ = x.shape
                    q = _module.to_q(x).reshape(batch, length, _module.num_heads, -1)
                    q = _module.q_rms_norm(q)
                    h = scaled_dot_product_attention(q, entry[1], entry[2])
                    return _module.to_out(h.reshape(batch, length, -1))

                self.restores.append((module, original))
                module.forward = forward

    def clear(self):
        for values in self.caches:
            values.clear()

    def close(self):
        for obj, original in reversed(self.restores):
            obj.forward = original
        self.clear()
