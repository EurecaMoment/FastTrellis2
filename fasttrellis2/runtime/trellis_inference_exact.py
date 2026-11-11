"""Request-scoped inference optimizations retaining native operator arithmetic."""
import numpy as np
import torch
import time


FLOW_NAMES = ('sparse_structure_flow_model', 'shape_slat_flow_model_512',
              'tex_slat_flow_model_512')


class ExactConditionCache:
    """Reuse image-only projections for one request; never reuse output assets."""
    def __init__(self, pipeline):
        self.restores = []
        self.caches = []
        self.hits = self.misses = 0
        for name in FLOW_NAMES:
            model = pipeline.models[name]
            original = model.forward
            casts = {}
            self.caches.append(casts)

            def forward(x, t, cond, _original=original, _casts=casts, _dtype=model.dtype, **kwargs):
                key = id(cond)
                entry = _casts.get(key)
                if entry is None:
                    entry = (cond, cond.to(dtype=_dtype))
                    _casts[key] = entry
                return _original(x, t, entry[1], **kwargs)

            self.restores.append((model, 'forward', original))
            model.forward = forward
            for module in model.modules():
                if getattr(module, '_type', None) != 'cross' or not hasattr(module, 'to_kv'):
                    continue
                linear = module.to_kv
                original_linear = linear.forward
                values = {}
                self.caches.append(values)

                def project(x, _original=original_linear, _values=values):
                    key = id(x)
                    entry = _values.get(key)
                    if entry is None:
                        self.misses += 1
                        entry = (x, _original(x))
                        _values[key] = entry
                    else:
                        self.hits += 1
                    return entry[1]

                self.restores.append((linear, 'forward', original_linear))
                linear.forward = project

    def clear(self):
        for cache in self.caches:
            cache.clear()

    def close(self):
        for obj, name, original in reversed(self.restores):
            setattr(obj, name, original)
        self.clear()


class SamplesOnly:
    """Keep the original 12 Euler/CFG steps and omit unused trajectory outputs."""
    def __init__(self, pipeline):
        from easydict import EasyDict
        self.restores = []
        for sampler in (pipeline.sparse_structure_sampler, pipeline.shape_slat_sampler, pipeline.tex_slat_sampler):
            original = sampler.sample

            def sample(model, noise, cond, neg_cond, steps=50, rescale_t=1.0,
                       guidance_strength=3.0, guidance_interval=(0.0, 1.0), verbose=True,
                       tqdm_desc='Sampling', _sampler=sampler, **kwargs):
                sequence = np.linspace(1, 0, steps+1)
                sequence = (rescale_t*sequence/(1+(rescale_t-1)*sequence)).tolist()
                value = noise
                for t, previous in zip(sequence[:-1], sequence[1:]):
                    velocity = _sampler._inference_model(model, value, t, cond, neg_cond=neg_cond,
                        guidance_strength=guidance_strength, guidance_interval=guidance_interval, **kwargs)
                    value = value - (t-previous)*velocity
                return EasyDict(samples=value, pred_x_t=[], pred_x_0=[])

            self.restores.append((sampler, original))
            sampler.sample = sample

    def close(self):
        for sampler, original in self.restores:
            sampler.sample = original


class ExactMetadataReuse:
    """Avoid redundant mask construction and materializing batch-one broadcasts."""
    def __init__(self, pipeline):
        from trellis2.modules.sparse.basic import VarLenTensor, SparseTensor
        import xformers.ops as xops
        self.restores = []
        self.masks = {}
        cls = xops.fmha.BlockDiagonalMask
        original_descriptor = cls.__dict__['from_seqlens']
        original = cls.from_seqlens

        def from_seqlens(_cls, q_seqlen, kv_seqlen=None, _original=original, **kwargs):
            q = tuple(q_seqlen)
            kv = None if kv_seqlen is None else tuple(kv_seqlen)
            key = (q, kv, str(kwargs.get('device')))
            if key not in self.masks:
                self.masks[key] = _original(q, kv, **kwargs)
            return self.masks[key]

        self.restores.append((cls, 'from_seqlens', original_descriptor))
        cls.from_seqlens = classmethod(from_seqlens)
        for cls in (VarLenTensor, SparseTensor):
            old = cls.__elemwise__

            def elementwise(value, other, op, _old=old):
                if isinstance(other, torch.Tensor) and value.shape[0] == 1:
                    try:
                        broadcast = torch.broadcast_to(other, value.shape)
                    except RuntimeError:
                        return _old(value, other, op)
                    return value.replace(op(value.feats, broadcast))
                return _old(value, other, op)

            self.restores.append((cls, '__elemwise__', old))
            cls.__elemwise__ = elementwise

        for name in FLOW_NAMES:
            embedder = pipeline.models[name].t_embedder
            original = embedder.timestep_embedding
            dim = embedder.frequency_embedding_size
            half = dim // 2
            frequencies = torch.exp(-np.log(10000) * torch.arange(0, half, dtype=torch.float32) / half)
            frequencies = frequencies.to(pipeline.models[name].device)

            def embedding(t, dimension, max_period=10000, _freq=frequencies, _dim=dim,
                          _original=original):
                if dimension != _dim or max_period != 10000:
                    return _original(t, dimension, max_period)
                phases = t[:, None].float() * _freq[None]
                result = torch.cat([torch.cos(phases), torch.sin(phases)], dim=-1)
                if dimension % 2:
                    result = torch.cat([result, torch.zeros_like(result[:, :1])], dim=-1)
                return result

            self.restores.append((embedder, 'timestep_embedding', original))
            embedder.timestep_embedding = embedding

    def clear(self):
        self.masks.clear()

    def close(self):
        for obj, name, original in reversed(self.restores):
            setattr(obj, name, original)
        self.clear()


class SparseRequestGraph:
    """Capture on actual request coordinates; include capture cost in latency.

    No padded tokens, changed attention kernel, or reuse of another object's
    coordinates. Each request releases its graph before the next request.
    """
    def __init__(self, model):
        self.model = model
        self.original = model.forward
        self.graph = None
        self.capture_seconds = 0.0
        self.calls = 0
        model.forward = self.forward

    def forward(self, x, t, cond, concat_cond=None, **kwargs):
        import time
        if kwargs:
            return self.original(x, t, cond, concat_cond=concat_cond, **kwargs)
        if self.graph is None:
            started = time.perf_counter()
            self.x = x.replace(x.feats.clone())
            self.t = t.clone()
            self.cond = cond.clone()
            self.concat = None if concat_cond is None else concat_cond.replace(concat_cond.feats.clone())
            # Fill spatial metadata on the original coordinates before capture.
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                self.original(self.x, self.t, self.cond, concat_cond=self.concat)
            torch.cuda.current_stream().wait_stream(stream)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, stream=stream):
                self.output = self.original(self.x, self.t, self.cond, concat_cond=self.concat)
            torch.cuda.synchronize()
            self.capture_seconds = time.perf_counter() - started
        if self.x.coords is not x.coords or self.x.feats.shape != x.feats.shape:
            raise RuntimeError('A request graph cannot be reused for new coordinates')
        self.x.feats.copy_(x.feats)
        self.t.copy_(t)
        self.cond.copy_(cond)
        if self.concat is not None:
            self.concat.feats.copy_(concat_cond.feats)
        self.graph.replay()
        self.calls += 1
        # Positive and negative CFG outputs must retain independent storage.
        return x.replace(self.output.feats.clone())

    def close(self):
        self.model.forward = self.original
        self.graph = None


def run_native512(pipeline, image, seed, check=None, fused_pointwise=False):
    """Native batch-one pipeline with request-scoped, equality-tested optimizations."""
    metadata = ExactMetadataReuse(pipeline)
    cache = ExactConditionCache(pipeline)
    sampler = SamplesOnly(pipeline)
    fused = None
    if fused_pointwise:
        from trellis_fused_exact import FusedBlocks
        fused = FusedBlocks(pipeline)
    timing = {}
    torch.cuda.synchronize()
    previous = time.perf_counter()

    def mark(name):
        nonlocal previous
        torch.cuda.synchronize()
        now = time.perf_counter()
        timing[name] = now-previous
        previous = now
        if check is not None:
            check()

    try:
        image = pipeline.preprocess_image(image)
        torch.manual_seed(seed)
        cond = pipeline.get_cond([image],512)
        mark('condition_seconds')
        coords = pipeline.sample_sparse_structure(cond,32,1,{'steps':12})
        mark('structure_seconds')
        shape = pipeline.sample_shape_slat(cond,pipeline.models['shape_slat_flow_model_512'],coords,{'steps':12})
        mark('shape_seconds')
        texture = pipeline.sample_tex_slat(cond,pipeline.models['tex_slat_flow_model_512'],shape,{'steps':12})
        mark('texture_seconds')
        mesh = pipeline.decode_latent(shape,texture,512)[0]
        mark('decode_seconds')
        return mesh,timing
    finally:
        if fused is not None:
            fused.close()
        sampler.close()
        cache.close()
        metadata.close()
