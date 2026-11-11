"""Experimental per-object constant negative cross-attention outputs."""
import torch


class ZeroConditionSparseCache:
    def __init__(self, model):
        self.restores = []
        self.values = []
        self.contexts = {}
        self.enabled = False
        self.hits = self.misses = self.context_checks = 0
        for block in model.blocks:
            module = block.cross_attn
            original = module.forward
            values = {}
            self.values.append(values)

            def forward(x, context=None, _original=original, _values=values):
                if not self.enabled or not isinstance(context, torch.Tensor) or context.shape[0] != 1:
                    return _original(x, context)
                context_key = id(context)
                condition = self.contexts.get(context_key)
                if condition is None:
                    self.context_checks += 1
                    condition = (context, bool(torch.count_nonzero(context).item() == 0))
                    self.contexts[context_key] = condition
                if not condition[1]:
                    return _original(x, context)
                key = (context_key, id(x.coords), tuple(x.feats.shape), x.feats.dtype)
                entry = _values.get(key)
                if entry is None:
                    result = _original(x, context)
                    # Isolate storage from downstream operators. Full native
                    # first-step output, with the actual sparse token count.
                    _values[key] = (context, x.coords, result.feats.clone())
                    self.misses += 1
                    return result
                self.hits += 1
                return x.replace(entry[2])

            self.restores.append((module, original))
            module.forward = forward

    def clear(self):
        for values in self.values:
            values.clear()
        self.contexts.clear()
        self.hits = self.misses = self.context_checks = 0

    def details(self):
        return dict(hits=self.hits, misses=self.misses, context_checks=self.context_checks,
            cached_output_bytes=sum(entry[2].numel()*entry[2].element_size()
                for values in self.values for entry in values.values()),
            scope='one object, fixed coordinates, zero negative condition only')

    def close(self):
        for module, original in reversed(self.restores):
            module.forward = original
        self.clear()
