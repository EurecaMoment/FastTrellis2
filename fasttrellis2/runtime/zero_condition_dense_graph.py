"""Experimental dense negative-condition graph; never enabled by the resident service."""
import torch

from dense_graph_static_kv import DenseGraphStaticKV


class ZeroConditionDenseGraph(DenseGraphStaticKV):
    """Cache the cross-attention output for an all-zero, fixed-length context.

    Equal context rows produce equal K/V rows. The attention output is therefore
    independent of Q in real arithmetic. Native floating-point equivalence is
    a separate acceptance requirement: this experiment compares complete native
    12-step latents before any service deployment may use the shortcut.
    """

    def __init__(self, model, conditioning_shape):
        super().__init__(model, conditioning_shape)
        self.constant_outputs = []
        self.zero_context_checks = 0
        self.constant_graph_replays = 0
        _, x, t, cond, buffers, _ = self.entries[1]
        restores = []
        stream = torch.cuda.Stream(device=model.device)
        stream.wait_stream(torch.cuda.current_stream(model.device))
        try:
            with torch.cuda.stream(stream):
                for block in model.blocks:
                    module = block.cross_attn
                    original = module.forward
                    dummy = torch.zeros((x.shape[0], model.resolution ** 3, module.channels),
                        device=model.device, dtype=model.dtype)
                    output = original(dummy, cond)
                    self.constant_outputs.append(output)
                    restores.append((module, original))
                    module.forward = lambda value, context=None, phases=None, _output=output: _output
                for _ in range(2):
                    self.raw_forward(x, t, cond)
            torch.cuda.current_stream(model.device).wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                result = self.raw_forward(x, t, cond)
            torch.cuda.synchronize(model.device)
            self.entries[1] = (graph, x, t, cond, buffers, result)
        finally:
            for module, original in restores:
                module.forward = original
        self.constant_output_bytes = sum(value.numel() * value.element_size() for value in self.constant_outputs)

    def forward(self, x, t, cond):
        if x.shape != self.entries[0][1].shape or cond.shape != self.entries[0][3].shape:
            return self.original(x, t, cond)
        key = id(cond)
        if key not in self.conditions and len(self.conditions) == 1:
            # Do not silently treat a second nonzero condition as unconditional.
            self.zero_context_checks += 1
            if torch.count_nonzero(cond).item() != 0:
                return self.original(x, t, cond)
        value = super().forward(x, t, cond)
        entry = self.conditions.get(key)
        if entry is not None and entry[1] == 1:
            self.constant_graph_replays += 1
        return value

    def close(self):
        torch.cuda.synchronize(self.model.device)
        super().close()
        self.constant_outputs.clear()
