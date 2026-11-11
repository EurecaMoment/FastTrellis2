"""Run independent native dense CFG evaluations on two CUDA streams."""
import torch


class ConcurrentDenseCFG:
    def __init__(self, model):
        from trellis2.pipelines.samplers.classifier_free_guidance_mixin import ClassifierFreeGuidanceSamplerMixin as Mixin
        self.restores = []
        self.stream = torch.cuda.Stream(device=model.device)
        self.paired_evaluations = 0
        self.enabled = True
        original = Mixin._inference_model
        self.restores.append((Mixin, '_inference_model', original))

        def inference(sampler, current_model, x_t, t, cond, neg_cond,
                      guidance_strength, guidance_rescale=0.0, **kwargs):
            if (not self.enabled or current_model is not model or not isinstance(x_t, torch.Tensor)
                    or guidance_strength in (0, 1)):
                return original(sampler, current_model, x_t, t, cond, neg_cond,
                    guidance_strength, guidance_rescale=guidance_rescale, **kwargs)
            caller = torch.cuda.current_stream(x_t.device)
            self.stream.wait_stream(caller)
            base = super(Mixin, sampler)._inference_model
            positive = base(current_model, x_t, t, cond, **kwargs)
            with torch.cuda.stream(self.stream):
                negative = base(current_model, x_t, t, neg_cond, **kwargs)
                x_t.record_stream(self.stream)
            caller.wait_stream(self.stream)
            negative.record_stream(caller)
            self.paired_evaluations += 1
            pred = guidance_strength * positive + (1 - guidance_strength) * negative
            if guidance_rescale > 0:
                x_0_pos = sampler._pred_to_xstart(x_t, t, positive)
                x_0_cfg = sampler._pred_to_xstart(x_t, t, pred)
                std_pos = x_0_pos.std(dim=list(range(1, x_0_pos.ndim)), keepdim=True)
                std_cfg = x_0_cfg.std(dim=list(range(1, x_0_cfg.ndim)), keepdim=True)
                x_0_rescaled = x_0_cfg * (std_pos / std_cfg)
                x_0 = guidance_rescale * x_0_rescaled + (1 - guidance_rescale) * x_0_cfg
                pred = sampler._xstart_to_pred(x_t, t, x_0)
            return pred

        Mixin._inference_model = inference

    def close(self):
        self.stream.synchronize()
        for obj, name, value in reversed(self.restores):
            setattr(obj, name, value)
