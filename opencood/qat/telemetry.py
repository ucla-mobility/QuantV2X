"""
telemetry.py — QAT training-health instrumentation.
====================================================

QAT failure modes are mostly SILENT: nothing NaNs, the loss even goes down a
little, and three days later the AP table explains that a third of the
layers collapsed to all-zeros in epoch 1. The counters here exist to make
those failure modes loud and immediate.

Metrics per wrapped layer
-------------------------
flip_rate     Fraction of weights whose ASSIGNED CODEWORD changed since the
              previous snapshot. The single most informative QAT health
              number:
                * healthy band ≈ 1e-3 .. 1e-2 per snapshot — shadow weights
                  are migrating between cells (learning IS happening at the
                  discrete level, not only in the invisible FP shadow).
                * ≈ 0 for hundreds of iters — lr too small, levels frozen,
                  or a severed gradient path (the spconv bug's signature!).
                * > 5e-2 — lr too large: codes thrash, block-recon targets
                  smear, training destabilizes.
zero_frac     Fraction of weights on the codeword nearest zero. Ternary
              networks WANT substantial zero mass (that's the sparsity win),
              but zero_frac → 1.0 means the layer collapsed to constant-zero
              output (dead layer), and zero_frac → 0.0 at K=3 means the
              zero level is unused (effectively binary — the layer's scale
              is probably mis-initialized).
levels        The learned codebook values — watching {-w_n, 0, +w_p} drift
              apart is direct evidence the TTQ asymmetry is being used.
scale_mean    Mean per-channel scale — should move slowly; jumps correlate
              with lr_quant being too high.
shadow_grad_norm  ||∂L/∂shadow||. Zero on a layer = severed autograd path;
              this is the runtime tripwire for the class of bug the spconv
              fix addressed.

The class is dependency-free (no tensorboard/wandb import): it RETURNS a
flat {metric_name: value} dict per step and optionally forwards it to a
user-supplied ``writer(tag, value, step)`` callable, so any logger plugs in.
"""

from typing import Callable, Dict, Optional

import torch

from opencood.qat.qat_module import QATQuantModule
from opencood.qat.qat_spconv import QATQuantSpconvModule


class QATTelemetry:
    """Snapshot-and-diff tracker over every QAT wrapper in a model.

    Usage:
        telem = QATTelemetry(qat_model)              # baseline snapshot
        ...
        if it % cfg['telemetry_every'] == 0:
            metrics = telem.step(it)                 # diff vs last snapshot
            alarms  = telem.alarms(metrics)          # actionable warnings
    """

    # Alarm thresholds (class attrs so an experiment can tune them).
    FLIP_HIGH = 5e-2      # codes thrashing -> halve lr_weight
    FLIP_LOW = 1e-6       # frozen codes    -> raise lr / check grads / levels
    ZERO_COLLAPSE = 0.98  # layer is (almost) all zero-level -> dead layer

    def __init__(self, qat_model, writer: Optional[Callable] = None):
        # Accept either a QATQuantModel or any nn.Module containing wrappers:
        root = getattr(qat_model, "model", qat_model)
        self.mods = {
            name: m for name, m in root.named_modules()
            if isinstance(m, (QATQuantModule, QATQuantSpconvModule))
        }
        if not self.mods:
            raise ValueError("QATTelemetry found no QAT wrappers — was the "
                             "model converted with QATQuantModel?")
        self.writer = writer
        # Baseline code snapshot. Codes (not shadow values) are what matter:
        # the shadow moves every step, but only a codeword flip changes the
        # deployed function.
        self.prev_codes: Dict[str, torch.Tensor] = {
            n: m.export_codes()[0] for n, m in self.mods.items()
        }

    @torch.no_grad()
    def step(self, global_iter: int) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for name, m in self.mods.items():
            codes, levels, scale = m.export_codes()

            flip = (codes != self.prev_codes[name]).float().mean().item()
            self.prev_codes[name] = codes

            # "zero level" = the codeword with the smallest |value| — after
            # level learning it may not be exactly 0.0 anymore.
            zero_idx = int(levels.abs().argmin().item())
            zero_frac = (codes == zero_idx).float().mean().item()

            g = m.weight.grad
            gnorm = float(g.norm().item()) if g is not None else -1.0

            layer = {
                f"{name}/flip_rate": flip,
                f"{name}/zero_frac": zero_frac,
                f"{name}/scale_mean": float(scale.mean().item()),
                f"{name}/shadow_grad_norm": gnorm,
            }
            for k, v in enumerate(levels.tolist()):
                layer[f"{name}/level_{k}"] = v
            out.update(layer)

            if self.writer is not None:
                for tag, val in layer.items():
                    self.writer(tag, val, global_iter)
        return out

    def alarms(self, metrics: Dict[str, float]) -> Dict[str, str]:
        """Turn raw metrics into actionable warnings (the Epic-2 runbook)."""
        warn: Dict[str, str] = {}
        for name in self.mods:
            flip = metrics.get(f"{name}/flip_rate", 0.0)
            zf = metrics.get(f"{name}/zero_frac", 0.0)
            gn = metrics.get(f"{name}/shadow_grad_norm", -1.0)
            if flip > self.FLIP_HIGH:
                warn[name] = (f"flip_rate {flip:.3%} > {self.FLIP_HIGH:.0%}: "
                              "codes thrashing — halve lr_weight")
            elif flip < self.FLIP_LOW:
                warn[name] = (f"flip_rate ~0: codes frozen — raise lr, check "
                              "learn_levels, or suspect a severed grad path")
            if zf > self.ZERO_COLLAPSE:
                warn[name] = (f"zero_frac {zf:.1%}: ternary collapse — "
                              "promote this layer to high-precision island")
            if gn == 0.0:
                warn[name] = ("shadow_grad_norm == 0 after backward: severed "
                              "autograd path (nn.Parameter-in-forward class "
                              "of bug) — investigate IMMEDIATELY")
        return warn
