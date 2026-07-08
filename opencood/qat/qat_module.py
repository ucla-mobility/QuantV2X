"""
qat_module.py — QAT wrapper for dense layers (Conv2d / ConvTranspose2d / Linear).
==================================================================================

Drop-in analogue of ``opencood.quant.quant_layer.QuantModule``. The API is
kept deliberately identical (fwd_func / fwd_kwargs / set_quant_state /
norm_function / activation_function / ignore_reconstruction) so that:
  * QATQuantModel can reuse the PTQ graph-surgery conventions, and
  * the blockwise reconstruction drivers (encoder_recon / block_recon /
    pyramid_recon, reused in Epic 1 as the Block-AP warm start) can treat
    QAT wrappers exactly like PTQ wrappers.

STRUCTURAL DIFFERENCES vs the PTQ QuantModule (the "why"):

  PTQ QuantModule                     | QATQuantModule
  ------------------------------------+-----------------------------------
  self.weight = org_module.weight     | self.weight = Parameter(clone())
  (shares the original Parameter,     | (an OWNED FP32 shadow copy — the
   plus org_weight backup for the     |  single master the optimizer sees;
   FP path)                           |  no org_weight backup needed since
                                      |  the shadow IS full precision)
  UniformAffineQuantizer (uniform     | TernaryCodebookQuantizer (learnable
  affine grid, 2..8 bit, AdaRound     | non-uniform codebook, K>=2, exact
  for weight rounding)                | level/scale gradients)
  quantizers OFF by default (PTQ      | weight quant ON by default (QAT
  calibrates first)                   | trains in the quantized regime)
"""

from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from opencood.qat.ternary_quant import (ActFakeQuant,
                                        TernaryCodebookQuantizer,
                                        nearest_level_indices)


class StraightThrough(nn.Module):
    """Identity placeholder (mirrors quant_layer.StraightThrough) — kept
    local to avoid importing the PTQ module (spconv/matplotlib at import
    time). Recon drivers overwrite norm_function / activation_function with
    real BN/ReLU instances when they restructure blocks."""

    def forward(self, x):
        return x


class QATQuantModule(nn.Module):
    """QAT wrapper holding FP32 shadow weights + a ternary codebook quantizer.

    Every forward re-materializes w_q = quantize(shadow); w_q is a temporary
    with a live grad_fn, never stored. Checkpoints therefore contain shadow
    weights + scales + levels; the discrete codes are an EXPORT artifact
    (export_codes), not training state.
    """

    def __init__(self,
                 org_module: Union[nn.Conv2d, nn.ConvTranspose2d, nn.Linear],
                 n_levels: int = 3,
                 act_bits: Optional[int] = None,
                 learn_levels: bool = True,
                 channel_wise: bool = True,
                 clip_grad: bool = True,
                 act_mode: str = "ema",
                 qdrop_prob: float = 1.0,
                 disable_act_quant: bool = False):
        super().__init__()

        # ---- capture the functional form of the wrapped op -----------------
        # We call F.conv2d/F.linear directly (rather than keeping the child
        # module) so the quantized weight is just *an argument* — no module
        # attribute surgery needed on the dense path.
        if isinstance(org_module, nn.Conv2d):
            self.fwd_kwargs = dict(stride=org_module.stride,
                                   padding=org_module.padding,
                                   dilation=org_module.dilation,
                                   groups=org_module.groups)
            self.fwd_func = F.conv2d
            channel_dim = 0            # Conv2d weight: [Cout, Cin, kH, kW]
        elif isinstance(org_module, nn.ConvTranspose2d):
            self.fwd_kwargs = dict(stride=org_module.stride,
                                   padding=org_module.padding,
                                   output_padding=org_module.output_padding,
                                   groups=org_module.groups,
                                   dilation=org_module.dilation)
            self.fwd_func = F.conv_transpose2d
            # ConvTranspose2d weight: [Cin, Cout, kH, kW] — the OUTPUT
            # channel lives at dim 1. A per-"channel" scale computed along
            # dim 0 would be per-INPUT-channel: mathematically valid but it
            # breaks the deployment contract (dequant folds into a
            # per-output-channel rescale) — hence channel_dim=1.
            channel_dim = 1
        elif isinstance(org_module, nn.Linear):
            self.fwd_kwargs = dict()
            self.fwd_func = F.linear
            channel_dim = 0            # Linear weight: [out_features, in]
        else:
            raise TypeError(
                f"QATQuantModule supports Conv2d/ConvTranspose2d/Linear, "
                f"got {type(org_module).__name__}")

        # ---- FP32 SHADOW WEIGHTS — the only master copy --------------------
        # detach().clone(): detach cuts any graph the checkpoint tensor may
        # carry; clone gives us owned storage so later in-place ops on the
        # original module cannot alias into our master.
        self.weight = nn.Parameter(org_module.weight.detach().clone())
        # Bias stays FP32 and unquantized (standard practice: biases are
        # O(Cout) numbers — negligible storage — but quantizing them costs
        # real accuracy because they set post-conv operating points).
        self.bias = (nn.Parameter(org_module.bias.detach().clone())
                     if org_module.bias is not None else None)

        # ---- quantizers -----------------------------------------------------
        self.weight_quantizer = TernaryCodebookQuantizer(
            n_levels=n_levels, channel_wise=channel_wise,
            learn_levels=learn_levels, clip_grad=clip_grad)
        # EAGER scale init: the scale Parameter must exist before the
        # optimizer is constructed, or it is silently never trained.
        self.weight_quantizer.init_scale(self.weight, channel_dim=channel_dim)

        self.act_quantizer = (ActFakeQuant(act_bits, mode=act_mode,
                                           prob=qdrop_prob)
                              if act_bits is not None else None)

        # ---- PTQ-API parity fields (recon drivers read/write these) --------
        self.norm_function = StraightThrough()
        self.activation_function = StraightThrough()
        self.ignore_reconstruction = False
        self.disable_act_quant = disable_act_quant
        self.trained = False

        # QAT trains in the quantized regime from step one (unlike PTQ,
        # which calibrates on the FP path first).
        self.use_weight_quant = True
        self.use_act_quant = act_bits is not None

    # -------------------------------------------------------------------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Re-quantize EVERY forward: w_q is a temporary carrying grad_fn back
        # to (shadow weight, levels, scale). Never cache it across steps —
        # the optimizer moves the shadow under our feet, which is the point.
        if self.use_weight_quant:
            w = self.weight_quantizer(self.weight)
        else:
            # FP bypass — must reproduce the wrapped module bit-exactly
            # (verified by tests): used for debugging and for FP-teacher
            # passes over the same graph.
            w = self.weight
        out = self.fwd_func(x, w, self.bias, **self.fwd_kwargs)

        # BN/ReLU that graph surgery may have absorbed (PTQ parity).
        if type(self.norm_function) == nn.BatchNorm1d:
            # BatchNorm1d over [N, C, L] needs channel-last permute — same
            # special case as the PTQ QuantModule.
            out = self.norm_function(out.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            out = self.norm_function(out)
        out = self.activation_function(out)

        if self.disable_act_quant:
            return out
        if self.use_act_quant and self.act_quantizer is not None:
            out = self.act_quantizer(out)
        return out

    # -------------------------------------------------------------------- #
    def set_quant_state(self, weight_quant: bool = True,
                        act_quant: bool = True):
        """Same signature/semantics as the PTQ QuantModule."""
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant

    # -------------------------------------------------------------------- #
    @torch.no_grad()
    def export_codes(self):
        """Deployment/telemetry form: (codes int8, levels [K], scale).

        codes[i] ∈ [0, K); the deployed weight is scale ⊙ levels[codes].
        int8 is safe because K ≤ 255 by construction. Bit-packing (2 bits or
        base-3 "trit" packing at 1.6 bits/weight) is Epic 5's export.py.
        """
        q = self.weight_quantizer
        idx = nearest_level_indices(
            self.weight / q.scale.clamp_min(1e-8), q.levels)
        return (idx.to(torch.int8),
                q.levels.detach().clone(),
                q.scale.detach().clone())

    def extra_repr(self) -> str:
        return (f"fwd={getattr(self.fwd_func, '__name__', '?')}, "
                f"w_quant={self.use_weight_quant}, "
                f"a_quant={self.use_act_quant}")
