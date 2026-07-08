"""
qat_spconv.py — autograd-safe QAT wrapper for sparse 3D convolutions.
======================================================================

THE BUG THIS FILE EXISTS TO NEVER REINTRODUCE
---------------------------------------------
spconv modules read ``self.weight`` inside their own forward — there is no
clean functional API like F.conv2d that takes the weight as an argument. So
a quantized weight must be *attached to the module* before calling it. The
original PTQ code did:

    self.spconv_module.weight = nn.Parameter(w_q)        # ← THE BUG

``nn.Parameter(t)`` constructs a NEW LEAF tensor: it shares storage with t
but has ``grad_fn=None`` and ``is_leaf=True``. Autograd sees a fresh,
unconnected variable — every gradient path from the loss back through the
sparse conv into the shadow weight, the codebook levels and the scales is
silently severed. Nothing crashes; gradients are simply zero. For PTQ's
AdaRound reconstruction this quietly degraded results; for QAT it would be
fatal (nothing would train at all).

THE FIX (already applied to opencood/quant's QuantSpconvModule, ported and
hardened here):
  1. DE-REGISTER the Parameter from the spconv module ONCE, at __init__
     (``del module._parameters['weight']``). After deregistration, 'weight'
     is an ordinary attribute slot on the object.
  2. Each forward, assign the freshly quantized tensor as a PLAIN attribute
     (``sp.weight = w_q``). nn.Module.__setattr__ only intercepts
     nn.Parameter values or names still present in _parameters/_buffers;
     a plain tensor on a free slot falls through to object.__setattr__, so
     ``w_q`` keeps its grad_fn (CodebookQuantSTEBackward) and the graph
     survives through spconv's implicit GEMM.
  3. NEVER construct nn.Parameter anywhere in forward. (Tests grep for it.)

CPU / no-spconv PORTABILITY
---------------------------
spconv's kernels require CUDA, and the package may be absent entirely (CI,
laptops). Import is therefore lazy, and type validation degrades to duck
typing: any module exposing a ``weight`` tensor, an ``out_channels`` int and
a forward that READS ``self.weight`` can be wrapped. The unit tests exploit
this with a mock sparse conv that reproduces spconv's attachment semantics
exactly — the same strategy as docs/qat_transition/test_spconv_grad_fix_cpu.py.
"""

from typing import Optional

import torch
import torch.nn as nn

from opencood.qat.ternary_quant import (ActFakeQuant,
                                        TernaryCodebookQuantizer,
                                        nearest_level_indices)

# ---- lazy spconv import (version 1.x and 2.x layouts both exist) -----------
SPCONV_AVAILABLE = False
_SPCONV_TYPES = ()
try:  # spconv 1.x
    from spconv import (SubMConv3d, SparseConv3d,  # noqa: F401
                        SparseInverseConv3d)
    _SPCONV_TYPES = (SubMConv3d, SparseConv3d, SparseInverseConv3d)
    SPCONV_AVAILABLE = True
except ImportError:
    try:  # spconv 2.x
        from spconv.pytorch import (SubMConv3d, SparseConv3d,  # noqa: F401
                                    SparseInverseConv3d)
        _SPCONV_TYPES = (SubMConv3d, SparseConv3d, SparseInverseConv3d)
        SPCONV_AVAILABLE = True
    except ImportError:
        pass  # duck-typed mode only


def _resolve_channel_dim(weight_shape, out_channels: int,
                         channel_dim: Optional[int]) -> int:
    """Locate Cout inside the spconv weight layout — WITHOUT hard-coding it.

    spconv 2.x stores weights as [Cout, k0, k1, k2, Cin]; spconv 1.x used
    [k0, k1, k2, Cin, Cout]. Rather than switching on library version
    strings (fragile), we look for out_channels in the shape:

      * exactly one match           -> that dim, unambiguous.
      * several matches (e.g. a     -> prefer dim 0 (the spconv 2.x
        3-output-channel layer with    convention, the only one QuantV2X's
        kernel size 3)                 pinned environment ships) if it
                                       matches, else fail loudly and demand
                                       an explicit channel_dim.

    A WRONG channel dim would not crash — it would give per-kernel-slice
    scales, silently degraded accuracy, and a very long debugging night.
    Loud failure is the feature.
    """
    if channel_dim is not None:
        assert weight_shape[channel_dim] == out_channels, (
            f"explicit channel_dim={channel_dim} inconsistent with weight "
            f"shape {tuple(weight_shape)} / out_channels={out_channels}")
        return channel_dim
    matches = [d for d, s in enumerate(weight_shape) if s == out_channels]
    if len(matches) == 1:
        return matches[0]
    if 0 in matches:
        return 0
    raise ValueError(
        f"Cannot unambiguously locate out_channels={out_channels} in weight "
        f"shape {tuple(weight_shape)} (candidates: {matches}). Pass "
        f"channel_dim explicitly via the qat config for this layer.")


class QATQuantSpconvModule(nn.Module):
    """Ternary-QAT wrapper for SubMConv3d / SparseConv3d / SparseInverseConv3d
    (or any duck-typed sparse conv that reads ``self.weight`` in forward)."""

    def __init__(self,
                 org_module: nn.Module,
                 n_levels: int = 3,
                 act_bits: Optional[int] = None,
                 learn_levels: bool = True,
                 channel_wise: bool = True,
                 clip_grad: bool = True,
                 act_mode: str = "ema",
                 qdrop_prob: float = 1.0,
                 channel_dim: Optional[int] = None,
                 disable_act_quant: bool = False):
        super().__init__()

        # Strict typing when spconv is importable; duck typing otherwise.
        if SPCONV_AVAILABLE and not isinstance(org_module, _SPCONV_TYPES):
            if not hasattr(org_module, "weight"):
                raise TypeError(
                    f"QATQuantSpconvModule expected an spconv conv or a "
                    f"duck-typed equivalent, got {type(org_module).__name__}")
        assert hasattr(org_module, "weight") and hasattr(org_module,
                                                         "out_channels"), \
            "wrapped module must expose .weight and .out_channels"

        # Keep the spconv op itself: it owns the indice-key/algo machinery
        # (rulebooks, kernel maps) that we must not reimplement.
        self.op = org_module

        # ---- 1) take ownership of the FP32 master copy ---------------------
        self.weight = nn.Parameter(self.op.weight.detach().clone())
        self.bias = (nn.Parameter(self.op.bias.detach().clone())
                     if getattr(self.op, "bias", None) is not None else None)

        # ---- 2) de-register from the wrapped op — ONCE, HERE, NEVER in
        #         forward ----------------------------------------------------
        # After this, self.op.weight is a plain attribute slot. Two wins:
        #   (a) forward can assign a graph tensor without nn.Module's
        #       __setattr__ rejecting it ("cannot assign Tensor as parameter")
        #   (b) the op's parameters() / state_dict() no longer report a
        #       second weight copy — the single-copy invariant the optimizer
        #       and the checkpoint format both rely on (tested).
        if "weight" in self.op._parameters:
            del self.op._parameters["weight"]
        if "bias" in self.op._parameters:
            del self.op._parameters["bias"]
        # Leave a valid attribute in the slot so the op remains usable even
        # before the first quantized forward (e.g. someone printing it).
        object.__setattr__(self.op, "weight", self.weight.data)
        if self.bias is not None:
            object.__setattr__(self.op, "bias", self.bias.data)

        # ---- 3) locate Cout in the (version-dependent) weight layout -------
        self.channel_dim = _resolve_channel_dim(
            list(self.weight.shape), int(self.op.out_channels), channel_dim)

        # ---- quantizers (eager scale init — see qat_module for why) --------
        self.weight_quantizer = TernaryCodebookQuantizer(
            n_levels=n_levels, channel_wise=channel_wise,
            learn_levels=learn_levels, clip_grad=clip_grad)
        self.weight_quantizer.init_scale(self.weight,
                                         channel_dim=self.channel_dim)
        self.act_quantizer = (ActFakeQuant(act_bits, mode=act_mode,
                                           prob=qdrop_prob)
                              if act_bits is not None else None)

        # ---- PTQ-API parity -------------------------------------------------
        self.norm_function = nn.Identity()
        self.activation_function = nn.Identity()
        self.ignore_reconstruction = False
        self.disable_act_quant = disable_act_quant
        self.trained = False
        self.use_weight_quant = True
        self.use_act_quant = act_bits is not None

    # -------------------------------------------------------------------- #
    def forward(self, x):
        """x: spconv.SparseConvTensor (or mock exposing .features /
        .replace_feature)."""
        if self.use_weight_quant:
            w = self.weight_quantizer(self.weight)   # grad_fn: CodebookQuantSTE
        else:
            w = self.weight                          # FP bypass (leaf is fine)

        # ---- THE AUTOGRAD-SAFE ATTACHMENT -----------------------------------
        # Plain attribute assignment. 'weight' was purged from _parameters at
        # __init__, so nn.Module.__setattr__ falls through to
        # object.__setattr__ and w KEEPS ITS grad_fn. Do NOT "clean this up"
        # to nn.Parameter(w): that constructs a new leaf and silently kills
        # every gradient in this subsystem (see module docstring).
        # Defensive re-check: state_dict loading or user code may have
        # re-registered a parameter behind our back.
        sp = self.op
        if "weight" in sp._parameters:                       # pragma: no cover
            del sp._parameters["weight"]
        sp.weight = w
        if self.bias is not None:
            if "bias" in sp._parameters:                     # pragma: no cover
                del sp._parameters["bias"]
            sp.bias = self.bias.to(x.features.device)

        out = sp(x)

        # BN/ReLU parity hooks operate on the dense feature matrix.
        out = out.replace_feature(self.norm_function(out.features))
        out = out.replace_feature(self.activation_function(out.features))

        if (self.use_act_quant and not self.disable_act_quant
                and self.act_quantizer is not None):
            out = out.replace_feature(self.act_quantizer(out.features))
        return out

    # -------------------------------------------------------------------- #
    def set_quant_state(self, weight_quant: bool = True,
                        act_quant: bool = True):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant

    @torch.no_grad()
    def export_codes(self):
        q = self.weight_quantizer
        idx = nearest_level_indices(
            self.weight / q.scale.clamp_min(1e-8), q.levels)
        return (idx.to(torch.int8),
                q.levels.detach().clone(),
                q.scale.detach().clone())

    def extra_repr(self) -> str:
        return (f"op={type(self.op).__name__}, "
                f"channel_dim={self.channel_dim}, "
                f"w_quant={self.use_weight_quant}")
