"""
qat_model.py — graph surgery: swap eligible layers for QAT wrappers.
=====================================================================

Mirrors the MECHANISM of ``opencood.quant.quant_model.QuantModel`` (recursive
``named_children`` walk + ``setattr`` swap) with three deliberate departures:

1.  NO BatchNorm folding, ever, at surgery time. The PTQ flow folds BN before
    calibration because its quantized weights are final. Under ternary QAT
    the weight distribution keeps moving for many epochs; a folded scale
    absorbs BN's γ/σ at time zero and then drifts away from reality. BN
    modules are left in the graph, live, tracking the quantized regime;
    folding happens once at export (Epic 3+).

2.  NO block-level specials (QuantPyramidFusion etc.) in Epic 0. The PTQ
    specials exist to give the *reconstruction drivers* block granularity.
    QAT's forward path only needs every conv/linear/spconv leaf wrapped —
    recursion reaches them inside PyramidFusion/ResNetBEVBackbone/... just
    fine. A ``qat_specials`` registry hook is kept (empty) for Epic 1, where
    the warm-start curriculum re-attaches block semantics.

3.  ReLU/BN are NOT absorbed into wrappers at surgery time (the PTQ walk
    moves them into activation_function/norm_function to define post-fusion
    quantization points). With activation quant OFF in Epic 0-2 that
    reshuffling buys nothing and costs graph-diff noise; the parity fields
    exist on the wrappers for the recon drivers to use when they need them.

Skip semantics are IDENTICAL to QuantModel._should_skip_quantization:
a name matches if it equals the local name, equals the full dotted name, or
is a dotted-prefix of the full name.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from opencood.qat.qat_config import get_qat_config
from opencood.qat.qat_module import QATQuantModule
from opencood.qat.qat_spconv import (QATQuantSpconvModule, SPCONV_AVAILABLE,
                                     _SPCONV_TYPES)

# Registry hook for Epic 1+ block specials (QATQuantPyramidFusion, ...).
qat_specials = {}


class QATQuantModel(nn.Module):
    """Wraps a full QuantV2X model for ternary/2-bit QAT.

    Additional sparse types (e.g. test mocks, future sparse ops) can be
    registered on the CLASS before construction:

        QATQuantModel.EXTRA_SPARSE_TYPES = (MockSubMConv3d,)
    """

    EXTRA_SPARSE_TYPES: Tuple[type, ...] = ()

    def __init__(self, model: nn.Module, qat_cfg: Optional[dict] = None):
        super().__init__()
        self.cfg = get_qat_config(qat_cfg)     # defaults + validation
        self.model = model
        # (full_name, disposition) log — the coverage report is a first-class
        # artifact: reviewing it is how a human confirms the surgery touched
        # exactly what was intended (gate QAT-0.6).
        self._coverage: List[Tuple[str, str]] = []
        # Track "first conv under each encoder_m*" for the first-layer rule.
        self._encoder_first_seen = set()
        self._refactor(self.model, parent_name="")

    # ------------------------------------------------------------------ #
    # policy resolution
    # ------------------------------------------------------------------ #
    def _matches(self, patterns, full_name: str, local_name: str) -> bool:
        for p in patterns:
            if local_name == p or full_name == p or \
                    full_name.startswith(f"{p}."):
                return True
        return False

    def _resolve_n_levels(self, full_name: str, local_name: str,
                          is_first_encoder_conv: bool) -> int:
        """high-precision islands get a K=255 uniform frozen grid (the
        'uniform grid is a codebook special case' trick) — everything else
        gets the configured ternary/2-bit codebook."""
        if is_first_encoder_conv and self.cfg["first_conv_high_precision"]:
            return self.cfg["high_precision_n_levels"]
        if self._matches(self.cfg["high_precision_names"], full_name,
                         local_name):
            return self.cfg["high_precision_n_levels"]
        return self.cfg["weight"]["n_levels"]

    def _first_encoder_conv(self, full_name: str) -> bool:
        """True exactly once per `encoder_m*` prefix, for the first eligible
        conv reached in named_children (= definition) order.

        CAVEAT (documented, accepted for Epic 0): definition order is almost
        always forward order in this codebase's encoders, but it is not
        *guaranteed* by nn.Module. The coverage report prints the resolved
        layer so a human can eyeball it; Epic 2 can pin exact names in
        `high_precision_names` if an encoder is ever reordered.
        """
        head = full_name.split(".")[0]
        if head.startswith("encoder_m") and head not in \
                self._encoder_first_seen:
            self._encoder_first_seen.add(head)
            return True
        return False

    def _sparse_types(self):
        base = tuple(_SPCONV_TYPES) if SPCONV_AVAILABLE else ()
        return base + tuple(self.EXTRA_SPARSE_TYPES)

    # ------------------------------------------------------------------ #
    # the recursive swap
    # ------------------------------------------------------------------ #
    def _refactor(self, module: nn.Module, parent_name: str):
        act_cfg = self.cfg["act"]
        act_bits = act_cfg["n_bits"] if act_cfg["enabled"] else None
        wcfg = self.cfg["weight"]

        for name, child in module.named_children():
            full_name = f"{parent_name}.{name}" if parent_name else name

            # ---- FP islands: never wrapped, never recursed into ----------
            if self._matches(self.cfg["skip_names"], full_name, name):
                self._coverage.append((full_name, "SKIPPED (FP32 island)"))
                continue

            # ---- registered block specials (empty in Epic 0) -------------
            if type(child) in qat_specials:
                setattr(module, name,
                        qat_specials[type(child)](child, self.cfg))
                self._coverage.append((full_name, "QAT-special"))
                continue

            # ---- sparse 3D convs ------------------------------------------
            if self._sparse_types() and isinstance(child,
                                                   self._sparse_types()):
                first = self._first_encoder_conv(full_name)
                K = self._resolve_n_levels(full_name, name, first)
                wrapped = QATQuantSpconvModule(
                    child, n_levels=K, act_bits=act_bits,
                    learn_levels=(wcfg["learn_levels"] and K <= 4),
                    channel_wise=wcfg["channel_wise"],
                    clip_grad=wcfg["clip_grad"],
                    act_mode=act_cfg["mode"],
                    qdrop_prob=act_cfg["qdrop_prob"])
                setattr(module, name, wrapped)
                self._coverage.append(
                    (full_name, f"QATQuantSpconvModule(K={K})"
                     + (" [first-conv HP]" if first and
                        self.cfg["first_conv_high_precision"] else "")))
                continue

            # ---- dense convs / linear --------------------------------------
            if isinstance(child, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                first = self._first_encoder_conv(full_name)
                K = self._resolve_n_levels(full_name, name, first)
                wrapped = QATQuantModule(
                    child, n_levels=K, act_bits=act_bits,
                    # a 255-level "uniform island" keeps frozen levels: with
                    # K that large the exact level gradients are individually
                    # tiny and learning them buys nothing.
                    learn_levels=(wcfg["learn_levels"] and K <= 4),
                    channel_wise=wcfg["channel_wise"],
                    clip_grad=wcfg["clip_grad"],
                    act_mode=act_cfg["mode"],
                    qdrop_prob=act_cfg["qdrop_prob"])
                setattr(module, name, wrapped)
                self._coverage.append(
                    (full_name, f"QATQuantModule(K={K})"
                     + (" [first-conv HP]" if first and
                        self.cfg["first_conv_high_precision"] else "")))
                continue

            # ---- BN stays LIVE (departure #1); recurse everything else ----
            self._refactor(child, full_name)

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def qat_modules(self):
        """(name, module) for every QAT wrapper — the canonical enumeration
        used by telemetry, param groups and phase policies."""
        for n, m in self.model.named_modules():
            if isinstance(m, (QATQuantModule, QATQuantSpconvModule)):
                yield n, m

    def set_quant_state(self, weight_quant: bool = True,
                        act_quant: bool = True):
        for _, m in self.qat_modules():
            m.set_quant_state(weight_quant, act_quant)

    def qat_param_groups(self, lr_weight: Optional[float] = None,
                         lr_quant: Optional[float] = None):
        """Two optimizer groups with different treatment — this split is not
        cosmetic:

        * shadow weights+biases: lr ~2e-5. They see high-variance STE
          gradients through a 3-level bottleneck; EfficientQAT's W2 ablations
          show larger lrs make codes thrash (flip-rate blowup).
        * scales+levels: lr ~1e-4. Few parameters, EXACT low-variance
          gradients — they tolerate and need a faster schedule.
        * weight_decay = 0.0 on BOTH, hard-coded: decay on `levels` drags
          the codebook toward 0 (ternary collapse: everything snaps to the
          zero level); decay on the shadow fights the tri-modal clustering
          QAT is trying to build; decay on `scale` shrinks the whole
          representable range. If regularization is ever wanted here it
          must be designed, not inherited from an optimizer default.
        """
        lr_w = self.cfg["lr_weight"] if lr_weight is None else lr_weight
        lr_q = self.cfg["lr_quant"] if lr_quant is None else lr_quant
        shadow, qparams = [], []
        for _, m in self.qat_modules():
            shadow.append(m.weight)
            if m.bias is not None:
                shadow.append(m.bias)
            qparams.append(m.weight_quantizer.scale)
            if m.weight_quantizer.levels.requires_grad:
                qparams.append(m.weight_quantizer.levels)
            aq = getattr(m, "act_quantizer", None)
            if aq is not None and getattr(aq, "step", None) is not None:
                qparams.append(aq.step)
        return [
            {"params": shadow, "lr": lr_w, "weight_decay": 0.0},
            {"params": qparams, "lr": lr_q, "weight_decay": 0.0},
        ]

    def coverage_report(self, printer=None) -> List[Tuple[str, str]]:
        """The human-review artifact for gate QAT-0.6."""
        if printer is not None:
            width = max((len(n) for n, _ in self._coverage), default=10)
            printer(f"{'module':<{width}}  disposition")
            printer("-" * (width + 30))
            for n, d in self._coverage:
                printer(f"{n:<{width}}  {d}")
        return list(self._coverage)
