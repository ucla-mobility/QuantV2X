"""
opencood.qat — Quantization-Aware Training at 1.58/2-bit for QuantV2X.
=======================================================================

This package is the QAT counterpart of ``opencood/quant/`` (the BRECQ/QDrop-
style PTQ stack). The PTQ package is deliberately left untouched: it is the
baseline we benchmark against, and its blockwise-reconstruction drivers
(encoder_recon / block_recon / pyramid_recon / ...) are reused by later QAT
epics as the warm-start curriculum.

Design contract shared by every module in this package
-------------------------------------------------------
1.  FP32 "shadow" weights are the ONLY master copy of each layer's weights.
    They live as ``nn.Parameter`` on the QAT wrapper, are updated by the
    optimizer in full precision, and are re-quantized on EVERY forward pass.
    The quantized tensor is a temporary — never stored as state, never
    checkpointed. (Why: a single SGD step is almost always too small to move
    a weight across a codeword boundary; it is the *accumulation* of many
    full-precision steps that eventually flips a codeword. If we overwrote
    the master with its quantized image, that accumulation — and therefore
    all learning — would be destroyed.)
2.  Gradients: the discrete assignment (argmin) gets a Straight-Through
    Estimator; the codebook levels and per-channel scales get EXACT
    gradients (see ternary_quant.CodebookQuantSTE for the derivation).
3.  API mirrors ``opencood.quant.quant_layer.QuantModule`` (org_module /
    fwd_func / fwd_kwargs / set_quant_state / norm_function /
    activation_function) so the existing recon tooling and graph-surgery
    conventions transfer with minimal friction.
4.  No BN folding at training time (``is_fusing=False`` semantics): BN must
    stay live so normalization statistics can track the quantized regime.
    Folding happens only at export (Epic 3+).
5.  No torch.fx: QuantV2X's forward is dynamically dispatched per modality
    (``eval(f"self.encoder_{modality}")``), takes dict inputs and branches on
    ``record_len`` — it is not symbolically traceable. Module swap only.
"""

from opencood.qat.qat_config import DEFAULT_QAT_CONFIG, get_qat_config
from opencood.qat.ternary_quant import (
    ActFakeQuant,
    CodebookQuantSTE,
    TernaryCodebookQuantizer,
    round_ste,
)
from opencood.qat.qat_module import QATQuantModule
from opencood.qat.qat_spconv import QATQuantSpconvModule, SPCONV_AVAILABLE
from opencood.qat.qat_model import QATQuantModel
from opencood.qat.telemetry import QATTelemetry

__all__ = [
    "DEFAULT_QAT_CONFIG",
    "get_qat_config",
    "CodebookQuantSTE",
    "TernaryCodebookQuantizer",
    "ActFakeQuant",
    "round_ste",
    "QATQuantModule",
    "QATQuantSpconvModule",
    "SPCONV_AVAILABLE",
    "QATQuantModel",
    "QATTelemetry",
]
