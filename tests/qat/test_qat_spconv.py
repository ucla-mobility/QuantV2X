"""
Gate QAT-0.5 — QATQuantSpconvModule: the autograd-safe sparse wrapper.

THE regression suite for the nn.Parameter-in-forward bug class. These tests
run on CPU against the mock sparse conv (conftest.py), which reproduces
spconv's weight-attachment semantics exactly; when real spconv + CUDA are
present, the same attachment assertions run against the real op too.
"""

import inspect

import pytest
import torch
import torch.nn as nn

from opencood.qat import qat_spconv as qat_spconv_mod
from opencood.qat.qat_spconv import (QATQuantSpconvModule, SPCONV_AVAILABLE,
                                     _resolve_channel_dim)

from conftest import MockSparseConvTensor, MockSubMConv3d


# ---------------------------------------------------------------------------
# Gate 1 — the regression guard: attached weight must carry grad_fn.
# ---------------------------------------------------------------------------
def test_attached_weight_preserves_grad_fn(mock_sparse_conv, sparse_input):
    qm = QATQuantSpconvModule(mock_sparse_conv, n_levels=3)
    qm(sparse_input)
    # after a quantized forward, the tensor sitting on the op must be a
    # NON-LEAF graph node (grad_fn present). nn.Parameter(w_q) would make it
    # a leaf with grad_fn=None — the exact PTQ bug.
    assert qm.op.weight.grad_fn is not None, \
        "REGRESSION: attached weight lost its grad_fn (Parameter re-registration?)"
    assert not qm.op.weight.is_leaf


def test_no_parameter_construction_in_forward_source():
    """Static tripwire: forward() must never construct nn.Parameter."""
    src = inspect.getsource(QATQuantSpconvModule.forward)
    assert "nn.Parameter(" not in src and "Parameter(" not in src, \
        "nn.Parameter constructed inside forward — this reintroduces the bug"


# ---------------------------------------------------------------------------
# Gate 2 — end-to-end backward through the sparse op.
# ---------------------------------------------------------------------------
def test_backward_reaches_all_three_paths(mock_sparse_conv, sparse_input):
    qm = QATQuantSpconvModule(mock_sparse_conv, n_levels=3)
    out = qm(sparse_input)
    out.features.sum().backward()
    assert qm.weight.grad is not None and qm.weight.grad.abs().sum() > 0, \
        "no gradient reached the shadow weight through the sparse op"
    assert qm.weight_quantizer.levels.grad is not None and \
        qm.weight_quantizer.levels.grad.abs().sum() > 0
    assert qm.weight_quantizer.scale.grad is not None and \
        qm.weight_quantizer.scale.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# Gate 3 — single-copy invariant.
# ---------------------------------------------------------------------------
def test_single_weight_copy(mock_sparse_conv):
    n_weight_numel = mock_sparse_conv.weight.numel()
    qm = QATQuantSpconvModule(mock_sparse_conv, n_levels=3)
    named = dict(qm.named_parameters())
    # exactly one weight parameter, owned by the wrapper — none on the op
    weight_params = [k for k in named if k.endswith("weight")]
    assert weight_params == ["weight"], f"unexpected weight params: {weight_params}"
    assert "op.weight" not in named
    assert named["weight"].numel() == n_weight_numel
    # the op's state_dict must not carry a stale second copy either
    assert "weight" not in qm.op.state_dict()


# ---------------------------------------------------------------------------
# Gate 4 — FP bypass equivalence vs the unwrapped op.
# ---------------------------------------------------------------------------
def test_fp_bypass_bit_exact(sparse_input):
    conv = MockSubMConv3d(4, 8)
    ref = conv(sparse_input).features.detach().clone()
    qm = QATQuantSpconvModule(conv, n_levels=3)
    qm.set_quant_state(False, False)
    out = qm(sparse_input).features
    assert torch.equal(out, ref)


# ---------------------------------------------------------------------------
# Gate 5 — optimizer visibility: a step moves the shadow AND the next
#          forward uses the moved value (no stale attachment).
# ---------------------------------------------------------------------------
def test_optimizer_step_propagates(mock_sparse_conv, sparse_input):
    qm = QATQuantSpconvModule(mock_sparse_conv, n_levels=3)
    opt = torch.optim.SGD(qm.parameters(), lr=1e-1)   # big lr: force flips
    w_before = qm.weight.detach().clone()
    out1 = qm(sparse_input).features.detach().clone()
    qm(sparse_input).features.sum().backward()
    opt.step()
    assert not torch.equal(qm.weight.detach(), w_before), \
        "optimizer did not move the shadow weight"
    out2 = qm(sparse_input).features
    assert not torch.equal(out2, out1), \
        "forward output unchanged after step — stale weight attachment"


# ---------------------------------------------------------------------------
# Gate 6 — channel-dim resolution.
# ---------------------------------------------------------------------------
def test_channel_dim_resolution():
    # spconv2-style [Cout=8, 3,3,3, Cin=4] -> unambiguous dim 0
    assert _resolve_channel_dim([8, 3, 3, 3, 4], 8, None) == 0
    # spconv1-style [3,3,3, Cin=4, Cout=8] -> unambiguous dim 4
    assert _resolve_channel_dim([3, 3, 3, 4, 8], 8, None) == 4
    # ambiguous with a dim-0 match -> spconv2 convention wins
    assert _resolve_channel_dim([3, 3, 3, 3, 4], 3, None) == 0
    # ambiguous, no dim-0 match -> loud failure demanding explicit config
    with pytest.raises(ValueError):
        _resolve_channel_dim([4, 3, 3, 3, 3], 3, None)
    # explicit override validated against the shape
    assert _resolve_channel_dim([4, 3, 3, 3, 3], 3, channel_dim=4) == 4
    with pytest.raises(AssertionError):
        _resolve_channel_dim([8, 3, 3, 3, 4], 8, channel_dim=2)


def test_scale_shape_follows_channel_dim(mock_sparse_conv):
    qm = QATQuantSpconvModule(mock_sparse_conv, n_levels=3)
    # mock layout [Cout=8, 3, 3, 3, Cin=4] -> scale [8, 1, 1, 1, 1]
    assert qm.channel_dim == 0
    assert qm.weight_quantizer.scale.shape == (8, 1, 1, 1, 1)


# ---------------------------------------------------------------------------
# Gate 7 — activation quant applies via replace_feature (features only).
# ---------------------------------------------------------------------------
def test_act_quant_via_replace_feature(sparse_input):
    qm = QATQuantSpconvModule(MockSubMConv3d(4, 8), n_levels=3, act_bits=8)
    qm.train()
    out = qm(sparse_input)
    assert isinstance(out, MockSparseConvTensor)
    assert out.features.shape == (32, 8)


# ---------------------------------------------------------------------------
# state_dict round-trip: shadow + quantizer persist; op carries no weight.
# ---------------------------------------------------------------------------
def test_state_dict_roundtrip(sparse_input):
    src = QATQuantSpconvModule(MockSubMConv3d(4, 8), n_levels=3)
    with torch.no_grad():
        src.weight += 0.03
        src.weight_quantizer.levels += 0.1
    dst = QATQuantSpconvModule(MockSubMConv3d(4, 8), n_levels=3)
    dst.load_state_dict(src.state_dict())
    assert torch.equal(src(sparse_input).features, dst(sparse_input).features)


# ---------------------------------------------------------------------------
# Real spconv (only on a CUDA box with spconv installed).
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not SPCONV_AVAILABLE, reason="spconv not installed")
def test_real_spconv_attachment():
    """CPU-safe subset against real spconv: attachment semantics only
    (running the kernel needs CUDA — see docs/qat_transition tests)."""
    from spconv.pytorch import SubMConv3d  # noqa: import guarded by skipif
    conv = SubMConv3d(4, 8, kernel_size=3, padding=1, bias=False)
    qm = QATQuantSpconvModule(conv, n_levels=3)
    w_q = qm.weight_quantizer(qm.weight)
    if "weight" in qm.op._parameters:
        del qm.op._parameters["weight"]
    qm.op.weight = w_q
    assert qm.op.weight.grad_fn is not None
    qm.op.weight.sum().backward()
    assert qm.weight.grad is not None and qm.weight.grad.abs().sum() > 0
