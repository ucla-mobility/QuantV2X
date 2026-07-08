"""
Gate QAT-0.3 — TernaryCodebookQuantizer + ActFakeQuant behavior.
"""

import io

import pytest
import torch
import torch.nn as nn

from opencood.qat.ternary_quant import ActFakeQuant, TernaryCodebookQuantizer


# ---------------------------------------------------------------------------
# TernaryCodebookQuantizer
# ---------------------------------------------------------------------------
def test_absmean_init_per_channel():
    q = TernaryCodebookQuantizer(n_levels=3, channel_wise=True)
    w = torch.randn(8, 4, 3, 3)
    q.init_scale(w, channel_dim=0)
    expected = w.abs().mean(dim=(1, 2, 3), keepdim=True)
    assert torch.allclose(q.scale.data, expected.clamp_min(1e-8))
    assert q.scale.shape == (8, 1, 1, 1)


def test_absmean_init_channel_dim1():
    """ConvTranspose2d layout [Cin, Cout, kH, kW]: scale along dim 1."""
    q = TernaryCodebookQuantizer(n_levels=3)
    w = torch.randn(4, 8, 2, 2)
    q.init_scale(w, channel_dim=1)
    assert q.scale.shape == (1, 8, 1, 1)


def test_dead_channel_no_div_by_zero():
    q = TernaryCodebookQuantizer(n_levels=3)
    w = torch.randn(4, 4)
    w[2] = 0.0                       # dead output channel
    q.init_scale(w, channel_dim=0)
    out = q(w)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("K,expected", [
    (3, [-1.0, 0.0, 1.0]),
    (4, [-1.5, -0.5, 0.5, 1.5]),
])
def test_level_init(K, expected):
    q = TernaryCodebookQuantizer(n_levels=K)
    assert q.levels.detach().tolist() == pytest.approx(expected)


def test_generic_K_uniform_grid_has_zero_when_odd():
    q = TernaryCodebookQuantizer(n_levels=255, learn_levels=False)
    # odd K on [-1,1] -> exact zero level (needed for W8-island sparsity)
    assert (q.levels.detach() == 0).any()
    assert not q.levels.requires_grad


def test_output_values_come_from_codebook():
    q = TernaryCodebookQuantizer(n_levels=3)
    w = torch.randn(8, 16)
    q.init_scale(w, channel_dim=0)
    out = q(w)
    # every output must equal scale_c * level_k for some k — per channel the
    # set of distinct values is at most K
    for c in range(8):
        assert out[c].unique().numel() <= 3


def test_state_dict_roundtrip():
    q1 = TernaryCodebookQuantizer(n_levels=3)
    w = torch.randn(4, 4)
    q1.init_scale(w, channel_dim=0)
    with torch.no_grad():
        q1.levels += 0.13            # perturb so defaults can't mask a bug

    q2 = TernaryCodebookQuantizer(n_levels=3)
    q2.init_scale(torch.randn(4, 4), channel_dim=0)   # allocate same shapes
    q2.load_state_dict(q1.state_dict())
    assert torch.equal(q2.levels.data, q1.levels.data)
    assert torch.equal(q2.scale.data, q1.scale.data)
    assert torch.equal(q1(w), q2(w))


def test_forward_before_init_raises():
    q = TernaryCodebookQuantizer(n_levels=3)
    with pytest.raises(AssertionError):
        q(torch.randn(3, 3))


# ---------------------------------------------------------------------------
# ActFakeQuant
# ---------------------------------------------------------------------------
def test_ema_quantization_grid():
    aq = ActFakeQuant(n_bits=8, mode="ema").eval()
    x = torch.rand(2, 4, 8, 8) * 6 - 3
    aq._calibrate(x)
    y = aq(x)
    # every output must sit on the affine grid lo + k*delta, k in [0, 255]
    n = 2 ** 8 - 1
    delta = (aq.running_hi - aq.running_lo) / n
    k = (y - aq.running_lo) / delta
    assert torch.allclose(k, k.round(), atol=1e-4)
    # 8-bit quantization error bounded by delta/2
    assert (y - x).abs().max() <= delta.item() / 2 + 1e-6


def test_lsq_creates_learnable_step_and_grad_flows():
    aq = ActFakeQuant(n_bits=4, mode="lsq")
    x = torch.randn(64, 8, requires_grad=True)
    y = aq(x)                                  # first call calibrates
    assert aq.step is not None and aq.step.requires_grad
    y.sum().backward()
    assert aq.step.grad is not None and torch.isfinite(aq.step.grad).all()
    assert x.grad is not None                  # STE path to activations


def test_lsq_unsigned_for_relu_outputs():
    aq = ActFakeQuant(n_bits=8, mode="lsq")
    aq(torch.rand(100) + 0.01)                 # strictly positive input
    assert aq.q_min.item() == 0.0 and aq.q_max.item() == 255.0


def test_lsq_signed_for_symmetric_inputs():
    aq = ActFakeQuant(n_bits=8, mode="lsq")
    aq(torch.randn(100))
    assert aq.q_min.item() == -128.0 and aq.q_max.item() == 127.0


def test_qdrop_bypasses_at_train_applies_at_eval():
    torch.manual_seed(1)
    aq = ActFakeQuant(n_bits=2, mode="ema", prob=0.5)   # coarse grid: visible error
    x = torch.randn(10_000)
    aq._calibrate(x)

    aq.train()
    y_train = aq(x)
    exact = (y_train == x).float().mean().item()
    # ~50% of elements bypass quantization (2-bit grid ~never lands exactly
    # on the input value, so equality <=> bypass)
    assert 0.4 < exact < 0.6

    aq.eval()
    y_eval = aq(x)
    assert (y_eval == x).float().mean().item() < 0.05   # always quantized
