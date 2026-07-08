"""
Gate QAT-0.2 — CodebookQuantSTE correctness.

The philosophy of these tests: the EXACT gradient paths (levels, scale) are
verified against torch's finite-difference oracle (gradcheck); the STE path
is verified against its own DEFINITION (identity inside the trust window,
zero outside) because the STE is a deliberate substitution — the true
derivative is 0 a.e., so gradcheck on the weight path would "fail" by
design, not by bug.
"""

import pytest
import torch

from opencood.qat.ternary_quant import (CodebookQuantSTE,
                                        nearest_level_indices)


def _safe_weights(shape, levels, scale, margin=0.15):
    """Random weights nudged AWAY from codeword-boundary midpoints.

    argmin assignment is discontinuous exactly at midpoints between levels;
    a finite-difference probe straddling a boundary would see the output
    jump by a whole quantization step and report a bogus mismatch. Keeping
    every normalized weight at least `margin`·cell away from any boundary
    makes the function locally smooth in (levels, scale) — the regime where
    the exact-gradient claim holds (a.e.).
    """
    w = torch.randn(*shape, dtype=torch.float64) * 0.1
    w_n = w / scale
    lv = levels.detach()
    mids = (lv[1:] + lv[:-1]) / 2
    for mid, lo, hi in zip(mids, lv[:-1], lv[1:]):
        cell = (hi - lo).item()
        near = (w_n - mid).abs() < margin * cell
        # push offenders toward their cell center
        w_n = torch.where(near, mid + margin * cell * torch.sign(w_n - mid + 1e-12), w_n)
    return (w_n * scale).detach()


# ---------------------------------------------------------------------------
# 1. gradcheck on the EXACT paths (levels, scale) — float64, tiny tensors.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("K", [3, 4])
def test_gradcheck_levels_and_scale(K):
    scale = torch.full((4, 1, 1, 1), 0.07, dtype=torch.float64,
                       requires_grad=True)
    levels = (torch.linspace(-1, 1, K, dtype=torch.float64)
              .clone().requires_grad_(True))
    weight = _safe_weights((4, 3, 3, 3), levels, scale.detach())
    weight.requires_grad_(False)   # STE path excluded from gradcheck — see module docstring

    # clip_grad=False: the clip window only affects the (excluded) weight path,
    # but keep the check honest by testing the pure math first.
    assert torch.autograd.gradcheck(
        lambda lv, s: CodebookQuantSTE.apply(weight, lv, s, False),
        (levels, scale), eps=1e-6, atol=1e-8), \
        "exact gradients to levels/scale disagree with finite differences"


# ---------------------------------------------------------------------------
# 2. STE definition checks on the weight path.
# ---------------------------------------------------------------------------
def test_ste_identity_unclipped():
    w = torch.randn(6, 5, requires_grad=True)
    levels = torch.tensor([-1.0, 0.0, 1.0])
    scale = torch.full((6, 1), 0.1)
    out = CodebookQuantSTE.apply(w, levels, scale, False)
    g = torch.randn_like(out)
    out.backward(g)
    # unclipped STE: gradient must pass through EXACTLY (identity, not approx)
    assert torch.equal(w.grad, g)


def test_ste_clip_window():
    levels = torch.tensor([-1.0, 0.0, 1.0])
    scale = torch.ones(1, 1)
    # normalized weights straddling the 1.5*outer-level window
    w = torch.tensor([[-2.0, -1.4, 0.0, 1.4, 2.0]], requires_grad=True)
    out = CodebookQuantSTE.apply(w, levels, scale, True)
    out.backward(torch.ones_like(out))
    expected_mask = torch.tensor([[0.0, 1.0, 1.0, 1.0, 0.0]])
    assert torch.equal(w.grad, expected_mask), \
        "clip window must zero gradients exactly outside 1.5x outer levels"


# ---------------------------------------------------------------------------
# 3. assignment correctness on hand-built values.
# ---------------------------------------------------------------------------
def test_assignment_hand_values():
    levels = torch.tensor([-1.0, 0.0, 1.0])
    w_n = torch.tensor([-0.9, -0.4, 0.1, 0.6, 100.0])
    idx = nearest_level_indices(w_n, levels)
    #  -0.9->-1(0)  -0.4->0? |-0.4-(-1)|=.6 vs |-0.4|=.4 -> 0(1)
    #   0.1->0(1)    0.6->1(2, |0.6-1|=.4 < .6)   100->1(2, clamped by argmin)
    assert idx.tolist() == [0, 1, 1, 2, 2]


def test_assignment_chunked_matches_unchunked():
    levels = torch.linspace(-1, 1, 255)
    w = torch.randn(10_000)
    a = nearest_level_indices(w, levels, chunk_numel=257)   # force many chunks
    b = nearest_level_indices(w, levels)                    # one chunk
    assert torch.equal(a, b)


# ---------------------------------------------------------------------------
# 4. gradient accounting: scatter_add must conserve gradient mass.
# ---------------------------------------------------------------------------
def test_grad_levels_mass_conservation():
    w = torch.randn(8, 4, 3, 3)
    levels = torch.tensor([-1.0, 0.0, 1.0], requires_grad=True)
    scale = torch.full((8, 1, 1, 1), 0.05)
    out = CodebookQuantSTE.apply(w, levels, scale, True)
    g = torch.randn_like(out)
    out.backward(g)
    # Σ_k grad_levels[k] must equal Σ_i g_i·scale_i — every element's
    # gradient lands on exactly one codeword, none lost, none duplicated.
    assert torch.allclose(levels.grad.sum(), (g * scale).sum(), atol=1e-5)


def test_grad_scale_formula():
    """grad_scale = Σ over channel of g · levels[idx] (NOT g · weight)."""
    w = torch.randn(3, 4)
    levels = torch.tensor([-1.0, 0.0, 1.0])
    scale = torch.full((3, 1), 0.1, requires_grad=True)
    out = CodebookQuantSTE.apply(w, levels, scale, True)
    g = torch.randn_like(out)
    out.backward(g)
    idx = nearest_level_indices(w / 0.1, levels)
    expected = (g * levels[idx]).sum(dim=1, keepdim=True)
    assert torch.allclose(scale.grad, expected, atol=1e-6)
