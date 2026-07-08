"""
numpy_reference_check.py — torch-free validation of the quantizer MATH.
========================================================================

Purpose: an independent oracle for CodebookQuantSTE's gradient formulas that
runs anywhere python+numpy exists (no torch, no GPU, no spconv). It mirrors
the forward/backward math of opencood/qat/ternary_quant.py in numpy and
checks the EXACT gradient claims against central finite differences.

This is NOT a substitute for tests/qat/ (which exercises the real autograd
plumbing) — it is the mathematical cross-check that the formulas the
autograd.Function implements are the correct derivatives in the first place.

Run:  python tests/qat/numpy_reference_check.py
"""

import numpy as np

rng = np.random.default_rng(0)


# ---- numpy mirror of the forward pass --------------------------------------
def forward(w, levels, scale):
    """w_q = scale * levels[argmin_k |w/scale - levels_k|]"""
    w_n = w / scale
    idx = np.abs(w_n[..., None] - levels.reshape(*([1] * w_n.ndim), -1)) \
        .argmin(axis=-1)
    return scale * levels[idx], idx


# ---- numpy mirror of the analytic backward ----------------------------------
def backward(g, w, levels, scale, idx):
    # (b) dL/dlevels[k] = sum over elements assigned to k of g * scale
    grad_levels = np.zeros_like(levels)
    np.add.at(grad_levels, idx.ravel(), (g * np.broadcast_to(scale, g.shape)).ravel())
    # (c) dL/dscale = sum over broadcast dims of g * levels[idx]
    gs = g * levels[idx]
    reduce_dims = tuple(d for d in range(g.ndim) if scale.shape[d] == 1)
    grad_scale = gs.sum(axis=reduce_dims, keepdims=True)
    return grad_levels, grad_scale


def loss_of(w, levels, scale, g):
    """Linear probe loss L = <g, w_q> so dL/dparam = backward's input grad."""
    w_q, _ = forward(w, levels, scale)
    return (g * w_q).sum()


def central_diff(f, x, eps=1e-6):
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        i = it.multi_index
        x[i] += eps
        hi = f()
        x[i] -= 2 * eps
        lo = f()
        x[i] += eps
        grad[i] = (hi - lo) / (2 * eps)
        it.iternext()
    return grad


def keep_off_boundaries(w, levels, scale, margin=0.1):
    """Nudge normalized weights away from assignment boundaries (the argmin
    is discontinuous there; the exact-gradient claim is 'almost everywhere')."""
    w_n = w / scale
    mids = (levels[1:] + levels[:-1]) / 2
    for j, mid in enumerate(mids):
        cell = levels[j + 1] - levels[j]
        near = np.abs(w_n - mid) < margin * cell
        w_n = np.where(near, mid + margin * cell * np.sign(w_n - mid + 1e-12),
                       w_n)
    return w_n * scale


checks = 0


def check(name, cond):
    global checks
    assert cond, f"FAILED: {name}"
    checks += 1
    print(f"  [ok] {name}")


print("== 1. assignment correctness on hand values ==")
levels = np.array([-1.0, 0.0, 1.0])
_, idx = forward(np.array([[-0.09, -0.04, 0.01, 0.06, 10.0]]),
                 levels, np.array([[0.1]]))
check("hand assignment [-0.9,-0.4,0.1,0.6,100] -> [0,1,1,2,2] (normalized)",
      idx.tolist() == [[0, 1, 1, 2, 2]])

print("== 2. exact level gradients vs central differences ==")
for K in (3, 4):
    levels = np.linspace(-1, 1, K)
    scale = np.full((4, 1), 0.07)
    w = keep_off_boundaries(rng.normal(0, 0.1, (4, 6)), levels, scale)
    g = rng.normal(size=(4, 6))
    _, idx = forward(w, levels, scale)
    grad_levels, grad_scale = backward(g, w, levels, scale, idx)

    num = central_diff(lambda: loss_of(w, levels, scale, g), levels)
    check(f"K={K}: analytic dL/dlevels matches finite diff "
          f"(max err {np.abs(grad_levels - num).max():.2e})",
          np.allclose(grad_levels, num, atol=1e-6))

    num_s = central_diff(lambda: loss_of(w, levels, scale, g), scale)
    check(f"K={K}: analytic dL/dscale matches finite diff "
          f"(max err {np.abs(grad_scale - num_s).max():.2e})",
          np.allclose(grad_scale, num_s, atol=1e-6))

print("== 3. gradient mass conservation (scatter accounting) ==")
levels = np.array([-1.0, 0.0, 1.0])
scale = np.full((8, 1), 0.05)
w = rng.normal(0, 0.1, (8, 20))
g = rng.normal(size=(8, 20))
_, idx = forward(w, levels, scale)
gl, _ = backward(g, w, levels, scale, idx)
check("sum(grad_levels) == sum(g*scale)",
      np.isclose(gl.sum(), (g * scale).sum()))

print("== 4. grad_scale uses the LEVEL, not the weight ==")
# construct a case where they'd differ visibly: w far from its level
w = np.array([[0.149]])          # scale 0.1 -> w_n=1.49 -> snaps to level 1.0
scale = np.array([[0.1]])
g = np.array([[2.0]])
_, idx = forward(w, levels, scale)
_, gs = backward(g, w, levels, scale, idx)
check("dL/dscale = g*level (2.0*1.0), NOT g*w_n (2.0*1.49)",
      np.isclose(gs[0, 0], 2.0))

print("== 5. absmean scale init ==")
w = rng.normal(size=(8, 4, 3, 3))
s = np.abs(w).mean(axis=(1, 2, 3), keepdims=True)
check("absmean init is per-out-channel mean of |W|",
      s.shape == (8, 1, 1, 1) and np.isclose(s[3, 0, 0, 0],
                                             np.abs(w[3]).mean()))

print("== 6. STE clip window definition ==")
w_n = np.array([-2.0, -1.4, 0.0, 1.4, 2.0])
inside = (w_n >= levels.min() * 1.5) & (w_n <= levels.max() * 1.5)
check("clip mask keeps exactly |w_n| <= 1.5*outer",
      inside.tolist() == [False, True, True, True, False])

print("== 7. trit packing round-trip (Epic-5 preview, base-3, 5/byte) ==")
codes = rng.integers(0, 3, 1000)
pad = (-len(codes)) % 5
flat = np.concatenate([codes, np.zeros(pad, dtype=int)])
packed = (flat.reshape(-1, 5) * (3 ** np.arange(5))).sum(1)
check("all packed bytes <= 242 (3^5-1)", packed.max() <= 242)
un = []
for b in packed:
    for _ in range(5):
        un.append(b % 3)
        b //= 3
check("unpack reproduces codes exactly",
      np.array_equal(np.array(un[:len(codes)]), codes))
check("effective storage ~1.6 bits/weight",
      abs(8 * len(packed) / len(codes) - 1.6) < 0.01)

print(f"\nALL {checks} NUMPY REFERENCE CHECKS PASSED")
