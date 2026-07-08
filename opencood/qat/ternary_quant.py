"""
ternary_quant.py — the quantizer core: codebook STE + activation fake-quant.
=============================================================================

Contents
--------
1. ``round_ste``            — the classic rounding STE (activations only!).
2. ``grad_scale``           — LSQ gradient-magnitude calibration helper.
3. ``nearest_level_indices``— memory-safe nearest-codeword assignment.
4. ``CodebookQuantSTE``     — custom autograd.Function: STE to shadow weights,
                              EXACT gradients to codebook levels and scales.
5. ``TernaryCodebookQuantizer`` — owns the learnable codebook + scale.
6. ``ActFakeQuant``         — uniform activation fake-quant (EMA or LSQ range)
                              with QDrop stochastic bypass.

The mathematical model (read this once, then the code reads itself)
-------------------------------------------------------------------
A weight tensor W is represented on-device as

    W_q = s ⊙ c[k*(W)]          where  k*(W) = argmin_k | W / s − c_k |     (1)

  * s    : per-output-channel positive scale (shape broadcastable to W),
  * c    : the codebook, a length-K vector of levels
           (K=3 ternary: c = {−w_n, 0, +w_p} after training; K=4: 2-bit),
  * k*(W): the *assignment* — the index of the nearest level to each
           normalized weight. This is the only non-differentiable piece.

Differentiating (1):

  ∂W_q/∂W       = 0 almost everywhere (piecewise-constant staircase), and
                  undefined (a Dirac) exactly at cell boundaries. Useless for
                  descent → we SUBSTITUTE the identity (the Straight-Through
                  Estimator), optionally windowed to the representable range
                  (see backward, path a). The substitution is principled: for
                  the *population* of weights in a cell, the average movement
                  of W_q under a small shift of W is exactly that shift, so
                  identity is the correct expectation over the layer even
                  though it is wrong pointwise.

  ∂W_q/∂c_k     = s · 1[k*(W) = k]   — EXACT. For fixed assignments, W_q is
                  *linear* in the levels; the gradient of level k is simply
                  the sum of (incoming grad × scale) over all weights
                  currently assigned to k. Assignment changes under an
                  infinitesimal level perturbation only for weights sitting
                  exactly on a boundary — a measure-zero set — so treating
                  the assignment as fixed is exact a.e. (This is precisely
                  the gradient VQ-VAE gives its codebook.)

  ∂W_q/∂s       = c[k*(W)]           — EXACT, same fixed-assignment argument.
                  Note it is the *level*, not the weight: channels whose
                  weights snap to ±w_p feel scale gradients proportionally.

Why the in-house ``round_ste`` trick cannot be used for weights here:
``(x.round() − x).detach() + x`` routes ALL gradient to x and, through the
``.detach()``, severs every other input — with a learnable non-uniform
codebook, levels and scales would silently receive zero gradient and never
train. Hence the explicit autograd.Function below.
"""

import math
from typing import Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 1. Rounding STE — kept for ACTIVATIONS ONLY (uniform grid, no learnable
#    codebook on that path in Epic 0; LSQ handles the learnable-scale case).
# ---------------------------------------------------------------------------
def round_ste(x: torch.Tensor) -> torch.Tensor:
    """Forward: round(x). Backward: identity to x (and ONLY to x).

    Identical to opencood.quant.quant_layer.round_ste; re-declared here so
    opencood/qat has no import-time dependency on the PTQ package (which
    pulls in spconv + matplotlib at module scope).
    """
    return (x.round() - x).detach() + x


# ---------------------------------------------------------------------------
# 2. LSQ gradient calibration.
# ---------------------------------------------------------------------------
def grad_scale(x: torch.Tensor, factor: float) -> torch.Tensor:
    """Return a tensor equal to x in the forward pass whose gradient is
    multiplied by ``factor`` in the backward pass.

    Identity trick:  y = (x − x·f).detach() + x·f
      forward :  y = x − x·f + x·f = x            (exact)
      backward:  dy/dx = f                        (only the undetached term)

    WHY (LSQ, Esser et al. 2020): the raw gradient to a quantization step
    size aggregates over every element of the tensor, so its magnitude grows
    with tensor size and bit-width while the parameter itself stays O(1).
    Scaling by 1/sqrt(N·Q_p) equalizes update magnitude across layers of
    different sizes — without it, one global lr cannot serve both a 64-ch
    and a 512-ch layer and training destabilizes.
    """
    return (x - x * factor).detach() + x * factor


# ---------------------------------------------------------------------------
# 3. Nearest-codeword assignment (shared by forward pass and export).
# ---------------------------------------------------------------------------
def nearest_level_indices(w_n: torch.Tensor, levels: torch.Tensor,
                          chunk_numel: int = 1 << 22) -> torch.Tensor:
    """argmin_k |w_n − levels_k| computed in flat chunks.

    The naive broadcast builds a [numel(w), K] distance tensor. For ternary
    (K=3) that is fine, but the same code path serves the K=255 "W8 island"
    fallback, where a 512x512x3x3 conv would need 512·512·9·255 floats
    (~2.4 GB) transiently. Chunking caps the transient at
    chunk_numel · K floats (default ~4M·K) with zero effect on the result.

    NOTE: torch.argmin breaks ties by returning the FIRST minimal index —
    deterministic, which the flip-rate telemetry relies on (a tie must not
    oscillate between equally-near codewords across identical forwards).
    """
    flat = w_n.reshape(-1)
    out = torch.empty_like(flat, dtype=torch.long)
    K = levels.numel()
    step = max(1, chunk_numel // max(K, 1))
    for start in range(0, flat.numel(), step):
        sl = flat[start:start + step]
        # [n, K] distances -> argmin over K
        out[start:start + step] = torch.argmin(
            (sl.unsqueeze(-1) - levels.view(1, -1)).abs(), dim=-1)
    return out.view_as(w_n)


# ---------------------------------------------------------------------------
# 4. The custom autograd.Function.
# ---------------------------------------------------------------------------
class CodebookQuantSTE(torch.autograd.Function):
    """w_q = scale * levels[argmin_k |w/scale − levels_k|]  (non-uniform snap).

    Three gradient paths (see module docstring for full derivations):
      (a) d L/d weight : STE (identity), optionally windowed to the
                         representable range — LSQ/PACT-style clipping.
      (b) d L/d levels : EXACT via scatter_add over assignments.
      (c) d L/d scale  : EXACT, reduced over the dims scale broadcasts over.
    """

    @staticmethod
    def forward(ctx,
                weight: torch.Tensor,   # FP32 shadow weights, any shape
                levels: torch.Tensor,   # [K] codebook, e.g. [-1, 0, +1]
                scale: torch.Tensor,    # positive, broadcastable to weight
                clip_grad: bool = True):
        # Normalize into "codebook space". Division (not multiplication)
        # so that levels stay O(1) regardless of the layer's weight
        # magnitude — a shared codebook parameterization across layers.
        w_n = weight / scale
        idx = nearest_level_indices(w_n, levels)
        # Reconstruct: levels[idx] is a *gather* — linear in `levels` for
        # fixed idx, which is what makes gradient path (b) exact.
        w_q = scale * levels[idx]

        # Save w_n (not weight) — backward needs it only for the clip window,
        # which is defined in normalized space. idx is long: allowed in
        # save_for_backward, excluded from differentiation automatically.
        ctx.save_for_backward(w_n, levels, scale, idx)
        ctx.clip_grad = clip_grad
        return w_q

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        w_n, levels, scale, idx = ctx.saved_tensors

        # ---- (a) STE to the shadow weights --------------------------------
        # True derivative is 0 a.e.; substitute identity so the task loss can
        # reshape the shadow distribution (weights migrate between cells over
        # many accumulated steps — the mechanism by which QAT finds a
        # quantization-flat minimum).
        grad_w = grad_out.clone()
        if ctx.clip_grad:
            # Window: keep gradient only where the normalized weight lies
            # within 1.5x the outermost codewords. Beyond that, updates can
            # never change the assignment (the weight is deep inside the
            # outermost cell) yet STE would keep pushing it outward,
            # inflating ||w|| and distorting the absmean scale. The 1.5
            # slack (vs clipping exactly at the outer level) leaves room for
            # weights to re-enter the active range. Same instinct as
            # LSQ/PACT clipped-gradient zones.
            lo = levels.min() * 1.5
            hi = levels.max() * 1.5
            grad_w = grad_w * ((w_n >= lo) & (w_n <= hi)).to(grad_out.dtype)

        # ---- (b) EXACT gradient to the codebook levels ---------------------
        # dw_q/dlevels[k] = scale for every element assigned to k, so
        # grad_levels[k] = Σ_{i: idx_i = k} grad_out_i · scale_i.
        # scatter_add over the flattened assignment implements the Σ.
        flat_g = (grad_out * scale.expand_as(grad_out)).reshape(-1)
        grad_levels = torch.zeros_like(levels).scatter_add_(
            0, idx.reshape(-1), flat_g)

        # ---- (c) EXACT gradient to the scale -------------------------------
        # dw_q/dscale = levels[idx] (the assigned level, NOT the weight).
        # scale broadcasts over every dim where its size is 1, so its
        # gradient is the sum over exactly those dims (standard broadcast
        # adjoint). keepdim=True preserves the [Cout,1,..,1] parameter shape.
        g_scale_full = grad_out * levels[idx]
        if scale.dim() == grad_out.dim():
            reduce_dims = [d for d in range(grad_out.dim())
                           if scale.shape[d] == 1]
            grad_scale_ = (g_scale_full.sum(dim=reduce_dims, keepdim=True)
                           if reduce_dims else g_scale_full)
        else:
            # Per-tensor scalar scale stored with fewer dims: reduce fully.
            grad_scale_ = g_scale_full.sum().view_as(scale)

        # One grad per forward input; clip_grad (a bool) gets None.
        return grad_w, grad_levels, grad_scale_, None


# ---------------------------------------------------------------------------
# 5. The weight quantizer module.
# ---------------------------------------------------------------------------
class TernaryCodebookQuantizer(nn.Module):
    """1.58-bit / 2-bit (or generic-K) non-uniform weight quantizer.

    Parameters (both trainable, both receiving EXACT gradients):
      * ``levels`` [K] — the codebook. K=3 initialized to {−1, 0, +1}
        (BitNet grid); training deforms it into the TTQ-style asymmetric
        {−w_n, 0, +w_p}. Freeze via ``learn_levels=False`` for a fixed grid.
      * ``scale``  — per-output-channel, initialized with the BitNet b1.58
        "absmean" rule s_c = mean|W_c|. Absmean (vs absmax) is robust to
        outlier weights and, for a roughly Laplacian weight distribution,
        puts the ±1 levels near the distribution's mass rather than its
        tails — empirically the right starting tessellation for ternary.

    Call signature matches UniformAffineQuantizer: ``w_q = quantizer(w)``.
    """

    def __init__(self,
                 n_levels: int = 3,
                 channel_wise: bool = True,
                 learn_levels: bool = True,
                 clip_grad: bool = True):
        super().__init__()
        assert isinstance(n_levels, int) and n_levels >= 2
        self.n_levels = n_levels
        if n_levels == 3:
            init = torch.tensor([-1.0, 0.0, 1.0])
        elif n_levels == 4:
            # Symmetric 2-bit grid with a half-step offset (no zero level:
            # for K=4 a zero level wastes one of only four slots on the
            # distribution's peak while leaving one tail 2x coarser).
            # Training deforms it freely if learn_levels.
            init = torch.tensor([-1.5, -0.5, 0.5, 1.5])
        else:
            # Generic uniform grid on [-1, 1] — the "special case" that
            # makes uniform quantization a subset of this quantizer. Used
            # for W8 islands (n_levels=255 keeps a true zero level since
            # 255 is odd -> symmetric zero-centered grid).
            init = torch.linspace(-1.0, 1.0, n_levels)
        self.levels = nn.Parameter(init, requires_grad=learn_levels)

        # Scale is created by init_scale because its SHAPE depends on the
        # weight tensor it will quantize. It must exist before the optimizer
        # is constructed (an nn.Parameter born after optimizer creation is
        # silently never updated) — the wrapper modules call init_scale
        # EAGERLY in their __init__ for exactly this reason.
        self.scale: Optional[nn.Parameter] = None
        self.channel_wise = channel_wise
        self.clip_grad = clip_grad
        self.inited = False
        self._channel_dim = 0

    @torch.no_grad()
    def init_scale(self, weight: torch.Tensor, channel_dim: int = 0):
        """Absmean scale init. ``channel_dim`` is parameterized because
        weight layouts differ: Conv2d/Linear put Cout at dim 0,
        ConvTranspose2d at dim 1, spconv layouts vary by version."""
        self._channel_dim = channel_dim
        if self.channel_wise:
            dims = [d for d in range(weight.dim()) if d != channel_dim]
            s = weight.abs().mean(dim=dims, keepdim=True)
        else:
            s = weight.abs().mean().view(*([1] * weight.dim()))
        # clamp_min: a dead channel (all-zero weights) would otherwise give
        # scale 0 -> division by zero in the forward normalization.
        self.scale = nn.Parameter(s.clamp_min(1e-8))
        self.inited = True

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        assert self.inited and self.scale is not None, (
            "TernaryCodebookQuantizer.init_scale(weight) must be called "
            "before the first forward AND before optimizer construction.")
        # clamp keeps the *effective* scale positive if the optimizer drives
        # the parameter through zero; gradient still reaches self.scale
        # (clamp is identity, grad 1, wherever scale > 1e-8).
        scale = self.scale.clamp_min(1e-8)
        return CodebookQuantSTE.apply(weight, self.levels, scale,
                                      self.clip_grad)

    @torch.no_grad()
    def assign(self, weight: torch.Tensor) -> torch.Tensor:
        """Discrete codes for telemetry/export (no autograd involvement)."""
        scale = self.scale.clamp_min(1e-8)
        return nearest_level_indices(weight / scale, self.levels)

    def extra_repr(self) -> str:
        return (f"n_levels={self.n_levels} "
                f"(~{math.log2(self.n_levels):.2f} bit), "
                f"channel_wise={self.channel_wise}, "
                f"learn_levels={self.levels.requires_grad}, "
                f"clip_grad={self.clip_grad}")


# ---------------------------------------------------------------------------
# 6. Activation fake-quant with QDrop.
# ---------------------------------------------------------------------------
class ActFakeQuant(nn.Module):
    """Uniform activation fake-quantizer, EMA- or LSQ-ranged, with QDrop.

    Activations use a UNIFORM grid (unlike weights) because (i) they are
    recomputed every frame so there is no storage win from a codebook, and
    (ii) integer GEMM/conv kernels need uniform activation grids to fold the
    dequant into a single per-channel rescale. 8/6/4-bit uniform is the
    deployable envelope; the ternary story is weights-only.

    Modes
    -----
    ema : asymmetric affine range from running min/max (momentum 0.1 —
          matches UniformAffineQuantizer.update_quantize_range). Robust
          bring-up mode; range params are buffers, not learned.
    lsq : symmetric/unsigned learned step size. The step parameter gets the
          1/sqrt(N·Q_p) gradient calibration (see grad_scale). Signedness is
          decided at calibration: if the observed minimum is >= 0 (post-ReLU
          feature maps — the common case in this codebase) use the unsigned
          range [0, 2^b − 1], otherwise signed [−2^{b−1}, 2^{b−1} − 1].

    QDrop: during TRAINING each element independently bypasses fake-quant
    with probability 1 − prob. Bypassed elements see the clean activation
    (and a clean gradient); quantized elements see the perturbation. The
    stochastic mixture prevents early training from overfitting to one
    fixed perturbation pattern. At eval, quantization always applies.
    """

    def __init__(self, n_bits: int = 8, mode: str = "ema",
                 momentum: float = 0.1, prob: float = 1.0):
        super().__init__()
        assert 2 <= n_bits <= 8
        assert mode in ("ema", "lsq")
        self.n_bits = n_bits
        self.mode = mode
        self.momentum = momentum
        self.prob = prob
        # EMA state: buffers (persisted in checkpoints, not optimized).
        self.register_buffer("running_lo", torch.zeros(1))
        self.register_buffer("running_hi", torch.zeros(1))
        # LSQ state: created at calibration (shape/signedness data-dependent).
        self.step: Optional[nn.Parameter] = None
        self.register_buffer("q_min", torch.zeros(1))
        self.register_buffer("q_max", torch.tensor(float(2 ** n_bits - 1)))
        self.inited = False

    @torch.no_grad()
    def _calibrate(self, x: torch.Tensor):
        lo, hi = x.min(), x.max()
        self.running_lo.fill_(lo.item())
        self.running_hi.fill_(hi.item())
        if self.mode == "lsq":
            if lo >= 0:                          # post-ReLU: unsigned grid
                q_min, q_max = 0.0, float(2 ** self.n_bits - 1)
            else:                                # signed symmetric grid
                q_min = -float(2 ** (self.n_bits - 1))
                q_max = float(2 ** (self.n_bits - 1) - 1)
            self.q_min.fill_(q_min)
            self.q_max.fill_(q_max)
            # LSQ init rule: step = 2·E|x| / sqrt(Q_p) — places the grid so
            # a roughly half-normal activation distribution spans it.
            step0 = 2.0 * x.abs().mean() / math.sqrt(max(q_max, 1.0))
            self.step = nn.Parameter(step0.clamp_min(1e-8).view(1))
        self.inited = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.inited:
            # First batch calibrates. NOTE: like the quantizer scale, if LSQ
            # mode is used the wrapper should force calibration BEFORE
            # optimizer construction (run one dummy batch) so self.step is
            # visible to the optimizer.
            self._calibrate(x)

        if self.mode == "ema":
            if self.training:
                # EMA tracks the drifting activation distribution during QAT
                # (ternary weights shift feature statistics — frozen PTQ
                # ranges are exactly what we are escaping).
                self.running_lo.mul_(1 - self.momentum).add_(
                    self.momentum * x.detach().min())
                self.running_hi.mul_(1 - self.momentum).add_(
                    self.momentum * x.detach().max())
            n = 2 ** self.n_bits - 1
            delta = ((self.running_hi - self.running_lo) / n).clamp_min(1e-8)
            # Asymmetric affine: x_int = round((x − lo)/Δ) ∈ [0, n].
            # round_ste is sufficient here: delta/lo are buffers (no gradient
            # wanted), so the detach() in round_ste severs nothing trainable.
            x_int = round_ste((x - self.running_lo) / delta).clamp(0, n)
            x_q = x_int * delta + self.running_lo
        else:  # lsq
            # Gradient-calibrated step (see grad_scale docstring for why).
            g = 1.0 / math.sqrt(x.numel() * float(self.q_max))
            step = grad_scale(self.step.clamp_min(1e-8), g)
            v = x / step
            # clamp BEFORE round: out-of-range elements get d/dx = 0 through
            # clamp's autograd (their value is pinned to the grid edge — the
            # LSQ-prescribed behavior) while d/dstep remains correct via the
            # v·step recomposition.
            v = v.clamp(self.q_min.item(), self.q_max.item())
            x_q = round_ste(v) * step

        if self.training and self.prob < 1.0:
            # QDrop: elementwise stochastic bypass (see class docstring).
            keep_q = torch.rand_like(x) < self.prob
            x_q = torch.where(keep_q, x_q, x)
        return x_q

    def extra_repr(self) -> str:
        return (f"n_bits={self.n_bits}, mode={self.mode}, "
                f"qdrop_prob={self.prob}")
