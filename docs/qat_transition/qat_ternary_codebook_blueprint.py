"""
QAT Ternary-Codebook Blueprint for QuantV2X
============================================

A minimal, idiomatic PyTorch mock-up of the QAT machinery needed to move
QuantV2X from its BRECQ/QDrop-style PTQ stack (opencood/quant/) to true
Quantization-Aware Training at 1.58-bit (ternary) and 2-bit, with a
LEARNABLE, NON-UNIFORM CODEBOOK instead of a uniform integer grid.

Design goals (mirrors opencood/quant/quant_layer.py conventions so it can
drop into the existing QuantModel graph-surgery machinery):

  1. Full-precision "shadow" weights are the ONLY master copy
     (`self.weight` is an nn.Parameter, updated by the optimizer).
     Quantized weights are re-materialized every forward pass and never
     stored as state.
  2. A custom torch.autograd.Function implements the Straight-Through
     Estimator (STE) for the weight path, while routing EXACT gradients
     to the codebook levels and the per-channel scale. This is the key
     difference vs. the vanilla `round_ste` trick
     (`(x.round() - x).detach() + x`), which would kill codebook grads.
  3. The codebook is shared per-layer (levels initialized to {-1, 0, +1}
     for 1.58-bit; {-2,-1,0,1}/{-1.5,-0.5,0.5,1.5} etc. for 2-bit) and its
     entries are trainable (TTQ-style asymmetric ternary is the special
     case levels = {-w_n, 0, +w_p}).
  4. API mirrors opencood.quant.quant_layer.QuantModule:
     (org_module, fwd_func, fwd_kwargs, set_quant_state, ...), so
     QuantModel.quant_module_refactor can be reused nearly unchanged.

Run this file directly for a self-test:
    python qat_ternary_codebook_blueprint.py
"""

from typing import Union, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# 1. The STE autograd Function with codebook-aware backward
# ----------------------------------------------------------------------------
class CodebookQuantSTE(torch.autograd.Function):
    """
    Forward:  w_q = scale * levels[argmin_k |w / scale - levels_k|]
              (nearest-codeword snap in the scale-normalized domain)

    Backward: three gradient paths --
      * d L / d w      : straight-through (identity), optionally clipped to
                         zero outside the representable range so shadow
                         weights far outside the codebook stop drifting
                         (LSQ/PACT-style gradient clipping).
      * d L / d levels : EXACT. w_q depends on levels via a gather, so
                         grad_levels[k] = sum over weights assigned to k of
                         (g * scale). Implemented with scatter_add (this is
                         the same gradient VQ-VAE gives its codebook).
      * d L / d scale  : EXACT. w_q = scale * levels[idx]
                         => grad_scale = sum(g * levels[idx]) per channel.
    """

    @staticmethod
    def forward(ctx,
                weight: torch.Tensor,     # [Cout, ...] FP32 shadow weights
                levels: torch.Tensor,     # [K] codebook (e.g. [-1, 0, 1])
                scale: torch.Tensor,      # [Cout, 1, 1, 1]-broadcastable, > 0
                clip_grad: bool = True):
        w_n = weight / scale                                   # normalized
        # [numel, K] distance to each codeword -> nearest assignment
        idx = torch.argmin(
            (w_n.unsqueeze(-1) - levels.view(*([1] * w_n.dim()), -1)).abs(),
            dim=-1)                                            # [same as w]
        w_q = scale * levels[idx]

        ctx.save_for_backward(w_n, levels, scale, idx)
        ctx.clip_grad = clip_grad
        return w_q

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        w_n, levels, scale, idx = ctx.saved_tensors

        # --- (a) STE path to the shadow weights -----------------------------
        grad_w = grad_out.clone()
        if ctx.clip_grad:
            # kill gradient where the shadow weight is far outside the
            # representable range (beyond outermost codewords +/- 50% slack)
            lo = levels.min() * 1.5
            hi = levels.max() * 1.5
            grad_w = grad_w * ((w_n >= lo) & (w_n <= hi)).to(grad_out.dtype)

        # --- (b) exact gradient to the codebook levels ----------------------
        # dw_q/dlevels[k] = scale for every element assigned to k
        flat_g = (grad_out * scale.expand_as(grad_out)).reshape(-1)
        grad_levels = torch.zeros_like(levels).scatter_add_(
            0, idx.reshape(-1), flat_g)

        # --- (c) exact gradient to the scale --------------------------------
        # dw_q/dscale = levels[idx]; reduce over all dims that scale broadcasts
        g_scale_full = grad_out * levels[idx]
        reduce_dims = [d for d in range(grad_out.dim())
                       if scale.shape[d] == 1] if scale.dim() == grad_out.dim() \
            else list(range(grad_out.dim()))
        grad_scale = g_scale_full.sum(dim=reduce_dims, keepdim=(scale.dim() == grad_out.dim()))

        return grad_w, grad_levels, grad_scale, None


# ----------------------------------------------------------------------------
# 2. Quantizer module: owns the codebook + scale, plugs in like
#    UniformAffineQuantizer (same call signature: quantizer(weight))
# ----------------------------------------------------------------------------
class TernaryCodebookQuantizer(nn.Module):
    """
    1.58-bit / 2-bit non-uniform weight quantizer.

    * levels: nn.Parameter [K]  -- the learnable codebook.
        K = 3 -> ternary (1.58 bit); K = 4 -> 2-bit.
        Freeze it (requires_grad_(False)) for a fixed BitNet-style grid.
    * scale:  nn.Parameter, per-output-channel, initialized with the
        BitNet b1.58 "absmean" rule: scale_c = mean(|W_c|)  (per channel).
    """

    def __init__(self,
                 n_levels: int = 3,
                 channel_wise: bool = True,
                 learn_levels: bool = True,
                 clip_grad: bool = True):
        super().__init__()
        assert n_levels in (3, 4), "ternary (3) or 2-bit (4) supported here"
        if n_levels == 3:
            init = torch.tensor([-1.0, 0.0, 1.0])
        else:  # symmetric 2-bit grid; will deform freely during training
            init = torch.tensor([-1.5, -0.5, 0.5, 1.5])
        self.levels = nn.Parameter(init, requires_grad=learn_levels)
        self.scale: Optional[nn.Parameter] = None      # lazy init from weights
        self.channel_wise = channel_wise
        self.clip_grad = clip_grad
        self.inited = False

    @torch.no_grad()
    def init_scale(self, weight: torch.Tensor):
        if self.channel_wise:
            dims = list(range(1, weight.dim()))
            s = weight.abs().mean(dim=dims, keepdim=True)  # absmean, per Cout
        else:
            s = weight.abs().mean().view(*([1] * weight.dim()))
        self.scale = nn.Parameter(s.clamp_min(1e-8))
        self.inited = True

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        if not self.inited:  # fallback; normally initialized eagerly by the wrapper
            self.init_scale(weight)
        # clamp keeps scale positive; gradient still flows through to the Parameter
        scale = self.scale.clamp_min(1e-8)
        return CodebookQuantSTE.apply(weight, self.levels, scale, self.clip_grad)

    def extra_repr(self):
        import math
        k = self.levels.numel()
        return f"n_levels={k} (~{math.log2(k):.2f} bit), " \
               f"channel_wise={self.channel_wise}, learn_levels={self.levels.requires_grad}"


class ActFakeQuant(nn.Module):
    """
    Minimal activation fake-quant (uniform, EMA range, STE via round trick).
    Activations stay at 8/6/4 bit -- only weights go ternary. Swappable with
    the existing UniformAffineQuantizer(leaf_param=True).
    """

    def __init__(self, n_bits: int = 8, momentum: float = 0.1):
        super().__init__()
        self.n_bits = n_bits
        self.momentum = momentum
        self.register_buffer("lo", torch.zeros(1))
        self.register_buffer("hi", torch.zeros(1))
        self.inited = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            lo, hi = x.detach().min(), x.detach().max()
            if not self.inited:
                self.lo.fill_(lo), self.hi.fill_(hi)
                self.inited = True
            else:
                self.lo.mul_(1 - self.momentum).add_(self.momentum * lo)
                self.hi.mul_(1 - self.momentum).add_(self.momentum * hi)
        n = 2 ** self.n_bits - 1
        delta = ((self.hi - self.lo) / n).clamp_min(1e-8)
        x_int = ((x - self.lo) / delta).round().clamp(0, n)
        x_q = x_int * delta + self.lo
        return (x_q - x).detach() + x       # plain STE is fine for activations


# ----------------------------------------------------------------------------
# 3. The QAT layer wrapper -- drop-in analogue of quant_layer.QuantModule
# ----------------------------------------------------------------------------
class QATQuantModule(nn.Module):
    """
    Wraps Conv2d / ConvTranspose2d / Linear.

    * `self.weight` IS the full-precision shadow weight (nn.Parameter,
      initialized from the pretrained FP module, updated by the optimizer).
    * Every forward pass re-quantizes the shadow weight through the
      ternary codebook; the quantized tensor is a temporary.
    * `set_quant_state(False)` bypasses quantization (FP debug path),
      matching the PTQ QuantModule API so pyramid_recon/encoder_recon
      tooling keeps working.
    """

    def __init__(self,
                 org_module: Union[nn.Conv2d, nn.ConvTranspose2d, nn.Linear],
                 n_levels: int = 3,
                 act_bits: Optional[int] = 8,
                 learn_levels: bool = True):
        super().__init__()
        if isinstance(org_module, nn.Conv2d):
            self.fwd_kwargs = dict(stride=org_module.stride, padding=org_module.padding,
                                   dilation=org_module.dilation, groups=org_module.groups)
            self.fwd_func = F.conv2d
        elif isinstance(org_module, nn.ConvTranspose2d):
            self.fwd_kwargs = dict(stride=org_module.stride, padding=org_module.padding,
                                   output_padding=org_module.output_padding,
                                   groups=org_module.groups, dilation=org_module.dilation)
            self.fwd_func = F.conv_transpose2d
        else:
            self.fwd_kwargs = dict()
            self.fwd_func = F.linear

        # ---- full-precision shadow weights (master copy) -------------------
        self.weight = nn.Parameter(org_module.weight.detach().clone())
        self.bias = (nn.Parameter(org_module.bias.detach().clone())
                     if org_module.bias is not None else None)

        self.weight_quantizer = TernaryCodebookQuantizer(
            n_levels=n_levels, learn_levels=learn_levels)
        # Eager scale init: MUST happen before the optimizer is constructed,
        # otherwise the lazily-created scale Parameter is never optimized.
        self.weight_quantizer.init_scale(self.weight)
        self.act_quantizer = ActFakeQuant(act_bits) if act_bits else None

        self.use_weight_quant = True
        self.use_act_quant = act_bits is not None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight_quantizer(self.weight) if self.use_weight_quant else self.weight
        out = self.fwd_func(x, w, self.bias, **self.fwd_kwargs)
        if self.use_act_quant and self.act_quantizer is not None:
            out = self.act_quantizer(out)
        return out

    def set_quant_state(self, weight_quant: bool = True, act_quant: bool = True):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant

    # ---- deployment: emit the discrete transmission/storage form -----------
    @torch.no_grad()
    def export_codes(self):
        """Returns (codes int8 [same shape as weight], levels [K], scale)."""
        q = self.weight_quantizer
        w_n = self.weight / q.scale.clamp_min(1e-8)
        idx = torch.argmin(
            (w_n.unsqueeze(-1) - q.levels.view(*([1] * w_n.dim()), -1)).abs(),
            dim=-1)
        return idx.to(torch.int8), q.levels.detach().clone(), q.scale.detach().clone()


# ----------------------------------------------------------------------------
# 4. Graph surgery -- mirrors QuantModel.quant_module_refactor
# ----------------------------------------------------------------------------
def convert_to_qat(module: nn.Module, n_levels: int = 3, act_bits: int = 8,
                   skip_names: tuple = ()) -> nn.Module:
    for name, child in module.named_children():
        if name in skip_names:
            continue
        if isinstance(child, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            setattr(module, name, QATQuantModule(child, n_levels=n_levels,
                                                 act_bits=act_bits))
        else:
            convert_to_qat(child, n_levels, act_bits, skip_names)
    return module


# ----------------------------------------------------------------------------
# 5. Self-test: verify all three gradient paths + end-to-end training step
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    # A stand-in for one BEV backbone block (conv -> relu -> conv)
    net = nn.Sequential(
        nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(),
        nn.Conv2d(16, 8, 3, padding=1),
    )
    net = convert_to_qat(net, n_levels=3, act_bits=8)
    print(net)

    x = torch.randn(4, 8, 32, 32)
    target = torch.randn(4, 8, 32, 32)

    opt = torch.optim.AdamW(net.parameters(), lr=1e-2)
    losses = []
    for step in range(50):
        opt.zero_grad()
        loss = F.mse_loss(net(x), target)
        loss.backward()

        if step == 0:
            m = net[0]
            assert m.weight.grad is not None and m.weight.grad.abs().sum() > 0, \
                "STE failed: no gradient reached shadow weights"
            assert m.weight_quantizer.levels.grad is not None and \
                   m.weight_quantizer.levels.grad.abs().sum() > 0, \
                "codebook received no gradient"
            assert m.weight_quantizer.scale.grad is not None and \
                   m.weight_quantizer.scale.grad.abs().sum() > 0, \
                "scale received no gradient"
            print("[ok] gradients flow to shadow weights, codebook levels, and scales")

        opt.step()
        losses.append(loss.item())

    m = net[0]
    codes, levels, scale = m.export_codes()
    uniq = codes.unique().tolist()
    zero_idx = int(levels.abs().argmin())           # codeword closest to 0
    zero_frac = (codes == zero_idx).float().mean().item()
    print(f"[ok] loss {losses[0]:.4f} -> {losses[-1]:.4f} over {len(losses)} steps")
    print(f"[ok] learned codebook levels: {levels.tolist()}")
    print(f"[ok] ternary code usage: unique={uniq}, zero fraction={zero_frac:.2%}")
    print(f"[ok] shadow weights remain FP32: dtype={m.weight.dtype}, "
          f"distinct values={m.weight.unique().numel()} (>3 as expected)")
    assert losses[-1] < losses[0], "training did not reduce loss"
    print("ALL CHECKS PASSED")
