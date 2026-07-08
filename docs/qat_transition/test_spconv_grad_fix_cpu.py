"""
CPU-only verification of the QuantSpconvModule autograd fix.

spconv's conv kernels need CUDA, but the bug/fix is purely about how the weight
tensor is attached to the module before the kernel runs. So we replicate that
attachment step and inspect the autograd graph directly — no GPU needed.

    python docs/qat_transition/test_spconv_grad_fix_cpu.py

The definitive end-to-end check (test_spconv_grad_fix.py) still needs a GPU box.
"""
import torch
import torch.nn as nn

try:
    from spconv import SubMConv3d
except ImportError:
    from spconv.pytorch import SubMConv3d

from opencood.quant.quant_layer import QuantSpconvModule

conv = SubMConv3d(4, 8, kernel_size=3, padding=1, bias=False)
qmod = QuantSpconvModule(conv, weight_quant_params=dict(n_bits=8),
                         act_quant_params=dict(n_bits=8))
qmod.set_quant_state(weight_quant=True, act_quant=False)

# --- replicate the exact weight path from the (fixed) forward ----------------
weight = qmod.weight_quantizer(qmod.weight)

# 1) the quantized weight must carry autograd history back to the shadow weight
assert weight.requires_grad, "quantized weight lost requires_grad"
assert weight.grad_fn is not None, "quantized weight has no grad_fn"

# 2) OLD (buggy) attachment: nn.Parameter() creates a fresh leaf — graph severed
old_style = nn.Parameter(weight)
assert old_style.grad_fn is None and old_style.is_leaf, \
    "sanity check failed: nn.Parameter should sever the graph (it does upstream)"

# 3) NEW (fixed) attachment: deregister + plain assignment preserves the graph
sp = qmod.spconv_module
if 'weight' in sp._parameters:
    del sp._parameters['weight']
sp.weight = weight
assert sp.weight.grad_fn is not None and not sp.weight.is_leaf, \
    "FAIL: fixed attachment still severed the autograd graph"

# 4) gradient actually reaches the shadow weight through the attached tensor
sp.weight.sum().backward()
g = qmod.weight.grad
assert g is not None and g.abs().sum() > 0, \
    "FAIL: no gradient reached QuantSpconvModule.weight"

# 5) idempotent: second pass re-attaches cleanly
qmod.weight.grad = None
w2 = qmod.weight_quantizer(qmod.weight)
if 'weight' in sp._parameters:
    del sp._parameters['weight']
sp.weight = w2
sp.weight.sum().backward()
assert qmod.weight.grad is not None and qmod.weight.grad.abs().sum() > 0

print("[ok] quantized weight keeps grad_fn through the fixed attachment")
print("[ok] old nn.Parameter() attachment confirmed to sever the graph")
print(f"[ok] gradient reached shadow weight: |g|_1={g.abs().sum().item():.4e}")
print("[ok] re-attachment on second forward works")
print("SPCONV FIX VERIFIED (CPU graph check) — run test_spconv_grad_fix.py "
      "on a GPU machine for the end-to-end numeric check")
