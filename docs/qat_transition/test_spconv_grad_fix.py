"""
Verifies the QuantSpconvModule autograd fix in opencood/quant/quant_layer.py.

Before the fix, `self.spconv_module.weight = nn.Parameter(weight)` created a new
leaf tensor every forward, so loss.backward() produced NO gradient on
QuantSpconvModule.weight (and none on quantizer params). After the fix, the
quantized weight tensor keeps its grad_fn and gradients flow.

Run inside the quantv2x env (needs spconv; CUDA if your spconv build is GPU-only):
    python docs/qat_transition/test_spconv_grad_fix.py
"""
import torch

try:  # spconv1 / spconv2, same pattern as quant_layer.py
    from spconv import SubMConv3d, SparseConvTensor
except ImportError:
    from spconv.pytorch import SubMConv3d, SparseConvTensor

from opencood.quant.quant_layer import QuantSpconvModule

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    print("WARNING: no CUDA — many spconv builds are GPU-only; "
          "if this crashes, rerun on a GPU machine.")

torch.manual_seed(0)

# --- tiny sparse input: 20 active voxels in a 16^3 grid, batch of 1 ------------
n_voxels, in_ch, out_ch = 20, 4, 8
features = torch.randn(n_voxels, in_ch, device=device)
coords = torch.cat([
    torch.zeros(n_voxels, 1, dtype=torch.int32, device=device),          # batch idx
    torch.randint(0, 16, (n_voxels, 3), dtype=torch.int32, device=device)
], dim=1)
x = SparseConvTensor(features, coords, spatial_shape=[16, 16, 16], batch_size=1)

conv = SubMConv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False).to(device)
qmod = QuantSpconvModule(conv, weight_quant_params=dict(n_bits=8),
                         act_quant_params=dict(n_bits=8)).to(device)
qmod.set_quant_state(weight_quant=True, act_quant=False)

out = qmod(x)
loss = out.features.pow(2).sum()
loss.backward()

g = qmod.weight.grad
assert g is not None, "FAIL: no gradient reached QuantSpconvModule.weight " \
                      "(autograd graph still severed)"
assert g.abs().sum() > 0, "FAIL: gradient is all zeros"
print(f"[ok] grad reached shadow weight: shape={tuple(g.shape)}, "
      f"|g|_1={g.abs().sum().item():.4e}")

# second forward must also work (parameter was deregistered on the first call)
qmod.weight.grad = None
loss2 = qmod(x).features.pow(2).sum()
loss2.backward()
assert qmod.weight.grad is not None and qmod.weight.grad.abs().sum() > 0
print("[ok] second forward/backward also flows — deregistration is idempotent")
print("SPCONV GRADIENT FIX VERIFIED")
