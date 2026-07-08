"""
Gate QAT-0.4 — QATQuantModule (dense wrappers).
"""

import copy

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from opencood.qat.qat_module import QATQuantModule


DENSE_CASES = [
    ("conv2d", lambda: nn.Conv2d(4, 8, 3, padding=1, bias=True),
     lambda: torch.randn(2, 4, 8, 8)),
    ("convT2d", lambda: nn.ConvTranspose2d(4, 8, 2, stride=2, bias=True),
     lambda: torch.randn(2, 4, 8, 8)),
    ("linear", lambda: nn.Linear(16, 8, bias=True),
     lambda: torch.randn(4, 16)),
]


# ---------------------------------------------------------------------------
# 1. FP bypass must be BIT-EXACT for all three layer types (gate #1).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name,make_layer,make_input", DENSE_CASES)
def test_fp_bypass_bit_exact(name, make_layer, make_input):
    layer = make_layer()
    x = make_input()
    ref = layer(x)
    qm = QATQuantModule(copy.deepcopy(layer), n_levels=3)
    qm.set_quant_state(False, False)
    assert torch.equal(qm(x), ref), f"{name}: FP bypass diverged from original"


# ---------------------------------------------------------------------------
# 2. gradients must reach ALL THREE parameter families (gate #2).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name,make_layer,make_input", DENSE_CASES)
def test_gradients_reach_all_paths(name, make_layer, make_input):
    qm = QATQuantModule(make_layer(), n_levels=3)
    out = qm(make_input())
    out.sum().backward()
    assert qm.weight.grad is not None and qm.weight.grad.abs().sum() > 0, \
        f"{name}: STE failed — no gradient on shadow weights"
    assert qm.weight_quantizer.levels.grad is not None and \
        qm.weight_quantizer.levels.grad.abs().sum() > 0, \
        f"{name}: codebook levels received no gradient"
    assert qm.weight_quantizer.scale.grad is not None and \
        qm.weight_quantizer.scale.grad.abs().sum() > 0, \
        f"{name}: scale received no gradient"


# ---------------------------------------------------------------------------
# 3. 10-step overfit: loss decreases WITH quantization enabled (gate #3).
# ---------------------------------------------------------------------------
def test_overfit_loss_decreases():
    net = nn.Sequential(
        QATQuantModule(nn.Conv2d(4, 16, 3, padding=1), n_levels=3),
        nn.ReLU(),
        QATQuantModule(nn.Conv2d(16, 4, 3, padding=1), n_levels=3),
    )
    x, target = torch.randn(4, 4, 8, 8), torch.randn(4, 4, 8, 8)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-2, weight_decay=0.0)
    losses = []
    for _ in range(10):
        opt.zero_grad()
        loss = F.mse_loss(net(x), target)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0], \
        f"training through ternary quant did not reduce loss: {losses}"


# ---------------------------------------------------------------------------
# 4. state_dict round-trip (gate #4).
# ---------------------------------------------------------------------------
def test_state_dict_roundtrip():
    src = QATQuantModule(nn.Conv2d(4, 8, 3, padding=1), n_levels=3)
    # perturb every learnable family so defaults can't hide a load bug
    with torch.no_grad():
        src.weight += 0.05
        src.weight_quantizer.levels += 0.1
        src.weight_quantizer.scale *= 1.3
    dst = QATQuantModule(nn.Conv2d(4, 8, 3, padding=1), n_levels=3)
    dst.load_state_dict(src.state_dict())
    x = torch.randn(2, 4, 8, 8)
    assert torch.equal(src(x), dst(x))


# ---------------------------------------------------------------------------
# 5. ConvTranspose2d per-OUTPUT-channel scale shape (gate #5).
# ---------------------------------------------------------------------------
def test_convtranspose_scale_along_output_dim():
    qm = QATQuantModule(nn.ConvTranspose2d(4, 8, 2, stride=2), n_levels=3)
    # weight layout [Cin=4, Cout=8, kH, kW] -> scale must be [1, 8, 1, 1]
    assert qm.weight_quantizer.scale.shape == (1, 8, 1, 1), \
        "ConvTranspose2d scale must be per-OUTPUT-channel (dim 1)"


# ---------------------------------------------------------------------------
# 6. shadow-weight contract: master stays FP and diverse; quantized image is
#    a temporary; export emits <= K distinct codes.
# ---------------------------------------------------------------------------
def test_shadow_weight_contract():
    qm = QATQuantModule(nn.Conv2d(4, 8, 3, padding=1), n_levels=3)
    x = torch.randn(2, 4, 8, 8)
    opt = torch.optim.SGD(qm.parameters(), lr=1e-2)
    for _ in range(5):
        opt.zero_grad()
        qm(x).sum().backward()
        opt.step()
    assert qm.weight.dtype == torch.float32
    assert qm.weight.unique().numel() > 3, \
        "shadow weights must remain full-precision (were overwritten?)"
    codes, levels, scale = qm.export_codes()
    assert codes.dtype == torch.int8
    assert codes.unique().numel() <= 3
    assert levels.numel() == 3 and scale.shape == (8, 1, 1, 1)


def test_bias_kept_fp_and_trainable():
    qm = QATQuantModule(nn.Conv2d(4, 8, 3, bias=True), n_levels=3)
    qm(torch.randn(2, 4, 8, 8)).sum().backward()
    assert qm.bias.grad is not None
    assert qm.bias.dtype == torch.float32


def test_rejects_unsupported_module():
    with pytest.raises(TypeError):
        QATQuantModule(nn.Conv3d(2, 2, 3), n_levels=3)
