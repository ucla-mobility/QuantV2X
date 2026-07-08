"""
Gate QAT-0.7 — QATTelemetry: flip rate, collapse detection, alarms.
"""

import pytest
import torch
import torch.nn as nn

from opencood.qat.qat_model import QATQuantModel
from opencood.qat.qat_module import QATQuantModule
from opencood.qat.telemetry import QATTelemetry

from conftest import ToyV2XModel


def _small_qat_net():
    return nn.Sequential(
        QATQuantModule(nn.Conv2d(4, 8, 3, padding=1), n_levels=3),
        nn.ReLU(),
        QATQuantModule(nn.Conv2d(8, 4, 3, padding=1), n_levels=3),
    )


def test_zero_flip_rate_when_nothing_moves():
    net = _small_qat_net()
    telem = QATTelemetry(net)
    metrics = telem.step(0)                    # no training in between
    flips = [v for k, v in metrics.items() if k.endswith("flip_rate")]
    assert flips and all(f == 0.0 for f in flips)


def test_flip_rate_detects_forced_flip():
    net = _small_qat_net()
    telem = QATTelemetry(net)
    m = net[0]
    with torch.no_grad():
        # push one weight across a codeword boundary deterministically:
        # scale * 2 lands a former ~0-level weight in the +1 cell
        m.weight[0, 0, 0, 0] = m.weight_quantizer.scale[0, 0, 0, 0] * 2.0
    metrics = telem.step(1)
    numel = m.weight.numel()
    key = [k for k in metrics if k.endswith("flip_rate")][0]
    # exactly the flipped fraction if that weight changed cells (>= 0, and
    # at most a few elements); assert it registered
    assert metrics[key] >= 1.0 / numel - 1e-9


def test_flip_rate_under_training_is_positive_and_sane():
    torch.manual_seed(0)
    net = _small_qat_net().train()
    telem = QATTelemetry(net)
    opt = torch.optim.AdamW(net.parameters(), lr=5e-3, weight_decay=0.0)
    x, y = torch.randn(4, 4, 8, 8), torch.randn(4, 4, 8, 8)
    for _ in range(50):
        opt.zero_grad()
        nn.functional.mse_loss(net(x), y).backward()
        opt.step()
    metrics = telem.step(50)
    flips = [v for k, v in metrics.items() if k.endswith("flip_rate")]
    assert any(f > 0 for f in flips), \
        "50 aggressive steps flipped no codewords — suspicious (dead grads?)"
    assert all(f < 0.5 for f in flips), "half the codes flipped — thrashing"


def test_zero_frac_and_levels_reported():
    net = _small_qat_net()
    metrics = QATTelemetry(net).step(0)
    zf = [v for k, v in metrics.items() if k.endswith("zero_frac")]
    assert zf and all(0.0 <= v <= 1.0 for v in zf)
    # ternary: three level_* entries per layer
    assert sum(1 for k in metrics if "/level_" in k) == 2 * 3


def test_shadow_grad_norm_tripwire():
    net = _small_qat_net()
    telem = QATTelemetry(net)
    # before any backward: grad is None -> sentinel -1
    m0 = telem.step(0)
    assert all(v == -1.0 for k, v in m0.items()
               if k.endswith("shadow_grad_norm"))
    net(torch.randn(2, 4, 8, 8)).sum().backward()
    m1 = telem.step(1)
    assert all(v > 0 for k, v in m1.items()
               if k.endswith("shadow_grad_norm"))


def test_alarms_fire_correctly():
    net = _small_qat_net()
    telem = QATTelemetry(net)
    name = list(telem.mods)[0]
    # synthetic metric dicts exercising each alarm branch
    assert name in telem.alarms({f"{name}/flip_rate": 0.10})       # thrash
    assert name in telem.alarms({f"{name}/flip_rate": 0.0})        # frozen
    assert name in telem.alarms({f"{name}/flip_rate": 5e-3,
                                 f"{name}/zero_frac": 0.99})       # collapse
    warn = telem.alarms({f"{name}/flip_rate": 5e-3,
                         f"{name}/zero_frac": 0.4,
                         f"{name}/shadow_grad_norm": 0.0})         # severed
    assert "severed" in warn[name]
    # healthy metrics -> no alarm for this layer
    healthy = telem.alarms({f"{name}/flip_rate": 5e-3,
                            f"{name}/zero_frac": 0.4,
                            f"{name}/shadow_grad_norm": 1.0})
    assert name not in healthy


def test_writer_callback_receives_every_metric():
    net = _small_qat_net()
    seen = []
    telem = QATTelemetry(net, writer=lambda tag, val, it: seen.append(tag))
    metrics = telem.step(3)
    assert sorted(seen) == sorted(metrics.keys())


def test_works_on_full_qat_model():
    qmodel = QATQuantModel(ToyV2XModel(), {"skip_names": ["aligner_m1"]})
    telem = QATTelemetry(qmodel)
    assert len(telem.mods) >= 5                # all wrapped layers found
    telem.step(0)


def test_raises_on_unconverted_model():
    with pytest.raises(ValueError):
        QATTelemetry(nn.Sequential(nn.Conv2d(2, 2, 3)))
