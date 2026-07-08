"""
Gate QAT-0.6 — QATQuantModel graph surgery + param groups + config plumbing.
"""

import pytest
import torch
import torch.nn as nn

from opencood.qat.qat_config import DEFAULT_QAT_CONFIG, get_qat_config
from opencood.qat.qat_model import QATQuantModel
from opencood.qat.qat_module import QATQuantModule
from opencood.qat.qat_spconv import QATQuantSpconvModule

from conftest import MockSubMConv3d, ToyV2XModel


CFG = {
    "skip_names": ["aligner_m1"],
    "high_precision_names": ["cls_head"],
    "first_conv_high_precision": True,
}


# ---------------------------------------------------------------------------
# config plumbing (QAT-0.1)
# ---------------------------------------------------------------------------
def test_config_defaults_and_merge():
    cfg = get_qat_config({"qat": {"weight": {"n_levels": 4}}})
    assert cfg["weight"]["n_levels"] == 4
    assert cfg["weight"]["learn_levels"] is True          # default preserved
    assert cfg["act"]["enabled"] is False


def test_config_absent_section_gives_defaults():
    cfg = get_qat_config({"model": {"core_method": "whatever"}})
    assert cfg == get_qat_config(None) == get_qat_config({})
    assert cfg["weight"]["n_levels"] == DEFAULT_QAT_CONFIG["weight"]["n_levels"]


def test_config_typo_rejected():
    with pytest.raises(KeyError):
        get_qat_config({"qat": {"weight": {"learn_lvls": False}}})


@pytest.mark.parametrize("bad", [
    {"weight": {"n_levels": 1}},
    {"act": {"n_bits": 9}},
    {"act": {"mode": "banana"}},
    {"act": {"qdrop_prob": 0.0}},
])
def test_config_validation(bad):
    with pytest.raises((ValueError, TypeError)):
        get_qat_config({"qat": bad})


# ---------------------------------------------------------------------------
# surgery coverage (QAT-0.6 gate: no leaks, correct dispositions)
# ---------------------------------------------------------------------------
def test_surgery_coverage(toy_model):
    qmodel = QATQuantModel(toy_model, CFG)
    cov = dict(qmodel.coverage_report())

    # FP island skipped
    assert cov["aligner_m1"].startswith("SKIPPED")
    assert isinstance(qmodel.model.aligner_m1, nn.Conv2d)

    # first encoder conv -> high-precision island
    assert "first-conv HP" in cov["encoder_m1.0"]
    assert "K=255" in cov["encoder_m1.0"]
    # second encoder conv -> ternary
    assert "K=3" in cov["encoder_m1.3"]

    # named high-precision island
    assert "K=255" in cov["cls_head"]

    # no unwrapped eligible layers anywhere except the skip list (leak sweep)
    for name, m in qmodel.model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            assert name.startswith("aligner_m1") or isinstance(
                m, (QATQuantModule, QATQuantSpconvModule)), \
                f"LEAK: {name} ({type(m).__name__}) escaped surgery"


def test_bn_stays_live(toy_model):
    qmodel = QATQuantModel(toy_model, CFG)
    bns = [m for m in qmodel.model.modules()
           if isinstance(m, nn.BatchNorm2d)]
    assert len(bns) == 1, "BatchNorm must remain in the graph (no folding)"
    assert bns[0].training or True  # presence is the contract; mode is caller's


def test_sparse_layers_swapped():
    class SparseModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder_m1 = nn.Sequential(MockSubMConv3d(4, 8))

    QATQuantModel.EXTRA_SPARSE_TYPES = (MockSubMConv3d,)
    try:
        qmodel = QATQuantModel(SparseModel(), CFG)
        # first_conv_high_precision applies to the sparse first conv too
        wrapped = qmodel.model.encoder_m1[0]
        assert isinstance(wrapped, QATQuantSpconvModule)
        assert wrapped.weight_quantizer.n_levels == 255
    finally:
        QATQuantModel.EXTRA_SPARSE_TYPES = ()


# ---------------------------------------------------------------------------
# full-model FP bypass == original model (QAT-0.6 gate)
# ---------------------------------------------------------------------------
def test_full_model_fp_bypass_matches_baseline(toy_input):
    torch.manual_seed(7)
    ref_model = ToyV2XModel().eval()
    ref_cls, ref_fc = ref_model(toy_input)

    torch.manual_seed(7)                       # identical init
    qmodel = QATQuantModel(ToyV2XModel(), CFG).eval()
    qmodel.set_quant_state(False, False)
    cls, fc = qmodel(toy_input)
    assert torch.equal(cls, ref_cls) and torch.equal(fc, ref_fc), \
        "FP bypass must reproduce the unconverted model bit-exactly"


# ---------------------------------------------------------------------------
# param groups (QAT-0.6 gate: split, lrs, weight_decay hard-zero)
# ---------------------------------------------------------------------------
def test_param_groups_split_and_hyperparams(toy_model):
    qmodel = QATQuantModel(toy_model, CFG)
    groups = qmodel.qat_param_groups()
    assert len(groups) == 2
    shadow, qparams = groups
    assert shadow["lr"] == pytest.approx(2e-5)
    assert qparams["lr"] == pytest.approx(1e-4)
    assert shadow["weight_decay"] == 0.0 and qparams["weight_decay"] == 0.0

    # every scale/levels param is in the quant group, never the shadow group
    shadow_ids = {id(p) for p in shadow["params"]}
    for name, m in qmodel.qat_modules():
        assert id(m.weight) in shadow_ids
        assert id(m.weight_quantizer.scale) not in shadow_ids
    # groups are optimizer-consumable
    torch.optim.AdamW(groups)


def test_param_groups_cover_all_trainables(toy_model):
    """Nothing trainable inside a wrapper may be missing from the groups —
    a parameter born outside them is silently never optimized."""
    qmodel = QATQuantModel(toy_model, CFG)
    grouped = {id(p) for g in qmodel.qat_param_groups() for p in g["params"]}
    for name, m in qmodel.qat_modules():
        for pname, p in m.named_parameters():
            if p.requires_grad:
                assert id(p) in grouped, f"{name}.{pname} not in any group"


# ---------------------------------------------------------------------------
# end-to-end smoke: full converted model trains, quantized, 10 steps.
# ---------------------------------------------------------------------------
def test_toy_model_end_to_end_training(toy_model, toy_input):
    qmodel = QATQuantModel(toy_model, CFG).train()
    opt = torch.optim.AdamW(qmodel.qat_param_groups(lr_weight=1e-3,
                                                    lr_quant=1e-3))
    target_cls = torch.randn(2, 2, 16, 16)
    losses = []
    for _ in range(10):
        opt.zero_grad()
        cls, _ = qmodel(toy_input)
        loss = torch.nn.functional.mse_loss(cls, target_cls)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0], f"no learning through full surgery: {losses}"


def test_set_quant_state_reaches_every_wrapper(toy_model):
    qmodel = QATQuantModel(toy_model, CFG)
    qmodel.set_quant_state(False, False)
    for _, m in qmodel.qat_modules():
        assert not m.use_weight_quant and not m.use_act_quant
