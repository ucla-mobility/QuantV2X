"""
Shared fixtures + CPU mocks for the Epic-0 QAT test suite.
===========================================================

Everything here is synthetic — no dataset, no GPU, no spconv required.

The MOCK SPARSE CONV deserves explanation: spconv's CUDA kernels cannot run
on CPU, but the entire class of bug QAT-0.5 guards against lives in how the
weight tensor is ATTACHED to the module before its forward runs (Parameter
re-registration severing grad_fn). The mock therefore reproduces spconv's
exact interface contract —

  * ``weight`` starts life as a registered nn.Parameter,
  * ``out_channels`` attribute (used for channel-dim resolution),
  * forward READS ``self.weight`` (not a captured local!) and computes a
    real differentiable op on ``x.features``,
  * returns an object with ``.features`` / ``.replace_feature`` semantics —

so gradients flow through genuine autograd machinery and the attachment
semantics under test are IDENTICAL to real spconv. If real spconv is
importable (GPU box), the same tests also run against it (see
test_qat_spconv.py::test_real_spconv_attachment).
"""

import random

import pytest
import torch
import torch.nn as nn


# --------------------------------------------------------------------------
# determinism: quantization tests compare exact values; seed everything.
# --------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _seed_everything():
    torch.manual_seed(0)
    random.seed(0)
    yield


# --------------------------------------------------------------------------
# mock sparse machinery
# --------------------------------------------------------------------------
class MockSparseConvTensor:
    """Duck-typed stand-in for spconv.SparseConvTensor: a dense [N, C]
    feature matrix + opaque indices, with the replace_feature contract."""

    def __init__(self, features: torch.Tensor, indices=None):
        self.features = features
        self.indices = indices

    def replace_feature(self, new_features: torch.Tensor):
        return MockSparseConvTensor(new_features, self.indices)


class MockSubMConv3d(nn.Module):
    """CPU mock of spconv 2.x SubMConv3d (weight layout [Cout, k,k,k, Cin]).

    Forward uses only the kernel's CENTER tap — a submanifold conv with an
    identity-footprint rulebook — because the test target is the autograd
    path through ``self.weight``, not sparse geometry. Crucially the weight
    is read from ``self.weight`` AT CALL TIME, exactly like real spconv.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, bias: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(
            torch.randn(out_channels, kernel_size, kernel_size, kernel_size,
                        in_channels) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

    def forward(self, x: MockSparseConvTensor) -> MockSparseConvTensor:
        k = self.kernel_size // 2
        w_center = self.weight[:, k, k, k, :]           # [Cout, Cin]
        out = x.features @ w_center.t()
        if self.bias is not None:
            out = out + self.bias
        return MockSparseConvTensor(out, x.indices)


@pytest.fixture
def mock_sparse_conv():
    return MockSubMConv3d(4, 8, kernel_size=3, bias=False)


@pytest.fixture
def sparse_input():
    # 32 active voxels, 4 channels; requires_grad False (weights are what we
    # differentiate — matches training reality).
    return MockSparseConvTensor(torch.randn(32, 4))


# --------------------------------------------------------------------------
# a toy multi-subsystem model shaped like QuantV2X's naming scheme, so the
# graph-surgery skip / high-precision / first-conv policies are exercised
# against realistic module paths (encoder_m1.*, aligner_m1, cls_head, ...).
# --------------------------------------------------------------------------
class ToyV2XModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_m1 = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),      # -> first-conv HP rule
            nn.BatchNorm2d(8),                  # -> must stay LIVE
            nn.ReLU(),
            nn.Conv2d(8, 8, 3, padding=1),      # -> ternary
        )
        self.backbone_m1 = nn.Sequential(
            nn.Conv2d(8, 16, 3, padding=1),     # -> ternary
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 2, 2),    # -> ternary, channel_dim=1
        )
        self.aligner_m1 = nn.Conv2d(8, 8, 1)    # -> skip list (FP island)
        self.cls_head = nn.Conv2d(8, 2, 1)      # -> high-precision island
        self.fc = nn.Linear(8, 4)               # -> ternary Linear

    def forward(self, x):
        f = self.backbone_m1(self.encoder_m1(x))
        f = self.aligner_m1(f)
        cls = self.cls_head(f)
        pooled = f.mean(dim=(2, 3))
        return cls, self.fc(pooled)


@pytest.fixture
def toy_model():
    return ToyV2XModel()


@pytest.fixture
def toy_input():
    return torch.randn(2, 3, 16, 16)
