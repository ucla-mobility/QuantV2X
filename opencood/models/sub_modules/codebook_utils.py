import math
from typing import Union
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from typing import Callable, Dict, List, Tuple, Union

from torch import nn

# from mcquic import Consts

__all__ = [
    "NonNegativeParametrizer",
    "LogExpMinusOne",
    "logExpMinusOne"
]


class _lowerBound(torch.autograd.Function):
    """Autograd function for the `LowerBound` operator."""

    @staticmethod
    def forward(ctx, input_, bound):
        ctx.save_for_backward(input_, bound)
        return torch.max(input_, bound)

    @staticmethod
    def backward(ctx, grad_output):
        input_, bound = ctx.saved_tensors
        pass_through_if = (input_ >= bound) | (grad_output < 0)
        return pass_through_if.type(grad_output.dtype) * grad_output, None


class LowerBound(nn.Module):
    """Lower bound operator, computes `torch.max(x, bound)` with a custom
    gradient.
    The derivative is replaced by the identity function when `x` is moved
    towards the `bound`, otherwise the gradient is kept to zero.
    """

    def __init__(self, bound: float):
        """Lower bound operator.

        Args:
            bound (float): The lower bound.
        """
        super().__init__()
        self.register_buffer("bound", torch.Tensor([float(bound)]))

    @torch.jit.unused
    def lower_bound(self, x):
        return _lowerBound.apply(x, self.bound)

    def forward(self, x):
        if torch.jit.is_scripting():
            return torch.max(x, self.bound)
        return self.lower_bound(x)


def gumbelSoftmax(logits: torch.Tensor, temperature: float = 1.0, hard: bool = True, dim: int = -1):
    eps = torch.finfo(logits.dtype).eps
    uniforms = torch.rand_like(logits).clamp_(eps, 1 - eps)
    gumbels = -((-(uniforms.log())).log())

    y_soft = ((logits + gumbels) / temperature).softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft  # previous version, bug?
        # ret = y_hard.detach() - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret


@dataclass
class CodeSize:
    """Latent code specification.
           Code in this paper is of shape: `[[1, m, h, w], [1, m, h, w] ... ]`
                                                            `↑ total length = L`

    Args:
        heights (List[int]): Latent height for each stage.
        widths (List[int]): Latent width for each stage.
        k (List[int]): [k1, k2, ...], codewords amount for each stage.
        m (int): M, multi-codebook amount.
    """
    m: int
    heights: List[int]
    widths: List[int]
    k: List[int]

    def __str__(self) -> str:
        sequence = ", ".join(f"[{w}x{h}, {k}]" for h, w, k in zip(self.heights, self.widths, self.k))
        return f"""
        {self.m} code-groups: {sequence}"""
