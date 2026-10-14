"""Float64 oracle for RMSNorm.

RMSNorm (Zhang and Sennrich, 2019) normalizes each row of ``x`` by its root
mean square and rescales every feature with a learned weight::

    y_i = x_i / sqrt(mean_j(x_j^2) + eps) * w_i

There is no mean subtraction, so one reduction per row is enough. That single
reduction is what makes the operation a clean study of memory-bound GPU code:
each element is read once, squared, summed, and written once.

The oracle computes the formula in float64 so that every implementation,
including the float32 eager baseline, is measured against the same target
rather than against each other.
"""
from __future__ import annotations

import torch


def _check_shapes(x: torch.Tensor, weight: torch.Tensor) -> None:
    if weight.dim() != 1:
        raise ValueError(f"weight must be 1-D, got shape {tuple(weight.shape)}")
    if x.dim() < 1 or x.shape[-1] != weight.shape[0]:
        raise ValueError(
            f"last dim of x ({tuple(x.shape)}) must match weight ({tuple(weight.shape)})"
        )


def rmsnorm_reference(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMSNorm in float64. Returns a float64 tensor of x's shape."""
    _check_shapes(x, weight)
    x64 = x.detach().to(torch.float64)
    w64 = weight.detach().to(torch.float64)
    mean_sq = x64.square().mean(dim=-1, keepdim=True)
    return x64 * torch.rsqrt(mean_sq + eps) * w64


def rmsnorm_residual_reference(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused residual-add + RMSNorm in float64.

    Returns ``(rmsnorm(x + residual), x + residual)``. A decoder layer feeds the
    second output (the updated residual stream) to the next layer, so an
    implementation has to produce both.
    """
    if residual.shape != x.shape:
        raise ValueError(f"residual shape {tuple(residual.shape)} != x shape {tuple(x.shape)}")
    s64 = x.detach().to(torch.float64) + residual.detach().to(torch.float64)
    return rmsnorm_reference(s64, weight, eps), s64
