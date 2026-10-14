"""Tolerances for comparing an implementation against the float64 oracle.

Every implementation here receives inputs that are exactly representable in
its dtype, so the only errors are the implementation's own: float32
accumulation of the sum of squares (relative error around sqrt(N) * 2^-24,
far below bf16 resolution) and the final rounding to the output dtype. One
rounding costs at most half an ulp; the Hugging Face ordering rounds twice.

The tolerance is therefore two machine epsilons, relative and absolute, which
is about four ulps at magnitude one and two ulps at magnitude eight. It is
deliberately tight: ``tests/test_tolerance.py`` checks that kernels with
classic bugs (mean over the padded block instead of N, eps dropped, weight
dropped, fp16 accumulation) fail it.
"""
from __future__ import annotations

import torch

_FP32_RTOL = 1e-5
_FP32_ATOL = 1e-6


def tolerance(dtype: torch.dtype) -> tuple[float, float]:
    """Return ``(rtol, atol)`` for comparing outputs of ``dtype`` to the oracle."""
    if dtype == torch.float32:
        return _FP32_RTOL, _FP32_ATOL
    if dtype == torch.float64:
        return 1e-12, 1e-12
    eps = torch.finfo(dtype).eps
    return 2 * eps, 2 * eps


def max_error(actual: torch.Tensor, expected64: torch.Tensor) -> tuple[float, float]:
    """``(max absolute error, max relative error)`` of ``actual`` against a float64 target.

    The relative error uses ``max(|expected|, 1)`` in the denominator so that
    near-zero outputs do not report meaningless ratios.
    """
    diff = (actual.detach().to(torch.float64) - expected64).abs()
    denom = expected64.abs().clamp_min(1.0)
    if diff.numel() == 0:
        return 0.0, 0.0
    return diff.max().item(), (diff / denom).max().item()


def is_close(actual: torch.Tensor, expected64: torch.Tensor, dtype: torch.dtype | None = None) -> bool:
    """``torch.allclose`` against the oracle with the dtype's tolerance; NaN never matches."""
    dtype = dtype or actual.dtype
    rtol, atol = tolerance(dtype)
    a = actual.detach().to(torch.float64)
    if torch.isnan(a).any() or torch.isinf(a).any():
        return False
    return torch.allclose(a, expected64, rtol=rtol, atol=atol)
