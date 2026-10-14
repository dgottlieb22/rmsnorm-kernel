"""The correctness harness has to reject wrong kernels, not just accept right ones.

Each mutant below is a plain-torch RMSNorm with one deliberate bug that real
kernels have shipped with. The test asserts the oracle comparison fails on
the shape and regime designed to expose it, and names the case that would
have let it through. Tolerances were tightened until all of these failed.
"""
from __future__ import annotations

import pytest
import torch

from rmsnorm.reference import rmsnorm_reference
from rmsnorm.tolerance import is_close
from rmsnorm.torch_impls import rmsnorm_torch_eager

from test_correctness import make_inputs


def mean_over_padded_block(x, w, eps):
    """Divides by the power-of-two block size instead of N (masked lanes counted as zeros)."""
    n = x.shape[-1]
    block = 1 << (n - 1).bit_length()
    h = x.float()
    mean_sq = h.pow(2).sum(-1, keepdim=True) / block
    return (h * torch.rsqrt(mean_sq + eps) * w.float()).to(x.dtype)


def eps_dropped(x, w, eps):
    h = x.float()
    return (h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True)) * w.float()).to(x.dtype)


def weight_dropped(x, w, eps):
    h = x.float()
    return (h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype)


def accumulate_in_storage_dtype(x, w, eps):
    """Squares and sums in the input dtype; fp16 overflows on large rows."""
    mean_sq = x.pow(2).mean(-1, keepdim=True)
    return (x * torch.rsqrt(mean_sq.float() + eps).to(x.dtype) * w)


def eps_inside_sqrt_missing_mean(x, w, eps):
    """Uses the sum of squares instead of the mean: off by sqrt(N)."""
    h = x.float()
    return (h * torch.rsqrt(h.pow(2).sum(-1, keepdim=True) + eps) * w.float()).to(x.dtype)


def test_padded_block_mean_caught_on_non_power_of_two(device):
    """N = 5120 pads to 8192; dividing by the block is a 20% error in rstd."""
    x, w, eps = make_inputs(4, 5120, torch.bfloat16, device)
    assert not is_close(mean_over_padded_block(x, w, eps), rmsnorm_reference(x, w, eps))


def test_padded_block_mean_slips_through_on_power_of_two(device):
    """Same bug, N = 4096: block == N, so the mutant is exact. This is why the
    shape matrix includes non-powers of two."""
    x, w, eps = make_inputs(4, 4096, torch.bfloat16, device)
    assert is_close(mean_over_padded_block(x, w, eps), rmsnorm_reference(x, w, eps))


def test_dropped_eps_caught_in_eps_dominant_regime(device):
    x, w, eps = make_inputs(4, 1000, torch.bfloat16, device, regime="eps_dominant")
    assert not is_close(eps_dropped(x, w, eps), rmsnorm_reference(x, w, eps))


def test_dropped_eps_slips_through_on_unit_scale_inputs(device):
    """With mean(x^2) ~ 1 and eps = 1e-6 the difference is below bf16 resolution,
    so unit-scale random inputs alone cannot validate eps handling."""
    x, w, eps = make_inputs(4, 1000, torch.bfloat16, device)
    assert is_close(eps_dropped(x, w, eps), rmsnorm_reference(x, w, eps))


def test_dropped_weight_caught(device):
    x, w, eps = make_inputs(4, 1000, torch.bfloat16, device)
    assert not is_close(weight_dropped(x, w, eps), rmsnorm_reference(x, w, eps))


def test_fp16_accumulation_caught_on_large_inputs(device):
    """x*x overflows fp16 for |x| > 256, mean becomes inf, rsqrt(inf) is 0 and the
    whole row comes out as zeros. No NaN, no error: silent zeros. The oracle
    comparison catches it; a finite-check alone would not."""
    x, w, eps = make_inputs(4, 4096, torch.float16, device, regime="large")
    y = accumulate_in_storage_dtype(x, w, eps)
    assert not is_close(y, rmsnorm_reference(x, w, eps))
    assert torch.isfinite(y.float()).all()
    overflowing_rows = (x.float().abs().amax(dim=-1) > 256)
    assert overflowing_rows.any()
    assert (y[overflowing_rows] == 0).all()


def test_fp16_accumulation_slips_through_on_unit_scale_inputs(device):
    """At unit scale the fp16 sum does not overflow and the error hides inside
    the tolerance; only the 'large' regime exposes this bug."""
    x, w, eps = make_inputs(4, 1000, torch.float16, device)
    assert is_close(accumulate_in_storage_dtype(x, w, eps), rmsnorm_reference(x, w, eps))


def test_sum_instead_of_mean_caught(device):
    x, w, eps = make_inputs(4, 1000, torch.bfloat16, device)
    assert not is_close(eps_inside_sqrt_missing_mean(x, w, eps), rmsnorm_reference(x, w, eps))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
def test_tolerance_admits_one_and_two_roundings(dtype, device):
    """The baseline (one rounding) and the HF ordering (two roundings) both pass;
    anything stricter would reject correct code."""
    from rmsnorm.torch_impls import rmsnorm_torch_eager_hf

    x, w, eps = make_inputs(64, 4096, dtype, device)
    ref = rmsnorm_reference(x, w, eps)
    assert is_close(rmsnorm_torch_eager(x, w, eps), ref)
    assert is_close(rmsnorm_torch_eager_hf(x, w, eps), ref)
