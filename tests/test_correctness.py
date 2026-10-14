"""Every implementation against the float64 oracle.

The matrix covers row lengths from 1 to 16384 including non-powers of two
(the masked-lane path), row counts that are not a multiple of any block
(the tail path of the multi-row kernel), three storage dtypes, and input
regimes that expose the classic failure modes: eps-dominated rows, rows that
overflow fp16 when squared, and all-zero rows.
"""
from __future__ import annotations

import pytest
import torch

import rmsnorm
from rmsnorm.reference import rmsnorm_reference, rmsnorm_residual_reference
from rmsnorm.tolerance import is_close, max_error, tolerance

from conftest import impl_items, interpreting

DTYPES = [torch.float32, torch.float16, torch.bfloat16]
ROW_LENGTHS = [1, 2, 3, 7, 16, 100, 512, 768, 1000, 1024, 2048, 2560, 4096, 5120]
LONG_ROWS = [8192, 11008, 16384]
ROW_COUNTS = [1, 37]

REGIMES = {
    # name: (scale of x, eps)
    "normal": (1.0, 1e-6),
    # mean(x^2) ~ 1e-6 == eps: a kernel that drops eps is off by sqrt(2).
    "eps_dominant": (1e-3, 1e-6),
    # |x| up to several hundred: x*x and the row sum overflow fp16 unless the
    # kernel squares and accumulates in float32.
    "large": (150.0, 1e-6),
}


def make_inputs(m, n, dtype, device, regime="normal", seed=0, weight_dtype=None):
    g = torch.Generator(device="cpu").manual_seed(seed)
    scale, eps = REGIMES[regime]
    x = (torch.randn(m, n, generator=g) * scale).to(dtype).to(device)
    w = torch.randn(n, generator=g).to(weight_dtype or dtype).to(device)
    return x, w, eps


def check(impl, x, w, eps, request=None):
    if impl.family == "compile" and request is not None:
        request.getfixturevalue("compile_available")
    y = impl.fn(x, w, eps)
    expected = rmsnorm_reference(x, w, eps)
    assert y.shape == x.shape
    assert y.dtype == x.dtype
    assert y.device == x.device
    abs_err, rel_err = max_error(y, expected)
    rtol, atol = tolerance(x.dtype)
    assert is_close(y, expected), (
        f"{impl.name} {tuple(x.shape)} {x.dtype}: max abs err {abs_err:.3e}, "
        f"max rel err {rel_err:.3e}, tolerance rtol={rtol:.1e} atol={atol:.1e}"
    )
    return y


@pytest.mark.parametrize("impl", impl_items(rmsnorm.IMPLEMENTATIONS))
@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d).split(".")[-1])
@pytest.mark.parametrize("n", ROW_LENGTHS)
@pytest.mark.parametrize("m", ROW_COUNTS)
def test_matches_oracle(impl, dtype, n, m, device, request):
    x, w, eps = make_inputs(m, n, dtype, device)
    check(impl, x, w, eps, request)


@pytest.mark.slow
@pytest.mark.parametrize("impl", impl_items(rmsnorm.IMPLEMENTATIONS))
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("n", LONG_ROWS)
def test_long_rows(impl, dtype, n, device, request):
    x, w, eps = make_inputs(3, n, dtype, device)
    check(impl, x, w, eps, request)


@pytest.mark.parametrize("impl", impl_items(rmsnorm.IMPLEMENTATIONS))
@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d).split(".")[-1])
@pytest.mark.parametrize("regime", ["eps_dominant", "large"])
@pytest.mark.parametrize("n", [1000, 4096])
def test_regimes(impl, dtype, regime, n, device, request):
    x, w, eps = make_inputs(5, n, dtype, device, regime=regime)
    check(impl, x, w, eps, request)


@pytest.mark.parametrize("impl", impl_items(rmsnorm.IMPLEMENTATIONS))
@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d).split(".")[-1])
def test_zero_rows_give_zero_not_nan(impl, dtype, device, request):
    if impl.family == "compile":
        request.getfixturevalue("compile_available")
    x = torch.zeros(4, 1000, dtype=dtype, device=device)
    w = torch.randn(1000, device=device).to(dtype)
    y = impl.fn(x, w, 1e-6)
    assert torch.equal(y, torch.zeros_like(x))


@pytest.mark.parametrize("impl", impl_items(rmsnorm.IMPLEMENTATIONS))
def test_empty_batch(impl, device, request):
    if impl.family == "compile":
        request.getfixturevalue("compile_available")
    x = torch.empty(0, 1024, dtype=torch.bfloat16, device=device)
    w = torch.ones(1024, dtype=torch.bfloat16, device=device)
    y = impl.fn(x, w, 1e-6)
    assert y.shape == (0, 1024) and y.dtype == torch.bfloat16


@pytest.mark.parametrize("impl", impl_items(rmsnorm.IMPLEMENTATIONS))
def test_three_dimensional_input(impl, device, request):
    """(batch, seq, hidden) input normalizes each (batch, seq) row."""
    if impl.family == "compile":
        request.getfixturevalue("compile_available")
    g = torch.Generator().manual_seed(1)
    x = torch.randn(2, 9, 1000, generator=g).to(torch.bfloat16).to(device)
    w = torch.randn(1000, generator=g).to(torch.bfloat16).to(device)
    y = impl.fn(x, w, 1e-6)
    assert y.shape == x.shape
    assert is_close(y, rmsnorm_reference(x, w, 1e-6))


@pytest.mark.parametrize("impl", impl_items(rmsnorm.IMPLEMENTATIONS))
def test_strided_rows(impl, device, request):
    """Rows that are a prefix slice of wider rows: unit stride along N, row stride 2N."""
    if impl.family == "compile":
        request.getfixturevalue("compile_available")
    g = torch.Generator().manual_seed(2)
    wide = torch.randn(13, 2 * 1000, generator=g).to(torch.bfloat16).to(device)
    x = wide[:, :1000]
    assert x.stride(1) == 1 and x.stride(0) == 2000 and not x.is_contiguous()
    w = torch.randn(1000, generator=g).to(torch.bfloat16).to(device)
    y = impl.fn(x, w, 1e-6)
    assert is_close(y, rmsnorm_reference(x, w, 1e-6))


@pytest.mark.parametrize("impl", impl_items(rmsnorm.IMPLEMENTATIONS))
def test_transposed_input(impl, device, request):
    """Non-unit stride along N is legal input; the kernels copy to a contiguous layout first."""
    if impl.family == "compile":
        request.getfixturevalue("compile_available")
    g = torch.Generator().manual_seed(3)
    xt = torch.randn(1000, 11, generator=g).to(torch.bfloat16).to(device)
    x = xt.t()
    assert x.stride(1) != 1
    w = torch.randn(1000, generator=g).to(torch.bfloat16).to(device)
    y = impl.fn(x, w, 1e-6)
    assert y.shape == (11, 1000)
    assert is_close(y, rmsnorm_reference(x, w, 1e-6))


@pytest.mark.parametrize("impl", impl_items(rmsnorm.IMPLEMENTATIONS))
def test_float32_weight_with_bf16_activations(impl, device, request):
    """A float32 weight with bf16 activations keeps the output in bf16."""
    if impl.family == "compile":
        request.getfixturevalue("compile_available")
    x, w, eps = make_inputs(6, 2048, torch.bfloat16, device, weight_dtype=torch.float32)
    y = impl.fn(x, w, eps)
    assert y.dtype == torch.bfloat16
    assert is_close(y, rmsnorm_reference(x, w, eps))


@pytest.mark.parametrize("impl", impl_items(rmsnorm.IMPLEMENTATIONS))
def test_deterministic(impl, device, request):
    """Two runs on the same inputs give bit-identical outputs."""
    if impl.family == "compile":
        request.getfixturevalue("compile_available")
    x, w, eps = make_inputs(16, 3000, torch.bfloat16, device)
    assert torch.equal(impl.fn(x, w, eps), impl.fn(x, w, eps))


@pytest.mark.parametrize("impl", impl_items(rmsnorm.IMPLEMENTATIONS))
def test_rejects_mismatched_weight(impl, device):
    x = torch.randn(2, 16, device=device)
    with pytest.raises(ValueError):
        impl.fn(x, torch.ones(8, device=device), 1e-6)


@pytest.mark.skipif(not rmsnorm.triton_impls.HAS_TRITON, reason="triton not installed")
@pytest.mark.parametrize("m,n", [(37, 512), (37, 1000), (8, 1024), (3, 2048), (37, 4096)])
def test_weight_tile_variant_is_bit_identical_to_multirow(m, n, device):
    """v3b changes only how the weight reaches the registers (a tile load instead of a 1-D
    load plus a shared-memory layout change, diary H14b); the arithmetic is the same, so the
    outputs match bit for bit, tail rows and masked columns included."""
    x, w, eps = make_inputs(m, n, torch.bfloat16, device, seed=6)
    v3 = rmsnorm.IMPLEMENTATIONS["triton_v3_multirow"].fn(x, w, eps)
    v3b = rmsnorm.IMPLEMENTATIONS["triton_v3b_wtile"].fn(x, w, eps)
    assert torch.equal(v3, v3b)


# --------------------------------------------------------------------------
# Residual-add + RMSNorm
# --------------------------------------------------------------------------
@pytest.mark.parametrize("impl", impl_items(rmsnorm.RESIDUAL_IMPLEMENTATIONS))
@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d).split(".")[-1])
@pytest.mark.parametrize("n", [1, 3, 1000, 4096, 5120])
@pytest.mark.parametrize("m", ROW_COUNTS)
def test_residual_matches_oracle(impl, dtype, n, m, device, request):
    if impl.family == "compile":
        request.getfixturevalue("compile_available")
    x, w, eps = make_inputs(m, n, dtype, device, seed=4)
    r = torch.randn(m, n, generator=torch.Generator().manual_seed(5)).to(dtype).to(device)
    y, s = impl.fn(x, r, w, eps)
    y_ref, s_ref = rmsnorm_residual_reference(x, r, w, eps)
    assert y.shape == x.shape and s.shape == x.shape
    assert y.dtype == dtype and s.dtype == dtype
    # The new residual is a single add rounded once; it has to match to the ulp.
    assert_rounded_sum_matches(s, s_ref, impl)
    assert is_close(y, y_ref), f"{impl.name} {(m, n)} {dtype}: {max_error(y, y_ref)}"


def truncate_to_bf16(t64: torch.Tensor) -> torch.Tensor:
    """float64 -> bf16 by dropping the low 16 bits of the float32 pattern (round toward zero)."""
    bits = t64.to(torch.float32).view(torch.int32) & ~0xFFFF
    return bits.view(torch.float32).to(torch.bfloat16)


def assert_rounded_sum_matches(s: torch.Tensor, s_ref64: torch.Tensor, impl) -> None:
    """``s`` must be the exact sum rounded once to its dtype.

    One exception: Triton's CPU interpreter converts float32 to bfloat16 by
    truncating the significand (``interpreter.py::_convert_float`` with no
    rounding mode) where the GPU uses round-to-nearest-even. Under the
    interpreter a Triton kernel's bf16 output is therefore the truncated sum,
    and the test pins exactly that rather than loosening to a tolerance.
    """
    expected = s_ref64.to(s.dtype)
    if torch.equal(s, expected):
        return
    if interpreting() and impl.family == "triton" and s.dtype == torch.bfloat16:
        assert torch.equal(s, truncate_to_bf16(s_ref64)), (
            f"{impl.name}: bf16 sum is neither the rounded nor the truncated exact sum"
        )
        return
    raise AssertionError(f"{impl.name}: residual sum differs from the rounded exact sum")


@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d).split(".")[-1])
def test_fused_residual_equals_unfused(dtype, device):
    """The fused kernel normalizes the rounded sum, so it agrees with add-then-norm to the tolerance."""
    if not rmsnorm.triton_impls.HAS_TRITON:
        pytest.skip("triton not installed")
    x, w, eps = make_inputs(37, 2500, dtype, device, seed=6)
    r = torch.randn(37, 2500, generator=torch.Generator().manual_seed(7)).to(dtype).to(device)
    y_fused, s_fused = rmsnorm.RESIDUAL_IMPLEMENTATIONS["triton_v4_fused_residual"].fn(x, r, w, eps)
    y_eager, s_eager = rmsnorm.RESIDUAL_IMPLEMENTATIONS["torch_eager_residual"].fn(x, r, w, eps)
    if interpreting() and dtype == torch.bfloat16:
        # Interpreter truncates the bf16 cast (see assert_rounded_sum_matches).
        assert torch.equal(s_fused, truncate_to_bf16(x.double() + r.double()))
    else:
        assert torch.equal(s_fused, s_eager)
    rtol, atol = tolerance(dtype)
    assert torch.allclose(y_fused.float(), y_eager.float(), rtol=rtol, atol=atol)
