"""Triton RMSNorm kernels, one per optimization step.

Each variant is its own ``@triton.jit`` kernel and Python wrapper so that the
benchmark can run them side by side and the diary (``docs/diary.md``) can
attribute every change in performance to exactly one change in code. The
variants, in the order they were written:

* ``v0_naive``: one program per row, the whole row in one block, default
  launch configuration (4 warps) regardless of the row length.
* ``v1_warps``: v0 with ``num_warps`` chosen from the row length so that each
  thread handles about 8 elements. Same code, different thread mapping.
* ``v2_looped``: the row is processed in fixed-size chunks in two passes
  (sum of squares, then normalize). Trades a second read of the row (served
  from L2) for a bounded register footprint on long rows.
* ``v3_multirow``: several short rows per program as a 2-D block, so the
  weight is loaded once per program and every thread has more independent
  loads in flight.
* ``v4_fused_residual``: residual add fused into the single-row kernel; also
  writes the updated residual stream.
* ``rmsnorm_triton``: the dispatcher that picks a variant by row length.

All kernels accumulate in float32 whatever the storage dtype, so fp16 inputs
cannot overflow in the square and tiny values cannot underflow (see
``tests/test_correctness.py::test_fp16_extremes``). Rows need unit stride
along the normalized dimension; other layouts are made contiguous first.

``triton`` is imported lazily so the PyTorch implementations in this package
work in an environment without it. Setting ``TRITON_INTERPRET=1`` before the
first import runs every kernel on the CPU through Triton's interpreter, which
is how the correctness tests run on a machine without a GPU.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .reference import _check_shapes

try:  # pragma: no cover - exercised implicitly by every Triton test
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None
    tl = None

HAS_TRITON = triton is not None

# Largest row a single-block kernel will take. Beyond this the block no longer
# fits in registers in any useful way and ``v2_looped`` is the right tool.
MAX_SINGLE_BLOCK_N = 65536


def _require_triton() -> None:
    if not HAS_TRITON:
        raise RuntimeError("triton is not installed; pip install triton")


# The most recent launch: (compiled kernel object or None, launch configuration).
# ``kernel_stats`` reads this to report registers, spills, shared memory and
# grid size for whatever the wrapper actually launched.
_LAST_LAUNCH: tuple[object, dict] | None = None


def _record(kernel, **config) -> None:
    global _LAST_LAUNCH
    _LAST_LAUNCH = (kernel, config)


def last_launch() -> tuple[object, dict] | None:
    return _LAST_LAUNCH


def _as_rows(x: torch.Tensor) -> torch.Tensor:
    """View ``x`` as (M, N) with unit stride along N, copying only if needed."""
    n = x.shape[-1]
    x2 = x.reshape(-1, n) if x.dim() != 2 else x
    if x2.stride(1) != 1:
        x2 = x2.contiguous()
    return x2


def _warps_for(block_n: int, target_per_thread: int = 8, max_warps: int = 16) -> int:
    """Pick ``num_warps`` so each thread owns about ``target_per_thread`` elements.

    Eight bf16 elements is one 128-bit load per thread, the widest a single
    thread can issue, so this keeps every load wide while leaving the SM room
    to hold several programs at once.
    """
    warps = block_n // (32 * target_per_thread)
    return max(1, min(max_warps, warps))


# --------------------------------------------------------------------------
# v0: one row per program, whole row in one block, fixed 4 warps.
# --------------------------------------------------------------------------
if HAS_TRITON:

    @triton.jit
    def _rmsnorm_row_kernel(
        x_ptr, w_ptr, y_ptr,
        stride_xm_div, stride_ym_div,
        N_div, eps,
        BLOCK_N: tl.constexpr, ALIGN: tl.constexpr,
    ):
        """One program normalizes one row held entirely in a BLOCK_N-wide block.

        ``BLOCK_N`` is the next power of two above N, so for N = 5120 a third
        of the lanes are masked off: they still take a slot in every
        instruction but load and store nothing. The float32 upcast happens in
        registers; nothing wider than the input dtype touches memory. The row
        offset is a 64-bit scalar so tensors past 2^31 elements index
        correctly; the column vector stays 32-bit.

        Strides and N arrive divided by ``ALIGN`` (the largest power of two up
        to 16 that divides all of them) and are multiplied back here. The
        product carries the divisibility into the compiler's alignment
        analysis, which is what lets it emit 128-bit loads and stores for a
        row length such as 1000 that the launcher's own divisible-by-16 hint
        does not cover (diary H12).
        """
        stride_xm = stride_xm_div * ALIGN
        stride_ym = stride_ym_div * ALIGN
        N = N_div * ALIGN
        row = tl.program_id(0).to(tl.int64)
        cols = tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(x_ptr + row * stride_xm + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        mean_sq = tl.sum(x * x, axis=0) / N
        rstd = tl.rsqrt(mean_sq + eps)
        y = x * rstd * w
        tl.store(y_ptr + row * stride_ym + cols, y.to(y_ptr.dtype.element_ty), mask=mask)

    @triton.jit
    def _rmsnorm_looped_kernel(
        x_ptr, w_ptr, y_ptr,
        stride_xm_div, stride_ym_div,
        N_div, eps,
        BLOCK_N: tl.constexpr, ALIGN: tl.constexpr,
    ):
        """Two passes over the row in BLOCK_N chunks.

        Pass 1 accumulates x*x into a BLOCK_N-wide float32 vector and reduces
        it once at the end (one cross-warp reduction per row, not one per
        chunk). Pass 2 re-reads the row, which by then sits in L2, and writes
        the output. Register use is bounded by BLOCK_N, not by N.
        """
        stride_xm = stride_xm_div * ALIGN
        stride_ym = stride_ym_div * ALIGN
        N = N_div * ALIGN
        row = tl.program_id(0).to(tl.int64)
        x_row = x_ptr + row * stride_xm
        y_row = y_ptr + row * stride_ym
        acc = tl.zeros([BLOCK_N], dtype=tl.float32)
        for start in range(0, N, BLOCK_N):
            cols = start + tl.arange(0, BLOCK_N)
            mask = cols < N
            x = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
            acc += x * x
        mean_sq = tl.sum(acc, axis=0) / N
        rstd = tl.rsqrt(mean_sq + eps)
        for start in range(0, N, BLOCK_N):
            cols = start + tl.arange(0, BLOCK_N)
            mask = cols < N
            x = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
            w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            y = x * rstd * w
            tl.store(y_row + cols, y.to(y_ptr.dtype.element_ty), mask=mask)

    @triton.jit
    def _rmsnorm_multirow_kernel(
        x_ptr, w_ptr, y_ptr,
        stride_xm_div, stride_ym_div,
        M, N_div, eps,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, ALIGN: tl.constexpr,
    ):
        """BLOCK_M rows per program as a [BLOCK_M, BLOCK_N] block.

        The weight is loaded once and broadcast across the rows; the
        reduction runs along axis 1 so each row still gets its own rstd.
        Rows past M are masked off in the last program.
        """
        stride_xm = stride_xm_div * ALIGN
        stride_ym = stride_ym_div * ALIGN
        N = N_div * ALIGN
        pid = tl.program_id(0).to(tl.int64)
        rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = tl.arange(0, BLOCK_N)
        row_mask = rows < M
        col_mask = cols < N
        mask = row_mask[:, None] & col_mask[None, :]
        x_ptrs = x_ptr + rows[:, None] * stride_xm + cols[None, :]
        x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
        mean_sq = tl.sum(x * x, axis=1) / N
        rstd = tl.rsqrt(mean_sq + eps)
        y = x * rstd[:, None] * w[None, :]
        y_ptrs = y_ptr + rows[:, None] * stride_ym + cols[None, :]
        tl.store(y_ptrs, y.to(y_ptr.dtype.element_ty), mask=mask)

    @triton.jit
    def _rmsnorm_fused_residual_kernel(
        x_ptr, r_ptr, w_ptr, y_ptr, s_ptr,
        stride_xm_div, stride_rm_div, stride_ym_div, stride_sm_div,
        N_div, eps,
        BLOCK_N: tl.constexpr, ALIGN: tl.constexpr,
    ):
        """y = rmsnorm(x + r); s = x + r, both from one read of x and r.

        The sum is rounded to the storage dtype before it is normalized, so
        the result matches the unfused add-then-norm sequence rather than a
        more precise value the unfused path never sees. The rounding is a
        register-only cast; it adds no memory traffic.
        """
        stride_xm = stride_xm_div * ALIGN
        stride_rm = stride_rm_div * ALIGN
        stride_ym = stride_ym_div * ALIGN
        stride_sm = stride_sm_div * ALIGN
        N = N_div * ALIGN
        row = tl.program_id(0).to(tl.int64)
        cols = tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(x_ptr + row * stride_xm + cols, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(r_ptr + row * stride_rm + cols, mask=mask, other=0.0).to(tl.float32)
        s = (x + r).to(s_ptr.dtype.element_ty)
        tl.store(s_ptr + row * stride_sm + cols, s, mask=mask)
        h = s.to(tl.float32)
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        mean_sq = tl.sum(h * h, axis=0) / N
        rstd = tl.rsqrt(mean_sq + eps)
        y = h * rstd * w
        tl.store(y_ptr + row * stride_ym + cols, y.to(y_ptr.dtype.element_ty), mask=mask)


@dataclass(frozen=True)
class LaunchPlan:
    """What a wrapper launches for a given (M, N): kernel, block constants, warps, grid.

    The runtime wrappers launch plans; ``aot_stats`` compiles the same plans
    offline (no GPU needed) to read registers and spills, so the two can never
    disagree about what "variant X at shape S" means.
    """
    variant: str
    kernel_name: str
    constexprs: dict
    num_warps: int
    grid: int

    @property
    def kernel(self):
        return globals()[self.kernel_name]

    @property
    def block_n(self) -> int:
        return self.constexprs["BLOCK_N"]

    @property
    def block_m(self) -> int:
        return self.constexprs.get("BLOCK_M", 1)


# Chunk width for the looped kernel. 4096 float32 accumulators over 8 warps is
# 16 per thread, which leaves room for the loaded chunk and the weight chunk.
LOOPED_BLOCK_N = 4096


def _rows_per_program(block_n: int, target_elems: int = 4096) -> int:
    """Rows per program so a program covers about ``target_elems`` elements."""
    return max(1, min(64, target_elems // block_n))


def alignment(*values: int, cap: int = 16) -> int:
    """Largest power of two up to ``cap`` that divides every value (1 if any is odd).

    Used as the ``ALIGN`` constexpr: strides and N are passed divided by it
    and multiplied back inside the kernel, so the compiler can see their
    divisibility. 8 bf16 elements are one 128-bit access, so 8 is enough for
    full-width loads; 16 is the launcher's own hint granularity.
    """
    a = cap
    while a > 1 and any(v % a for v in values):
        a //= 2
    return a


def plan_row(m: int, n: int, num_warps: int | None = None, variant: str = "triton_v1_warps",
             row_strides: tuple[int, ...] = ()) -> LaunchPlan:
    """Single-row kernel. ``num_warps=None`` applies the 8-elements-per-thread heuristic (v1)."""
    if n > MAX_SINGLE_BLOCK_N:
        raise ValueError(f"N={n} exceeds the single-block limit {MAX_SINGLE_BLOCK_N}; use the looped kernel")
    block_n = triton.next_power_of_2(n)
    warps = _warps_for(block_n) if num_warps is None else num_warps
    align = alignment(n, *(row_strides or (n,)))
    return LaunchPlan(variant, "_rmsnorm_row_kernel", {"BLOCK_N": block_n, "ALIGN": align}, warps, m)


def plan_looped(m: int, n: int, block_n: int = LOOPED_BLOCK_N, row_strides: tuple[int, ...] = ()) -> LaunchPlan:
    if block_n & (block_n - 1):
        raise ValueError(f"block_n must be a power of two, got {block_n}")
    block_n = min(block_n, triton.next_power_of_2(n))
    align = alignment(n, *(row_strides or (n,)))
    return LaunchPlan("triton_v2_looped", "_rmsnorm_looped_kernel", {"BLOCK_N": block_n, "ALIGN": align},
                      _warps_for(block_n), m)


def plan_multirow(m: int, n: int, block_m: int | None = None, row_strides: tuple[int, ...] = ()) -> LaunchPlan:
    block_n = triton.next_power_of_2(n)
    if block_m is None:
        block_m = _rows_per_program(block_n)
    if block_m & (block_m - 1):
        raise ValueError(f"block_m must be a power of two, got {block_m}")
    align = alignment(n, *(row_strides or (n,)))
    return LaunchPlan("triton_v3_multirow", "_rmsnorm_multirow_kernel",
                      {"BLOCK_M": block_m, "BLOCK_N": block_n, "ALIGN": align},
                      _warps_for(block_m * block_n), triton.cdiv(m, block_m))


def plan_fused_residual(m: int, n: int, row_strides: tuple[int, ...] = ()) -> LaunchPlan:
    if n > MAX_SINGLE_BLOCK_N:
        raise ValueError(f"N={n} exceeds the single-block limit {MAX_SINGLE_BLOCK_N}")
    block_n = triton.next_power_of_2(n)
    align = alignment(n, *(row_strides or (n,)))
    return LaunchPlan("triton_v4_fused_residual", "_rmsnorm_fused_residual_kernel",
                      {"BLOCK_N": block_n, "ALIGN": align}, _warps_for(block_n), m)


def plan_for(variant: str, m: int, n: int, row_strides: tuple[int, ...] = ()) -> LaunchPlan:
    """The plan each registered variant launches at (M, N), including the dispatcher's choice."""
    if variant == "triton_v0_naive":
        return plan_row(m, n, num_warps=4, variant=variant, row_strides=row_strides)
    if variant == "triton_v1_warps":
        return plan_row(m, n, row_strides=row_strides)
    if variant == "triton_v2_looped":
        return plan_looped(m, n, row_strides=row_strides)
    if variant == "triton_v3_multirow":
        return plan_multirow(m, n, row_strides=row_strides)
    if variant == "triton_v4_fused_residual":
        return plan_fused_residual(m, n, row_strides=row_strides)
    if variant == "triton":
        if n <= MULTIROW_MAX_N:
            return plan_multirow(m, n, row_strides=row_strides)
        if n >= LOOPED_MIN_N:
            return plan_looped(m, n, row_strides=row_strides)
        return plan_row(m, n, row_strides=row_strides)
    raise KeyError(variant)


def _launch(plan: LaunchPlan, *args):
    kernel = plan.kernel[(plan.grid,)](*args, **plan.constexprs, num_warps=plan.num_warps)
    _record(kernel, kernel_name=plan.kernel_name, grid=plan.grid, num_warps=plan.num_warps, **plan.constexprs)
    return kernel


def _run_plain(plan_fn, x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    x2 = _as_rows(x)
    m, n = x2.shape
    y = torch.empty_like(x2)
    if m == 0:
        return y.view(x.shape)
    plan = plan_fn(m, n, (x2.stride(0), y.stride(0)))
    a = plan.constexprs["ALIGN"]
    if plan.kernel_name == "_rmsnorm_multirow_kernel":
        args = (x2, weight.contiguous(), y, x2.stride(0) // a, y.stride(0) // a, m, n // a, eps)
    else:
        args = (x2, weight.contiguous(), y, x2.stride(0) // a, y.stride(0) // a, n // a, eps)
    _launch(plan, *args)
    return y.view(x.shape)


def rmsnorm_triton_v0_naive(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """One row per program, whole row in one block, 4 warps for every N."""
    _require_triton()
    _check_shapes(x, weight)
    return _run_plain(lambda m, n, s: plan_row(m, n, num_warps=4, variant="triton_v0_naive", row_strides=s),
                      x, weight, eps)


def rmsnorm_triton_v1_warps(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """v0 with ``num_warps`` scaled to the row: about 8 elements per thread."""
    _require_triton()
    _check_shapes(x, weight)
    return _run_plain(lambda m, n, s: plan_row(m, n, row_strides=s), x, weight, eps)


def rmsnorm_triton_v2_looped(
    x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6, block_n: int = LOOPED_BLOCK_N
) -> torch.Tensor:
    """Two passes over each row in ``block_n`` chunks; register use independent of N."""
    _require_triton()
    _check_shapes(x, weight)
    return _run_plain(lambda m, n, s: plan_looped(m, n, block_n, row_strides=s), x, weight, eps)


def rmsnorm_triton_v3_multirow(
    x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6, block_m: int | None = None
) -> torch.Tensor:
    """Several rows per program; meant for short rows (N <= 1024 or so)."""
    _require_triton()
    _check_shapes(x, weight)
    return _run_plain(lambda m, n, s: plan_multirow(m, n, block_m, row_strides=s), x, weight, eps)


def rmsnorm_triton_v4_fused_residual(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns ``(rmsnorm(x + residual), x + residual)`` from one kernel."""
    _require_triton()
    _check_shapes(x, weight)
    if residual.shape != x.shape:
        raise ValueError(f"residual shape {tuple(residual.shape)} != x shape {tuple(x.shape)}")
    if residual.dtype != x.dtype:
        raise ValueError("x and residual must share a dtype")
    x2 = _as_rows(x)
    r2 = _as_rows(residual)
    m, n = x2.shape
    y = torch.empty_like(x2)
    s = torch.empty_like(x2)
    if m == 0:
        return y.view(x.shape), s.view(x.shape)
    strides = (x2.stride(0), r2.stride(0), y.stride(0), s.stride(0))
    plan = plan_fused_residual(m, n, row_strides=strides)
    a = plan.constexprs["ALIGN"]
    _launch(plan, x2, r2, weight.contiguous(), y, s, *(st // a for st in strides), n // a, eps)
    return y.view(x.shape), s.view(x.shape)


# Dispatch thresholds. These are starting guesses from the performance model
# (docs/performance-model.md); the diary records the measured crossovers and
# the values here follow the measurements, not the other way round.
MULTIROW_MAX_N = 1024
LOOPED_MIN_N = 8192


def rmsnorm_triton(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Pick a kernel by row length: multirow for short rows, looped for long ones."""
    _require_triton()
    _check_shapes(x, weight)
    return _run_plain(lambda m, n, s: plan_for("triton", m, n, row_strides=s), x, weight, eps)
