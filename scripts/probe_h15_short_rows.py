"""Diary H15: where the instructions go on short rows, and what each fix removes (compile-time).

Reading the SASS of ``triton_v1_warps`` at N = 512 (``scripts/sass_mix.py --dump``) gives this
split of its 128 executed instructions per thread (8 bf16 elements of x and 8 of w):

    ~29  prologue: parameter loads, program id, 64-bit address math, mask setup, register
         zeroing for the masked loads, the two 128-bit loads
    ~24  bf16 -> f32 converts: PRMT + SHF.L for the high half of each pair, SHF.L for the low
         half: 1.5 instructions per value, 16 values
     8   square-accumulate (FMUL + 7 FFMA)
    10   intra-warp reduction (5 SHFL.BFLY + 5 FADD)
    ~20  cross-warp reduction: shared-memory store/load, two BAR.SYNC, a second shuffle level
    11   mean and rsqrt: I2F of N, an IEEE-exact reciprocal (MUFU.RCP with 2 FSETP + 4 FMUL
         range fixups), FFMA for eps, MUFU.RSQ
    16   x * rstd * w
     4   f32 -> bf16 packs (one per pair)
    ~6   store address (uniform datapath), exit guard, store

So about 55 instructions are per-element work (6.75 per element) and about 73 are per-thread
fixed cost. Each probe below removes one piece and is compiled on its own so the contribution
is measured, not inferred:

    inv_n      pass 1/N from the host: drops the I2F and the exact-reciprocal sequence
    1warp      one warp per row (16 elements per thread at N = 512): drops the cross-warp
               reduction and halves the fixed cost per element; occupancy is then capped at
               50% by the 32-blocks-per-SM limit
    nomask     N == BLOCK_N known at compile time: drops the mask compare, the predicates and
               the register zeroing
    rowloop    ROWS rows per program in one warp, weight loaded and converted once: amortizes
               the prologue and the weight work over ROWS rows

Usage, from the repository root with TRITON_INTERPRET unset::

    .venv/bin/python scripts/probe_h15_short_rows.py [--n 512] [--arch sm_80] [--rows 8]
"""
from __future__ import annotations

import argparse
import sys

import torch
import triton
import triton.language as tl

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from sass_mix import classify, sass_for  # noqa: E402

from rmsnorm.aot_stats import compile_plan, ptxas_resources  # noqa: E402
from rmsnorm.triton_impls import LaunchPlan, alignment, plan_row  # noqa: E402


@triton.jit
def _row_inv_n_kernel(x_ptr, w_ptr, y_ptr, stride_xm_div, stride_ym_div, N_div, inv_n, eps,
                      BLOCK_N: tl.constexpr, ALIGN: tl.constexpr):
    """v1 with 1/N passed from the host instead of dividing by N."""
    stride_xm = stride_xm_div * ALIGN
    stride_ym = stride_ym_div * ALIGN
    N = N_div * ALIGN
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(x_ptr + row * stride_xm + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.rsqrt(tl.sum(x * x, axis=0) * inv_n + eps)
    tl.store(y_ptr + row * stride_ym + cols, (x * rstd * w).to(y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _row_nomask_kernel(x_ptr, w_ptr, y_ptr, stride_xm_div, stride_ym_div, N_div, eps,
                       BLOCK_N: tl.constexpr, ALIGN: tl.constexpr):
    """v1 for N == BLOCK_N: no column mask anywhere."""
    stride_xm = stride_xm_div * ALIGN
    stride_ym = stride_ym_div * ALIGN
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_N)
    x = tl.load(x_ptr + row * stride_xm + cols).to(tl.float32)
    w = tl.load(w_ptr + cols).to(tl.float32)
    rstd = tl.rsqrt(tl.sum(x * x, axis=0) / BLOCK_N + eps)
    tl.store(y_ptr + row * stride_ym + cols, (x * rstd * w).to(y_ptr.dtype.element_ty))


@triton.jit
def _rowloop_kernel(x_ptr, w_ptr, y_ptr, stride_xm_div, stride_ym_div, M, N_div, inv_n, eps,
                    ROWS: tl.constexpr, BLOCK_N: tl.constexpr, ALIGN: tl.constexpr, MASKED: tl.constexpr):
    """ROWS rows per program, one after another; the weight is loaded and converted once.

    ``MASKED=False`` is the fully specialized path for N == BLOCK_N and M % ROWS == 0:
    no column or row mask anywhere.
    """
    stride_xm = stride_xm_div * ALIGN
    stride_ym = stride_ym_div * ALIGN
    N = N_div * ALIGN
    cols = tl.arange(0, BLOCK_N)
    row0 = tl.program_id(0).to(tl.int64) * ROWS
    if MASKED:
        cmask = cols < N
        w = tl.load(w_ptr + cols, mask=cmask, other=0.0).to(tl.float32)
        for i in tl.static_range(ROWS):
            row = row0 + i
            mask = cmask & (row < M)
            x = tl.load(x_ptr + row * stride_xm + cols, mask=mask, other=0.0).to(tl.float32)
            rstd = tl.rsqrt(tl.sum(x * x, axis=0) * inv_n + eps)
            tl.store(y_ptr + row * stride_ym + cols, (x * rstd * w).to(y_ptr.dtype.element_ty), mask=mask)
    else:
        w = tl.load(w_ptr + cols).to(tl.float32)
        for i in tl.static_range(ROWS):
            row = row0 + i
            x = tl.load(x_ptr + row * stride_xm + cols).to(tl.float32)
            rstd = tl.rsqrt(tl.sum(x * x, axis=0) * inv_n + eps)
            tl.store(y_ptr + row * stride_ym + cols, (x * rstd * w).to(y_ptr.dtype.element_ty))


@triton.jit
def _rowloop_unpack_kernel(x_ptr, w_ptr, y_ptr, stride_xm_div, stride_ym_div, M, N_div, inv_n, eps,
                           ROWS: tl.constexpr, BLOCK_N: tl.constexpr, ALIGN: tl.constexpr, JOIN: tl.constexpr):
    """The unmasked row loop with the bf16 pairs unpacked and packed by hand.

    ``.to(tl.float32)`` on a loaded bf16 vector compiles to PRMT + SHF.L for the
    high half of each 32-bit pair and SHF.L for the low half: three instructions
    per pair. Loading the pairs as int32 and unpacking with one shift (low) and
    one mask (high) is two. ``JOIN=True`` packs the result with ``tl.join`` and a
    2-D store; ``JOIN=False`` packs with one ``cvt.rn.bf16x2.f32`` per pair via
    inline PTX and stores int32 pairs, so nothing changes layout.
    """
    stride_xm = stride_xm_div * ALIGN
    stride_ym = stride_ym_div * ALIGN
    HALF: tl.constexpr = BLOCK_N // 2
    xi_ptr = x_ptr.to(tl.pointer_type(tl.int32))
    wi_ptr = w_ptr.to(tl.pointer_type(tl.int32))
    yi_ptr = y_ptr.to(tl.pointer_type(tl.int32))
    pairs = tl.arange(0, HALF)
    wv = tl.load(wi_ptr + pairs)
    w_lo = (wv << 16).to(tl.float32, bitcast=True)
    w_hi = (wv & -65536).to(tl.float32, bitcast=True)
    row0 = tl.program_id(0).to(tl.int64) * ROWS
    for i in tl.static_range(ROWS):
        row = row0 + i
        xv = tl.load(xi_ptr + row * (stride_xm // 2) + pairs)
        x_lo = (xv << 16).to(tl.float32, bitcast=True)
        x_hi = (xv & -65536).to(tl.float32, bitcast=True)
        ss = tl.sum(x_lo * x_lo + x_hi * x_hi, axis=0)
        rstd = tl.rsqrt(ss * inv_n + eps)
        y_lo = x_lo * rstd * w_lo
        y_hi = x_hi * rstd * w_hi
        if JOIN:
            y = tl.join(y_lo, y_hi).to(tl.bfloat16)   # [HALF, 2]
            tl.store(y_ptr + row * stride_ym + (pairs[:, None] * 2 + tl.arange(0, 2)[None, :]), y)
        else:
            yv = tl.inline_asm_elementwise("cvt.rn.bf16x2.f32 $0, $1, $2;", "=r,r,r", [y_hi, y_lo],
                                           dtype=tl.int32, is_pure=True, pack=1)
            tl.store(yi_ptr + row * (stride_ym // 2) + pairs, yv)


def describe(label: str, plan: LaunchPlan, n: int, arch: str, rows_per_thread_elems: float) -> None:
    compiled = compile_plan(plan, torch.bfloat16, 4096, n, arch)
    res = ptxas_resources(compiled.asm["ptx"], arch)
    shared = int(getattr(compiled.metadata, "shared", 0) or 0)
    by_class, _ = classify(sass_for(plan, n, arch))
    total = sum(by_class.values())
    print(f"| {label} | {plan.num_warps} | {rows_per_thread_elems:g} | {total} | {total / rows_per_thread_elems:.2f} | "
          f"{res['registers']} | {shared} | {by_class.get('barrier', 0)} | {by_class.get('shared', 0)} | "
          f"{by_class.get('global', 0)} | {by_class.get('shuffle', 0)} | {by_class.get('convert', 0)} | "
          f"{by_class.get('fp math', 0)} | {by_class.get('int/addr/pred', 0)} | {by_class.get('control/uniform', 0)} |")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--arch", default="sm_80")
    ap.add_argument("--rows", type=int, default=8)
    args = ap.parse_args()
    n, m = args.n, 4096
    block_n = triton.next_power_of_2(n)
    align = alignment(n)
    base = plan_row(m, n)  # v1: the 8-elements-per-thread heuristic
    one_warp = plan_row(m, n, num_warps=1)
    elems2 = block_n / (32 * base.num_warps)
    elems1 = block_n / 32
    cx = {"BLOCK_N": block_n, "ALIGN": align}
    print(f"N={n}, M={m}, {args.arch}, bf16. Per thread, static SASS count up to the last EXIT.")
    print("| kernel | warps | elems/thread | insns | insns/elem | regs | smem B | BAR | LDS+STS | LDG+STG | SHFL | convert | fp | int/addr/pred | uniform/ctrl |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    describe("v1 (baseline)", base, n, args.arch, elems2)
    describe("inv_n", LaunchPlan("probe", "", cx, base.num_warps, m, kernel_fn=_row_inv_n_kernel), n, args.arch, elems2)
    describe("1warp", one_warp, n, args.arch, elems1)
    for w in (2, 4):
        # v3's kernel with as many warps as rows: does each warp get its own row (no cross-warp reduction)?
        describe(f"v3 kernel, BLOCK_M={w}, {w} warps",
                 LaunchPlan("probe", "_rmsnorm_multirow_kernel", {"BLOCK_M": w, **cx}, w, m // w), n, args.arch, elems1)
    if block_n == n:
        describe("nomask", LaunchPlan("probe", "", cx, base.num_warps, m, kernel_fn=_row_nomask_kernel), n, args.arch, elems2)
    for warps, label in ((1, "rowloop 1warp"), (base.num_warps, f"rowloop {base.num_warps}warps")):
        elems = args.rows * block_n / (32 * warps)
        for masked in (True, False):
            if not masked and block_n != n:
                continue
            plan = LaunchPlan("probe", "", {**cx, "ROWS": args.rows, "MASKED": masked}, warps,
                              triton.cdiv(m, args.rows), kernel_fn=_rowloop_kernel)
            describe(f"{label} x{args.rows} {'masked' if masked else 'nomask'}", plan, n, args.arch, elems)
    if block_n == n:
        for join in (True, False):
            plan = LaunchPlan("probe", "", {**cx, "ROWS": args.rows, "JOIN": join}, 1, triton.cdiv(m, args.rows),
                              kernel_fn=_rowloop_unpack_kernel)
            describe(f"rowloop 1warp x{args.rows} nomask, hand unpack, {'join' if join else 'cvt.bf16x2'}",
                     plan, n, args.arch, args.rows * block_n / 32)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
