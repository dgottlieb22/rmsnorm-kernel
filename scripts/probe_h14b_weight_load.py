"""Diary H14b: how the multirow kernel should load the weight vector (compile-time, no GPU).

The shipped ``triton_v3_multirow`` loads ``w`` as a 1-D vector of BLOCK_N
elements. With 16 warps that is one element per thread at N = 512, and the
compiler then moves the vector through shared memory into the layout of the
[BLOCK_M, BLOCK_N] x tile before the multiply (diary H14). This script compiles
two alternatives for the same launch plans and prints what changed:

(a) ``w`` as a [1, BLOCK_N] tensor, broadcast against the tile;
(b) ``w`` as a full [BLOCK_M, BLOCK_N] tile, every row of the block reading the
    same BLOCK_N weights.

Usage, from the repository root with TRITON_INTERPRET unset::

    .venv/bin/python scripts/probe_h14b_weight_load.py [--arch sm_80] [--n 512 --n 1024 ...]
"""
from __future__ import annotations

import argparse

import torch
import triton
import triton.language as tl

from rmsnorm.aot_stats import access_widths, compile_plan, ptxas_resources
from rmsnorm.triton_impls import LaunchPlan, plan_for


@triton.jit
def _w_as_1xn_kernel(x_ptr, w_ptr, y_ptr, stride_xm_div, stride_ym_div, M, N_div, eps,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, ALIGN: tl.constexpr):
    """(a): the weight is loaded as a [1, BLOCK_N] block and broadcast."""
    stride_xm = stride_xm_div * ALIGN
    stride_ym = stride_ym_div * ALIGN
    N = N_div * ALIGN
    pid = tl.program_id(0).to(tl.int64)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    row_mask = rows < M
    col_mask = cols < N
    mask = row_mask[:, None] & col_mask[None, :]
    x = tl.load(x_ptr + rows[:, None] * stride_xm + cols[None, :], mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols[None, :], mask=col_mask[None, :], other=0.0).to(tl.float32)
    mean_sq = tl.sum(x * x, axis=1) / N
    rstd = tl.rsqrt(mean_sq + eps)
    y = x * rstd[:, None] * w
    tl.store(y_ptr + rows[:, None] * stride_ym + cols[None, :], y.to(y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _w_as_tile_kernel(x_ptr, w_ptr, y_ptr, stride_xm_div, stride_ym_div, M, N_div, eps,
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, ALIGN: tl.constexpr):
    """(b): the weight is loaded as a [BLOCK_M, BLOCK_N] tile; every row reads the same addresses."""
    stride_xm = stride_xm_div * ALIGN
    stride_ym = stride_ym_div * ALIGN
    N = N_div * ALIGN
    pid = tl.program_id(0).to(tl.int64)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    row_mask = rows < M
    col_mask = cols < N
    mask = row_mask[:, None] & col_mask[None, :]
    x = tl.load(x_ptr + rows[:, None] * stride_xm + cols[None, :], mask=mask, other=0.0).to(tl.float32)
    zero_rows = tl.zeros([BLOCK_M], dtype=tl.int32)
    w = tl.load(w_ptr + zero_rows[:, None] + cols[None, :], mask=mask, other=0.0).to(tl.float32)
    mean_sq = tl.sum(x * x, axis=1) / N
    rstd = tl.rsqrt(mean_sq + eps)
    y = x * rstd[:, None] * w
    tl.store(y_ptr + rows[:, None] * stride_ym + cols[None, :], y.to(y_ptr.dtype.element_ty), mask=mask)


def describe(label: str, plan: LaunchPlan, n: int, arch: str, show_ir: bool) -> None:
    compiled = compile_plan(plan, torch.bfloat16, 4096, n, arch)
    ptx, ttgir = compiled.asm["ptx"], compiled.asm["ttgir"]
    res = ptxas_resources(ptx, arch)
    gl, gs = access_widths(ptx)
    sl, ss = access_widths(ptx, "shared")
    shared = int(getattr(compiled.metadata, "shared", 0) or 0)
    converts = [line.strip() for line in ttgir.splitlines() if "ttg.convert_layout" in line]
    print(f"  {label:<10} regs={res['registers']:>3} spill={res['spill_stores']:>3}B smem={shared:>5}B "
          f"global ld[{gl}] st[{gs}]  shared ld[{sl}] st[{ss}]  "
          f"convert_layout={len(converts)} bar.sync={ptx.count('bar.sync')}")
    if show_ir:
        for line in ttgir.splitlines():
            if line.startswith("#blocked"):
                print("            ", line.strip())
        for line in converts:
            print("            ", line[:200])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", default="sm_80")
    ap.add_argument("--n", action="append", type=int, default=None)
    ap.add_argument("--ir", action="store_true", help="also print the TTGIR layouts and convert_layout ops")
    args = ap.parse_args()
    for n in args.n or (512, 1024, 2048, 4096):
        base = plan_for("triton_v3_multirow", 4096, n)
        print(f"N={n} block={base.block_m}x{base.block_n} warps={base.num_warps} ({args.arch}, bf16)")
        describe("shipped", base, n, args.arch, args.ir)
        describe("(a) 1xN", LaunchPlan("probe", "", base.constexprs, base.num_warps, base.grid,
                                       kernel_fn=_w_as_1xn_kernel), n, args.arch, args.ir)
        describe("(b) tile", LaunchPlan("probe", "", base.constexprs, base.num_warps, base.grid,
                                        kernel_fn=_w_as_tile_kernel), n, args.arch, args.ir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
