"""Diary H16: an explicit layout that gives each warp its own row, several warps per program.

Standard Triton chooses the thread mapping of a [W, BLOCK_N] block itself and
spreads every row across all the program's warps, so the row reduction
crosses warps (shared memory, two barriers). The one-warp row loop (v5,
diary H15) avoids that only by running one warp per program, which the
32-blocks-per-SM limit caps at 50% occupancy. Gluon exposes the layout: with
``BlockedLayout(size_per_thread=[1, BLOCK_N // 32], threads_per_warp=[1, 32],
warps_per_cta=[W, 1])`` warp k holds row k entirely, the reduction along the
row is intra-warp, and the program still has W warps. This probe compiles that
kernel and prints the same SASS summary as the H15 probe. Compile-time only:
Triton's CPU interpreter does not run Gluon kernels, so correctness waits for
a GPU along with the timing.

Usage, from the repository root with TRITON_INTERPRET unset::

    .venv/bin/python scripts/probe_h16_gluon_rows.py [--n 512] [--rows 2] [--arch sm_80]
"""
from __future__ import annotations

import argparse
import sys

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.language import static_range  # gluon has no static_range of its own; the code generator accepts this one

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from sass_mix import classify, sass_for  # noqa: E402

from rmsnorm.aot_stats import ARCHS, compile_plan, ptxas_resources  # noqa: E402
from rmsnorm.kernel_stats import occupancy  # noqa: E402
from rmsnorm.triton_impls import LaunchPlan, alignment, plan_for  # noqa: E402


@gluon.jit
def _rows_per_warp_kernel(x_ptr, w_ptr, y_ptr, stride_xm_div, stride_ym_div, M, N_div, inv_n, eps,
                          ROWS: gl.constexpr, BLOCK_N: gl.constexpr, ALIGN: gl.constexpr, num_warps: gl.constexpr):
    """Warp k of the program owns row k of each group of num_warps rows; ROWS groups per program.

    No mask: N == BLOCK_N and M % (num_warps * ROWS) == 0 are the caller's
    responsibility, as in v5's unmasked path.
    """
    layout: gl.constexpr = gl.BlockedLayout(size_per_thread=[1, BLOCK_N // 32], threads_per_warp=[1, 32],
                                            warps_per_cta=[num_warps, 1], order=[1, 0])
    row_layout: gl.constexpr = gl.SliceLayout(1, layout)
    col_layout: gl.constexpr = gl.SliceLayout(0, layout)
    stride_xm = stride_xm_div * ALIGN
    stride_ym = stride_ym_div * ALIGN
    cols = gl.arange(0, BLOCK_N, layout=col_layout)
    w = gl.load(w_ptr + cols).to(gl.float32)
    w2 = gl.expand_dims(w, 0)
    base = gl.program_id(0).to(gl.int64) * (num_warps * ROWS)
    lane_rows = gl.arange(0, num_warps, layout=row_layout)
    for i in static_range(ROWS):
        rows = base + i * num_warps + lane_rows
        rows2 = gl.expand_dims(rows, 1)
        cols2 = gl.expand_dims(cols, 0)
        x = gl.load(x_ptr + rows2 * stride_xm + cols2).to(gl.float32)
        rstd = gl.rsqrt(gl.sum(x * x, axis=1) * inv_n + eps)
        y = x * gl.expand_dims(rstd, 1) * w2
        gl.store(y_ptr + rows2 * stride_ym + cols2, y.to(gl.bfloat16))


def describe(label: str, plan: LaunchPlan, n: int, arch: str, elems: float) -> None:
    compiled = compile_plan(plan, torch.bfloat16, 4096, n, arch)
    res = ptxas_resources(compiled.asm["ptx"], arch)
    shared = int(getattr(compiled.metadata, "shared", 0) or 0)
    programs, occ, limiter = occupancy(res["registers"], plan.num_warps, shared, ARCHS[arch])
    by_class, _ = classify(sass_for(plan, n, arch))
    total = sum(by_class.values())
    print(f"| {label} | {plan.num_warps} | {elems:g} | {total} | {total / elems:.2f} | {res['registers']} | {shared} | "
          f"{by_class.get('barrier', 0)} | {by_class.get('shared', 0)} | {occ:.0%} ({limiter}) | {plan.grid} |")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--rows", type=int, default=2, help="row groups per program (rows per warp)")
    ap.add_argument("--arch", default="sm_80")
    args = ap.parse_args()
    n, m = args.n, 4096
    block_n = triton.next_power_of_2(n)
    assert block_n == n, "this probe is the unmasked path: N must be a power of two"
    elems_per_thread = args.rows * block_n / 32
    print(f"N={n}, M={m}, {args.arch}, bf16, {args.rows} rows per warp. Per thread, executed SASS.")
    print("| kernel | warps | elems/thread | insns | insns/elem | regs | smem B | BAR | LDS+STS | theoretical occupancy | grid |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    v5 = plan_for("triton_v5_rowloop", m, n)
    describe(f"v5 rowloop (standard Triton, 1 warp, ROWS={v5.block_m})", v5, n, args.arch, v5.block_m * block_n / 32)
    for warps in (1, 2, 4, 8):
        plan = LaunchPlan("probe", "", {"ROWS": args.rows, "BLOCK_N": block_n, "ALIGN": alignment(n), "num_warps": warps},
                          warps, m // (warps * args.rows), kernel_fn=_rows_per_warp_kernel)
        describe(f"gluon rows-per-warp, {warps} warps x {args.rows} rows", plan, n, args.arch, elems_per_thread)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
