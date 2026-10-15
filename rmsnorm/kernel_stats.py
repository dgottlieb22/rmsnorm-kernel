"""Registers, spills, shared memory, grid size and theoretical occupancy per variant and shape.

Triton exposes the compiled kernel's resource usage (``n_regs``, ``n_spills``,
``metadata.shared``) after a launch. Together with the launch configuration
that the wrapper recorded, that is enough to compute the occupancy a kernel
can reach on the current device and how many waves its grid needs, which is
the first thing to look at when a shape runs below the roofline:

* registers per thread x threads per program, against the SM's register file,
  bounds programs per SM; so does the per-SM block limit;
* programs per SM x warps per program, against the SM's warp limit, is the
  achievable occupancy;
* grid / (programs per SM x SMs) is the number of waves; a grid smaller than
  one wave leaves SMs idle no matter how good the kernel is.

Spills (``n_spills`` > 0) mean the compiler ran out of registers and is
staging values through local memory, which goes through L1/L2: the usual
reason a whole-row kernel collapses on long rows.
"""
from __future__ import annotations

import csv
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch

from . import IMPLEMENTATIONS, RESIDUAL_IMPLEMENTATIONS, triton_impls
from .bench import Shape, make_inputs


@dataclass
class KernelStats:
    impl: str
    m: int
    n: int
    dtype: str
    kernel: str
    grid: int
    num_warps: int
    threads: int
    block_n: int
    block_m: int
    n_regs: int
    n_spills: int
    shared_bytes: int
    programs_per_sm: int
    occupancy: float  # achieved warps per SM / max warps per SM
    waves: float
    limiter: str


def device_limits(device) -> dict:
    p = torch.cuda.get_device_properties(device)
    return {
        "sm_count": p.multi_processor_count,
        "regs_per_sm": getattr(p, "regs_per_multiprocessor", 65536),
        "max_threads_per_sm": getattr(p, "max_threads_per_multi_processor", 2048),
        "max_blocks_per_sm": 32 if p.major >= 8 else 16,
        "shared_per_sm": getattr(p, "shared_memory_per_multiprocessor", 100 * 1024),
    }


def occupancy(n_regs: int, num_warps: int, shared_bytes: int, limits: dict) -> tuple[int, float, str]:
    """Programs per SM, occupancy (fraction of the warp limit) and the binding limit."""
    threads = 32 * num_warps
    max_warps = limits["max_threads_per_sm"] // 32
    # Registers are allocated per warp in units of 256; round the per-thread
    # count up to a multiple of 8 to mirror that.
    regs_per_thread = max(8, (n_regs + 7) // 8 * 8)
    by_regs = limits["regs_per_sm"] // (regs_per_thread * threads) if regs_per_thread * threads else 0
    by_warps = max_warps // num_warps
    by_blocks = limits["max_blocks_per_sm"]
    by_shared = limits["shared_per_sm"] // shared_bytes if shared_bytes else by_blocks + 1  # no shared memory: never binds
    candidates = {"registers": by_regs, "warps": by_warps, "blocks": by_blocks, "shared": by_shared}
    programs = max(0, min(candidates.values()))
    limiter = "+".join(k for k, v in candidates.items() if v == programs)
    return programs, programs * num_warps / max_warps, limiter


def stats_for(impl_name: str, shape: Shape, dtype: torch.dtype, device, kind: str = "rmsnorm") -> KernelStats | None:
    registry = RESIDUAL_IMPLEMENTATIONS if kind == "residual" else IMPLEMENTATIONS
    impl = registry[impl_name]
    if impl.family != "triton":
        return None
    inputs = make_inputs(shape, dtype, device, kind)
    if kind == "residual":
        x, r, w = inputs
        impl.fn(x, r, w, 1e-6)
    else:
        x, w = inputs
        impl.fn(x, w, 1e-6)
    launched = triton_impls.last_launch()
    if launched is None:
        return None
    kernel, cfg = launched
    n_regs = int(getattr(kernel, "n_regs", 0) or 0)
    n_spills = int(getattr(kernel, "n_spills", 0) or 0)
    meta = getattr(kernel, "metadata", None)
    shared = int(getattr(meta, "shared", 0) or 0)
    limits = device_limits(device)
    programs, occ, limiter = occupancy(n_regs, cfg["num_warps"], shared, limits)
    waves = cfg["grid"] / (programs * limits["sm_count"]) if programs else float("inf")
    return KernelStats(
        impl=impl_name, m=shape.m, n=shape.n, dtype=str(dtype).replace("torch.", ""),
        kernel=cfg["kernel_name"], grid=cfg["grid"], num_warps=cfg["num_warps"], threads=32 * cfg["num_warps"],
        block_n=cfg.get("BLOCK_N", 0), block_m=cfg.get("BLOCK_M", 1), n_regs=n_regs, n_spills=n_spills,
        shared_bytes=shared, programs_per_sm=programs, occupancy=occ, waves=waves, limiter=limiter,
    )


def collect(shapes: Sequence[Shape], dtypes: Sequence[torch.dtype], device, impl_names: Sequence[str] | None = None,
            kind: str = "rmsnorm") -> list[KernelStats]:
    registry = RESIDUAL_IMPLEMENTATIONS if kind == "residual" else IMPLEMENTATIONS
    names = impl_names or [n for n, i in registry.items() if i.family == "triton"]
    out = []
    for dtype in dtypes:
        for shape in shapes:
            for name in names:
                s = stats_for(name, shape, dtype, device, kind)
                if s is not None:
                    out.append(s)
    return out


def write_csv(path: Path, rows: Sequence[KernelStats]) -> None:
    fields = [f.name for f in dataclasses.fields(KernelStats)]
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(dataclasses.asdict(r))


def render_markdown(rows: Sequence[KernelStats]) -> str:
    lines = ["| impl | shape | dtype | grid | warps | BLOCK | regs | spills | smem B | programs/SM | occupancy | waves | limiter |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        block = f"{r.block_m}x{r.block_n}" if r.block_m > 1 else str(r.block_n)
        lines.append(f"| {r.impl} | {r.m}x{r.n} | {r.dtype} | {r.grid} | {r.num_warps} | {block} | {r.n_regs} | "
                     f"{r.n_spills} | {r.shared_bytes} | {r.programs_per_sm} | {r.occupancy:.0%} | {r.waves:.2f} | {r.limiter} |")
    return "\n".join(lines)
