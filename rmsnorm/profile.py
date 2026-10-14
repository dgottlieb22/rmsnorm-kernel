"""Profiling helpers.

``run_target`` is the process that ``nsys`` and ``ncu`` attach to: it runs one
implementation on one shape a fixed number of times inside an NVTX range, so
the profiler can be told to capture exactly those launches (see
``scripts/ncu.sh`` and ``scripts/nsys.sh``).

``kernel_table`` uses ``torch.profiler`` to list the CUDA kernels one call
launches, with their device time. For the eager baseline this is the count
of separate kernels (cast, square, mean, add, rsqrt, multiplies, cast) and the
gaps between them; for a Triton variant it is one kernel. It is the quickest
way to see when an operation is launch-bound: many short kernels with device
time well below the wall time per call.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from . import IMPLEMENTATIONS, RESIDUAL_IMPLEMENTATIONS
from .bench import Shape, make_inputs


def _callable(impl_name: str, shape: Shape, dtype: torch.dtype, device, kind: str):
    registry = RESIDUAL_IMPLEMENTATIONS if kind == "residual" else IMPLEMENTATIONS
    impl = registry[impl_name]
    inputs = make_inputs(shape, dtype, device, kind)
    if kind == "residual":
        x, r, w = inputs
        return lambda: impl.fn(x, r, w, 1e-6)
    x, w = inputs
    return lambda: impl.fn(x, w, 1e-6)


def run_target(impl_name: str, shape: Shape, dtype: torch.dtype, device, *, kind: str = "rmsnorm",
               warmup: int = 10, iters: int = 20) -> None:
    """Warm up, then run ``iters`` calls inside an NVTX range named after the implementation."""
    fn = _callable(impl_name, shape, dtype, device, kind)
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push(f"{impl_name}_{shape.label}_{str(dtype).replace('torch.', '')}")
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()


@dataclass
class KernelRow:
    name: str
    count: int
    device_us_total: float

    @property
    def device_us_each(self) -> float:
        return self.device_us_total / self.count if self.count else 0.0


def kernel_table(impl_name: str, shape: Shape, dtype: torch.dtype, device, *, kind: str = "rmsnorm",
                 iters: int = 10) -> tuple[list[KernelRow], float]:
    """CUDA kernels launched per call (name, count per call, device time) and the wall time per call."""
    fn = _callable(impl_name, shape, dtype, device, kind)
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
    wall_us_per_call = start.elapsed_time(end) * 1000.0 / iters
    rows: dict[str, KernelRow] = {}
    for evt in prof.key_averages():
        if evt.device_type != torch.autograd.DeviceType.CUDA:
            continue
        # device_time_total is in microseconds; counts are over all iters.
        total = getattr(evt, "device_time_total", None)
        if total is None:  # older torch
            total = getattr(evt, "cuda_time_total", 0.0)
        rows[evt.key] = KernelRow(name=evt.key, count=evt.count // iters, device_us_total=total / iters)
    return sorted(rows.values(), key=lambda r: -r.device_us_total), wall_us_per_call


def render_kernel_table(rows: list[KernelRow], wall_us: float) -> str:
    total_dev = sum(r.device_us_total for r in rows)
    lines = [f"wall time per call: {wall_us:.1f} us; device time per call: {total_dev:.1f} us; "
             f"kernels per call: {sum(r.count for r in rows)}",
             "", "| kernel | launches/call | device us/call |", "|---|---|---|"]
    for r in rows:
        name = r.name if len(r.name) <= 90 else r.name[:87] + "..."
        lines.append(f"| `{name}` | {r.count} | {r.device_us_total:.1f} |")
    return "\n".join(lines)
