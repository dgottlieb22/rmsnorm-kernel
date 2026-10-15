"""Benchmark harness.

What is measured, and how, is the content of ``docs/methodology.md``; this
module is the executable version of it.

* GPU time per call comes from CUDA events around each call. Between calls
  in ``cold`` mode a 256 MB buffer is written to evict the inputs from L2, so
  every timed call reads its inputs from HBM: the conservative number.
  Whether a forward pass finds the norm's input still in L2 depends on the
  shape and the preceding kernels and is not measured here. ``warm`` mode
  skips the flush and shows what a microbenchmark that forgets to flush
  reports; the two bound the real case.
* Measurements run in rounds. Each round does its own warmup and takes
  ``iters`` samples; the reported latency is the median over all samples and
  the round-to-round spread of the per-round medians is reported next to it.
  A spread of a few percent is normal; more means the number is not stable.
* CPU launch cost is the wall time of issuing a call without synchronizing.
  When it exceeds the GPU time the operation is launch-bound: the GPU idles
  waiting for the host.
* Batched GPU time brackets ``iters`` back-to-back calls with one event pair.
  Launches overlap with execution, so this approaches the kernel's execution
  time when the host keeps up and the host's issue rate when it does not.
* Effective bandwidth is the operation's unavoidable traffic (inputs read
  once, outputs written once, weight read once) divided by GPU time. It is
  compared with a measured device-to-device copy, not a spec sheet.

Every implementation is checked against the float64 oracle on each shape
before it is timed. A wrong kernel is reported, not benchmarked.
"""
from __future__ import annotations

import csv
import dataclasses
import json
import platform
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

import torch

from . import IMPLEMENTATIONS, RESIDUAL_IMPLEMENTATIONS, Impl
from .reference import rmsnorm_reference, rmsnorm_residual_reference
from .tolerance import is_close, max_error

L2_FLUSH_BYTES = 256 * 1024 * 1024


@dataclass(frozen=True)
class Shape:
    m: int
    n: int

    @property
    def label(self) -> str:
        return f"{self.m}x{self.n}"


QUICK_SHAPES = [Shape(m, n) for m in (1, 64, 4096) for n in (512, 4096, 16384)]
FULL_SHAPES = [
    Shape(m, n)
    for m in (1, 8, 64, 512, 4096, 16384)
    # 1000 is not a multiple of 16: its row stride loses the launcher's alignment hint (diary H12).
    for n in (512, 768, 1000, 1024, 2048, 4096, 5120, 8192, 16384)
]
MATRICES = {"quick": QUICK_SHAPES, "full": FULL_SHAPES}
MAX_ELEMENTS = 1 << 29  # 512M elements: 1 GiB in bf16 per tensor


@dataclass
class Timing:
    gpu_us_median: float
    gpu_us_p25: float
    gpu_us_p75: float
    gpu_us_min: float
    round_spread: float  # (max - min) of per-round medians, relative to the median
    cpu_us: float
    batched_gpu_us: float
    rounds: int
    iters: int


@dataclass
class Result:
    impl: str
    family: str
    kind: str  # "rmsnorm" | "residual"
    m: int
    n: int
    dtype: str
    cache: str
    bytes_moved: int
    gpu_us_median: float
    gpu_us_p25: float
    gpu_us_p75: float
    gpu_us_min: float
    round_spread: float
    cpu_us: float
    batched_gpu_us: float
    gbps: float
    elements_per_us: float
    max_abs_err: float
    max_rel_err: float
    rounds: int
    iters: int
    error: str = ""


def bytes_for(kind: str, m: int, n: int, dtype: torch.dtype, weight_dtype: torch.dtype) -> int:
    """Unavoidable traffic: every input element read once, every output written once."""
    e = torch.tensor([], dtype=dtype).element_size()
    we = torch.tensor([], dtype=weight_dtype).element_size()
    if kind == "rmsnorm":
        return m * n * (e + e) + n * we
    if kind == "residual":
        return m * n * (2 * e + 2 * e) + n * we
    raise ValueError(kind)


def _percentile(samples: Sequence[float], q: float) -> float:
    s = sorted(samples)
    if not s:
        return float("nan")
    k = (len(s) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


class L2Flusher:
    def __init__(self, device):
        self.buf = torch.empty(L2_FLUSH_BYTES // 4, dtype=torch.int32, device=device)

    def flush(self):
        self.buf.zero_()


def time_gpu(fn: Callable[[], object], *, device, cache: str, rounds: int, iters: int, warmup: int) -> Timing:
    """Time ``fn`` on a CUDA device; see the module docstring for the protocol."""
    if device.type != "cuda":
        raise RuntimeError("GPU timing needs a CUDA device; this host has none")
    flusher = L2Flusher(device) if cache == "cold" else None
    all_samples: list[float] = []
    round_medians: list[float] = []
    for _ in range(rounds):
        for _ in range(warmup):
            fn()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for i in range(iters):
            if flusher is not None:
                flusher.flush()
            starts[i].record()
            fn()
            ends[i].record()
        torch.cuda.synchronize()
        samples = [s.elapsed_time(e) * 1000.0 for s, e in zip(starts, ends)]
        all_samples.extend(samples)
        round_medians.append(statistics.median(samples))

    # Host issue cost: wall time per call with nothing waiting on the GPU.
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    cpu_us = (time.perf_counter() - t0) / iters * 1e6
    torch.cuda.synchronize()

    # Back-to-back calls under one event pair: launches overlap execution.
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    batched_us = start.elapsed_time(end) * 1000.0 / iters

    med = statistics.median(all_samples)
    return Timing(
        gpu_us_median=med,
        gpu_us_p25=_percentile(all_samples, 0.25),
        gpu_us_p75=_percentile(all_samples, 0.75),
        gpu_us_min=min(all_samples),
        round_spread=(max(round_medians) - min(round_medians)) / med if med else float("nan"),
        cpu_us=cpu_us,
        batched_gpu_us=batched_us,
        rounds=rounds,
        iters=iters,
    )


def measure_copy_bandwidth(device, nbytes: int = 1 << 30, iters: int = 20) -> float:
    """Device-to-device copy bandwidth in GB/s (read + write counted), cold L2 irrelevant at this size."""
    src = torch.empty(nbytes // 2, dtype=torch.int16, device=device)
    dst = torch.empty_like(src)
    for _ in range(3):
        dst.copy_(src)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        dst.copy_(src)
    end.record()
    torch.cuda.synchronize()
    seconds = start.elapsed_time(end) / 1000.0 / iters
    return 2 * nbytes / seconds / 1e9


# Vendor peak HBM bandwidth in GB/s, by substring of torch.cuda.get_device_name().
# Only used as a reference line; the measured copy bandwidth is the roofline.
SPEC_BANDWIDTH_GBPS = {
    "H100 SXM": 3352.0,
    "H100 PCIe": 2039.0,
    "H100 NVL": 3938.0,
    "H200": 4800.0,
    "A100-SXM4-80GB": 2039.0,
    "A100-SXM4-40GB": 1555.0,
    "A100 80GB PCIe": 1935.0,
    "A100-PCIE-40GB": 1555.0,
    "A10G": 600.0,
    "L4": 300.0,
    "L40S": 864.0,
    "T4": 320.0,
    "RTX 4090": 1008.0,
    "RTX 3090": 936.0,
}


def spec_bandwidth(device_name: str) -> float | None:
    for key, gbps in SPEC_BANDWIDTH_GBPS.items():
        if key.lower() in device_name.lower():
            return gbps
    return None


def environment(device) -> dict:
    info = {
        "torch": torch.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cuda_available": torch.cuda.is_available(),
    }
    try:
        import triton

        info["triton"] = triton.__version__
    except ImportError:
        info["triton"] = None
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        info.update(
            {
                "device_name": props.name,
                "sm_count": props.multi_processor_count,
                "total_memory_gb": round(props.total_memory / 1e9, 1),
                "compute_capability": f"{props.major}.{props.minor}",
                "cuda_runtime": torch.version.cuda,
                "spec_bandwidth_gbps": spec_bandwidth(props.name),
            }
        )
    return info


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


DTYPE_BY_NAME = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32,
                 "bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def make_inputs(shape: Shape, dtype: torch.dtype, device, kind: str, seed: int = 0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(shape.m, shape.n, generator=g).to(dtype).to(device)
    w = (1.0 + 0.1 * torch.randn(shape.n, generator=g)).to(dtype).to(device)
    if kind == "residual":
        r = torch.randn(shape.m, shape.n, generator=g).to(dtype).to(device)
        return x, r, w
    return x, w


def check_once(impl: Impl, kind: str, inputs, eps: float) -> tuple[float, float]:
    """Run once and compare against the oracle; returns (max abs err, max rel err) or raises."""
    if kind == "residual":
        x, r, w = inputs
        y, s = impl.fn(x, r, w, eps)
        y_ref, _ = rmsnorm_residual_reference(x, r, w, eps)
    else:
        x, w = inputs
        y = impl.fn(x, w, eps)
        y_ref = rmsnorm_reference(x, w, eps)
    abs_err, rel_err = max_error(y, y_ref)
    if not is_close(y, y_ref):
        raise AssertionError(f"{impl.name} is wrong on {tuple(x.shape)} {x.dtype}: abs {abs_err:.3e} rel {rel_err:.3e}")
    return abs_err, rel_err


def bench_one(
    impl: Impl, kind: str, shape: Shape, dtype: torch.dtype, device, *,
    cache: str, rounds: int, iters: int, warmup: int, eps: float = 1e-6,
) -> Result:
    inputs = make_inputs(shape, dtype, device, kind)
    nbytes = bytes_for(kind, shape.m, shape.n, dtype, dtype)
    base = dict(
        impl=impl.name, family=impl.family, kind=kind, m=shape.m, n=shape.n,
        dtype=_dtype_name(dtype), cache=cache, bytes_moved=nbytes,
    )
    try:
        abs_err, rel_err = check_once(impl, kind, inputs, eps)
        if kind == "residual":
            x, r, w = inputs
            fn = lambda: impl.fn(x, r, w, eps)  # noqa: E731
        else:
            x, w = inputs
            fn = lambda: impl.fn(x, w, eps)  # noqa: E731
        t = time_gpu(fn, device=device, cache=cache, rounds=rounds, iters=iters, warmup=warmup)
    except Exception as e:  # noqa: BLE001 - record and move on; the run must finish
        nan = float("nan")
        return Result(**base, gpu_us_median=nan, gpu_us_p25=nan, gpu_us_p75=nan, gpu_us_min=nan,
                      round_spread=nan, cpu_us=nan, batched_gpu_us=nan, gbps=nan, elements_per_us=nan,
                      max_abs_err=nan, max_rel_err=nan, rounds=rounds, iters=iters,
                      error=f"{type(e).__name__}: {e}"[:500])
    seconds = t.gpu_us_median * 1e-6
    return Result(
        **base,
        gpu_us_median=t.gpu_us_median, gpu_us_p25=t.gpu_us_p25, gpu_us_p75=t.gpu_us_p75,
        gpu_us_min=t.gpu_us_min, round_spread=t.round_spread, cpu_us=t.cpu_us,
        batched_gpu_us=t.batched_gpu_us,
        gbps=nbytes / seconds / 1e9,
        elements_per_us=shape.m * shape.n / t.gpu_us_median,
        max_abs_err=abs_err, max_rel_err=rel_err, rounds=t.rounds, iters=t.iters,
    )


def select_impls(names: Iterable[str] | None, kind: str) -> list[Impl]:
    registry = RESIDUAL_IMPLEMENTATIONS if kind == "residual" else IMPLEMENTATIONS
    if names:
        missing = [n for n in names if n not in registry]
        if missing:
            raise KeyError(f"unknown implementation(s) {missing}; known: {sorted(registry)}")
        return [registry[n] for n in names]
    return list(registry.values())


def run_matrix(
    *, shapes: Sequence[Shape], dtypes: Sequence[torch.dtype], impls: Sequence[Impl], kind: str,
    device, caches: Sequence[str] = ("cold",), rounds: int = 5, iters: int = 100, warmup: int = 25,
    out_dir: Path | None = None, log: Callable[[str], None] = print,
) -> list[Result]:
    results: list[Result] = []
    env = environment(device)
    env["copy_bandwidth_gbps"] = measure_copy_bandwidth(device)
    log(f"device: {env.get('device_name')} copy bandwidth {env['copy_bandwidth_gbps']:.0f} GB/s")
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "env.json").write_text(json.dumps(env, indent=2))
    for dtype in dtypes:
        for shape in shapes:
            if shape.m * shape.n > MAX_ELEMENTS:
                log(f"skip {shape.label}: more than {MAX_ELEMENTS} elements")
                continue
            for cache in caches:
                for impl in impls:
                    r = bench_one(impl, kind, shape, dtype, device, cache=cache, rounds=rounds, iters=iters, warmup=warmup)
                    results.append(r)
                    if r.error:
                        log(f"{_dtype_name(dtype)} {shape.label:>12} {cache} {impl.name:<24} ERROR {r.error}")
                    else:
                        log(f"{_dtype_name(dtype)} {shape.label:>12} {cache} {impl.name:<24} "
                            f"{r.gpu_us_median:9.2f} us  {r.gbps:8.0f} GB/s  cpu {r.cpu_us:7.1f} us  spread {r.round_spread:.1%}")
                    if out_dir is not None:
                        write_csv(out_dir / "raw.csv", results)
    return results


def write_csv(path: Path, results: Sequence[Result]) -> None:
    fields = [f.name for f in dataclasses.fields(Result)]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow(dataclasses.asdict(r))


def read_csv(path: Path) -> list[Result]:
    out = []
    with path.open() as fh:
        for row in csv.DictReader(fh):
            kwargs = {}
            for f in dataclasses.fields(Result):
                v = row[f.name]
                if f.type in ("int",) or f.name in ("m", "n", "bytes_moved", "rounds", "iters"):
                    kwargs[f.name] = int(v)
                elif f.name in ("impl", "family", "kind", "dtype", "cache", "error"):
                    kwargs[f.name] = v
                else:
                    kwargs[f.name] = float(v) if v not in ("", None) else float("nan")
            out.append(Result(**kwargs))
    return out
