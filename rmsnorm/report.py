"""Render a benchmark run (``raw.csv`` + ``env.json``) into ``report.md`` and plots.

Tables are per dtype, cache mode and row length N: rows are the row counts M,
columns the implementations, each cell the median GPU time in microseconds
with the effective bandwidth. A second table gives speedup against the
``torch_eager`` baseline, a third the fraction of the measured copy
bandwidth each implementation reaches, and a fourth the host launch cost
next to the GPU time for the smallest M, which is where launch overhead
decides the result.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Sequence

from .bench import Result, read_csv

BASELINE = "torch_eager"
RESIDUAL_BASELINE = "torch_eager_residual"


def _fmt(v: float, nd: int = 1) -> str:
    return "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{nd}f}"


def _group(results: Sequence[Result]):
    g = defaultdict(list)
    for r in results:
        g[(r.kind, r.dtype, r.cache)].append(r)
    return g


def _impl_order(results: Sequence[Result]) -> list[str]:
    seen = []
    for r in results:
        if r.impl not in seen:
            seen.append(r.impl)
    return seen


def _cell(r: Result | None, copy_gbps: float | None) -> str:
    if r is None:
        return "-"
    if r.error:
        return "ERR"
    pct = f" ({r.gbps / copy_gbps:.0%})" if copy_gbps else ""
    return f"{_fmt(r.gpu_us_median, 1)} us, {_fmt(r.gbps, 0)} GB/s{pct}"


def render_tables(results: Sequence[Result], env: dict) -> str:
    copy_gbps = env.get("copy_bandwidth_gbps")
    lines: list[str] = []
    for (kind, dtype, cache), rs in sorted(_group(results).items()):
        impls = _impl_order(rs)
        baseline = RESIDUAL_BASELINE if kind == "residual" else BASELINE
        by_key = {(r.impl, r.m, r.n): r for r in rs}
        ns = sorted({r.n for r in rs})
        ms = sorted({r.m for r in rs})
        lines.append(f"## {kind}, {dtype}, {cache} L2\n")
        lines.append("Median GPU time per call and effective bandwidth; the percentage is of the "
                     f"measured device copy bandwidth ({_fmt(copy_gbps, 0)} GB/s).\n")
        for n in ns:
            lines.append(f"### N = {n}\n")
            lines.append("| M | " + " | ".join(impls) + " |")
            lines.append("|---|" + "---|" * len(impls))
            for m in ms:
                cells = [_cell(by_key.get((i, m, n)), copy_gbps) for i in impls]
                lines.append(f"| {m} | " + " | ".join(cells) + " |")
            lines.append("")
            if baseline in impls:
                lines.append(f"Speedup over `{baseline}` (baseline time / time):\n")
                lines.append("| M | " + " | ".join(i for i in impls if i != baseline) + " |")
                lines.append("|---|" + "---|" * (len(impls) - 1))
                for m in ms:
                    b = by_key.get((baseline, m, n))
                    cells = []
                    for i in impls:
                        if i == baseline:
                            continue
                        r = by_key.get((i, m, n))
                        if b is None or r is None or b.error or r.error:
                            cells.append("-")
                        else:
                            cells.append(f"{b.gpu_us_median / r.gpu_us_median:.2f}x")
                    lines.append(f"| {m} | " + " | ".join(cells) + " |")
                lines.append("")
        # Launch-cost table at the smallest M.
        m0 = ms[0]
        lines.append(f"### Host launch cost vs GPU time at M = {m0}\n")
        lines.append("A row where the host time exceeds the GPU time is launch-bound: the GPU "
                     "waits for the host, and the batched column (back-to-back calls) shows the "
                     "achievable rate.\n")
        lines.append("| N | impl | GPU us | host us | batched GPU us | launch-bound |")
        lines.append("|---|---|---|---|---|---|")
        for n in ns:
            for i in impls:
                r = by_key.get((i, m0, n))
                if r is None or r.error:
                    continue
                bound = "yes" if r.cpu_us > r.gpu_us_median else "no"
                lines.append(f"| {n} | {i} | {_fmt(r.gpu_us_median)} | {_fmt(r.cpu_us)} | {_fmt(r.batched_gpu_us)} | {bound} |")
        lines.append("")
        # Stability.
        worst = max((r for r in rs if not r.error), key=lambda r: r.round_spread, default=None)
        if worst is not None:
            lines.append(f"Largest round-to-round spread in this group: {worst.round_spread:.1%} "
                         f"({worst.impl}, {worst.m}x{worst.n}).\n")
        errors = [r for r in rs if r.error]
        if errors:
            lines.append("Errors:\n")
            for r in errors:
                lines.append(f"- {r.impl} {r.m}x{r.n}: {r.error}")
            lines.append("")
    return "\n".join(lines)


def render_env(env: dict) -> str:
    keys = ["device_name", "sm_count", "total_memory_gb", "compute_capability", "cuda_runtime",
            "torch", "triton", "python", "spec_bandwidth_gbps", "copy_bandwidth_gbps"]
    lines = ["## Environment\n", "| key | value |", "|---|---|"]
    for k in keys:
        if k in env:
            v = env[k]
            if isinstance(v, float):
                v = f"{v:.0f}"
            lines.append(f"| {k} | {v} |")
    lines.append("")
    return "\n".join(lines)


def render_report(run_dir: Path) -> Path:
    results = read_csv(run_dir / "raw.csv")
    env = json.loads((run_dir / "env.json").read_text()) if (run_dir / "env.json").exists() else {}
    text = f"# Benchmark report: {run_dir.name}\n\n" + render_env(env) + "\n" + render_tables(results, env)
    out = run_dir / "report.md"
    out.write_text(text)
    return out


def plot(run_dir: Path) -> list[Path]:
    """Bandwidth against N (one panel per M) and against M (at N = 4096), per dtype and cache mode."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        return []
    results = read_csv(run_dir / "raw.csv")
    env = json.loads((run_dir / "env.json").read_text()) if (run_dir / "env.json").exists() else {}
    copy_gbps = env.get("copy_bandwidth_gbps")
    out_dir = run_dir / "plots"
    out_dir.mkdir(exist_ok=True)
    written = []
    for (kind, dtype, cache), rs in sorted(_group(results).items()):
        rs = [r for r in rs if not r.error]
        if not rs:
            continue
        impls = _impl_order(rs)
        ms = sorted({r.m for r in rs})
        fig, axes = plt.subplots(1, len(ms), figsize=(4 * len(ms), 3.6), sharey=True, squeeze=False)
        for ax, m in zip(axes[0], ms):
            for impl in impls:
                pts = sorted((r.n, r.gbps) for r in rs if r.impl == impl and r.m == m)
                if pts:
                    ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=impl)
            if copy_gbps:
                ax.axhline(copy_gbps, color="k", linestyle="--", linewidth=0.8, label="device copy")
            ax.set_xscale("log", base=2)
            ax.set_title(f"M = {m}")
            ax.set_xlabel("N (hidden size)")
            ax.grid(True, alpha=0.3)
        axes[0][0].set_ylabel("effective GB/s")
        axes[0][-1].legend(fontsize=7, loc="lower right")
        fig.suptitle(f"{kind} {dtype} ({cache} L2): bandwidth vs hidden size")
        fig.tight_layout()
        p = out_dir / f"{kind}_{dtype}_{cache}_bandwidth_vs_n.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        written.append(p)
    return written
