"""Compile-time kernel statistics without a GPU.

Triton's compiler runs on the CPU and ships NVIDIA's ``ptxas``, so a kernel
can be compiled for a chosen SM and its register count, spill bytes and
shared-memory use read off the assembler without a device. For a
bandwidth-bound kernel that is most of what decides occupancy, and it is the
evidence for the register-pressure claims in the diary (H3, H4) before any
timing exists.

For each registered variant and shape this module builds the same
``LaunchPlan`` the runtime wrapper would launch, compiles it with the same
argument specialization the launcher would apply (16-byte divisibility hints
on aligned pointers and on integers divisible by 16), runs ``ptxas -v`` on
the PTX for the target, and parses::

    ptxas info    : Used 104 registers, used 1 barriers, 400 bytes cmem[0]
    ptxas info    : 192 bytes stack frame, 188 bytes spill stores, 188 bytes spill loads

Theoretical occupancy and waves then follow from the per-SM limits of the
target. The PTX is also scanned for global and shared-memory loads and stores
by access width, and ``dump_dir`` keeps the TTGIR and PTX of every compile so
a layout question (where did that shared memory come from?) can be answered
by reading the IR (diary H14). What this cannot see: achieved occupancy,
memory throughput, or anything that depends on how the kernel runs. Those
need ``kernel_stats`` and the profiler on a GPU.
"""
from __future__ import annotations

import csv
import dataclasses
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch

from . import triton_impls
from .kernel_stats import occupancy
from .triton_impls import LaunchPlan, plan_for

# Target architectures: (compute capability, ptxas gpu name, per-SM limits, SM count of a
# representative part). Limits: register file, max resident threads, max resident blocks.
ARCHS = {
    "sm_80": dict(cc=80, gpu_name="sm_80", sm_count=108, regs_per_sm=65536, max_threads_per_sm=2048,
                  max_blocks_per_sm=32, shared_per_sm=160 * 1024, part="A100"),
    "sm_86": dict(cc=86, gpu_name="sm_86", sm_count=80, regs_per_sm=65536, max_threads_per_sm=1536,
                  max_blocks_per_sm=16, shared_per_sm=100 * 1024, part="A10G"),
    "sm_89": dict(cc=89, gpu_name="sm_89", sm_count=128, regs_per_sm=65536, max_threads_per_sm=1536,
                  max_blocks_per_sm=24, shared_per_sm=100 * 1024, part="RTX 4090 / L4 (58 SMs)"),
    "sm_90": dict(cc=90, gpu_name="sm_90a", sm_count=132, regs_per_sm=65536, max_threads_per_sm=2048,
                  max_blocks_per_sm=32, shared_per_sm=228 * 1024, part="H100 SXM"),
}

TRITON_DTYPE = {torch.bfloat16: "bf16", torch.float16: "fp16", torch.float32: "fp32"}


@dataclass
class AotStats:
    variant: str
    kernel: str
    m: int
    n: int
    dtype: str
    arch: str
    specialized: bool
    grid: int
    num_warps: int
    block_m: int
    block_n: int
    align: int
    registers: int
    spill_stores: int
    spill_loads: int
    stack_bytes: int
    shared_bytes: int
    programs_per_sm: int
    occupancy: float
    waves: float
    limiter: str
    global_loads: str = ""   # PTX global loads by width in bits, "128:8 16:2"
    global_stores: str = ""
    shared_loads: str = ""   # PTX shared-memory loads and stores by width; "-" when none
    shared_stores: str = ""
    layout_converts: int = 0  # ttg.convert_layout ops in the TTGIR (some lower to nothing)


def _ptxas_path() -> str:
    import triton

    return os.path.join(os.path.dirname(triton.__file__), "backends", "nvidia", "bin", "ptxas")


def signature_and_attrs(plan: LaunchPlan, dtype: torch.dtype, m: int, n: int, specialize: bool = True) -> tuple[dict, dict]:
    """Mirror the JIT launcher: pointer args get the dtype, ints become i32, eps fp32, constexprs marked.

    Specialization: the launcher adds a divisibility-by-16 attribute to every
    tensor pointer (torch allocations are 256-byte aligned) and to every
    integer divisible by 16. Row strides equal N for contiguous inputs, so the
    hints depend on N; the compiler uses them to decide load widths. With
    ``specialize=False`` no hints are passed, which is what a launcher would
    do for misaligned pointers or odd strides.
    """
    kernel = plan.kernel
    dt = TRITON_DTYPE[dtype]
    align = plan.constexprs.get("ALIGN", 1)
    # Contiguous rows: every row stride equals N. Strides and N are passed
    # divided by ALIGN, as the wrappers do.
    values = {name: n // align for name in ("stride_xm_div", "stride_rm_div", "stride_ym_div", "stride_sm_div", "N_div")}
    values["M"] = m
    signature: dict = {}
    attrs: dict = {}
    for idx, name in enumerate(kernel.arg_names):
        if name in plan.constexprs:
            signature[name] = "constexpr"
        elif name.endswith("_ptr"):
            signature[name] = f"*{dt}"
            if specialize:
                attrs[(idx,)] = [["tt.divisibility", 16]]
        elif name == "eps":
            signature[name] = "fp32"
        else:
            signature[name] = "i32"
            if specialize and values[name] % 16 == 0:
                attrs[(idx,)] = [["tt.divisibility", 16]]
    return signature, attrs


def compile_plan(plan: LaunchPlan, dtype: torch.dtype, m: int, n: int, arch: str, specialize: bool = True):
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    if os.environ.get("TRITON_INTERPRET") == "1":
        raise RuntimeError("unset TRITON_INTERPRET before compiling offline; the interpreter patches the compiler")
    signature, attrs = signature_and_attrs(plan, dtype, m, n, specialize)
    src = ASTSource(fn=plan.kernel, signature=signature, constexprs=plan.constexprs, attrs=attrs)
    target = GPUTarget("cuda", ARCHS[arch]["cc"], 32)
    return triton.compile(src, target=target, options={"num_warps": plan.num_warps})


_USED_RE = re.compile(r"Used (\d+) registers")
_SPILL_RE = re.compile(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads")
_ACCESS_RE = {
    space: re.compile(rf"\b(ld|st)\.{space}(?:\.[a-z]+)*\.(v\d\.)?(?:b|u|s|f)(\d+)")
    for space in ("global", "shared")
}


def access_widths(ptx: str, space: str = "global") -> tuple[str, str]:
    """Load and store instructions to ``space`` in the PTX by access width in bits, e.g. ``"128:8"``.

    One 128-bit access per thread per 8 bf16 elements is the widest the
    hardware offers and what a coalesced kernel should show; 16-bit entries
    mean scalar bf16 accesses, eight times the instructions for the same bytes.
    For ``space="shared"`` the counts show what the kernel stages through
    shared memory: the row kernels only exchange one float per warp for the
    reduction (two 32-bit stores and loads); anything wider is a layout
    conversion (diary H14).
    """
    loads: dict[int, int] = {}
    stores: dict[int, int] = {}
    for op, vec, bits in _ACCESS_RE[space].findall(ptx):
        width = int(bits) * (int(vec[1]) if vec else 1)
        target = loads if op == "ld" else stores
        target[width] = target.get(width, 0) + 1
    fmt = lambda d: " ".join(f"{w}:{c}" for w, c in sorted(d.items(), reverse=True)) or "-"  # noqa: E731
    return fmt(loads), fmt(stores)


def ptxas_resources(ptx: str, arch: str) -> dict:
    """Run the bundled ptxas on PTX text and parse registers, stack and spill bytes."""
    with tempfile.NamedTemporaryFile("w", suffix=".ptx", delete=False) as fh:
        fh.write(ptx)
        path = fh.name
    try:
        out = subprocess.run(
            [_ptxas_path(), "-v", f"--gpu-name={ARCHS[arch]['gpu_name']}", "-o", os.devnull, path],
            capture_output=True, text=True, check=False,
        )
    finally:
        os.unlink(path)
    if out.returncode != 0:
        raise RuntimeError(f"ptxas failed for {arch}: {out.stderr.strip()[:500]}")
    text = out.stderr + out.stdout
    used = _USED_RE.search(text)
    spill = _SPILL_RE.search(text)
    return {
        "registers": int(used.group(1)) if used else 0,
        "stack_bytes": int(spill.group(1)) if spill else 0,
        "spill_stores": int(spill.group(2)) if spill else 0,
        "spill_loads": int(spill.group(3)) if spill else 0,
    }


def stats_for(variant: str, m: int, n: int, dtype: torch.dtype, arch: str, specialize: bool = True,
              dump_dir: Path | None = None) -> AotStats:
    plan = plan_for(variant, m, n)
    compiled = compile_plan(plan, dtype, m, n, arch, specialize)
    ptx, ttgir = compiled.asm["ptx"], compiled.asm["ttgir"]
    if dump_dir is not None:
        dump_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{variant}_{m}x{n}_{TRITON_DTYPE[dtype]}_{arch}" + ("" if specialize else "_nohints")
        (dump_dir / f"{stem}.ttgir").write_text(ttgir)
        (dump_dir / f"{stem}.ptx").write_text(ptx)
    res = ptxas_resources(ptx, arch)
    loads, stores = access_widths(ptx)
    shared_loads, shared_stores = access_widths(ptx, "shared")
    shared = int(getattr(compiled.metadata, "shared", 0) or 0)
    limits = ARCHS[arch]
    programs, occ, limiter = occupancy(res["registers"], plan.num_warps, shared, limits)
    waves = plan.grid / (programs * limits["sm_count"]) if programs else float("inf")
    return AotStats(
        variant=variant, kernel=plan.kernel_name, m=m, n=n, dtype=TRITON_DTYPE[dtype], arch=arch,
        specialized=specialize,
        grid=plan.grid, num_warps=plan.num_warps, block_m=plan.block_m, block_n=plan.block_n,
        align=plan.constexprs.get("ALIGN", 1),
        registers=res["registers"], spill_stores=res["spill_stores"], spill_loads=res["spill_loads"],
        stack_bytes=res["stack_bytes"], shared_bytes=shared, programs_per_sm=programs, occupancy=occ,
        waves=waves, limiter=limiter, global_loads=loads, global_stores=stores,
        shared_loads=shared_loads, shared_stores=shared_stores,
        layout_converts=ttgir.count("ttg.convert_layout"),
    )


DEFAULT_VARIANTS = ["triton_v0_naive", "triton_v1_warps", "triton_v2_looped", "triton_v3_multirow",
                    "triton_v4_fused_residual"]
DEFAULT_NS = [512, 1024, 2048, 4096, 5120, 8192, 16384]


def collect(variants: Sequence[str] = DEFAULT_VARIANTS, ns: Sequence[int] = DEFAULT_NS, m: int = 4096,
            dtypes: Sequence[torch.dtype] = (torch.bfloat16,), archs: Sequence[str] = ("sm_80", "sm_90"),
            specialize: bool = True, log=None, dump_dir: Path | None = None) -> list[AotStats]:
    rows = []
    for arch in archs:
        for dtype in dtypes:
            for n in ns:
                for variant in variants:
                    s = stats_for(variant, m, n, dtype, arch, specialize, dump_dir)
                    rows.append(s)
                    if log:
                        log(f"{arch} {s.dtype} N={n:>6} {variant:<26} warps={s.num_warps:>2} block={s.block_m}x{s.block_n:<6} "
                            f"align={s.align:>2} regs={s.registers:>3} spill={s.spill_stores:>4}B smem={s.shared_bytes:>5}B "
                            f"occ={s.occupancy:.0%} ({s.limiter}) ld[{s.global_loads}] st[{s.global_stores}] "
                            f"shared ld[{s.shared_loads}] st[{s.shared_stores}]")
    return rows


def write_csv(path: Path, rows: Sequence[AotStats]) -> None:
    fields = [f.name for f in dataclasses.fields(AotStats)]
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(dataclasses.asdict(r))


def render_markdown(rows: Sequence[AotStats]) -> str:
    lines = ["| arch | variant | dtype | hints | M x N | grid | warps | block | ALIGN | regs | spill B (st/ld) | smem B | programs/SM | occupancy | waves | limiter | global ld (bits:count) | global st | shared ld | shared st | layout converts |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        block = f"{r.block_m}x{r.block_n}" if r.block_m > 1 else str(r.block_n)
        hints = "yes" if r.specialized else "no"
        lines.append(f"| {r.arch} | {r.variant} | {r.dtype} | {hints} | {r.m}x{r.n} | {r.grid} | {r.num_warps} | {block} | {r.align} | {r.registers} | "
                     f"{r.spill_stores}/{r.spill_loads} | {r.shared_bytes} | {r.programs_per_sm} | {r.occupancy:.0%} | "
                     f"{r.waves:.2f} | {r.limiter} | {r.global_loads} | {r.global_stores} | {r.shared_loads} | {r.shared_stores} | "
                     f"{r.layout_converts} |")
    return "\n".join(lines)


__all__ = ["ARCHS", "AotStats", "collect", "compile_plan", "ptxas_resources", "render_markdown", "stats_for",
           "write_csv"]

# Keep a reference so linters see the module is used for its side effects (kernel definitions).
_ = triton_impls
