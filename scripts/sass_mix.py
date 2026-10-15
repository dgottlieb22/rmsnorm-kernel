"""SASS instruction mix per thread for a variant at given N (sm_80, bf16, M = 4096). No GPU needed.

Compiles the launch plan, assembles the PTX with the bundled ptxas, disassembles the
cubin with the bundled nvdisasm, and counts instructions by class. The count is
STATIC: for the loop-free kernels (v0, v1, v3, v3b, v4) it equals what one thread
executes; for v2 the loop bodies run N / BLOCK_N times and the static count is a
lower bound.

Instructions per element is the metric for the instruction-bound regime (short
rows): bytes moved per instruction issued. The floor for a bf16 element is about
one load and one store per 8 elements plus 2 converts, 1 square-accumulate, 2
multiplies, so roughly 5 to 6 instructions per element before any address,
predicate or reduction work.

Usage, from the repository root with TRITON_INTERPRET unset::

    .venv/bin/python scripts/sass_mix.py --impl triton_v1_warps --n 512 --n 4096 [--arch sm_80] [--dump]
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import tempfile
from collections import Counter

import torch

from rmsnorm.aot_stats import ARCHS, compile_plan
from rmsnorm.triton_impls import plan_for

CLASSES = [
    ("global", ("LDG", "STG")),
    ("shared", ("LDS", "STS")),
    ("barrier", ("BAR",)),
    ("shuffle", ("SHFL",)),
    ("convert", ("F2F", "F2FP", "I2F", "F2I", "PRMT")),
    ("fp math", ("FADD", "FMUL", "FFMA", "MUFU", "FSEL", "FMNMX", "FSETP", "FCHK")),
    ("int/addr/pred", ("IMAD", "IADD", "LEA", "SHF", "LOP", "SEL", "ISETP", "PLOP", "MOV", "IABS", "I2I", "VOTE", "P2R", "R2P")),
    ("control/uniform", ("BRA", "EXIT", "NOP", "S2R", "CS2R", "ULDC", "UMOV", "USHF", "UIADD", "UISETP", "ULOP", "USEL", "UIMAD", "BSSY", "BSYNC", "DEPBAR", "WARPSYNC", "ERRBAR", "MEMBAR", "RET", "BPT", "YIELD")),
]
_INSN = re.compile(r"^\s*/\*[0-9a-f]+\*/\s+(?:@!?U?P\d\s+)?([A-Z0-9_.]+)")


def _bin(name: str) -> str:
    import triton

    return os.path.join(os.path.dirname(triton.__file__), "backends", "nvidia", "bin", name)


def sass_for(plan, n: int, arch: str, dump: str | None = None) -> str:
    compiled = compile_plan(plan, torch.bfloat16, 4096, n, arch)
    with tempfile.TemporaryDirectory() as d:
        ptx = os.path.join(d, "k.ptx")
        cubin = os.path.join(d, "k.cubin")
        with open(ptx, "w") as fh:
            fh.write(compiled.asm["ptx"])
        subprocess.run([_bin("ptxas"), f"--gpu-name={ARCHS[arch]['gpu_name']}", "-o", cubin, ptx], check=True,
                       capture_output=True)
        sass = subprocess.run([_bin("nvdisasm"), "-c", cubin], check=True, capture_output=True, text=True).stdout
    if dump:
        os.makedirs(dump, exist_ok=True)
        with open(os.path.join(dump, f"{plan.variant}_{n}_{arch}.sass"), "w") as fh:
            fh.write(sass)
    return sass


def classify(sass: str) -> tuple[Counter, Counter]:
    """Count executed instructions by class and by opcode.

    Everything after the last ``EXIT`` is alignment padding (a self-branch and
    NOPs) that never issues, so counting stops there.
    """
    lines = sass.splitlines()
    last_exit = max((i for i, l in enumerate(lines) if _INSN.match(l) and _INSN.match(l).group(1).startswith("EXIT")),
                    default=len(lines) - 1)
    by_class: Counter = Counter()
    by_opcode: Counter = Counter()
    for line in lines[: last_exit + 1]:
        m = _INSN.match(line)
        if not m:
            continue
        op = m.group(1)
        by_opcode[op] += 1
        base = op.split(".")[0]
        for cls, prefixes in CLASSES:
            if base.startswith(prefixes):
                by_class[cls] += 1
                break
        else:
            by_class["other"] += 1
    return by_class, by_opcode


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--impl", action="append", default=None)
    ap.add_argument("--n", action="append", type=int, default=None)
    ap.add_argument("--arch", default="sm_80")
    ap.add_argument("--dump", default=None, metavar="DIR", help="write the SASS text per compile")
    ap.add_argument("--top", type=int, default=0, help="also print the N most frequent opcodes")
    args = ap.parse_args()
    impls = args.impl or ["triton_v1_warps"]
    ns = args.n or [512, 1024, 4096]
    cols = [c for c, _ in CLASSES] + ["other"]
    print("| variant | N | block | warps | elems/thread | insns/thread | insns/elem | " + " | ".join(cols) + " |")
    print("|---|---|---|---|---|---|---|" + "---|" * len(cols))
    for n in ns:
        for impl in impls:
            try:
                plan = plan_for(impl, 4096, n)
            except ValueError as e:   # the variant does not cover this N
                print(f"| {impl} | {n} | skipped: {e} |")
                continue
            by_class, by_opcode = classify(sass_for(plan, n, args.arch, args.dump))
            total = sum(by_class.values())
            threads = 32 * plan.num_warps
            elems = plan.block_m * plan.block_n / threads
            block = f"{plan.block_m}x{plan.block_n}" if plan.block_m > 1 else str(plan.block_n)
            print(f"| {impl} | {n} | {block} | {plan.num_warps} | {elems:g} | {total} | {total / elems:.2f} | "
                  + " | ".join(str(by_class.get(c, 0)) for c in cols) + " |")
            if args.top:
                print("    " + ", ".join(f"{op} {c}" for op, c in by_opcode.most_common(args.top)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
