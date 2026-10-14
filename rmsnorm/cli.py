"""Command line: ``python -m rmsnorm.cli <command>``.

bench        time implementations over a shape matrix (GPU only), write raw.csv + env.json
report       render a run directory into report.md and plots
stats        registers / spills / occupancy per Triton variant and shape (GPU only)
kernels      torch.profiler kernel table for one implementation and shape (GPU only)
target       run one implementation repeatedly for nsys / ncu to attach to (GPU only)
env          print the environment the benchmark would record
list         list implementations
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import torch

from . import IMPLEMENTATIONS, RESIDUAL_IMPLEMENTATIONS
from .bench import DTYPE_BY_NAME, MATRICES, Shape, environment, run_matrix, select_impls


def _device(require: bool = True):
    if torch.cuda.is_available():
        return torch.device("cuda")
    if require:
        sys.exit("This command needs a CUDA device. Correctness tests run on CPU (make test); timing does not.")
    return torch.device("cpu")


def _parse_shapes(args) -> list[Shape]:
    if args.shape:
        out = []
        for s in args.shape:
            m, n = s.lower().split("x")
            out.append(Shape(int(m), int(n)))
        return out
    return MATRICES[args.matrix]


def cmd_bench(args) -> int:
    device = _device()
    shapes = _parse_shapes(args)
    dtypes = [DTYPE_BY_NAME[d] for d in args.dtype]
    impls = select_impls(args.impl, args.kind)
    if not args.include_compile:
        impls = [i for i in impls if i.family != "compile"]
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    name = torch.cuda.get_device_name(device).replace(" ", "_")
    out_dir = Path(args.out) / f"{stamp}-{name}-{args.kind}-{args.matrix if not args.shape else 'custom'}"
    print(f"writing to {out_dir}")
    run_matrix(shapes=shapes, dtypes=dtypes, impls=impls, kind=args.kind, device=device,
               caches=args.cache, rounds=args.rounds, iters=args.iters, warmup=args.warmup, out_dir=out_dir)
    from .report import plot, render_report

    print(f"report: {render_report(out_dir)}")
    for p in plot(out_dir):
        print(f"plot: {p}")
    return 0


def cmd_report(args) -> int:
    from .report import plot, render_report

    run_dir = Path(args.run_dir)
    print(render_report(run_dir))
    for p in plot(run_dir):
        print(p)
    return 0


def cmd_stats(args) -> int:
    from . import kernel_stats

    device = _device()
    shapes = _parse_shapes(args)
    dtypes = [DTYPE_BY_NAME[d] for d in args.dtype]
    rows = kernel_stats.collect(shapes, dtypes, device, args.impl or None, kind=args.kind)
    print(json.dumps(kernel_stats.device_limits(device)))
    print(kernel_stats.render_markdown(rows))
    if args.out:
        kernel_stats.write_csv(Path(args.out), rows)
    return 0


def cmd_kernels(args) -> int:
    from .profile import kernel_table, render_kernel_table

    device = _device()
    m, n = args.shape.lower().split("x")
    rows, wall = kernel_table(args.impl, Shape(int(m), int(n)), DTYPE_BY_NAME[args.dtype], device, kind=args.kind)
    print(f"# {args.impl} {args.shape} {args.dtype}")
    print(render_kernel_table(rows, wall))
    return 0


def cmd_target(args) -> int:
    from .profile import run_target

    device = _device()
    m, n = args.shape.lower().split("x")
    run_target(args.impl, Shape(int(m), int(n)), DTYPE_BY_NAME[args.dtype], device, kind=args.kind,
               warmup=args.warmup, iters=args.iters)
    return 0


def cmd_env(args) -> int:
    print(json.dumps(environment(_device(require=False)), indent=2))
    return 0


def cmd_list(args) -> int:
    for title, reg in (("rmsnorm", IMPLEMENTATIONS), ("residual", RESIDUAL_IMPLEMENTATIONS)):
        print(f"{title}:")
        for impl in reg.values():
            print(f"  {impl.name:<26} [{impl.family}] {impl.description}")
    return 0


def _add_shape_args(p):
    p.add_argument("--matrix", choices=sorted(MATRICES), default="quick")
    p.add_argument("--shape", action="append", help="MxN, repeatable; overrides --matrix")
    p.add_argument("--dtype", action="append", default=None, help="bf16|fp16|fp32, repeatable (default bf16)")
    p.add_argument("--kind", choices=["rmsnorm", "residual"], default="rmsnorm")
    p.add_argument("--impl", action="append", default=None, help="implementation name, repeatable (default all)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="rmsnorm", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("bench", help="time implementations over a shape matrix")
    _add_shape_args(b)
    b.add_argument("--cache", action="append", default=None, choices=["cold", "warm"], help="L2 state (default cold)")
    b.add_argument("--rounds", type=int, default=5)
    b.add_argument("--iters", type=int, default=100)
    b.add_argument("--warmup", type=int, default=25)
    b.add_argument("--include-compile", action="store_true", help="also time torch.compile (one compile per shape)")
    b.add_argument("--out", default="results")
    b.set_defaults(fn=cmd_bench)

    r = sub.add_parser("report", help="render a run directory")
    r.add_argument("run_dir")
    r.set_defaults(fn=cmd_report)

    s = sub.add_parser("stats", help="registers, spills, occupancy per Triton variant")
    _add_shape_args(s)
    s.add_argument("--out", default=None, help="CSV path")
    s.set_defaults(fn=cmd_stats)

    k = sub.add_parser("kernels", help="torch.profiler kernel table for one call")
    k.add_argument("--impl", required=True)
    k.add_argument("--shape", required=True, help="MxN")
    k.add_argument("--dtype", default="bf16")
    k.add_argument("--kind", choices=["rmsnorm", "residual"], default="rmsnorm")
    k.set_defaults(fn=cmd_kernels)

    t = sub.add_parser("target", help="profiler target: run one implementation repeatedly")
    t.add_argument("--impl", required=True)
    t.add_argument("--shape", required=True, help="MxN")
    t.add_argument("--dtype", default="bf16")
    t.add_argument("--kind", choices=["rmsnorm", "residual"], default="rmsnorm")
    t.add_argument("--warmup", type=int, default=10)
    t.add_argument("--iters", type=int, default=20)
    t.set_defaults(fn=cmd_target)

    sub.add_parser("env", help="print environment").set_defaults(fn=cmd_env)
    sub.add_parser("list", help="list implementations").set_defaults(fn=cmd_list)

    args = ap.parse_args(argv)
    if hasattr(args, "dtype") and args.cmd in ("bench", "stats"):
        args.dtype = args.dtype or ["bf16"]
    if args.cmd == "bench":
        args.cache = args.cache or ["cold"]
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
