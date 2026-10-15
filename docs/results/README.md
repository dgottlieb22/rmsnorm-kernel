# Curated results

Files here are the ones the README and diary cite. Each says how it was
produced.

- `aot_stats_bf16.csv` / `.md`: compile-time resources per variant (v0, v1,
  v2, v3, v3b, v5 up to N = 1024, v4) and shape for sm_80 and sm_90, bf16,
  M = 4096, N in {512, 1000, 1024, 2048, 4096, 5120, 8192, 16384}, with the
  launcher's alignment hints and the kernels' ALIGN rescaling. Produced on a
  machine without a GPU by
  `python -m rmsnorm.cli aot --n 512 --n 1000 --n 1024 --n 2048 --n 4096 --n 5120 --n 8192 --n 16384 --out docs/results/aot_stats_bf16.csv`
  (Triton 3.4.0, bundled ptxas). Registers and spill bytes are ptxas output;
  global and shared-memory load/store widths are counted in the PTX;
  `layout_converts` counts `ttg.convert_layout` ops in the TTGIR (one can
  lower to nothing, see diary H14); occupancy and waves are computed from
  the per-SM limits in `rmsnorm/aot_stats.py::ARCHS`.
- `aot_stats_bf16_nohints.csv` / `.md`: v0 and v1 at N = 4096 and 16384 on
  sm_80 compiled without the pointer and integer divisibility hints
  (`--no-specialize`), the code a misaligned input gets. Diary H12.
- The IR behind diary H14 and H14b is not committed; regenerate it with
  `python -m rmsnorm.cli aot --impl triton_v3_multirow --n 512 --arch sm_80 --dump-ir <dir>`
  and `python scripts/probe_h14b_weight_load.py --ir`.
- `sass_mix_bf16_sm80.md`: executed SASS instructions per thread by class for
  the loop-free variants (v0, v1, v3, v3b, v5), sm_80, bf16, M = 4096, from
  `python scripts/sass_mix.py --impl triton_v0_naive --impl triton_v1_warps --impl triton_v3_multirow --impl triton_v3b_wtile --impl triton_v5_rowloop --n 512 --n 1000 --n 1024 --n 2048 --n 4096 --n 8192 --n 16384`.
  Static count up to the last EXIT; equal to the executed count for these
  kernels because none has a runtime loop. v2 is left out: its loop bodies
  appear once in the static count however many times they run.
- `h15_short_rows_n512.md`, `h15_short_rows_n1024.md`: the per-fix
  instruction counts behind diary H15, from
  `python scripts/probe_h15_short_rows.py --n 512 --rows 8` and `--n 1024`.

No timing or profiler data yet. Benchmark runs land under `results/` (not
committed); the ones the text relies on get copied here after being read.
