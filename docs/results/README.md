# Curated results

Files here are the ones the README and diary cite. Each says how it was
produced.

- `aot_stats_bf16.csv` / `.md`: compile-time resources per variant and shape
  for sm_80 and sm_90, bf16, M = 4096, with the launcher's alignment hints.
  Produced on a machine without a GPU by
  `python -m rmsnorm.cli aot --out docs/results/aot_stats_bf16.csv`
  (Triton 3.4.0, bundled ptxas). Registers and spill bytes are ptxas output;
  occupancy and waves are computed from the per-SM limits in
  `rmsnorm/aot_stats.py::ARCHS`.
- `aot_stats_bf16_nohints.csv` / `.md`: v0 and v1 at N = 4096 and 16384 on
  sm_80 compiled without the hints (`--no-specialize`), for diary H12.

No timing or profiler data yet. Benchmark runs land under `results/` (not
committed); the ones the text relies on get copied here after being read.
