#!/usr/bin/env bash
# Everything that needs a GPU, in order. Run from the repo root on a CUDA host:
#
#   scripts/run_all.sh            # full matrix (bf16; add fp16/fp32 with DTYPES)
#   QUICK=1 scripts/run_all.sh    # quick matrix only
#
# Steps:
#   1. environment and tests (including the GPU-only tests)
#   2. benchmarks: rmsnorm and residual kinds, cold L2 (plus warm on the quick
#      matrix, to show what skipping the flush does), torch.compile included
#   3. kernel resource stats (registers, spills, occupancy) per Triton variant
#   4. torch.profiler kernel tables for the baseline and the dispatcher
#   5. Nsight Compute / Systems on the shapes the README discusses, if installed
# Outputs land under results/<timestamp>-<device>-...; copy the ones the README
# cites into docs/results/ by hand, after reading them.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
DTYPES=${DTYPES:-bf16}
dtype_args=()
for d in ${DTYPES}; do dtype_args+=(--dtype "$d"); done

echo "== environment"
"${PY}" -m rmsnorm.cli env

echo "== tests"
"${PY}" -m pytest -q

matrix=full
if [[ -n "${QUICK:-}" ]]; then matrix=quick; fi

echo "== bench rmsnorm (${matrix}, cold L2)"
"${PY}" -m rmsnorm.cli bench --matrix "${matrix}" "${dtype_args[@]}" --kind rmsnorm --include-compile --cache cold

echo "== bench rmsnorm (quick, warm vs cold L2)"
"${PY}" -m rmsnorm.cli bench --matrix quick "${dtype_args[@]}" --kind rmsnorm --cache cold --cache warm

echo "== bench residual (${matrix}, cold L2)"
"${PY}" -m rmsnorm.cli bench --matrix "${matrix}" "${dtype_args[@]}" --kind residual --include-compile --cache cold

mkdir -p results/stats
echo "== kernel stats"
"${PY}" -m rmsnorm.cli stats --matrix "${matrix}" "${dtype_args[@]}" --out results/stats/rmsnorm.csv | tee results/stats/rmsnorm.md
"${PY}" -m rmsnorm.cli stats --matrix "${matrix}" "${dtype_args[@]}" --kind residual --out results/stats/residual.csv | tee results/stats/residual.md

mkdir -p results/kernels
echo "== torch.profiler kernel tables"
for shape in 1x4096 64x4096 4096x4096 4096x512; do
  for impl in torch_eager torch_compiled triton_v0_naive triton; do
    "${PY}" -m rmsnorm.cli kernels --impl "${impl}" --shape "${shape}" --dtype bf16 | tee -a results/kernels/tables.md
    echo >> results/kernels/tables.md
  done
done

if command -v ncu >/dev/null 2>&1; then
  echo "== Nsight Compute"
  for shape in 4096x512 4096x4096 64x16384 1x4096; do
    for impl in triton_v0_naive triton_v1_warps triton_v2_looped triton_v3_multirow; do
      scripts/ncu.sh "${impl}" "${shape}" bf16 || echo "ncu failed for ${impl} ${shape}"
    done
  done
  scripts/ncu.sh torch_eager 4096x4096 bf16 || true
else
  echo "ncu not found; skipping Nsight Compute"
fi

if command -v nsys >/dev/null 2>&1; then
  echo "== Nsight Systems"
  for impl in torch_eager torch_compiled triton; do
    scripts/nsys.sh "${impl}" 1x4096 bf16 || echo "nsys failed for ${impl}"
    scripts/nsys.sh "${impl}" 4096x4096 bf16 || echo "nsys failed for ${impl}"
  done
else
  echo "nsys not found; skipping Nsight Systems"
fi

echo "== done; results under results/"
