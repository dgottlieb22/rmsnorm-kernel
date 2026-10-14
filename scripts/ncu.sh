#!/usr/bin/env bash
# Profile one implementation at one shape with Nsight Compute.
#
#   scripts/ncu.sh <impl> <MxN> [dtype=bf16] [kind=rmsnorm] [launch_count=3]
#
# The target (rmsnorm.cli target) warms up, then runs 20 calls inside an NVTX
# range named <impl>_<MxN>_<dtype>. ncu is told to profile only kernels inside
# that range, so the eager baseline's several kernels and a Triton variant's
# single kernel are captured the same way. Output: results/ncu/<name>.ncu-rep
# and a CSV of the "details" page that scripts/ncu_summary.py turns into a
# markdown table of the metrics the README discusses.
set -euo pipefail
cd "$(dirname "$0")/.."

impl=${1:?impl}
shape=${2:?MxN}
dtype=${3:-bf16}
kind=${4:-rmsnorm}
count=${5:-3}
PY=${PY:-.venv/bin/python}

mkdir -p results/ncu
name="${impl}_${shape}_${dtype}"
range="${name}/"

ncu --nvtx --nvtx-include "${range}" \
    --launch-count "${count}" \
    --set full --import-source yes \
    -f -o "results/ncu/${name}" \
    "${PY}" -m rmsnorm.cli target --impl "${impl}" --shape "${shape}" --dtype "${dtype}" --kind "${kind}"

ncu -i "results/ncu/${name}.ncu-rep" --csv --page details > "results/ncu/${name}.details.csv"
"${PY}" scripts/ncu_summary.py "results/ncu/${name}.details.csv" | tee "results/ncu/${name}.summary.md"
