#!/usr/bin/env bash
# Trace one implementation at one shape with Nsight Systems.
#
#   scripts/nsys.sh <impl> <MxN> [dtype=bf16] [kind=rmsnorm]
#
# Captures CUDA and NVTX activity only inside the target's NVTX range, then
# prints the per-kernel summary (count, average, min, max) and the GPU trace.
# For the eager baseline the trace shows the chain of short kernels and the
# gaps between them; that gap is the launch overhead the fused kernels remove.
set -euo pipefail
cd "$(dirname "$0")/.."

impl=${1:?impl}
shape=${2:?MxN}
dtype=${3:-bf16}
kind=${4:-rmsnorm}
PY=${PY:-.venv/bin/python}

mkdir -p results/nsys
name="${impl}_${shape}_${dtype}"

nsys profile -t cuda,nvtx --capture-range=nvtx --nvtx-capture "${name}" \
    --capture-range-end=stop -f true -o "results/nsys/${name}" \
    "${PY}" -m rmsnorm.cli target --impl "${impl}" --shape "${shape}" --dtype "${dtype}" --kind "${kind}"

nsys stats --report cuda_gpu_kern_sum --format csv --output "results/nsys/${name}" "results/nsys/${name}.nsys-rep"
nsys stats --report cuda_gpu_trace --format csv --output "results/nsys/${name}" "results/nsys/${name}.nsys-rep"
nsys stats --report cuda_gpu_kern_sum "results/nsys/${name}.nsys-rep"
