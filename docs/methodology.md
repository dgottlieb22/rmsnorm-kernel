# Methodology

## Correctness before speed

Every implementation is compared with a float64 oracle
(`rmsnorm/reference.py`), not with another implementation, so a shared bug
cannot hide. The matrix (`tests/test_correctness.py`) covers:

- row lengths 1, 2, 3, 7, 16, 100, 512, 768, 1000, 1024, 2048, 2560, 4096,
  5120, and (marked slow) 8192, 11008, 16384. The non-powers of two exercise
  the masked lanes of block-wide kernels; 11008 is the Llama-2 7B MLP width;
- row counts 1 and 37 (37 is not a multiple of any block, so the multi-row
  kernel's tail mask runs); an empty batch; a 3-D `(batch, seq, hidden)` input;
- float32, float16, bfloat16 activations, and a float32 weight with bf16
  activations;
- input regimes: unit scale; `eps`-dominated (`mean(x^2) ~ eps`, where a
  kernel that drops `eps` is off by sqrt(2)); large values that overflow
  fp16 when squared or summed; all-zero rows (must give zeros, not NaN);
- layouts: a prefix slice of wider rows (unit stride along N, row stride 2N)
  and a transposed view (non-unit stride along N);
- determinism: two runs give identical bits.

Tolerances (`rmsnorm/tolerance.py`) are two machine epsilons of the output
dtype, relative and absolute. `tests/test_tolerance.py` holds mutants, plain
PyTorch RMSNorms with one real bug each, and asserts the harness rejects them:
dividing by the padded block width instead of N (caught at N = 5120, invisible
at N = 4096: hence the non-power-of-two shapes), dropping `eps` (caught only in
the eps-dominated regime), dropping the weight, summing in fp16 (produces
silent zeros, because `rsqrt(inf) = 0`, caught only with large inputs), and
summing instead of averaging. The same file shows that the baseline's single
rounding and Hugging Face's double rounding both pass, so the tolerance is as
tight as it can be without rejecting correct code.

On a machine without a GPU the Triton kernels run through Triton's CPU
interpreter. One artifact matters: the interpreter converts float32 to bf16 by
truncation (`triton/runtime/interpreter.py::_convert_float` with no rounding
mode), where the GPU rounds to nearest even. bf16 outputs under the
interpreter can therefore be one ulp low. The tolerance absorbs this; the
bit-exact residual-sum check pins the truncated value explicitly under the
interpreter and the rounded value on a GPU.

## Timing

Protocol, implemented in `rmsnorm/bench.py`:

1. Check the implementation against the oracle on the exact inputs about to
   be timed. Wrong output is reported as an error row; it is not timed.
2. Warm up: 25 calls (JIT compilation, autotuning caches, allocator).
3. For each of 5 rounds: 25 more warmup calls, then 100 timed calls. Each call
   is bracketed by its own pair of CUDA events. In `cold` mode a 256 MB buffer
   is zeroed before each call so the inputs are evicted from L2 and come from
   HBM. This is the conservative number, not a claim about what a forward
   pass sees: a norm's input is the residual stream the preceding kernel just
   wrote (32 MB at 4096 x 4096 bf16, against a 50 MB L2 on an A100 or H100),
   and how much of it is still resident depends on the shape and on what ran
   in between, which this harness does not measure. `warm` mode skips the
   flush; the two together bound the real case.
4. Report the median over the 500 samples, the interquartile range, the
   minimum, and the spread of the 5 per-round medians. A spread above a few
   percent means the number is not stable (clock changes, other tenants) and
   the run is repeated.
5. Host cost: wall time per call when the GPU is idle and nothing
   synchronizes, measured over 100 calls. This is the Python wrapper plus the
   launch. It is reported next to the GPU time; where it is larger, the
   operation is launch-bound and the GPU time alone overstates what a caller
   would see.
6. Batched time: 100 back-to-back calls under one event pair, divided by 100.
   With launches overlapping execution this approaches the kernel's execution
   time when the host keeps up, and the host's issue rate when it does not.

Effective bandwidth is the traffic model from `performance-model.md`
(each input read once, each output written once, the weight once) divided by
the median GPU time. It is reported as GB/s and as a fraction of a
device-to-device copy of 1 GiB measured at the start of the run; that copy is
the roofline. Spec-sheet bandwidth is recorded for reference only.

Throughput is reported as elements per microsecond (rows x hidden per
microsecond). Speedup is baseline median / implementation median against
`torch_eager` for the plain operation and against the eager add-then-norm pair
for the fused one.

## Things this harness refuses to do

Each of these produces a flattering and wrong number:

- Timing with a warm L2 and calling it bandwidth. A 2 MB working set never
  leaves the 50 MB L2; the "bandwidth" reported is L2 bandwidth, not HBM's.
  How far off that is for this operation is H10. The harness defaults to cold
  and reports warm separately so the difference is visible.
- Timing with the host clock and no synchronization: measures launch cost,
  not the kernel.
- Including the first call: measures compilation.
- One round of a few iterations: no estimate of variance.
- One convenient shape: the dispatcher's thresholds exist because the winner
  changes with N; a single shape hides that.
- Comparing only with a slow baseline. The eager float32 chain is the
  starting point, not the bar. `torch.compile`, `F.rms_norm`, and the vendor
  kernels (vLLM, Liger, flash-attn, when installed) are timed on the same
  matrix, and the report prints them whether or not they win.
- Reporting speedups without absolute times and bandwidth. A 5x speedup at
  M = 1 could be, say, 40 us of eager launches against 8 us for one launch
  (illustrative numbers, not measured), and would say nothing about the
  kernel.
- Dropping the Python overhead. At small shapes the wrapper is the cost; it
  is measured and reported.

## Profiling

`rmsnorm.cli kernels` lists the CUDA kernels one call launches with their
device time (via `torch.profiler`), which shows the eager baseline's chain of
short kernels against the fused kernels' single launch. `scripts/nsys.sh`
records a timeline of one implementation at one shape inside an NVTX range
and prints the kernel summary; the gaps between the eager kernels on the
timeline are the launch overhead. `scripts/ncu.sh` runs Nsight Compute with
the full section set on the same NVTX range and `scripts/ncu_summary.py`
reduces the export to the metrics `performance-model.md` section 5 reads:
DRAM throughput and bytes, occupancy and its limiter, registers, sectors per
request, local memory traffic (spills), and stall reasons.
`rmsnorm.cli stats` reads registers, spills and shared memory straight from
the compiled Triton kernels and computes theoretical occupancy and waves for
every variant and shape without a profiler.
