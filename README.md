# rmsnorm-kernels: RMSNorm from a PyTorch baseline to Triton, one change at a time

A GPU performance engineering study of one transformer inference operation,
RMSNorm, taken from a straightforward PyTorch implementation through
`torch.compile` to a sequence of hand-written Triton kernels, each differing
from the previous one by a single change. The deliverable is not a speedup
number. It is being able to say, for every implementation and every shape,
why it runs as fast as it does, and to show the measurement that supports
the claim.

Status, stated plainly: the implementations, the correctness suite (1141
tests; 994 run on a CPU-only host, the rest need a GPU or a working
`torch.compile`), the benchmark harness, the profiling tooling and the
performance model are complete. This was built on a machine without a GPU;
the Triton kernels were verified through Triton's CPU interpreter, and their
register, spill and occupancy figures were obtained by compiling them offline
for sm_80 and sm_90 (section 5). No benchmark or profile has been run. Every
section that needs measured timings says PENDING HARDWARE, and
`docs/diary.md` holds the hypotheses written down before the measurements so
that the predictions cannot be adjusted after the fact (one already had to
be corrected by the compile-time data). `scripts/run_all.sh` produces
everything on a CUDA host.

## 1. The operation and why it matters

RMSNorm (Zhang and Sennrich, 2019) normalizes each row of activations by its
root mean square and rescales each feature with a learned weight:

```
y[i] = x[i] / sqrt(mean_j(x[j]^2) + eps) * w[i]
```

Unlike LayerNorm it subtracts no mean, so a row needs one reduction, not
two. Llama, Mistral, Gemma, Qwen and most current decoder models use it
twice per layer (before attention and before the MLP), and in a decoder
layer it follows the residual add, so production servers fuse
`residual + x` into it (vLLM's `fused_add_rms_norm`).

It matters for inference for two reasons that pull in opposite directions.
It is small: 4 float32 operations per element against 4 bytes of traffic in
bf16, so it is memory-bound on any GPU by a factor of 10-20 (see
`docs/performance-model.md`), and a good kernel simply streams the row at
HBM bandwidth. And it runs at every shape the model sees: at decode (one
token per sequence, M in the tens) it is a few kilobytes and the launch is
the cost; at prefill (M in the thousands) it is tens of megabytes and the
memory system is the cost. The same kernel moves between regimes as the
shape changes, which is what makes it a good object of study: a kernel that
is "fast" at one shape and the explanation for why it is slow at another are
both one page of code.

## 2. Implementations

All of them share one contract: `f(x, weight, eps) -> y` with `x` of shape
`(..., N)`, any dtype in {fp32, fp16, bf16}, output in x's dtype; the fused
form is `f(x, residual, weight, eps) -> (y, x + residual)`. The registry in
`rmsnorm/__init__.py` lists them; tests and the benchmark iterate over it.

### Baseline: eager PyTorch (`torch_eager`)

```python
def rmsnorm_torch_eager(x, weight, eps=1e-6):
    h = x.to(torch.float32)
    mean_sq = h.pow(2).mean(dim=-1, keepdim=True)
    h = h * torch.rsqrt(mean_sq + eps)
    return (h * weight.to(torch.float32)).to(x.dtype)
```

Correct, readable, and the way most model code is written. Every line is a
separate CUDA kernel and every intermediate is a full-size float32 tensor
that goes out to HBM and comes back. The profiler section counts the kernels
and the bytes.

Two more PyTorch baselines sit beside it: `torch_eager_hf`, the Hugging Face
`LlamaRMSNorm` ordering (round to bf16, then multiply by the weight: two
roundings), and `torch_rms_norm`, `torch.nn.functional.rms_norm` from
torch >= 2.4, the library's own implementation.

### torch.compile (`torch_compiled`)

`torch.compile(rmsnorm_torch_eager)`. On CUDA, TorchInductor emits a Triton
kernel for the fused graph, so this comparison is Triton against Triton: the
compiler's generic reduction template against kernels written for this one
operation.

### Triton kernels (`rmsnorm/triton_impls.py`)

Each variant is a separate kernel so the diary can attribute every change in
performance to one change in code.

- `triton_v0_naive`: one program per row, the whole row in one block of
  `next_power_of_2(N)` lanes, 4 warps for every N. The first kernel anyone
  writes; its failure modes at N = 512 and N = 16384 are the first two
  diary entries.

  ```python
  @triton.jit
  def _rmsnorm_row_kernel(x_ptr, w_ptr, y_ptr, stride_xm_div, stride_ym_div, N_div, eps,
                          BLOCK_N: tl.constexpr, ALIGN: tl.constexpr):
      stride_xm = stride_xm_div * ALIGN   # strides and N arrive divided by ALIGN;
      stride_ym = stride_ym_div * ALIGN   # the product makes their divisibility
      N = N_div * ALIGN                   # visible to the compiler (diary H12)
      row = tl.program_id(0).to(tl.int64)
      cols = tl.arange(0, BLOCK_N)
      mask = cols < N
      x = tl.load(x_ptr + row * stride_xm + cols, mask=mask, other=0.0).to(tl.float32)
      w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
      mean_sq = tl.sum(x * x, axis=0) / N
      rstd = tl.rsqrt(mean_sq + eps)
      tl.store(y_ptr + row * stride_ym + cols, (x * rstd * w).to(y_ptr.dtype.element_ty), mask=mask)
  ```

- `triton_v1_warps`: the same kernel with `num_warps` chosen so each thread
  owns about 8 elements (one 16-byte load). Thread mapping only.
- `triton_v2_looped`: the row processed in 4096-wide chunks in two passes
  (sum of squares, then normalize). Register use no longer grows with N; the
  second pass re-reads the row from L2.
- `triton_v3_multirow`: several short rows per program as a 2-D block; the
  weight is loaded once per program and every thread has more independent
  loads in flight. The once-per-program weight load is narrow (one element
  per thread at N = 512) and the compiler moves it through shared memory
  into the tile's layout, with two extra barriers; diary H14 traces that in
  the IR and H14b compiles the alternatives.
- `triton_v3b_wtile`: v3 with the weight read as a `[BLOCK_M, BLOCK_N]` tile
  (row stride 0), so it arrives in the x tile's layout: no shared-memory
  layout change, two fewer barriers, BLOCK_M times the weight-load requests
  from L1. Same arithmetic as v3, bit-identical output; which is faster is
  H6c.
- `triton_v5_rowloop`: one warp per program, ROWS rows one after another,
  weight loaded and converted once, 1/N from the host, no mask when the shape
  allows it. Built from the SASS counts in diary H15: a one-warp row has no
  cross-warp reduction (no shared memory, no barrier), and the loop amortizes
  the per-program prologue. 7.8 instructions per element at 4096 x 512
  against v1's 16.0; the price is a grid of M / ROWS warps and 50% (block
  limit) or 39% (registers, N = 1024) theoretical occupancy. N <= 1024.
- `triton_v4_fused_residual`: the residual add inside the row kernel, writing
  both the normalized output and the new residual stream. The sum is rounded
  to the storage dtype before it is normalized so the result matches the
  unfused add-then-norm sequence bit for bit on the residual and within
  tolerance on the output.
- `triton`: the dispatcher. Multi-row for N <= 1024, looped for N >= 8192,
  single row in between. The thresholds are the model's guesses until the
  diary's measurements replace them.

All kernels square and accumulate in float32 whatever the storage dtype, and
every load is masked to N so non-power-of-two hidden sizes (5120, 11008)
are handled without padding the tensors. Strides and N are passed divided by
a compile-time `ALIGN` and multiplied back inside the kernel, so the compiler
can prove 16-byte alignment for hidden sizes like 1000 that the launcher's
own divisible-by-16 hint misses (section 5 and diary H12 show what that
changes in the generated code).

### Vendor kernels (`rmsnorm/external.py`)

vLLM's CUDA `rms_norm` and `fused_add_rms_norm`, Liger Kernel's Triton
RMSNorm, and flash-attn's `rms_norm_fn` are registered automatically when
their packages import. They appear in the same tables as everything else,
whether or not they win.

## 3. Correctness

Details in `docs/methodology.md`. In short: every implementation is checked
against a float64 oracle, not against another implementation; the matrix
covers 17 row lengths from 1 to 16384 including non-powers of two, row
counts that exercise tail masks, three dtypes, mixed weight dtype, strided
and transposed layouts, empty batches, 3-D inputs, all-zero rows, and three
input regimes (unit scale, `eps`-dominated, fp16-overflowing). Tolerances
are two machine epsilons of the output dtype.

The tolerance is shown to have teeth, not asserted to: `tests/test_tolerance.py`
runs five RMSNorms with one real bug each and checks the harness rejects
them. These are results obtained on this machine, not predictions:

| mutant | caught by | would slip through on |
|---|---|---|
| mean over the padded block width instead of N | N = 5120 | N = 4096 (block == N) |
| `eps` dropped | the `eps`-dominated regime (`mean(x^2) ~ 1e-6`) | unit-scale inputs: the error is below bf16 resolution |
| weight dropped | any random weight | |
| square and sum in fp16 | large inputs: `x*x` overflows, `rsqrt(inf) = 0`, the row comes out as silent zeros | unit-scale inputs |
| sum instead of mean | any N > 1 | |

Two numerics findings from the same suite, also measured here (CPU, torch
2.8.0): `F.rms_norm` multiplies by the weight before rounding to the input
dtype and matches the single-rounding baseline bit for bit, and the Hugging
Face ordering (round first, then multiply) differs from it in about 26% of
bf16 and fp16 outputs on random data. Both orderings are inside the
tolerance; a kernel that promises to match Hugging Face bit for bit has to
pick the HF order.

One tooling finding: Triton's CPU interpreter converts float32 to bf16 by
truncation, where the GPU rounds to nearest even. Anyone validating bf16
kernels through `TRITON_INTERPRET=1` sees results up to one ulp low. The
suite pins the truncated value under the interpreter and the rounded value
on a GPU rather than loosening the tolerance.

## 4. Benchmark methodology

Full text in `docs/methodology.md`; implementation in `rmsnorm/bench.py`.

- Correctness check on the exact inputs before any timing; a wrong kernel
  becomes an error row, never a number.
- CUDA events around each call; 5 rounds x 100 timed calls after warmup;
  median, interquartile range, minimum, and the spread of per-round medians
  (a stability check).
- Cold L2 by default: a 256 MB buffer is written before every call so the
  inputs come from HBM. Warm mode is reported separately to show what the
  flush changes.
- Host launch cost (wall time per call, no synchronization) and batched GPU
  time (100 back-to-back calls under one event pair) next to the GPU time,
  so launch-bound shapes are identified instead of flattered.
- Effective bandwidth from the operation's unavoidable traffic (inputs once,
  outputs once, weight once), as GB/s and as a fraction of a device-to-device
  copy measured on the same GPU at the start of the run. Spec-sheet
  bandwidth is recorded but not used as the roofline.
- Shape matrix: M in {1, 8, 64, 512, 4096, 16384} x N in {512, 768, 1000,
  1024, 2048, 4096, 5120, 8192, 16384}, bf16 by default, fp16 and fp32 on
  request. M = 1 and 8 are decode, 64 and 512 small batches, 4096 and 16384
  prefill; N spans small models to Llama 70B's 8192, two non-powers of two,
  and one size (1000) whose row stride is not a multiple of 16 bytes.

## 5. Results

### Measured here: compile-time resources (no GPU needed)

Triton compiles on the CPU and ships NVIDIA's `ptxas`, so every variant can be
compiled for a target SM with the same argument specialization the launcher
applies, and registers, spills, shared memory and theoretical occupancy read
off the assembler (`python -m rmsnorm.cli aot`, module `rmsnorm/aot_stats.py`).
These are facts about the generated code, not timings. sm_80 (A100), bf16,
M = 4096; the full table for sm_80 and sm_90 is `docs/results/aot_stats_bf16.md`.

| N | v0 naive (4 warps) | v1 warps | v2 looped (4096 chunks) | v3 multirow | v3b wtile | v5 rowloop (1 warp) |
|---|---|---|---|---|---|---|
| 512 | 20 regs, 100% occ, 64-bit accesses | 2 warps, 24 regs, 100%, 128-bit | 32 regs, 100% | 8x512 block, 16 warps, 27 regs, 100%, grid 512, 1 KB smem | 30 regs, 100%, 64 B smem | 2 rows, 48 regs, 50% (block limit), 0 smem |
| 1000 | 26 regs, 100%, 128-bit (ALIGN 8) | 4 warps, 26 regs, 100% | 32 regs, 100% | 4x1024, 28 regs, 100%, 2 KB smem | 29 regs, 100%, 64 B | 2 rows, 72 regs, 44%, masked |
| 1024 | same as 1000 | same as 1000 | 32 regs, 100% | same as 1000 | same as 1000 | 2 rows, 80 regs, 39% |
| 2048 | 38 regs, 75% | 8 warps, 26 regs, 100% | 32 regs, 100% | 2x2048, 29 regs, 100%, 4 KB smem | 28 regs, 100%, 64 B | not covered |
| 4096 | 62 regs, 50% | 16 warps, 26 regs, 100% | 32 regs, 100% | 1x4096, 28 regs, 100% | same as v3 | not covered |
| 5120 / 8192 | 127 regs, 25% | 16 warps, 38 regs, 75% | 32 regs, 100% | 38 regs, 75% | 40 regs, 75% | not covered |
| 16384 | 218 regs, 12.5% | 16 warps, 63 regs, 50% | 32 regs, 100% | 63 regs, 50% | same as v3 | not covered |

Executed SASS instructions per element (static count, loop-free kernels;
`scripts/sass_mix.py`, full table `docs/results/sass_mix_bf16_sm80.md`):

| N | v0 | v1 | v3 | v3b | v5 |
|---|---|---|---|---|---|
| 512 | 25.5 | 16.0 | 17.5 | 16.6 | 7.8 |
| 1024 | 16.2 | 16.2 | 17.9 | 16.9 | 7.0 |
| 4096 | 9.8 | 16.8 | 16.1 | 16.1 | |
| 16384 | 8.5 | 10.1 | 10.1 | 9.9 | |

Every access in this table is a 128-bit vector per thread except where
marked. No variant spills at any shape in the matrix. Six things this
already settles (diary H3, H4, H12, H13, H14, H15):

- The naive kernel's problem on long rows is register-capped occupancy, not
  spilling: 218 registers per thread at N = 16384 leaves room for 2 programs
  of 4 warps per SM. The model's first guess (spills) was wrong and is
  corrected in the diary, not erased.
- The 8-elements-per-thread heuristic holds registers at 26 wherever it can
  reach 8; once the 16-warp cap forces 16 and 32 elements per thread
  (N >= 8192) occupancy drops to 75% and 50%. The looped kernel is flat at 32
  registers and 100% at every N.
- Alignment hints are part of the kernel. The launcher marks 16-byte-aligned
  pointers and integers divisible by 16; with them `v1` at N = 4096 is two
  128-bit loads and one 128-bit store per thread, without them (a misaligned
  pointer) it is sixteen 16-bit loads and eight 16-bit stores, and at
  N = 16384 `v0` goes from 218 registers to 255 with 240 bytes of spills.
  A hidden size of 1000, whose stride is not divisible by 16, originally
  compiled to 16-bit accesses plus a 2 KB shared-memory layout conversion.
  One change fixed it: the wrappers pass strides and N divided by a
  compile-time `ALIGN` (the largest power of two up to 16 dividing them) and
  the kernel multiplies back, which the compiler's divisibility analysis can
  see. N = 1000 now compiles to the same code as N = 1024. `tl.multiple_of`
  on the scalar arguments was tried first and changed nothing.
- `v0` at N = 512 issues 64-bit accesses because each of its 128 threads
  owns only 4 elements; `v1` drops to 2 warps there so every thread owns 8
  and the accesses are 128-bit. Whether that matters at 1 KB per row is a
  timing question.
- The multirow kernel's 1 to 4 KB of shared memory is the weight vector. Its
  1-D load is spread one element per thread across all 512 threads (a
  16-bit load at N = 512), and the compiler moves it through shared memory
  (2 bytes x N) into the 2-D tile's layout: one extra store and load per
  thread and two extra barriers per program, read off the TTGIR and PTX
  (diary H14). Loading the weight as a [1, N] block does not remove the
  conversion and doubles the shared memory; loading it as a full
  [BLOCK_M, N] tile removes it at the cost of BLOCK_M times the weight-load
  requests from L1 (H14b). Which is faster is a timing question, so the
  tile load is a candidate variant, not a change to v3.
- On short rows the instruction stream is mostly fixed cost, and the
  multirow design does not touch it. Reading the SASS of v1 at N = 512: 128
  instructions per thread for 8 elements, of which about 65 are prologue,
  cross-warp reduction (shared memory and two barriers) and the division by
  N, and 8 per element are the work. v3's thread does the same work plus the
  layout change, 17.5 per element against v1's 16.0, so H6's predicted 1.3-2x
  has no mechanism and is withdrawn. Three fixes measured one at a time:
  1/N from the host (-7 instructions), no mask when N is a power of two
  (-15), one warp per row (-31% per element, no shared memory, no barrier).
  A loop over rows in that one warp amortizes the rest: 6.6 per element at 8
  rows, 7.8 at the 2 rows that keep 2048 warps in flight at M = 4096. That
  shipped as `triton_v5_rowloop` (diary H15); the dispatcher waits for the
  timing. Hand-unpacking the bf16 pairs saved 1.6% and was not adopted.

### Pending hardware: timing and profiles

`scripts/run_all.sh` writes `results/<run>/report.md` with,
per dtype and cache mode and per N, a table of M rows x implementations
(median us, GB/s, % of copy bandwidth), a speedup table against the eager
baseline, a host-vs-GPU time table at the smallest M, and plots of bandwidth
against N for each M. Curated copies the text below refers to will live
under `docs/results/`.

The sections that will be filled in from those files:

- Latency and bandwidth table, 4096 x N for all N, all implementations.
- Bandwidth against M at N = 4096: where each implementation leaves the
  launch-bound regime and where it reaches the roofline.
- The N = 512, 1000 and 5120 columns against their neighbors (H6, H7, H12).
- Fused residual against the unfused pair and against vLLM (H9).
- Cold against warm L2 (H10).
- Where PyTorch, Inductor or a vendor kernel is faster than ours, with the
  profile that says why.

## 6. Profiling

PENDING HARDWARE for the runtime data; the tooling is in place, and one
tool already produces data here:

- `python -m rmsnorm.cli aot --arch sm_80 --arch sm_90`: compiles every
  variant offline for the target SM with the launcher's specialization,
  runs the bundled `ptxas -v`, and reports registers, spill bytes, shared
  memory, programs per SM, theoretical occupancy with its limiter, waves,
  and the global and shared-memory accesses in the PTX by width.
  `--no-specialize` shows the kernel a misaligned input gets; `--dump-ir DIR`
  keeps each compile's TTGIR and PTX for reading. Results in section 5.
- `python scripts/sass_mix.py --impl <variant> --n <N>`: assembles the PTX
  with the bundled `ptxas`, disassembles with `nvdisasm`, and counts the
  executed SASS instructions per thread by class (loads and stores, shared
  memory, barriers, shuffles, converts, floating point, integer and
  predicate, uniform). Instructions per element is the compile-time proxy for
  the instruction-bound regime (diary H15).
- `python -m rmsnorm.cli kernels --impl torch_eager --shape 4096x4096`:
  CUDA kernels launched per call with their device time (`torch.profiler`).
  For the baseline this is the count of separate kernels and the difference
  between device time and wall time per call.
- `python -m rmsnorm.cli stats --matrix full`: registers, spills and shared
  memory read from the compiled Triton kernels, plus theoretical occupancy,
  its limiter (registers, warps, blocks or shared memory) and the number of
  waves the grid needs on this device.
- `scripts/nsys.sh <impl> <MxN>`: Nsight Systems timeline of the
  implementation's NVTX range and a kernel summary; the gaps between the
  eager baseline's kernels are the launch overhead.
- `scripts/ncu.sh <impl> <MxN>`: Nsight Compute with the full section set on
  the same NVTX range, reduced by `scripts/ncu_summary.py` to DRAM throughput
  and bytes, L2 throughput, SM throughput, achieved and theoretical
  occupancy and limiters, registers per thread, global load sectors per
  request (coalescing), local memory traffic (spills), shared memory, and
  warp stall reasons.

`docs/performance-model.md` section 5 says in what order to read those
numbers and what each one implies for this operation.

## 7. Performance model

`docs/performance-model.md`. The short version: RMSNorm moves 4 bytes per
element in bf16 and does 4 float32 operations on it, an arithmetic intensity
of about 1 flop per byte against ridge points of 10 (A100) to 20 (H100)
flop per byte, so it is memory-bound by construction. The time floor is
bytes / achievable bandwidth: 67 MB for 4096 x 4096, about 38 us on an A100
and 23 us on an H100 at typical copy bandwidth. Where a kernel does not
reach the floor, the model names four regimes and what to measure in each:
bandwidth-bound (coalescing, redundant traffic), parallelism-bound (bytes in
flight per Little's law: occupancy, loads per thread, grid smaller than the
machine), launch-bound (host issue cost against a kernel of a few
microseconds), and instruction-bound (short rows where fixed per-program
work and masked lanes dominate). Section 4 of that document predicts, per
variant, where each should win and lose; the diary checks every prediction.

## 8. Failed and abandoned experiments

None yet; nothing has been run. `docs/diary.md` has a section reserved for
them, and the candidates not yet built (autotuning, a CUDA kernel, a
split-row kernel for small M with long rows) are listed there with the
measurement that would justify each.

## 9. Conclusions

Pending the measurements. What can be said from the model alone: the
operation cannot be compute-bound on current GPUs; the eager baseline's
cost is launches and float32 intermediates, not arithmetic; the whole-row
Triton design is the right one for N between about 1024 and 16384 and
the wrong one below (per-program overhead) and above (registers); fusing
the residual add is worth at most 1.25x in the bandwidth-bound regime; and
at decode shapes no kernel change matters as much as the number of launches
per layer.

## 10. Interview questions this project answers

| question | where the answer is |
|---|---|
| Why is this kernel memory-bound? | model section 1 (1 flop/byte vs a 10-20 flop/byte ridge); ncu DRAM throughput vs SM throughput (H3) |
| What changed when you altered the tile size? | H4 (warps per row), H5b (chunk width), H6b (rows per program); `stats` output for registers and occupancy |
| What does occupancy mean here? | model section 3 (resident warps vs the limit, which resource binds, Little's law for bytes in flight); `stats` limiter column |
| Why is your kernel faster at hidden size 4096 but slower at 512? | model section 3 (instruction-bound regime), H3 and H6 |
| Why does the naive kernel degrade at N = 16384? | H3, measured at compile time: 218 registers per thread caps residency at 2 programs x 4 warps per SM (12.5% occupancy); no spills. The timing half is pending |
| Why do alignment hints matter, and what did you do about them? | H12, measured at compile time: with the hints each thread does 128-bit accesses; without them 16-bit scalars, 255 registers and spills on the longest row. Passing strides and N divided by a compile-time ALIGN restored 128-bit accesses for N = 1000 |
| When does fusion help and by how much? | model section 1 (4 vs 5 bytes per element), H9 |
| Why not trust a warm-cache benchmark? | H10 and the methodology |
| Why is M = 1 slow for every implementation? | H11: launch-bound; host time vs GPU time table |
| How do you know your tolerance is right? | section 3: mutants that fail, correct orderings that pass |

## 11. How to run

CPU (correctness only; Triton kernels run through the interpreter):

```
make venv-cpu      # torch 2.8.0+cpu from the PyTorch CPU index, triton 3.4.0
make test          # 994 pass here; torch.compile tests skip when the host cannot build C++ extensions
```

GPU:

```
make venv-gpu
make test          # includes tests/test_gpu.py
scripts/run_all.sh # tests, benchmarks (rmsnorm and residual, cold and warm), stats, kernel tables, ncu/nsys
QUICK=1 scripts/run_all.sh
DTYPES="bf16 fp16 fp32" scripts/run_all.sh
```

Single commands: `python -m rmsnorm.cli --help`; `bench`, `report`, `stats`,
`kernels`, `target`, `env`, `list`.

## 12. Layout

```
rmsnorm/reference.py      float64 oracle (plain and fused-residual)
rmsnorm/torch_impls.py    eager, HF ordering, F.rms_norm, torch.compile
rmsnorm/triton_impls.py   kernels v0-v4, dispatcher, launch recording for stats
rmsnorm/external.py       vLLM / Liger / flash-attn adapters (optional)
rmsnorm/tolerance.py      dtype-derived tolerances and error metrics
rmsnorm/bench.py          timing protocol, traffic model, copy roofline, CSV
rmsnorm/report.py         markdown tables and plots from a run directory
rmsnorm/kernel_stats.py   registers / spills / occupancy / waves per variant (on a GPU)
rmsnorm/aot_stats.py      the same from offline compilation + ptxas, for a chosen SM, no GPU needed
rmsnorm/profile.py        nsys/ncu target, torch.profiler kernel table
rmsnorm/cli.py            command line
tests/                    correctness matrix, mutants, numerics pins, harness, GPU-only
scripts/                  run_all.sh, ncu.sh, nsys.sh, ncu_summary.py
docs/performance-model.md predictions and the regimes
docs/methodology.md       how correctness and timing are done, and what is refused
docs/diary.md             one hypothesis per experiment; compile-time results in; timings pending hardware
docs/results/             curated data the text cites, each file with its provenance
```
