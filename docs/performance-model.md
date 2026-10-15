# Performance model for RMSNorm

This document predicts, before any measurement, how fast each shape can run
and what will limit it. The diary (`diary.md`) checks these predictions on
hardware; where they are wrong the model gets corrected, not the data.

Hardware figures below are vendor spec-sheet values used for orientation. The
benchmark compares against a device-to-device copy measured on the same GPU,
which is the bandwidth a kernel can actually reach.

## 0. What this document assumes

Every number in sections 1-3 that is not spec-sheet arithmetic comes from this
table. Nothing in it has been measured by this project; the last column says
which diary entry will.

| assumption | value used | status |
|---|---|---|
| HBM bandwidth and fp32 rate | A100 80 GB: 2039 GB/s, 19.5 TFLOP/s; H100 SXM: 3352 GB/s, 67 TFLOP/s; L4: 300 GB/s | vendor spec sheets |
| achievable copy bandwidth | 85-90% of spec | assumed; the benchmark measures it at the start of every run and uses the measured value as the roofline |
| DRAM latency | 600-800 ns | assumed, uncited; sets the bytes-in-flight target in section 3 and the "16 warps per SM" figure in diary H15 |
| outstanding 16-byte loads per thread | "a few" | assumed; the compile-time tables show loads per thread in the PTX, not how many are in flight |
| kernel launch cost | a few us of host time, a few us of GPU-side ramp | assumed; H11 measures host time per call |
| Triton wrapper host cost | tens of us | assumed; H11 |
| eager kernels per call | 8-10 | assumed; H1 counts them |
| L2 bandwidth relative to HBM | 2-3x | assumed; H10 measures the warm-vs-cold ratio for this operation |
| bandwidth-bound threshold | traffic above ~100 MB with M in the thousands | rule of thumb |
| "done" threshold | 85-90% of measured copy bandwidth | a stopping rule for this project, not a fact about the hardware |

## 1. What the operation has to move

For an input of M rows and N columns in a storage dtype of `e` bytes
(bf16: 2), with a weight of N elements:

| quantity | per element | total |
|---|---|---|
| bytes read | `e` (x) | `M*N*e + N*e` (weight, once) |
| bytes written | `e` (y) | `M*N*e` |
| float32 ops | square, accumulate, scale, weight multiply: 4 | `4*M*N` + one rsqrt per row |

Traffic is `2*e` bytes per element, 4 B for bf16. Arithmetic intensity is
about 1 flop per byte. The ridge point of an A100 is 19.5 TFLOP/s (fp32,
non-tensor) / 2.0 TB/s = 10 flop/B; an H100 SXM is 67 TFLOP/s / 3.35 TB/s =
20 flop/B. RMSNorm sits 10 to 20 times below the ridge: it cannot be
compute-bound on these GPUs. When it fails to reach the bandwidth roofline,
the reason is somewhere else, and finding that reason is the project.

The fused residual-add variant moves `4*e` bytes per element (read x and r,
write y and s) against the unfused sequence's `5*e` (add: read 2, write 1;
norm: read 1, write 1). The best case for fusion is therefore a 20% reduction
in bytes, so at the bandwidth limit the fused kernel is at most 1.25x faster
than a perfect pair of kernels. Anything beyond that comes from removing a
launch and a short kernel, which only matters at small shapes.

## 2. Time floor

`t_min = bytes / BW_achievable`. For M = 4096, N = 4096, bf16: 67.1 MB.

| device | spec BW | copy BW at 85-90% of spec (assumed) | floor |
|---|---|---|---|
| A100 80 GB | 2039 GB/s | ~1750 GB/s | ~38 us |
| H100 SXM | 3352 GB/s | ~2900 GB/s | ~23 us |
| L4 | 300 GB/s | ~260 GB/s | ~260 us |

The stopping rule for this project: a kernel at 90% of the measured copy
bandwidth on this shape counts as done, and the remaining 10% is not chased.

## 3. The four regimes

Which limit applies depends on the shape, and the same kernel moves between
regimes as M and N change. This is the answer to "why is your kernel faster
at hidden size 4096 but slower at 512": the kernel did not change, the
regime did.

### Bandwidth-bound: total traffic well above the L2 and the grid fills the GPU

As a rule of thumb, `M*N*2e` above ~100 MB and M in the thousands. The floor
from section 2 applies. What matters: every load fully coalesced and as wide
as possible (16 B per thread), no redundant traffic (the weight read per row
should be served from L2, which is unmeasured, and costs L1/L2 bandwidth and
instructions either way), enough bytes in flight to cover DRAM latency (next
regime).

### Parallelism-bound: not enough bytes in flight

Little's law: to sustain bandwidth `BW` with memory latency `L`, the GPU
needs `BW * L` bytes in flight. Taking `L` as 600-800 ns (assumed, section
0), an H100 needs about 3.35 TB/s x 0.7 us = 2.3 MB in flight, roughly 18 KB
per SM. If a thread holds only a few outstanding 16-byte loads, the rest
has to come from having many warps resident. At 100% occupancy (64 warps,
2048 threads per SM) one 16 B load per thread is 32 KB per SM, enough; at
25% occupancy it is 8 KB. The model's expectation is that bandwidth then
drops unless each thread keeps several loads in flight. Whether it does is
open: diary H3 shows `triton_v0_naive` at 12.5% theoretical occupancy with
16 vector loads per thread at N = 16384, and only the timing can say whether
that costs anything. How many of the looped kernel's per-iteration loads the
compiler keeps in flight has not been read from its SASS.

Occupancy here means resident warps per SM divided by the maximum (64 on
A100/H100, 48 on consumer Ampere/Ada). It is limited by registers per
thread (65536 per SM: 64 registers x 256 threads x 4 programs = full),
by the warp limit (programs x warps per program <= 64), by the block limit
(32 per SM), and by shared memory. `rmsnorm.cli stats` reports which one
binds for each variant and shape.

The second parallelism failure is a grid smaller than the machine. One row
per program with M = 64 is 64 programs for 132 SMs: 68 SMs have no work, and
the model expects bandwidth to be capped near the fraction of SMs that do,
before anything else is considered; whether 64 programs can pull more than
that is H8. The multi-row
kernel makes this worse (fewer programs); the fix would be splitting a row
across programs, which needs a cross-program reduction. The diary tests
whether it is worth it (H8).

### Launch-bound: the host cannot issue work fast enough

The model takes a kernel launch as a few microseconds of host time and a few
microseconds of GPU-side ramp (assumed, section 0). For M = 1 (one decode
token through one layer), N = 4096 bf16 is 16 KB of traffic: 5 ns at
bandwidth. The whole call is overhead. If eager PyTorch runs eight to ten
kernels for RMSNorm (cast, square, mean, add, rsqrt, weight cast, two
multiplies, cast; `rmsnorm.cli kernels` counts them in H1), its floor is that
many launches, around 30-50 us on those assumptions. A fused kernel is one
launch, but its Python wrapper (shape checks, `reshape`, Triton's launcher)
is assumed to cost tens of microseconds; the benchmark records host time next
to GPU time (H11) so that this cost is visible rather than hidden behind a
GPU-only timer. Below a few hundred rows of N = 4096 the right comparison is
latency, not bandwidth, and the right optimization is fewer launches and a
thinner wrapper (or CUDA graphs around the whole layer, which is out of scope
here).

### Instruction-bound: too much work per byte that is not the operation

Short rows. With N = 512 and one row per 128-thread program, each thread
holds 4 elements; the fixed per-program work (index math, the cross-warp
reduction with its barrier, the weight load, the rsqrt) is amortized over
2 KB of traffic. The weight read is as large as the row itself, so the
kernel issues 1.5x the loads the operation needs. Masked lanes add to this:
for N = 5120 the block is 8192 wide and 37.5% of the lanes of every
per-element instruction are predicated off (the fixed per-program
instructions, which H15 measured at half the stream for 8 elements per
thread, are not). The multi-row kernel was written for this regime, to load
the weight once per BLOCK_M rows and shrink the grid; the SASS counts in H15
found it does not reduce per-thread work (section 4), and the row-loop kernel
`triton_v5_rowloop` is the one that does.

## 4. Per-variant predictions

| variant | expected to win | expected to lose | why |
|---|---|---|---|
| torch_eager | nowhere | everywhere | 8-10 kernels, each with full-size float32 intermediates: several times the traffic and the launches |
| torch_compiled | mid and large shapes, close to our kernels | M = 1 (same launch floor as any single kernel; guards add host time) | Inductor fuses the graph into one or two Triton kernels with reasonable defaults |
| triton_v0_naive | N in [1024, 4096], M large | N = 512 (too little per program); N >= 8192 (4 warps hold 64-128 elements per thread: 125-218 registers, so 4 then 2 programs per SM; the original guess was spills, and the offline compile in diary H3 showed register-capped occupancy instead, with spills starting at N = 32768) | fixed 4 warps regardless of N |
| triton_v1_warps | N in [2048, 8192] | N = 512 (still one row per program) | 8 elements per thread: one 16 B load each, room for occupancy |
| triton_v2_looped | N >= 8192 | N <= 4096 (two passes and loop overhead for no benefit) | bounded registers; second pass re-reads from L2 |
| triton_v3_multirow | N <= 1024 | large N (fewer programs, bigger blocks) | amortizes weight load and per-program overhead. Revised after H15: no per-thread saving over v1 (17.5 against 16.0 instructions per element at N = 512), expect within a few percent of v1 |
| triton_v3b_wtile | N <= 1024, by a few percent over v3 if its two extra barriers cost more than BLOCK_M re-reads of the weight from L1 | nowhere distinct from v3; identical code at BLOCK_M = 1 | same tile as v3, weight read in the tile's layout: no shared-memory layout change (diary H14, H14b) |
| triton_v5_rowloop | N <= 1024 at M in the thousands and up: the short-row kernel to beat | M = 64 (a grid of 64 warps); N > 1024 (not covered) | one warp per row, no barrier, prologue and weight amortized over ROWS rows: half of v1's instructions per element, at 50% (block limit) or 39% (registers, N = 1024) theoretical occupancy (diary H15) |
| triton (dispatch) | all N, by construction | M = 1 vs the best single kernel (dispatch costs host time) | chooses by N |
| triton_v4_fused_residual | all shapes vs the unfused pair | vs vLLM's CUDA kernel: unknown | 20% fewer bytes, one launch instead of two or more |

Where these predictions fail, the diary says so and explains what the profile
showed instead.

## 5. What to read in the profiler

For each kernel and shape, the Nsight Compute summary
(`scripts/ncu_summary.py`) is read in this order:

1. DRAM throughput as a fraction of peak. Above ~85% (the stopping rule in
   section 0): bandwidth-bound, done.
2. DRAM bytes read and written against the traffic model. More bytes than the
   model means redundant traffic (re-reads, spills, uncoalesced sectors).
3. Achieved occupancy against theoretical, and the limiter. Low theoretical
   occupancy with high register count points at the whole-row design on long
   rows; low achieved against theoretical points at a grid that does not fill
   the device (waves < 1).
4. Sectors per request for global loads (`scripts/ncu_summary.py` derives it
   from the sector and request sums). A fully coalesced warp request uses
   one 32-byte sector per 32 bytes it asks for, so the ideal equals the bytes
   per lane: 4 for 32-bit lanes, 16 for the 128-bit loads these kernels
   issue. Higher means wasted sectors.
5. Local memory traffic: non-zero means register spills.
6. Warp stall reasons: `long_scoreboard` is waiting on memory (expected and
   fine when throughput is high; a symptom when it is not).
