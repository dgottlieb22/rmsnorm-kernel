# Performance diary

One entry per experiment. Each states the hypothesis before the measurement,
the single change made, the command that produces the evidence, and then the
result and the explanation. Entries whose Result says PENDING HARDWARE have
not been run: this repository was built on a machine without a GPU, and no
number below is invented. Failed experiments stay in the diary when they
teach something; see the last section.

Conventions: bf16 unless stated; "copy BW" is the device-to-device copy
bandwidth the benchmark measures at the start of each run; shapes are M x N.

---

## H1. The eager baseline is launch- and traffic-bound, not compute-bound

Hypothesis: `torch_eager` launches 8-10 kernels per call and moves several
times the operation's traffic because every intermediate (`x.float()`,
`x*x`, the normalized tensor, the scaled tensor) is a full-size float32
tensor written to and read back from HBM. At 4096 x 4096 its device time
should be 4-6x the traffic floor; at 1 x 4096 it should be dominated by
launch gaps.

Change: none (baseline measurement).

Measurement:
```
python -m rmsnorm.cli kernels --impl torch_eager --shape 4096x4096
python -m rmsnorm.cli kernels --impl torch_eager --shape 1x4096
scripts/nsys.sh torch_eager 1x4096
scripts/ncu.sh torch_eager 4096x4096
```
Read: kernel count per call; sum of device time vs wall time per call;
DRAM bytes read+written vs the 67.1 MB model.

Result: PENDING HARDWARE.

Explanation: PENDING HARDWARE.

---

## H2. torch.compile closes most of the gap, except at M = 1

Hypothesis: Inductor fuses the graph into one reduction kernel (possibly two:
a reduction and a pointwise). At 4096 x 4096 it reaches within 10-20% of the
best hand-written kernel. At 1 x 4096 its host-side guards and the generated
wrapper make it no faster than eager per call in host time, even though GPU
time drops to one kernel.

Change: `torch.compile(rmsnorm_torch_eager)`.

Measurement:
```
python -m rmsnorm.cli bench --matrix quick --impl torch_eager --impl torch_compiled --impl triton --include-compile
python -m rmsnorm.cli kernels --impl torch_compiled --shape 4096x4096
```
Read: GPU us and host us columns at M = 1 and M = 4096.

Result: PENDING HARDWARE.

Explanation: PENDING HARDWARE.

---

## H3. The naive whole-row kernel is already near the roofline at N = 4096, and collapses at N = 16384

Hypothesis: `triton_v0_naive` (one row per program, 4 warps for every N)
reaches 70-85% of copy BW at 4096 x 4096: 128 threads each own 32 elements,
loads are 16 B per thread and fully coalesced, and the row fits in registers.
At N = 16384 the same 128 threads own 128 elements each; x and w together are
256 float32 values per thread, above the 255-register limit, so the compiler
spills and bandwidth drops by half or more. At N = 512 each thread owns 4
elements and the fixed per-program cost dominates: well under 50% of copy BW.

Change: first Triton kernel.

Measurement:
```
python -m rmsnorm.cli bench --matrix full --impl torch_eager --impl triton_v0_naive
python -m rmsnorm.cli stats --matrix full --impl triton_v0_naive
scripts/ncu.sh triton_v0_naive 4096x16384
```
Read: `n_spills` and `n_regs` at N = 16384 vs 4096; local memory sectors in
the ncu summary; bandwidth at N = 512.

Result: PENDING HARDWARE.

Explanation: PENDING HARDWARE.

---

## H4. Scaling num_warps to the row (8 elements per thread) removes the spills and changes little else

Hypothesis: `triton_v1_warps` picks `num_warps = BLOCK_N / 256`, capped at 16.
At N = 16384 that is 16 warps: 32 elements per thread, about 100 registers,
no spills, bandwidth back above 80% of copy BW. At N = 4096 it is 16 warps vs
v0's 4: the same bytes with 4x the threads and a quarter of the elements each.
Expect a small change either way (more warps per program means fewer programs
per SM for the same occupancy; the reduction's cross-warp step grows). At
N = 512 it is 2 warps (64 threads): more programs can be resident, but each
is tiny; expect no improvement.

Change: `num_warps` from a constant to `_warps_for(BLOCK_N)`. Nothing else.

Measurement:
```
python -m rmsnorm.cli bench --matrix full --impl triton_v0_naive --impl triton_v1_warps
python -m rmsnorm.cli stats --matrix full --impl triton_v0_naive --impl triton_v1_warps
```
Read: spills gone at 16384; per-N speedup of v1 over v0; occupancy and
limiter columns.

Result: PENDING HARDWARE.

Explanation: PENDING HARDWARE.

---

## H5. Two passes over the row (looped kernel) only pay off where the single block spills

Hypothesis: `triton_v2_looped` (chunks of 4096, sum-of-squares pass then
normalize pass) bounds registers independent of N. Where v1 does not spill
(N <= 16384 with 16 warps), v2 should be slightly slower: the second pass
re-reads the row (from L2, so no extra HBM bytes, but extra L1/L2 requests
and instructions) and the loop adds overhead. v2 should win only where the
whole-row design cannot hold the row: N = 32768 and above, which the full
matrix does not include. If this holds, the dispatcher's `LOOPED_MIN_N`
moves from 8192 to above 16384.

Sub-experiment H5b: chunk width 1024 / 2048 / 4096 / 8192 at N = 16384. Larger
chunks mean fewer loop iterations and more bytes in flight per thread; smaller
chunks mean more programs per SM. Expect 2048-4096 to be flat and 1024 to
lose.

Change: loop over the row; `BLOCK_N` becomes a chunk width.

Measurement:
```
python -m rmsnorm.cli bench --shape 4096x8192 --shape 4096x16384 --shape 512x16384 --impl triton_v1_warps --impl triton_v2_looped
scripts/ncu.sh triton_v2_looped 4096x16384
```
For H5b, call `rmsnorm_triton_v2_looped(x, w, eps, block_n=...)` directly
from a short script (the CLI exposes the default only).

Result: PENDING HARDWARE.

Explanation: PENDING HARDWARE.

---

## H6. Several rows per program wins at N <= 1024 and loses at small M

Hypothesis: `triton_v3_multirow` with BLOCK_M = 4096 / BLOCK_N rows per
program (8 rows at N = 512) loads the weight once per 8 rows instead of once
per row, amortizes the per-program fixed cost over 8 KB instead of 1 KB, and
gives each thread 8 independent elements to load. Expect 1.3-2x over v1 at
4096 x 512 and 16384 x 512. At 64 x 512 the grid shrinks to 8 programs on a
100+ SM device: expect a loss against v1's 64 programs. Expect a loss at
N >= 2048, where BLOCK_M falls to 1-2 and the 2-D indexing is pure overhead.

Sub-experiment H6b ("what changed when you altered the tile size"): BLOCK_M in
{1, 2, 4, 8, 16, 32} at 4096 x 512 and 16384 x 1024. Expect a plateau once a
program covers ~4-8 KB and a decline when BLOCK_M x BLOCK_N exceeds what the
threads can hold in registers (tile too tall: spills again).

Change: 2-D block, reduction along axis 1.

Measurement:
```
python -m rmsnorm.cli bench --shape 64x512 --shape 4096x512 --shape 16384x512 --shape 4096x1024 --shape 4096x2048 --impl triton_v1_warps --impl triton_v3_multirow
python -m rmsnorm.cli stats --shape 4096x512 --impl triton_v1_warps --impl triton_v3_multirow
```
For H6b, call `rmsnorm_triton_v3_multirow(x, w, eps, block_m=...)` directly.

Result: PENDING HARDWARE.

Explanation: PENDING HARDWARE.

---

## H7. Masked lanes at non-power-of-two N cost instructions, not bandwidth

Hypothesis: at N = 5120 the single-block kernels use BLOCK_N = 8192 and 37.5%
of every lane is predicated off. Masked lanes move no bytes, so in the
bandwidth-bound regime (4096 x 5120, 84 MB) the cost should be small (under
10%) relative to the interpolation between N = 4096 and N = 8192. The looped
kernel with a chunk of 1024 covers 5120 in five exact chunks with no masking
and should match or beat the single block there, at the price of the loop.

Change: none in code; the comparison is between existing variants and the
`block_n` parameter of v2.

Measurement:
```
python -m rmsnorm.cli bench --shape 4096x4096 --shape 4096x5120 --shape 4096x8192 --impl triton_v1_warps --impl triton_v2_looped
scripts/ncu.sh triton_v1_warps 4096x5120
```
Read: achieved bandwidth at 5120 against neighbors; issued instructions and
predicated-off lanes in ncu's instruction statistics.

Result: PENDING HARDWARE.

Explanation: PENDING HARDWARE.

---

## H8. Small M with long rows is grid-bound; only splitting rows across programs can fix it

Hypothesis: at 64 x 16384 (2 MB), one row per program is 64 programs for
108-132 SMs. Every variant stays under 50% of copy BW regardless of its
inner loop, and `stats` reports waves well below 1. The multi-row kernel
makes it worse. A split-row design (each program reduces a chunk, partial
sums combined through a second tiny kernel or atomics, then a normalize
pass) would use every SM at the cost of a second launch or a global
synchronization; whether that is worth it for a 2 MB operation that takes a
few microseconds is doubtful, and the answer decides whether a v5 is built.

Change: none yet.

Measurement:
```
python -m rmsnorm.cli bench --shape 8x16384 --shape 64x16384 --shape 512x16384 --impl triton_v1_warps --impl triton_v2_looped --impl triton_v3_multirow --impl torch_compiled --include-compile
python -m rmsnorm.cli stats --shape 64x16384
```
Read: waves column; bandwidth vs M at fixed N.

Result: PENDING HARDWARE.

Explanation: PENDING HARDWARE.

---

## H9. Fusing the residual add saves 20% of the bytes and one launch

Hypothesis: the unfused pair moves 5 bytes per element per byte of dtype
(add: 2 in, 1 out; norm: 1 in, 1 out), the fused kernel 4. In the
bandwidth-bound regime `triton_v4_fused_residual` is 1.15-1.25x faster than
`torch_eager_residual`'s add plus the best plain kernel, and more at small
shapes where the removed launch matters. Against `torch_compiled_residual`
(Inductor fuses the add too) expect parity within 10%. Against vLLM's CUDA
`fused_add_rms_norm`, if installed, expect parity; it is the same algorithm.

Change: residual add inside the row kernel, rounded to the storage dtype
before normalization so the result matches the unfused sequence.

Measurement:
```
python -m rmsnorm.cli bench --matrix full --kind residual --include-compile
```
Read: speedup column against `torch_eager_residual`; bandwidth vs copy BW.

Result: PENDING HARDWARE.

Explanation: PENDING HARDWARE.

---

## H10. A warm L2 overstates bandwidth by 2-3x on shapes that fit

Hypothesis: for 64 x 4096 (1 MB) the warm-L2 measurement reports "bandwidth"
above HBM peak, because the inputs come from L2; for 4096 x 16384 (256 MB)
warm and cold agree within noise. This is the justification for the cold
default and the reason bandwidth numbers at small shapes are not reported as
bandwidth.

Change: none; `--cache warm` vs `--cache cold`.

Measurement:
```
python -m rmsnorm.cli bench --matrix quick --cache cold --cache warm --impl triton_v1_warps
```

Result: PENDING HARDWARE.

Explanation: PENDING HARDWARE.

---

## H11. At M = 1 every implementation is launch-bound and the Python wrapper is the cost

Hypothesis: at 1 x 4096 the GPU time of a Triton kernel is 3-6 us but the
host time of its wrapper (shape checks, `reshape`, Triton's launcher) is
20-40 us; the batched column equals the host time, not the GPU time. The
dispatcher adds a few microseconds over the variant it picks. `torch_eager`
is 8-10 launches of host time. The practical conclusion for a serving stack
is CUDA graphs around the whole decode step, not a faster norm kernel.

Change: none.

Measurement:
```
python -m rmsnorm.cli bench --shape 1x4096 --shape 8x4096 --shape 64x4096 --include-compile
```
Read: the "Host launch cost vs GPU time" table in the report.

Result: PENDING HARDWARE.

Explanation: PENDING HARDWARE.

---

## Candidates not yet built

- `triton.autotune` over `num_warps` and `BLOCK_M`, to check whether the
  heuristics in `_warps_for` and `_rows_per_program` leave anything on the
  table, and what autotuning costs at first call per shape.
- A CUDA kernel with warp-shuffle reductions and 16-byte vector loads. Built
  only if the Triton kernels sit more than ~10% below copy BW in the
  bandwidth-bound regime after H3-H7; the point would be to see what Triton's
  code generation leaves out, not to win.
- The split-row kernel from H8, if the measurement says the grid-bound regime
  matters for a realistic decode batch.

## Failed and abandoned experiments

None recorded yet. This section is for changes that did not help, kept when
they explain something about the hardware or the compiler.
