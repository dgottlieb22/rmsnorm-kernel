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
python -m rmsnorm.cli aot --impl triton_v0_naive --arch sm_80 --arch sm_90
scripts/ncu.sh triton_v0_naive 4096x16384
```
Read: `n_spills` and `n_regs` at N = 16384 vs 4096; local memory sectors in
the ncu summary; bandwidth at N = 512.

Result (compile-time half, measured here with `aot`, Triton 3.4.0, bf16;
full table in `docs/results/aot_stats_bf16.md`):

| N | BLOCK_N | warps | registers (sm_80 / sm_90) | spills | programs per SM | theoretical occupancy |
|---|---|---|---|---|---|---|
| 512 | 512 | 4 | 20 / 22 | 0 | 16 | 100% |
| 2048 | 2048 | 4 | 38 / 38 | 0 | 12 | 75% |
| 4096 | 4096 | 4 | 62 / 60 | 0 | 8 | 50% |
| 5120, 8192 | 8192 | 4 | 125 / 127 | 0 | 4 | 25% |
| 16384 | 16384 | 4 | 218 / 216 | 0 | 2 | 12.5% |

(The v0 row at 16384 read 213 registers before the ALIGN change of H12 added
three multiplies; the occupancy did not move.)

The spill prediction was wrong: with the alignment hints the launcher passes
(see H12), the compiler fits N = 16384 in about 215 registers with no spill. What
does happen is that registers cap residency at 2 programs x 4 warps = 8
warps per SM, 12.5% occupancy, falling by half at every doubling of N from
2048 up. Each of those 128 threads keeps 128 elements (16 vector loads) in
flight, so bytes in flight per SM are not obviously short of Little's law
(see the model, section 3); whether 12.5% occupancy costs bandwidth here is
exactly what the timing has to decide, and the revised prediction is: a
moderate loss at 16384 (fewer warps to cover the reduction's latency and the
tail of each program), not the collapse first predicted.

Result (timing): PENDING HARDWARE.

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
python -m rmsnorm.cli aot --impl triton_v0_naive --impl triton_v1_warps --arch sm_80 --arch sm_90
```
Read: spills gone at 16384; per-N speedup of v1 over v0; occupancy and
limiter columns.

Result (compile-time half, measured here with `aot`, sm_80, bf16):

| N | v0 warps / regs / occupancy | v1 warps / regs / occupancy |
|---|---|---|
| 512 | 4 / 20 / 100% | 2 / 24 / 100% (32 programs per SM, block limit) |
| 2048 | 4 / 38 / 75% | 8 / 26 / 100% |
| 4096 | 4 / 62 / 50% | 16 / 26 / 100% |
| 8192 | 4 / 125 / 25% | 16 / 38 / 75% |
| 16384 | 4 / 218 / 12.5% | 16 / 63 / 50% |

Registers per thread scale with elements per thread, as expected: 8 elements
per thread is 26 registers at every N where the heuristic reaches it. From
N = 8192 up the cap of 16 warps means 16 and 32 elements per thread and the
occupancy ceiling drops to 75% and 50%. There were no spills to remove (H3),
so the change is from register-capped occupancy to warp-capped occupancy.
Revised prediction: v1 over v0 is a clear win at N >= 8192 if occupancy
matters there, close to flat at 4096 (50% occupancy with 32 elements per
thread in flight against 100% with 8), and a coin flip at 512.

Result (timing): PENDING HARDWARE.

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

Sub-experiment H6c (added after H14b): `triton_v3b_wtile` against v3 at the
same shapes. v3b reads the weight as a tile and skips v3's shared-memory
layout change and two barriers, at BLOCK_M times the weight-load requests
from L1 (H14b). Expect the difference to be a few percent at most, in v3b's
favor if the barriers cost more than the L1 re-reads; the ncu comparison to
read is L1 request count and `barrier` stall reasons.

Revised after H15 (SASS counts): the 1.3-2x prediction has no
instruction-level mechanism. v3's thread holds one row's 8 columns like v1's,
converts its own 8 weights, runs the same prologue and the same cross-warp
reduction, and issues more instructions per element than v1 (17.5 against
16.0 at N = 512). What v3 changes is bytes requested for the weight (once per
8 rows) and the program count; neither is a per-thread saving. The revised
expectation is v3 within a few percent of v1 either way, and the kernel to
beat them both on short rows is `triton_v5_rowloop` (H15), which halves the
instructions per element and has no barrier.

Change: 2-D block, reduction along axis 1.

Measurement:
```
python -m rmsnorm.cli bench --shape 64x512 --shape 4096x512 --shape 16384x512 --shape 4096x1024 --shape 4096x2048 --impl triton_v1_warps --impl triton_v3_multirow --impl triton_v3b_wtile --impl triton_v5_rowloop
python -m rmsnorm.cli stats --shape 4096x512 --impl triton_v1_warps --impl triton_v3_multirow --impl triton_v3b_wtile --impl triton_v5_rowloop
scripts/ncu.sh triton_v3_multirow 4096x512
scripts/ncu.sh triton_v3b_wtile 4096x512
scripts/ncu.sh triton_v5_rowloop 4096x512
```
For H6b, call `rmsnorm_triton_v3_multirow(x, w, eps, block_m=...)` directly;
for v5's ROWS sweep, `rmsnorm_triton_v5_rowloop(x, w, eps, rows=...)`.

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

## H12. The launcher's alignment hints change the kernel, not just its arguments

Hypothesis (formed after the first offline compile, which had no hints and
showed spills that the hinted compile did not): Triton's launcher marks every
16-byte-aligned pointer and every integer divisible by 16 with
`tt.divisibility = 16`. The compiler uses the hints to emit 128-bit loads and
stores; without them it emits narrower accesses, more address arithmetic and
more live values. The same source therefore compiles to a worse kernel for a
misaligned input (a slice starting at an odd element) or for a row stride not
divisible by 16 (a hidden size such as 1000).

Measurement (compile-time, `aot`, sm_80, bf16; the `aot` table now counts
global load and store instructions in the PTX by width):
```
python -m rmsnorm.cli aot --impl triton_v0_naive --impl triton_v1_warps --n 1000 --n 1024 --n 4096 --n 16384 --arch sm_80
python -m rmsnorm.cli aot --impl triton_v0_naive --impl triton_v1_warps --n 4096 --n 16384 --arch sm_80 --no-specialize
```

Result, part 1 (before any code change): the mechanism is exactly the access
width. With hints, `triton_v1_warps` at N = 4096 is two 128-bit loads and one
128-bit store per thread (x, w; y). Without hints it is sixteen 16-bit loads
and eight 16-bit stores: the same bytes as scalar bf16 accesses, eight times
the instructions, each with its own address and predicate. Registers follow:

| kernel | hints | registers | spills | occupancy | global accesses per thread |
|---|---|---|---|---|---|
| v0, N = 4096, 4 warps | yes | 62 | 0 | 50% | 8 x 128-bit loads, 4 stores |
| v0, N = 4096, 4 warps | no | 120 | 0 | 25% | 64 x 16-bit loads, 32 stores |
| v0, N = 16384, 4 warps | yes | 218 | 0 | 12.5% | 32 x 128-bit, 16 |
| v0, N = 16384, 4 warps | no | 255 | 240 B stores, 240 B loads | 12.5% | 256 x 16-bit, 128 |
| v1, N = 16384, 16 warps | yes | 63 | 0 | 50% | 8 x 128-bit, 4 |
| v1, N = 16384, 16 warps | no | 127 | 0 | 25% | 64 x 16-bit, 32 |

And N = 1000 against 1024 before the fix: the pointers kept their hints but
the stride (1000) and N lost theirs, so the row base offset was unprovable:
16-bit accesses, 30 registers against 26, and 2048 bytes of shared memory
against 16 (a layout conversion between the load and store layouts the
compiler had to pick without the hint).

Change (one, in all four kernels): the wrappers compute `ALIGN`, the largest
power of two up to 16 dividing N and every row stride, pass the strides and
N divided by it, and the kernel multiplies them back (`N = N_div * ALIGN`).
The multiplication by a compile-time constant is what the compiler's
divisibility analysis can see, where `tl.multiple_of` on the scalar
arguments did nothing (tried first; the PTX did not change). The launcher's
own granularity is 16 elements; 8 bf16 elements are already one 128-bit
access, so ALIGN = 8 is enough.

Result, part 2 (after the change): N = 1000 compiles to the same code as
N = 1024 for the row kernels: 26 registers, 16 bytes of shared memory, two
128-bit loads and one 128-bit store per thread (`tests/test_aot_stats.py`
pins this). N = 1004 (ALIGN = 4) gets 64-bit accesses and keeps the 2 KB
layout conversion. Cost: v0 at N = 16384 went from 213 to 218 registers for
the three extra multiplies; occupancy unchanged. All hinted shapes otherwise
compile to identical access patterns.

What the change cannot fix: a misaligned pointer. The `--no-specialize`
column above is now that case (ALIGN still carries the stride information),
and it stays scalar. The remedy there is a contiguous copy before the
kernel, whose cost against the scalar kernel is a timing question, PENDING
HARDWARE, along with the runtime effect of the N = 1000 fix (the full
matrix now contains N = 1000).

Explanation: the hints are what let the compiler prove that a 16-byte vector
access is in bounds and aligned. Without proof it falls back to scalar
accesses, each with its own address and predicate, which is where the extra
registers go; on the longest row that pushes past the register file and into
spills.

---

## H13. The looped kernel's register footprint is flat in N

Hypothesis: `triton_v2_looped` holds one chunk at a time, so registers do not
depend on N and occupancy stays at 100%.

Result (compile-time, measured here, sm_80 and sm_90, bf16): 32 registers at
N = 4096, 8192 and 16384 (16 warps, 4096-wide chunks), 0 spills, 4 programs
per SM, 100% theoretical occupancy at every N. Confirmed. Whether the
second pass over the row costs more than the occupancy buys back is H5,
PENDING HARDWARE.

---

## H14. The multirow kernel's weight load is narrow and goes through shared memory

Hypothesis (formed from the `aot` table before reading any IR): at N = 512
`triton_v3_multirow` shows `128:1 16:1` global loads and 1024 bytes of shared
memory, where the row kernels show `128:2` and 8 to 64 bytes. The x tile is
8 elements per thread at every N (one 128-bit load), so the 16-bit load must
be the weight: a 1-D vector of BLOCK_N elements spread over all 512 threads
is one element per thread at N = 512. The shared memory is 2 bytes x N
(1024, 2048, 4096 at N = 512, 1024, 2048) and does not move with BLOCK_M,
which fits a layout conversion of the bf16 weight vector into the tile's
layout.

Change: none in the kernels. The `aot` table gained `shared ld` / `shared st`
(shared-memory accesses in the PTX by width), `layout converts`
(`ttg.convert_layout` ops in the TTGIR) and `--dump-ir DIR`, which writes
each compile's TTGIR and PTX.

Measurement:
```
python -m rmsnorm.cli aot --impl triton_v1_warps --impl triton_v3_multirow --n 512 --n 1024 --n 2048 --n 4096 --arch sm_80 --dump-ir /tmp/ir
```

Result: confirmed in the IR. At 4096 x 512 (8 x 512 block, 16 warps) the
TTGIR holds exactly one `ttg.convert_layout`, on the weight, in bf16, before
the cast to float32:

```
#blocked  = sizePerThread = [1],    threadsPerWarp = [32],    warpsPerCTA = [16]      // how w is loaded: 1 element per thread
#blocked1 = sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [8, 2]    // the x tile: 8 contiguous per thread, warps = 8 rows x 2 column halves
%39 = ttg.convert_layout %38 : tensor<512xbf16, #blocked> -> tensor<512xbf16, #ttg.slice<{dim = 0, parent = #blocked1}>>
```

The target is the column distribution of the x tile: each thread needs the 8
contiguous weights that match its 8 x elements, and the 8 row-warps all need
the same weights. The compiler loads w coalesced across all 512 threads, one
bf16 each, writes it to shared memory and reads it back in the tile's
layout. In the PTX that is one `st.shared.b16` and one `ld.shared.v4.b32`
per thread with a barrier on each side: 4 `bar.sync` per program against
the row kernels' 2. Shared memory is exactly 2 bytes x N, the vector in
bf16; the reduction's one float per warp (64 B) is not live at the same time
and reuses the space.

| N | block | w global load | shared B | shared st / ld per thread, beyond the reduction's 2 x b32 each | bar.sync |
|---|---|---|---|---|---|
| 512 | 8 x 512 | 16-bit | 1024 | b16 / v4.b32 | 4 |
| 1024 | 4 x 1024 | 32-bit | 2048 | b32 / v4.b32 | 4 |
| 2048 | 2 x 2048 | 64-bit | 4096 | b64 / v4.b32 | 4 |
| 4096 | 1 x 4096 | 128-bit | 64 | none | 2 |

At N = 4096 (BLOCK_M = 1) the op is still in the TTGIR (`layout converts`
reads 1) but its source and target layouts are the same distribution (8 per
thread, 32 threads, 16 warps), so it lowers to nothing and the kernel
compiles like v1. `tests/test_aot_stats.py` pins the 2N bytes, the 16-bit
load, the convert count, and the row kernels' 4 bytes per warp.

Explanation: a 1-D `tl.load` gets its own layout from the coalescing pass,
chosen for that load alone: contiguous elements spread across every thread
of the program. The 2-D tile's layout gives each thread 8 contiguous columns
and replicates the columns across the row-warps. The two agree only when the
tile has one row. For every other BLOCK_M the compiler inserts a conversion,
and a conversion that moves data between threads goes through shared
memory. The cost per program is 2N bytes of shared memory, one extra store
and load per thread and two barriers, set against the design's stated
benefit of loading the weight once per BLOCK_M rows. The weight is 1 KB per
program at N = 512 against an 8 KB x tile, so this is a fixed cost per
program, not a bandwidth term; whether two barriers show up at all is for
the timing in H6.

---

## H14b. Two other ways to load the weight: one worse, one with no conversion

Hypothesis: (a) loading w as a [1, BLOCK_N] block and broadcasting it lets
the compiler choose a layout that matches the tile, removing the conversion.
(b) loading w as a full [BLOCK_M, BLOCK_N] tile, every row of the block
reading the same N weights, removes the conversion by construction, at the
price of BLOCK_M times the weight-load requests, served from L1 after the
first warp touches each line.

Change: none to the shipped kernel. Both alternatives are compiled by
`scripts/probe_h14b_weight_load.py` with the shipped kernel's launch plans.

Measurement:
```
python scripts/probe_h14b_weight_load.py --ir
```

Result (compile-time, sm_80, bf16):

| N | kernel | regs | shared B | global ld per thread | shared st / ld | converts | bar.sync |
|---|---|---|---|---|---|---|---|
| 512 | shipped | 27 | 1024 | 128:1 16:1 | 16:1 32:2 / 128:1 32:2 | 1 | 4 |
| 512 | (a) 1 x N | 28 | 2048 | 128:1 16:1 | 32:3 / 128:2 32:2 | 1 | 4 |
| 512 | (b) tile | 30 | 64 | 128:2 | 32:2 / 32:2 | 0 | 2 |
| 1024 | shipped | 28 | 2048 | 128:1 32:1 | 32:3 / 128:1 32:2 | 1 | 4 |
| 1024 | (a) 1 x N | 27 | 4096 | 128:1 32:1 | 64:1 32:2 / 128:2 32:2 | 1 | 4 |
| 1024 | (b) tile | 29 | 64 | 128:2 | 32:2 / 32:2 | 0 | 2 |
| 2048 | shipped | 29 | 4096 | 128:1 64:1 | 64:1 32:2 / 128:1 32:2 | 1 | 4 |
| 2048 | (a) 1 x N | 30 | 8192 | 128:1 64:1 | 128:1 32:2 / 128:2 32:2 | 1 | 4 |
| 2048 | (b) tile | 28 | 64 | 128:2 | 32:2 / 32:2 | 0 | 2 |

(a) is worse. The [1, BLOCK_N] load gets the same one-element-per-thread
distribution (`sizePerThread = [1, 1], warpsPerCTA = [1, 16]`), so the
conversion stays. It now happens after the cast to float32, on the broadcast
operand, so shared memory doubles to 4 bytes x N. The conversion comes from
the mismatch between a load spread across every thread and a tile that
replicates columns across row-warps; the tensor's rank does not enter.

(b) compiles to no conversion: two 128-bit global loads per thread (x and
w), the row kernels' shared-memory footprint (64 B, reduction only), 2
barriers, registers within 3 of the shipped kernel. Each program now issues
BLOCK_M times the weight-load requests: at N = 512 the 8 row-warps each
request the same 1 KB of weights, so load requests per program go from 9 KB
(8 KB of x, 1 KB of w) to 16 KB, all but the first touch of each line an L1
hit, DRAM traffic unchanged. Per thread, the weight costs the shipped kernel
one global load, one shared store, one shared load and two barriers; it
costs (b) one global load. Whether seven extra L1 hits per weight line cost
more than a shared round trip and two barriers is a timing question.

Result (timing): PENDING HARDWARE. (b) shipped as `triton_v3b_wtile`
(`_rmsnorm_multirow_wtile_kernel`, same plan as v3 with a different kernel),
so H6 can time the two side by side; the probe script now compiles (b) from
the registry. The dispatcher still picks v3 for N <= 1024 until the timing
says otherwise. `tests/test_correctness.py` pins that v3 and v3b agree bit
for bit (tail rows and masked columns included) and `tests/test_aot_stats.py`
pins the compile-time facts above, N = 1000 (ALIGN 8) included.

---

## H15. Short rows: where the instructions go, and a kernel that halves them

This entry is the first run of a stricter loop: measure, interpret, name the
limiter, list three fixes with a predicted contribution each, build them one
at a time, measure each on its own. The measurement is static SASS
(`scripts/sass_mix.py`: ptxas to a cubin, nvdisasm, count executed
instructions up to the last EXIT, by class). For the loop-free kernels the
static count is what one thread executes. Full tables:
`docs/results/sass_mix_bf16_sm80.md`, `docs/results/h15_short_rows_n512.md`,
`docs/results/h15_short_rows_n1024.md`.

### Stage 0: the kernels as they were

Instructions per thread and per element, sm_80, bf16, M = 4096:

| N | v0 naive | v1 warps | v3 multirow | v3b wtile |
|---|---|---|---|---|
| 512 | 102 / 25.5 | 128 / 16.0 | 140 / 17.5 | 133 / 16.6 |
| 1024 | 130 / 16.2 | 130 / 16.2 | 143 / 17.9 | 135 / 16.9 |
| 4096 | 313 / 9.8 | 134 / 16.8 | 129 / 16.1 | 129 / 16.1 |
| 16384 | 1088 / 8.5 | 323 / 10.1 | 322 / 10.1 | 318 / 9.9 |

Interpretation. Fitting count = fixed + rate x elements across v0 and v1
gives 8.1 instructions per element and 65 per thread of fixed cost. With 8
elements per thread (the v1 heuristic, which keeps every load 128-bit) the
fixed cost is half the instruction stream. Reading the SASS of v1 at N = 512
(128 instructions) says where it goes:

| instructions | what |
|---|---|
| ~29 | prologue: parameter loads, program id, 64-bit address math, mask setup, register zeroing for the masked loads, two 128-bit loads |
| ~24 | bf16 to f32 converts: PRMT + SHF.L for the high half of a pair, SHF.L (as IMAD.U32) for the low half: 1.5 per value, 8 x and 8 w |
| 8 | square-accumulate: FMUL + 7 FFMA |
| 10 | intra-warp reduction: 5 SHFL.BFLY + 5 FADD |
| ~20 | cross-warp reduction: STS, BAR.SYNC, LDS, a second shuffle level, STS, BAR.SYNC, LDS |
| 11 | mean and rsqrt: I2F of N, an IEEE-exact reciprocal (MUFU.RCP with 2 FSETP and 4 range-fixup FMULs), FFMA for eps, MUFU.RSQ |
| 16 | x * rstd * w |
| 4 | f32 to bf16 packs (F2FP.BF16.PACK_AB, one per pair) |
| ~6 | store address (uniform datapath), exit guard, store |

Two things the table settles before any timing. First, the multirow kernel
does not reduce per-thread work: at N = 512 its thread holds one row's 8
columns exactly as v1's does, converts its own 8 weights, runs the same
prologue and the same cross-warp reduction, and pays the H14 layout change
on top. v3 issues more instructions per element than v1 (17.5 against 16.0),
not fewer. H6's "amortizes the weight load and the per-program fixed cost
over 8 rows" was about bytes and programs, not instructions, and the
predicted 1.3-2x has no instruction-level mechanism behind it. Second, the
bottleneck on short rows, if the row count is large enough to fill the
machine, is per-thread fixed cost: prologue, reduction and the division by
N are 60% of the stream at 8 elements per thread.

The limiter, named: issue rate is a candidate but not a proven one. At 16
instructions per element and 4 bytes per element, holding an A100's 1.75
TB/s copy bandwidth needs 7.0e12 thread-instructions per second, 36% of the
part's peak issue rate (108 SMs x 4 schedulers x 32 lanes x 1.41 GHz); at
7 per element it is 17%. So a perfectly fed GPU would not be issue-bound
either way, and if N = 512 is slow on hardware the likelier causes are the
two barriers per row (a warp waiting at BAR.SYNC issues nothing), the long
dependent chain per row (load, convert, reduce, shuffle x5, shared memory,
barrier, rsqrt, multiply, store) with few rows in flight per SM, and block
scheduling of 4096 two-warp programs. Every fix below shortens that chain or
removes barriers as well as cutting instructions.

### Stage 1: three fixes, each measured alone (N = 512, baseline v1 = 128 instructions, 16.0 per element)

| fix | mechanism | instructions | per element | occupancy effect |
|---|---|---|---|---|
| inv_n: pass 1/N from the host | drops I2F and the exact-reciprocal sequence (fp math 39 to 32) | 121 (-7) | 15.1 | none |
| nomask: N == BLOCK_N known at compile time (includes the constant 1/N) | drops ISETP, predicates, CS2R zeroing | 113 (-15) | 14.1 | none |
| 1warp: one warp per row, 16 elements per thread | no shared memory, no barrier, no second shuffle level; fixed cost over twice the elements | 176 for 16 | 11.0 (-31%) | capped at 50% by the 32-blocks-per-SM limit; bytes in flight unchanged (two 128-bit loads per thread) |

Taken alone, the one-warp change is worth more than the other two together.
The two cheap ones compose with it. As a check that the limit is the
decomposition and not the kernel text, v3's own kernel compiled with
BLOCK_M = 2 and 2 warps (one warp per row, 16 elements per thread) comes out
at 189 instructions: 11.8 per element, the one-warp number plus the 2-D
indexing, and still carrying the cross-warp reduction code because the
compiler does not know the rows and the warps line up.

### Stage 2: amortize the fixed cost across rows

With one warp per row the remaining fixed cost is per program: prologue,
weight load and converts, rsqrt setup. A loop over ROWS rows in the same
warp pays it once (`_rowloop_kernel` in `scripts/probe_h15_short_rows.py`,
`tl.static_range` so the count is exact). Weight hoisted, 1/N from the host,
masked and unmasked:

| ROWS | masked | unmasked | registers |
|---|---|---|---|
| 1 | 166 / 10.4 | 155 / 9.7 | 37 |
| 2 | 275 / 8.6 | 249 / 7.8 | 47 |
| 4 | 489 / 7.6 | 454 / 7.1 | 54 |
| 8 | 913 / 7.1 | 850 / 6.6 | 47 |
| 16 | 1825 / 7.1 | 1690 / 6.6 | 64 |

The unmasked fit is 6.6 per element and 50 per program; the asymptote is
reached by ROWS = 8. The same loop with 2 warps per row is 9.6 / 8.8 at
ROWS = 8: the per-row barrier (23 BAR.SYNC for 8 rows) eats the gain, so the
loop only works with one warp. The remaining 6.6 per element is close to the
work itself: 1.5 converts, 1 FFMA, 2 FMUL, 0.5 pack, 0.25 load/store, and
the reduction's 10 instructions over 16 elements.

The trade is the grid. One warp per program and ROWS rows per program means
M / ROWS warps in the whole launch; at M = 4096 and ROWS = 8 that is 512
warps, under 5 per SM, far below the ~16 per SM the parallelism model wants.
So ROWS has to follow M: `_rows_for` doubles ROWS only while the grid keeps
at least 2048 programs (16 warps per SM on a 128-SM part), which gives
ROWS = 2 at M = 4096 (7.8 per element) and ROWS = 8 from M = 16384 (6.6).
This is the same trade H6 noted for v3's grid, now with a number on both
sides.

### Stage 3: the converts

With the reduction and the prologue gone, bf16 to f32 conversion is the
largest remaining item: Triton's `.to(tl.float32)` on a packed pair is
PRMT + SHF.L for the high half and IMAD.U32 for the low half, three
instructions per pair. Loading the pairs as int32 and unpacking by hand
(`xv << 16`, `xv & 0xFFFF0000`) is two. Two ways to pack the result back:

| pack | instructions (ROWS = 8) | per element | why |
|---|---|---|---|
| `tl.join` of the two halves, 2-D store | 1048 | 8.2 | the join is a layout change: 1 KB of shared memory, 128 LDS/STS and 31 barriers per program (H14's mechanism again) |
| inline PTX `cvt.rn.bf16x2.f32` on int32 pairs | 836 | 6.5 | one instruction per pair, no layout change |

The converts drop from 136 to 64, but fp math rises from 440 to 496: writing
the sum of squares as `x_lo*x_lo + x_hi*x_hi` turns one FFMA chain into FMUL
+ FFMA per pair plus a separate reduction of the pair vector. Net gain 14
instructions in 850, 1.6%, for inline assembly. Not adopted; recorded.

### What shipped from H15

`triton_v5_rowloop` (`_rmsnorm_rowloop_kernel`): one warp per program, ROWS
rows in sequence, weight hoisted, 1/N from the host, no mask when N is a
power of two and M a multiple of ROWS, ROWS from `_rows_for(M)`, N <= 1024
(32 elements per thread in one warp at 1024; 80 registers there, 39%
theoretical occupancy, register-limited). Registered, in the full
correctness matrix (shapes past 1024 are reported as skips with the reason),
pinned in `tests/test_aot_stats.py`: no shared memory, no barrier, and under
half of v1's instructions per element at 512 and 1024 (7.8 and 7.0 at
M = 4096 against 16.0 and 16.3). The dispatcher still picks v3 for short
rows: H15 has shown that v3 cannot beat v1 by the mechanism H6 claimed, and
that v5 issues half the instructions, but which is faster is the timing's
call (H6, with v5 added to its measurement list).

Result (timing): PENDING HARDWARE. What to read: at 4096 x 512 and
16384 x 512, v5 against v1 and v3; ncu `smsp__inst_executed` to confirm the
static counts, `smsp__warps_issue_stalled_barrier` and `_long_scoreboard` to
see whether the barriers or the dependent chain were the cost, and achieved
occupancy for v5 (theoretical 50% at 512, 39% at 1024).

---

## H16. An explicit layout removes the one-warp trade (compile-time, Gluon)

Hypothesis: v5 runs one warp per program only because standard Triton
chooses the thread mapping of a 2-D block itself and spreads every row
across all the program's warps (H15 stage 1: v3's kernel with BLOCK_M = 2
and 2 warps still carried the cross-warp reduction). Triton 3.4's Gluon
dialect exposes the layout. `BlockedLayout(size_per_thread = [1, BLOCK_N / 32],
threads_per_warp = [1, 32], warps_per_cta = [W, 1])` makes warp k hold row k
of each group of W rows, so the reduction along the row stays inside a warp
while the program has W warps. Prediction: v5's instruction count, no
shared memory, no barrier, and the occupancy cap moves from the 32-blocks
limit (50%) toward 100%.

Change: none to the shipped kernels; `scripts/probe_h16_gluon_rows.py`
compiles the Gluon kernel for W in {1, 2, 4, 8} and ROWS in {1, 2, 4}. The
offline compile needed one addition in `aot_stats`: Gluon's `make_ir` asks
the active GPU driver for the target name, so the probe path supplies the
target it already has.

Result (compile-time, sm_80, bf16, N = 512, M = 4096; full table
`docs/results/h16_gluon_rows_n512.md`):

| kernel | warps x rows | instructions / per element | registers | smem, barriers | theoretical occupancy |
|---|---|---|---|---|---|
| v5 (standard Triton) | 1 x 2 | 249 / 7.78 | 48 | 0, 0 | 50% (blocks) |
| gluon | 1 x 2 | 249 / 7.78 | 48 | 0, 0 | 50% (blocks) |
| gluon | 4 x 2 | 260 / 8.12 | 48 | 0, 0 | 62% (registers) |
| gluon | 4 x 1 | 155 / 9.69 | 36 | 0, 0 | 75% (registers) |
| gluon | 4 x 4 | 474 / 7.41 | 48 | 0, 0 | 62% (registers) |

With one warp the Gluon kernel compiles to the same code as v5 (249
instructions, a check that the explicit layout is what Triton picked on its
own there). With 4 warps per program the row reduction is still intra-warp,
shared memory and barriers stay at zero, and the per-element count rises by
11 instructions per program for the per-warp row index (8.12 against 7.78).
The occupancy cap does move off the block limit, but not to 100%: 48
registers per thread cap it at 62%, and 36 registers (ROWS = 1) at 75%. The
registers are the two rows of x held in float32 plus the hoisted weight; the
way to 100% would be 32 registers per thread, which means giving back either
the hoisted weight (convert it per row, which H15 stage 2 paid to avoid) or
the second row in flight. So the trade does not vanish, it changes shape:
blocks versus barriers becomes registers versus instructions. At N = 1024
the registers already bind at one warp (80 registers, 39%), and Gluon
changes nothing there (457 against 450 instructions, 38%).

Explanation: the one-warp design was never the goal, only Triton's only
route to an intra-warp reduction on a 2-D block. Given the layout, the
remaining limits are the ones the arithmetic predicts from register count,
and a 16-element row slice in float32 plus its weights is 32 registers
before any address or temporary.

Not shipped. The CPU interpreter does not run Gluon kernels, so the kernel
cannot go through the correctness matrix on this machine, and a variant that
cannot be tested here does not get registered. On a GPU: run the probe's
kernel through the oracle first (the H15 identity test pattern against v5),
then time W = 4 against v5 at 4096 x 512 and 16384 x 512; the question is
whether 62% against 50% theoretical occupancy shows up at all when both
kernels keep two 128-bit loads per thread in flight.

Result (timing): PENDING HARDWARE.

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

Changes that did not help, kept when they explain something about the
hardware or the compiler.

- H14b (a): loading the multirow kernel's weight as a [1, BLOCK_N] block
  instead of a 1-D vector. Hypothesis was that the 2-D shape would let the
  compiler pick the tile's layout for the load. It did not: the load still
  gets one element per thread across all 16 warps, the conversion stays, and
  because it now runs after the cast to float32 the shared memory doubles
  (2048 B against 1024 B at N = 512). The lesson is that the conversion is
  about how the elements are distributed over threads, not about rank.
- H12, first attempt: `tl.multiple_of` on the stride and N scalar arguments
  to recover the alignment hints at N = 1000. The PTX did not change. The
  divisibility analysis follows a multiplication by a `constexpr`, which is
  what the ALIGN rescaling gives it.
- H15, row loop with 2 warps per row: the per-row cross-warp reduction puts a
  barrier pair in every iteration (23 BAR.SYNC for 8 rows) and the count
  lands at 8.8-9.6 per element against 6.6-7.1 with one warp. The loop only
  pays with a warp that owns the whole row.
- H15, hand-unpacked bf16 with `tl.join` for the pack: interleaving two
  vectors with `tl.join` is a layout change, and the compiler routes it
  through 1 KB of shared memory with 31 barriers per program. The same pack
  through inline `cvt.rn.bf16x2.f32` has no layout change and saves 1.6% of
  the instructions, which did not justify inline assembly in the shipped
  kernel. The lesson is the one from H14 again: any op that changes which
  thread holds which element costs shared memory and barriers, and a
  one-line `join` is such an op.
