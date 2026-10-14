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

Result (timing): PENDING HARDWARE. (b) is a candidate variant, not a change
to v3: the shipped kernel is the one the design text describes, and the two
should be timed side by side in H6 before either is dropped.

---

## Candidates not yet built

- `triton.autotune` over `num_warps` and `BLOCK_M`, to check whether the
  heuristics in `_warps_for` and `_rows_per_program` leave anything on the
  table, and what autotuning costs at first call per shape.
- A multirow variant that loads the weight as a [BLOCK_M, BLOCK_N] tile
  (H14b (b)): no shared-memory conversion and two fewer barriers per program,
  BLOCK_M times the weight-load requests from L1. Time it against the shipped
  v3 in H6 at 4096 x 512 and 16384 x 512; the compile-time facts are in H14b.
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
