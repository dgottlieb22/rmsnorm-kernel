N=512, M=4096, sm_80, bf16. Per thread, static SASS count up to the last EXIT.
| kernel | warps | elems/thread | insns | insns/elem | regs | smem B | BAR | LDS+STS | LDG+STG | SHFL | convert | fp | int/addr/pred | uniform/ctrl |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v1 (baseline) | 2 | 8 | 128 | 16.00 | 24 | 8 | 2 | 4 | 3 | 6 | 13 | 39 | 32 | 23 |
| inv_n | 2 | 8 | 121 | 15.12 | 24 | 8 | 2 | 4 | 3 | 6 | 12 | 32 | 33 | 23 |
| 1warp | 1 | 16 | 176 | 11.00 | 37 | 0 | 0 | 0 | 6 | 5 | 25 | 62 | 45 | 27 |
| v3 kernel, BLOCK_M=2, 2 warps | 2 | 16 | 189 | 11.81 | 32 | 16 | 2 | 6 | 5 | 11 | 21 | 72 | 57 | 14 |
| v3 kernel, BLOCK_M=4, 4 warps | 4 | 16 | 226 | 14.12 | 39 | 1024 | 4 | 8 | 5 | 11 | 21 | 72 | 82 | 20 |
| nomask | 2 | 8 | 113 | 14.12 | 24 | 8 | 2 | 4 | 3 | 6 | 12 | 32 | 32 | 16 |
| rowloop 1warp x8 masked | 1 | 128 | 913 | 7.13 | 54 | 0 | 0 | 0 | 34 | 40 | 136 | 440 | 218 | 44 |
| rowloop 1warp x8 nomask | 1 | 128 | 850 | 6.64 | 47 | 0 | 0 | 0 | 34 | 40 | 136 | 440 | 183 | 13 |
| rowloop 2warps x8 masked | 2 | 64 | 612 | 9.56 | 40 | 8 | 23 | 32 | 17 | 48 | 68 | 256 | 141 | 26 |
| rowloop 2warps x8 nomask | 2 | 64 | 566 | 8.84 | 40 | 8 | 23 | 32 | 17 | 48 | 68 | 256 | 116 | 4 |
| rowloop 1warp x8 nomask, hand unpack, join | 1 | 128 | 1048 | 8.19 | 54 | 1024 | 31 | 128 | 82 | 40 | 64 | 496 | 190 | 13 |
| rowloop 1warp x8 nomask, hand unpack, cvt.bf16x2 | 1 | 128 | 836 | 6.53 | 49 | 0 | 0 | 0 | 34 | 40 | 64 | 496 | 184 | 14 |
