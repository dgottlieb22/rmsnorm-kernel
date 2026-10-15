N=1024, M=4096, sm_80, bf16. Per thread, static SASS count up to the last EXIT.
| kernel | warps | elems/thread | insns | insns/elem | regs | smem B | BAR | LDS+STS | LDG+STG | SHFL | convert | fp | int/addr/pred | uniform/ctrl |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v1 (baseline) | 4 | 8 | 130 | 16.25 | 26 | 16 | 2 | 4 | 3 | 7 | 13 | 40 | 32 | 23 |
| inv_n | 4 | 8 | 123 | 15.38 | 26 | 16 | 2 | 4 | 3 | 7 | 12 | 33 | 33 | 23 |
| 1warp | 1 | 32 | 298 | 9.31 | 61 | 0 | 0 | 0 | 12 | 5 | 49 | 110 | 81 | 35 |
| v3 kernel, BLOCK_M=2, 2 warps | 2 | 32 | 297 | 9.28 | 56 | 16 | 2 | 6 | 10 | 11 | 41 | 120 | 87 | 19 |
| v3 kernel, BLOCK_M=4, 4 warps | 4 | 32 | 325 | 10.16 | 48 | 64 | 2 | 10 | 9 | 22 | 37 | 139 | 74 | 27 |
| nomask | 4 | 8 | 115 | 14.38 | 24 | 16 | 2 | 4 | 3 | 7 | 12 | 33 | 32 | 16 |
| rowloop 1warp x8 masked | 1 | 256 | 1756 | 6.86 | 80 | 0 | 0 | 0 | 68 | 40 | 272 | 824 | 435 | 111 |
| rowloop 1warp x8 nomask | 1 | 256 | 1550 | 6.05 | 80 | 0 | 0 | 0 | 68 | 40 | 272 | 824 | 329 | 13 |
| rowloop 4warps x8 masked | 4 | 64 | 628 | 9.81 | 40 | 16 | 23 | 32 | 17 | 56 | 68 | 264 | 141 | 26 |
| rowloop 4warps x8 nomask | 4 | 64 | 582 | 9.09 | 40 | 16 | 23 | 32 | 17 | 56 | 68 | 264 | 116 | 4 |
| rowloop 1warp x8 nomask, hand unpack, join | 1 | 256 | 2023 | 7.90 | 80 | 1024 | 63 | 256 | 164 | 40 | 128 | 944 | 343 | 51 |
| rowloop 1warp x8 nomask, hand unpack, cvt.bf16x2 | 1 | 256 | 1528 | 5.97 | 88 | 0 | 0 | 0 | 68 | 40 | 128 | 944 | 330 | 14 |
