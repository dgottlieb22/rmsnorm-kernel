N=512, M=4096, sm_80, bf16, 2 rows per warp. Per thread, executed SASS.
| kernel | warps | elems/thread | insns | insns/elem | regs | smem B | BAR | LDS+STS | theoretical occupancy | grid |
|---|---|---|---|---|---|---|---|---|---|---|
| v5 rowloop (standard Triton, 1 warp, ROWS=2) | 1 | 32 | 249 | 7.78 | 48 | 0 | 0 | 0 | 50% (blocks) | 2048 |
| gluon rows-per-warp, 1 warps x 2 rows | 1 | 32 | 249 | 7.78 | 48 | 0 | 0 | 0 | 50% (blocks) | 2048 |
| gluon rows-per-warp, 2 warps x 2 rows | 2 | 32 | 260 | 8.12 | 48 | 0 | 0 | 0 | 66% (registers) | 1024 |
| gluon rows-per-warp, 4 warps x 2 rows | 4 | 32 | 260 | 8.12 | 48 | 0 | 0 | 0 | 62% (registers) | 512 |
| gluon rows-per-warp, 8 warps x 2 rows | 8 | 32 | 260 | 8.12 | 48 | 0 | 0 | 0 | 62% (registers) | 256 |
| gluon rows-per-warp, 1 warps x 1 rows | 1 | 16 | 155 | 9.69 | 37 | 0 | 0 | 0 | 50% (blocks) | 4096 |
| gluon rows-per-warp, 2 warps x 1 rows | 2 | 16 | 153 | 9.56 | 39 | 0 | 0 | 0 | 78% (registers) | 2048 |
| gluon rows-per-warp, 4 warps x 1 rows | 4 | 16 | 155 | 9.69 | 36 | 0 | 0 | 0 | 75% (registers) | 1024 |
| gluon rows-per-warp, 8 warps x 1 rows | 8 | 16 | 155 | 9.69 | 36 | 0 | 0 | 0 | 75% (registers) | 512 |
| gluon rows-per-warp, 1 warps x 4 rows | 1 | 64 | 454 | 7.09 | 50 | 0 | 0 | 0 | 50% (blocks) | 1024 |
| gluon rows-per-warp, 2 warps x 4 rows | 2 | 64 | 474 | 7.41 | 48 | 0 | 0 | 0 | 66% (registers) | 512 |
| gluon rows-per-warp, 4 warps x 4 rows | 4 | 64 | 474 | 7.41 | 48 | 0 | 0 | 0 | 62% (registers) | 256 |
| gluon rows-per-warp, 8 warps x 4 rows | 8 | 64 | 474 | 7.41 | 48 | 0 | 0 | 0 | 62% (registers) | 128 |
