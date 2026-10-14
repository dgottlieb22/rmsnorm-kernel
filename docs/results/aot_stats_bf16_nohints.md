| arch | variant | dtype | hints | M x N | grid | warps | block | ALIGN | regs | spill B (st/ld) | smem B | programs/SM | occupancy | waves | limiter | global ld (bits:count) | global st | shared ld | shared st | layout converts |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sm_80 | triton_v0_naive | bf16 | no | 4096x4096 | 4096 | 4 | 4096 | 16 | 120 | 0/0 | 16 | 4 | 25% | 9.48 | registers | 16:64 | 16:32 | 32:2 | 32:2 | 0 |
| sm_80 | triton_v1_warps | bf16 | no | 4096x4096 | 4096 | 16 | 4096 | 16 | 32 | 0/0 | 64 | 4 | 100% | 9.48 | registers+warps | 16:16 | 16:8 | 32:2 | 32:2 | 0 |
| sm_80 | triton_v0_naive | bf16 | no | 4096x16384 | 4096 | 4 | 16384 | 16 | 255 | 240/240 | 16 | 2 | 12% | 18.96 | registers | 16:256 | 16:128 | 32:2 | 32:2 | 0 |
| sm_80 | triton_v1_warps | bf16 | no | 4096x16384 | 4096 | 16 | 16384 | 16 | 127 | 0/0 | 64 | 1 | 25% | 37.93 | registers | 16:64 | 16:32 | 32:2 | 32:2 | 0 |
