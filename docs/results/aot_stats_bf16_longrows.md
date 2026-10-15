| arch | variant | dtype | hints | M x N | grid | warps | block | ALIGN | regs | spill B (st/ld) | smem B | programs/SM | occupancy | waves | limiter | global ld (bits:count) | global st | shared ld | shared st | layout converts |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sm_80 | triton_v0_naive | bf16 | yes | 4096x16384 | 4096 | 4 | 16384 | 16 | 218 | 0/0 | 16 | 2 | 12% | 18.96 | registers | 128:32 | 128:16 | 32:2 | 32:2 | 0 |
| sm_80 | triton_v1_warps | bf16 | yes | 4096x16384 | 4096 | 16 | 16384 | 16 | 63 | 0/0 | 64 | 2 | 50% | 18.96 | registers | 128:8 | 128:4 | 32:2 | 32:2 | 0 |
| sm_80 | triton_v2_looped | bf16 | yes | 4096x16384 | 4096 | 16 | 4096 | 16 | 32 | 0/0 | 64 | 4 | 100% | 9.48 | registers+warps | 128:3 | 128:1 | 32:2 | 32:2 | 0 |
| sm_80 | triton_v0_naive | bf16 | yes | 4096x32768 | 4096 | 4 | 32768 | 16 | 32 | 3122/3104 | 16 | 16 | 100% | 2.37 | registers+warps | 128:64 | 128:32 | 32:2 | 32:2 | 0 |
| sm_80 | triton_v1_warps | bf16 | yes | 4096x32768 | 4096 | 16 | 32768 | 16 | 121 | 0/0 | 64 | 1 | 25% | 37.93 | registers | 128:16 | 128:8 | 32:2 | 32:2 | 0 |
| sm_80 | triton_v2_looped | bf16 | yes | 4096x32768 | 4096 | 16 | 4096 | 16 | 32 | 0/0 | 64 | 4 | 100% | 9.48 | registers+warps | 128:3 | 128:1 | 32:2 | 32:2 | 0 |
| sm_80 | triton_v0_naive | bf16 | yes | 4096x65536 | 4096 | 4 | 65536 | 16 | 32 | 6014/5928 | 16 | 16 | 100% | 2.37 | registers+warps | 128:128 | 128:64 | 32:2 | 32:2 | 0 |
| sm_80 | triton_v1_warps | bf16 | yes | 4096x65536 | 4096 | 16 | 65536 | 16 | 128 | 300/300 | 64 | 1 | 25% | 37.93 | registers | 128:32 | 128:16 | 32:2 | 32:2 | 0 |
| sm_80 | triton_v2_looped | bf16 | yes | 4096x65536 | 4096 | 16 | 4096 | 16 | 32 | 0/0 | 64 | 4 | 100% | 9.48 | registers+warps | 128:3 | 128:1 | 32:2 | 32:2 | 0 |
