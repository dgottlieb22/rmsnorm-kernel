| arch | variant | dtype | hints | M x N | grid | warps | block | regs | spill B (st/ld) | smem B | programs/SM | occupancy | waves | limiter |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sm_80 | triton_v0_naive | bf16 | no | 4096x4096 | 4096 | 4 | 4096 | 104 | 0/0 | 16 | 4 | 25% | 9.48 | registers |
| sm_80 | triton_v1_warps | bf16 | no | 4096x4096 | 4096 | 16 | 4096 | 32 | 0/0 | 64 | 4 | 100% | 9.48 | registers+warps |
| sm_80 | triton_v0_naive | bf16 | no | 4096x16384 | 4096 | 4 | 16384 | 255 | 188/188 | 16 | 2 | 12% | 18.96 | registers |
| sm_80 | triton_v1_warps | bf16 | no | 4096x16384 | 4096 | 16 | 16384 | 110 | 0/0 | 64 | 1 | 25% | 37.93 | registers |
