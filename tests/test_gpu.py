"""Tests that need a CUDA device. Skipped automatically elsewhere."""
from __future__ import annotations

import pytest
import torch

from rmsnorm import IMPLEMENTATIONS, bench, kernel_stats, profile
from rmsnorm.bench import Shape

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def cuda():
    return torch.device("cuda")


def test_time_gpu_returns_consistent_statistics(cuda):
    x = torch.randn(256, 4096, device=cuda, dtype=torch.bfloat16)
    w = torch.ones(4096, device=cuda, dtype=torch.bfloat16)
    fn = lambda: IMPLEMENTATIONS["triton_v0_naive"].fn(x, w, 1e-6)  # noqa: E731
    t = bench.time_gpu(fn, device=cuda, cache="cold", rounds=2, iters=20, warmup=5)
    assert t.gpu_us_min <= t.gpu_us_p25 <= t.gpu_us_median <= t.gpu_us_p75
    assert t.gpu_us_median > 0 and t.cpu_us > 0 and t.batched_gpu_us > 0
    assert t.rounds == 2 and t.iters == 20


def test_cold_is_not_faster_than_warm_on_an_l2_resident_shape(cuda):
    """A 2 MB working set lives in L2; flushing it must not make the call faster."""
    x = torch.randn(256, 4096, device=cuda, dtype=torch.bfloat16)
    w = torch.ones(4096, device=cuda, dtype=torch.bfloat16)
    fn = lambda: IMPLEMENTATIONS["triton_v1_warps"].fn(x, w, 1e-6)  # noqa: E731
    cold = bench.time_gpu(fn, device=cuda, cache="cold", rounds=3, iters=50, warmup=10)
    warm = bench.time_gpu(fn, device=cuda, cache="warm", rounds=3, iters=50, warmup=10)
    assert cold.gpu_us_median >= 0.9 * warm.gpu_us_median


def test_copy_bandwidth_is_plausible(cuda):
    gbps = bench.measure_copy_bandwidth(cuda, nbytes=1 << 28, iters=5)
    assert 50 < gbps < 10000


def test_bench_one_checks_correctness_and_fills_bandwidth(cuda):
    r = bench.bench_one(IMPLEMENTATIONS["triton"], "rmsnorm", Shape(64, 4096), torch.bfloat16, cuda,
                        cache="cold", rounds=1, iters=10, warmup=3)
    assert not r.error
    assert r.gbps > 0 and r.max_rel_err < 0.02


def test_kernel_stats_report_registers_and_occupancy(cuda):
    rows = kernel_stats.collect([Shape(64, 4096), Shape(8, 16384)], [torch.bfloat16], cuda,
                                ["triton_v0_naive", "triton_v2_looped"])
    assert len(rows) == 4
    for r in rows:
        assert r.n_regs > 0 and 0 < r.occupancy <= 1.0 and r.programs_per_sm >= 1
    naive_long = next(r for r in rows if r.impl == "triton_v0_naive" and r.n == 16384)
    looped_long = next(r for r in rows if r.impl == "triton_v2_looped" and r.n == 16384)
    # The looped kernel's register footprint does not grow with N; the naive one's does.
    assert looped_long.n_regs <= naive_long.n_regs


def test_kernel_table_counts_launches(cuda):
    rows, wall = profile.kernel_table("torch_eager", Shape(64, 4096), torch.bfloat16, cuda, iters=5)
    assert sum(r.count for r in rows) >= 5, "eager RMSNorm is several kernels"
    rows, wall = profile.kernel_table("triton_v1_warps", Shape(64, 4096), torch.bfloat16, cuda, iters=5)
    assert sum(r.count for r in rows) == 1
