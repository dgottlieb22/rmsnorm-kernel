"""Benchmark harness pieces that do not need a GPU.

The timing path itself needs CUDA events and is covered by ``test_gpu.py``.
The numbers used here are synthetic inputs to the rendering code, never
measurements, and nothing from this file is written under ``docs/``.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from rmsnorm import bench, cli, kernel_stats, report
from rmsnorm.bench import Result, Shape


def test_bytes_model_counts_each_tensor_once():
    # 64 rows x 4096 bf16: read x, write y (2 bytes each), read 4096 bf16 weights.
    assert bench.bytes_for("rmsnorm", 64, 4096, torch.bfloat16, torch.bfloat16) == 64 * 4096 * 4 + 4096 * 2
    # Residual: read x and r, write y and s.
    assert bench.bytes_for("residual", 64, 4096, torch.bfloat16, torch.bfloat16) == 64 * 4096 * 8 + 4096 * 2
    # fp32 weight with bf16 activations.
    assert bench.bytes_for("rmsnorm", 1, 8, torch.bfloat16, torch.float32) == 8 * 4 + 8 * 4


def test_percentile_interpolates():
    assert bench._percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5
    assert bench._percentile([5.0], 0.25) == 5.0
    assert math.isnan(bench._percentile([], 0.5))


def test_time_gpu_refuses_cpu():
    with pytest.raises(RuntimeError, match="CUDA"):
        bench.time_gpu(lambda: None, device=torch.device("cpu"), cache="cold", rounds=1, iters=1, warmup=0)


def test_spec_bandwidth_lookup():
    assert bench.spec_bandwidth("NVIDIA A100-SXM4-80GB") == 2039.0
    assert bench.spec_bandwidth("NVIDIA H100 80GB HBM3") is None  # not in the table: rely on the measured copy


def test_quick_matrix_is_subset_of_full():
    assert set(bench.QUICK_SHAPES) <= set(bench.FULL_SHAPES)
    assert all(s.m * s.n <= bench.MAX_ELEMENTS for s in bench.FULL_SHAPES)


def _synthetic_results() -> list[Result]:
    rows = []
    nan = float("nan")
    for n in (512, 4096):
        for m in (1, 4096):
            nbytes = bench.bytes_for("rmsnorm", m, n, torch.bfloat16, torch.bfloat16)
            for impl, us in (("torch_eager", 40.0), ("triton_v0_naive", 10.0), ("triton", 8.0)):
                rows.append(Result(
                    impl=impl, family="torch" if impl.startswith("torch") else "triton", kind="rmsnorm",
                    m=m, n=n, dtype="bfloat16", cache="cold", bytes_moved=nbytes,
                    gpu_us_median=us, gpu_us_p25=us * 0.98, gpu_us_p75=us * 1.02, gpu_us_min=us * 0.95,
                    round_spread=0.03, cpu_us=12.0, batched_gpu_us=us * 0.9,
                    gbps=nbytes / (us * 1e-6) / 1e9, elements_per_us=m * n / us,
                    max_abs_err=1e-3, max_rel_err=1e-3, rounds=5, iters=100,
                ))
    rows.append(Result(impl="broken", family="triton", kind="rmsnorm", m=1, n=512, dtype="bfloat16", cache="cold",
                       bytes_moved=0, gpu_us_median=nan, gpu_us_p25=nan, gpu_us_p75=nan, gpu_us_min=nan,
                       round_spread=nan, cpu_us=nan, batched_gpu_us=nan, gbps=nan, elements_per_us=nan,
                       max_abs_err=nan, max_rel_err=nan, rounds=5, iters=100, error="AssertionError: wrong"))
    return rows


def test_csv_round_trip(tmp_path: Path):
    rows = _synthetic_results()
    bench.write_csv(tmp_path / "raw.csv", rows)
    back = bench.read_csv(tmp_path / "raw.csv")
    assert len(back) == len(rows)
    assert back[0] == rows[0]
    assert back[-1].error == "AssertionError: wrong" and math.isnan(back[-1].gpu_us_median)


def test_report_renders_tables_speedups_and_launch_costs(tmp_path: Path):
    bench.write_csv(tmp_path / "raw.csv", _synthetic_results())
    (tmp_path / "env.json").write_text(json.dumps({"device_name": "synthetic", "copy_bandwidth_gbps": 2000.0}))
    out = report.render_report(tmp_path)
    text = out.read_text()
    assert "## rmsnorm, bfloat16, cold L2" in text
    assert "### N = 4096" in text
    assert "5.00x" in text  # 40 us baseline / 8 us triton
    assert "launch-bound" in text and "| yes |" in text  # host 12 us > GPU 10 us for v0 at M=1
    assert "ERR" in text and "AssertionError: wrong" in text
    pngs = report.plot(tmp_path)
    assert pngs and all(p.exists() and p.stat().st_size > 0 for p in pngs)


def test_occupancy_calculator_a100_like():
    limits = {"sm_count": 108, "regs_per_sm": 65536, "max_threads_per_sm": 2048, "max_blocks_per_sm": 32,
              "shared_per_sm": 160 * 1024}
    # 64 regs x 128 threads = 8192 regs per program -> 8 programs by registers; 16 by warps.
    programs, occ, limiter = kernel_stats.occupancy(64, 4, 0, limits)
    assert (programs, limiter) == (8, "registers")
    assert occ == pytest.approx(8 * 4 / 64)
    # 16 warps per program: 32 regs x 512 threads fills the register file at 4 programs, and
    # 64 / 16 warps is also 4, so both limits bind.
    programs, occ, limiter = kernel_stats.occupancy(32, 16, 0, limits)
    assert (programs, occ, limiter) == (4, 1.0, "registers+warps")
    # Fewer registers: only the warp limit binds.
    programs, occ, limiter = kernel_stats.occupancy(16, 16, 0, limits)
    assert (programs, occ, limiter) == (4, 1.0, "warps")
    # Register rounding: 33 regs allocates as 40.
    programs, _, _ = kernel_stats.occupancy(33, 4, 0, limits)
    assert programs == 65536 // (40 * 128)


def test_cli_list_and_env_run_on_cpu(capsys):
    assert cli.main(["list"]) == 0
    out = capsys.readouterr().out
    assert "triton_v0_naive" in out and "torch_eager" in out
    assert cli.main(["env"]) == 0
    env = json.loads(capsys.readouterr().out)
    assert "torch" in env and "cuda_available" in env


def test_cli_bench_refuses_cpu():
    if torch.cuda.is_available():
        pytest.skip("GPU present")
    with pytest.raises(SystemExit):
        cli.main(["bench", "--shape", "1x512"])


def test_make_inputs_shapes():
    x, w = bench.make_inputs(Shape(3, 10), torch.float16, "cpu", "rmsnorm")
    assert x.shape == (3, 10) and w.shape == (10,) and x.dtype == w.dtype == torch.float16
    x, r, w = bench.make_inputs(Shape(3, 10), torch.float32, "cpu", "residual")
    assert r.shape == (3, 10)
