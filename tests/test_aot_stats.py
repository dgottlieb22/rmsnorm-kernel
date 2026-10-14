"""Compile-time resource facts from offline compilation (no GPU needed).

These pin what ``docs/diary.md`` H3, H4, H12 and H14 cite: numbers read from
ptxas and the PTX Triton 3.4.0 emits for sm_80. They are facts about this
compiler version, so a Triton upgrade that changes code generation fails here
on purpose; the diary then needs a new look, not a looser test.

The test session enables Triton's interpreter for the runtime tests, and
that patch cannot be undone in-process, so the compiler runs in a clean
subprocess and the test reads its CSV.
"""
from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow

ROOT = Path(__file__).resolve().parents[1]


def _run_aot(tmp_path: Path, *args: str) -> dict:
    out = tmp_path / "aot.csv"
    env = {k: v for k, v in os.environ.items() if k != "TRITON_INTERPRET"}
    cmd = [sys.executable, "-m", "rmsnorm.cli", "aot", "--arch", "sm_80", "--out", str(out), *args]
    res = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)
    assert res.returncode == 0, res.stderr[-2000:]
    rows = {}
    with out.open() as fh:
        for row in csv.DictReader(fh):
            rows[(row["variant"], int(row["n"]), row["specialized"] == "True")] = row
    return rows


@pytest.fixture(scope="module")
def hinted(tmp_path_factory):
    return _run_aot(tmp_path_factory.mktemp("aot"),
                    "--impl", "triton_v0_naive", "--impl", "triton_v1_warps", "--impl", "triton_v2_looped",
                    "--impl", "triton_v3_multirow",
                    "--n", "512", "--n", "1000", "--n", "1024", "--n", "4096", "--n", "8192", "--n", "16384")


@pytest.fixture(scope="module")
def unhinted(tmp_path_factory):
    return _run_aot(tmp_path_factory.mktemp("aot_nohints"), "--no-specialize",
                    "--impl", "triton_v0_naive", "--impl", "triton_v1_warps", "--n", "4096", "--n", "16384")


def test_hidden_size_1000_compiles_like_1024(hinted):
    """The ALIGN constexpr carries the stride's divisibility by 8, so N = 1000 gets the same
    128-bit accesses, registers and 16 bytes of shared memory as N = 1024 (diary H12)."""
    for variant in ("triton_v0_naive", "triton_v1_warps"):
        a, b = hinted[(variant, 1000, True)], hinted[(variant, 1024, True)]
        assert a["align"] == "8" and b["align"] == "16"
        assert (a["registers"], a["shared_bytes"], a["global_loads"], a["global_stores"]) == \
               (b["registers"], b["shared_bytes"], b["global_loads"], b["global_stores"])
        assert a["global_loads"] == "128:2" and a["global_stores"] == "128:1"


def test_access_widths_with_and_without_pointer_hints(hinted, unhinted):
    """Aligned pointers: one 128-bit access per 8 elements per thread. Misaligned: 16-bit scalars."""
    v1 = hinted[("triton_v1_warps", 4096, True)]
    assert (v1["global_loads"], v1["global_stores"]) == ("128:2", "128:1")
    v1p = unhinted[("triton_v1_warps", 4096, False)]
    assert (v1p["global_loads"], v1p["global_stores"]) == ("16:16", "16:8")
    # v0 at N = 512 has 4 elements per thread: 64-bit accesses, half the width.
    v0 = hinted[("triton_v0_naive", 512, True)]
    assert v0["global_loads"] == "64:2"


def test_naive_long_row_is_register_limited_not_spilling(hinted):
    """v0 at N = 16384 with 4 warps: ~213 registers, no spills, 2 programs per SM (12.5% occupancy)."""
    s = hinted[("triton_v0_naive", 16384, True)]
    assert int(s["registers"]) >= 200 and int(s["spill_stores"]) == 0
    assert int(s["programs_per_sm"]) == 2 and s["limiter"] == "registers"
    assert float(s["occupancy"]) == pytest.approx(8 / 64)


def test_warps_heuristic_restores_occupancy_on_long_rows(hinted):
    """v1 at N = 16384 (16 warps) uses under a third of v0's registers, no spills, 50% occupancy."""
    v0 = hinted[("triton_v0_naive", 16384, True)]
    v1 = hinted[("triton_v1_warps", 16384, True)]
    assert int(v1["registers"]) < int(v0["registers"]) / 3
    assert int(v1["spill_stores"]) == 0
    assert float(v1["occupancy"]) >= 0.5


def test_looped_kernel_registers_do_not_grow_with_n(hinted):
    rows = [hinted[("triton_v2_looped", n, True)] for n in (4096, 8192, 16384)]
    assert len({r["registers"] for r in rows}) == 1
    assert all(float(r["occupancy"]) == 1.0 and int(r["spill_stores"]) == 0 for r in rows)


def test_naive_kernel_at_4096_is_half_occupancy_from_registers(hinted):
    s = hinted[("triton_v0_naive", 4096, True)]
    assert int(s["num_warps"]) == 4 and s["limiter"] == "registers" and float(s["occupancy"]) == pytest.approx(0.5)
    assert float(hinted[("triton_v1_warps", 4096, True)]["occupancy"]) == 1.0


def test_dropping_alignment_hints_costs_registers_and_spills(hinted, unhinted):
    """Without the pointers' 16-byte divisibility hints (a misaligned input) the same v0 kernel at
    N = 16384 needs all 255 registers and spills; v1's register count doubles."""
    h = hinted[("triton_v0_naive", 16384, True)]
    p = unhinted[("triton_v0_naive", 16384, False)]
    assert int(p["registers"]) > int(h["registers"])
    assert int(p["spill_stores"]) > 0 and int(h["spill_stores"]) == 0
    v1h = hinted[("triton_v1_warps", 16384, True)]
    v1p = unhinted[("triton_v1_warps", 16384, False)]
    assert int(v1p["registers"]) > 1.5 * int(v1h["registers"])


def test_multirow_block_covers_eight_short_rows(hinted):
    s = hinted[("triton_v3_multirow", 512, True)]
    assert (int(s["block_m"]), int(s["block_n"]), int(s["num_warps"])) == (8, 512, 16)
    assert int(s["grid"]) == 4096 // 8 and float(s["occupancy"]) == 1.0


def test_row_kernels_share_one_float_per_warp(hinted):
    """The single-row kernels' only shared memory is the cross-warp reduction: 4 bytes per warp,
    written and read as 32-bit words. Anything wider is a layout conversion (diary H14)."""
    for n in (512, 1024, 4096, 16384):
        s = hinted[("triton_v1_warps", n, True)]
        assert int(s["shared_bytes"]) == 4 * int(s["num_warps"])
        assert (s["shared_loads"], s["shared_stores"]) == ("32:2", "32:2")


def test_multirow_restages_weight_vector_through_shared_memory(hinted):
    """With BLOCK_M > 1 the weight is loaded one element per thread and moved through shared
    memory into the tile's column layout: 2 bytes x N of shared memory, a 16-bit store and a
    128-bit load per thread, one ttg.convert_layout (diary H14)."""
    s = hinted[("triton_v3_multirow", 512, True)]
    assert int(s["shared_bytes"]) == 2 * 512
    assert s["global_loads"] == "128:1 16:1" and int(s["layout_converts"]) == 1
    assert "16:1" in s["shared_stores"].split() and "128:1" in s["shared_loads"].split()
    assert int(hinted[("triton_v3_multirow", 1024, True)]["shared_bytes"]) == 2 * 1024
    # At BLOCK_M = 1 the load layout already matches the tile's; the op stays in the TTGIR but
    # lowers to nothing, and the kernel compiles like v1.
    one = hinted[("triton_v3_multirow", 4096, True)]
    v1 = hinted[("triton_v1_warps", 4096, True)]
    assert int(one["block_m"]) == 1 and int(one["layout_converts"]) == 1
    assert (one["shared_bytes"], one["shared_loads"], one["shared_stores"], one["global_loads"]) == \
           (v1["shared_bytes"], v1["shared_loads"], v1["shared_stores"], v1["global_loads"])
