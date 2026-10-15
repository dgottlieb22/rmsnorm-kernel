"""The ncu details-CSV summarizer, on a hand-written CSV in the format `ncu --csv --page details` emits.

The values here are placeholders for the parser, not measurements.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ncu_summary.py"

spec = importlib.util.spec_from_file_location("ncu_summary", SCRIPT)
ncu_summary = importlib.util.module_from_spec(spec)
sys.modules["ncu_summary"] = ncu_summary
spec.loader.exec_module(ncu_summary)

HEADER = '"ID","Process ID","Process Name","Host Name","Kernel Name","Context","Stream","Block Size","Grid Size","Device","CC","Section Name","Metric Name","Metric Unit","Metric Value"\n'


def _row(lid, kernel, section, name, unit, value):
    return f'"{lid}","1","python","host","{kernel}","1","7","(128, 1, 1)","(4096, 1, 1)","0","9.0","{section}","{name}","{unit}","{value}"\n'


def test_summary_table_from_details_csv(tmp_path: Path):
    csv_path = tmp_path / "x.details.csv"
    lines = ["==PROF== Disconnected from process 1\n", HEADER]
    k = "_rmsnorm_row_kernel"
    lines.append(_row(0, k, "GPU Speed Of Light Throughput", "gpu__time_duration.sum", "usecond", "12.34"))
    lines.append(_row(0, k, "GPU Speed Of Light Throughput", "dram__throughput.avg.pct_of_peak_sustained_elapsed", "%", "87.5"))
    lines.append(_row(0, k, "Memory Workload Analysis", "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum", "sector", "4,194,304"))
    lines.append(_row(0, k, "Memory Workload Analysis", "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum", "request", "262,144"))
    lines.append(_row(0, k, "Memory Workload Analysis", "smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct", "%", "100"))
    lines.append(_row(0, k, "Launch Statistics", "launch__registers_per_thread", "register/thread", "40"))
    csv_path.write_text("".join(lines))
    out = ncu_summary.render(ncu_summary.load(csv_path))
    assert "`_rmsnorm_row_kernel`" in out
    assert "| Duration | 12.34 usecond |" in out
    assert "| DRAM throughput % peak | 87.5 % |" in out
    assert "| Registers / thread | 40 register/thread |" in out
    assert "| Achieved occupancy % | - |" in out  # absent metrics render as '-'
    # The pct metric is a bytes-used percentage and is labeled as such, never as a sector count.
    assert "| Global load: bytes used per sector (%) | 100 % |" in out
    assert "sectors / request | 100" not in out
    # 16 sectors per request is what fully coalesced 128-bit lanes produce (32 lanes x 16 B / 32 B).
    assert "| Sectors per global load request (derived; fully coalesced = bytes per lane, 16 for 128-bit loads) | 16.00 |" in out
