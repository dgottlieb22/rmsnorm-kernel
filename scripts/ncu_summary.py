#!/usr/bin/env python
"""Turn ``ncu --csv --page details`` output into a markdown table of the metrics that matter here.

Usage: scripts/ncu_summary.py results/ncu/<name>.details.csv

The details page lists one row per (kernel launch, section, metric). The
metrics below are the ones the README's profiler section interprets:

- DRAM throughput as a percentage of peak, and bytes read and written: is the
  kernel memory-bound, and does it move more bytes than the operation needs?
- SM throughput: compute pressure (expected low for RMSNorm).
- Achieved and theoretical occupancy, registers per thread, the occupancy
  limiter: can the SM hold enough warps to cover memory latency?
- L1/TEX sectors per request for global loads: 4.0 means perfectly coalesced
  128-byte requests; higher means wasted sectors.
- Local memory traffic: register spills show up here.
- Duration, so the table stands on its own.

Metric names change between ncu versions; a metric missing from the export is
reported as "-" rather than failing the run. Check the first real export against
this list and update it if the names moved.
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

METRICS = [
    ("Duration", "gpu__time_duration.sum"),
    ("DRAM throughput % peak", "dram__throughput.avg.pct_of_peak_sustained_elapsed"),
    ("DRAM bytes read", "dram__bytes_read.sum"),
    ("DRAM bytes written", "dram__bytes_write.sum"),
    ("L2 throughput % peak", "lts__throughput.avg.pct_of_peak_sustained_elapsed"),
    ("SM throughput % peak", "sm__throughput.avg.pct_of_peak_sustained_elapsed"),
    ("Achieved occupancy %", "sm__warps_active.avg.pct_of_peak_sustained_active"),
    ("Theoretical occupancy %", "sm__maximum_warps_per_active_cycle_pct"),
    ("Registers / thread", "launch__registers_per_thread"),
    ("Block size", "launch__block_size"),
    ("Grid size", "launch__grid_size"),
    ("Waves per SM", "launch__waves_per_multiprocessor"),
    ("Occupancy limit: registers", "launch__occupancy_limit_registers"),
    ("Occupancy limit: warps", "launch__occupancy_limit_warps"),
    ("Occupancy limit: blocks", "launch__occupancy_limit_blocks"),
    ("Global load sectors / request", "smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct"),
    ("L1 global load sectors", "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum"),
    ("L1 global load requests", "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum"),
    ("Local load sectors (spills)", "l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum"),
    ("Local store sectors (spills)", "l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum"),
    ("Shared memory / block (bytes)", "launch__shared_mem_per_block"),
    ("Warp cycles per issued instruction", "smsp__average_warp_latency_per_inst_issued.ratio"),
    ("Stall: long scoreboard (%)", "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio"),
]


def load(path: Path):
    """{(launch id, kernel name): {metric name: (value, unit)}}"""
    launches = defaultdict(dict)
    with path.open(newline="") as fh:
        lines = [ln for ln in fh if not ln.startswith("==")]  # ncu prefixes banners with ==
    reader = csv.DictReader(lines)
    for row in reader:
        key = (row.get("ID", ""), row.get("Kernel Name", ""))
        launches[key][row.get("Metric Name", "")] = (row.get("Metric Value", ""), row.get("Metric Unit", ""))
    return launches


def render(launches) -> str:
    out = []
    for (lid, kernel), metrics in launches.items():
        short = kernel if len(kernel) <= 80 else kernel[:77] + "..."
        out.append(f"### launch {lid}: `{short}`\n")
        out.append("| metric | value |")
        out.append("|---|---|")
        for label, name in METRICS:
            if name in metrics:
                v, u = metrics[name]
                out.append(f"| {label} | {v} {u} |")
            else:
                out.append(f"| {label} | - |")
        # Derived: sectors per request for global loads, if both are present.
        s = metrics.get("l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum")
        r = metrics.get("l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum")
        if s and r:
            try:
                ratio = float(s[0].replace(",", "")) / float(r[0].replace(",", ""))
                out.append(f"| Sectors per global load request (derived; 4 = fully coalesced 32-bit lanes) | {ratio:.2f} |")
            except (ValueError, ZeroDivisionError):
                pass
        out.append("")
    return "\n".join(out)


def main(argv):
    if len(argv) != 2:
        print(__doc__)
        return 2
    print(render(load(Path(argv[1]))))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
