"""对比1b: joint allocation vs. each site's own best FIXED alpha (from
scan_allocation.py's completed scan), at a larger search budget than the
original 对比1 (which only had an arbitrary alpha=0.5 control available).

Only the 3 sites the alpha scan actually completed have a scan-determined
best alpha (data/processed/allocation_scan.csv) - beijing_plain's alpha
scan never finished (all combos hit a confirmed AquaCrop-internal hang
tied to year 2013 + its short wheat cultivar, see scan_allocation.py and
论文一) and ningxia_irrigation is single-crop with no split to compare, so
both are out of scope for this specific comparison. beijing_plain still
has its original joint-vs-0.5 对比1 result (cross_season_pareto_
beijing_plain_loam.csv) - just not the strengthened fixed-best control.

Larger budget than the original run (pop=48/gen=30 vs 32/20, ~2.25x
evaluations) since this is the version meant to be the paper's headline
number, not a first pass.
"""

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))

from optimize_cross_season import BEST_FIXED_ALPHA
from run_manifest import csv_result_complete

SCRIPT = Path(__file__).resolve().parent / "optimize_cross_season.py"
PYTHON = sys.executable  # P1 (docs/审计修复计划.md): not a hardcoded venv path
OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

POP_SIZE = 48
N_GEN = 30

# audit-v2 (P0-13): a result file that merely exists is not "done".
RESULT_COLS = ["total_yield_t_ha", "total_irrigation_mm", "water_loss_mm", "yield_cv", "mode"]


def main():
    # optimize_cross_season.py's --site mode always writes to
    # cross_season_pareto_{site}_loam.csv unconditionally (no skip-if-exists
    # check) - that's the same path the original 对比1 run already used, so
    # move the original aside before launching and restore it after, instead
    # of letting the subprocess silently overwrite paper 1's existing result.
    procs = {}
    backups = {}
    for site_id in BEST_FIXED_ALPHA:
        out_path = OUT_DIR / f"cross_season_pareto_1b_{site_id}_loam.csv"
        if csv_result_complete(out_path, RESULT_COLS, min_rows=1):
            print(f"skip {site_id}, complete result exists")
            continue
        default_out = OUT_DIR / f"cross_season_pareto_{site_id}_loam.csv"
        if default_out.exists():
            backup = OUT_DIR / f"_orig_cross_season_pareto_{site_id}_loam.csv"
            default_out.replace(backup)
            backups[site_id] = backup
        cmd = [
            str(PYTHON), str(SCRIPT), "--site", site_id,
            "--pop-size", str(POP_SIZE), "--n-gen", str(N_GEN),
            "--modes", "joint", "fixed_best",
        ]
        log_path = OUT_DIR / f"_log_cross_season_1b_{site_id}.txt"
        procs[site_id] = (subprocess.Popen(cmd, stdout=open(log_path, "w"), stderr=subprocess.STDOUT), out_path, default_out)
        print(f"launched {site_id} (pid {procs[site_id][0].pid})")

    for site_id, (proc, expected_out, default_out) in procs.items():
        rc = proc.wait()
        if rc == 0 and default_out.exists():
            default_out.replace(expected_out)
        if site_id in backups:
            backups[site_id].replace(default_out)
        print(f"{site_id} finished with exit code {rc}")


if __name__ == "__main__":
    main()
