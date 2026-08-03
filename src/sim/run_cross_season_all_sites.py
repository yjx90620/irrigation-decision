"""Launch cross-season optimization for every site as parallel processes.

Site-level process parallelism is the right granularity here: thread-based
parallelism inside a single optimization gives no speedup (measured 18.5s
vs 18.8s for 12 individuals with 12 threads vs 1) because AquaCrop's numba
kernels hold the GIL, but separate processes scale fine and this machine
has cores to spare.
"""

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))

from config import SITES
from cropping_systems import is_double_crop
from run_manifest import csv_result_complete

SCRIPT = Path(__file__).resolve().parent / "optimize_cross_season.py"
PYTHON = sys.executable  # P1 (docs/审计修复计划.md): not a hardcoded venv path
OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

POP_SIZE = 32
N_GEN = 20

# audit-v2 (P0-13): a CSV that merely exists is not "done" - a crashed or
# NaN-filled optimization must be re-run, not skipped forever.
RESULT_COLS = ["total_yield_t_ha", "total_irrigation_mm", "water_loss_mm", "yield_cv", "mode"]


def main():
    procs = {}
    for site_id in SITES:
        # Cross-season allocation needs two crops sharing a quota; Ningxia
        # is single spring maize (see cropping_systems.py).
        if not is_double_crop(site_id):
            print(f"skip {site_id}: single-crop site, no cross-season allocation")
            continue
        out_path = OUT_DIR / f"cross_season_pareto_{site_id}_loam.csv"
        if csv_result_complete(out_path, RESULT_COLS, min_rows=1):
            print(f"skip {site_id}, complete result exists")
            continue
        cmd = [str(PYTHON), str(SCRIPT), "--site", site_id, "--pop-size", str(POP_SIZE), "--n-gen", str(N_GEN)]
        procs[site_id] = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        print(f"launched {site_id} (pid {procs[site_id].pid})")

    for site_id, proc in procs.items():
        rc = proc.wait()
        print(f"{site_id} finished with exit code {rc}")


if __name__ == "__main__":
    main()
