"""Launch scan_allocation.py per double-cropping site as parallel
processes - same site-level-parallelism pattern as
run_cross_season_all_sites.py, for the same reason: AquaCrop's numba
kernels hold the GIL, so process-level parallelism is what actually
scales, and this machine has plenty of idle cores to use for it.
"""

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))

from config import SITES
from cropping_systems import is_double_crop

SCRIPT = Path(__file__).resolve().parent / "scan_allocation.py"
PYTHON = Path(__file__).resolve().parents[2] / ".venv" / "Scripts" / "python.exe"
OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


def main():
    procs = {}
    for site_id in SITES:
        if not is_double_crop(site_id):
            continue
        if (OUT_DIR / f"allocation_scan_{site_id}.csv").exists():
            print(f"skip {site_id}, already done")
            continue
        cmd = [str(PYTHON), str(SCRIPT), "--site", site_id]
        procs[site_id] = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        print(f"launched {site_id} (pid {procs[site_id].pid})")

    for site_id, proc in procs.items():
        rc = proc.wait()
        print(f"{site_id} finished with exit code {rc}")

    # merge per-site files into the combined allocation_scan.csv
    import pandas as pd

    frames = [pd.read_csv(f) for f in OUT_DIR.glob("allocation_scan_*.csv")]
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(OUT_DIR / "allocation_scan.csv", index=False)
        print(f"merged -> {OUT_DIR / 'allocation_scan.csv'}")


if __name__ == "__main__":
    main()
