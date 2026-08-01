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

from config import SITES

SCRIPT = Path(__file__).resolve().parent / "optimize_cross_season.py"
PYTHON = Path(__file__).resolve().parents[2] / ".venv" / "Scripts" / "python.exe"
OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

POP_SIZE = 32
N_GEN = 20


def main():
    procs = {}
    for site_id in SITES:
        out_path = OUT_DIR / f"cross_season_pareto_{site_id}_loam.csv"
        if out_path.exists():
            print(f"skip {site_id}, already done")
            continue
        cmd = [str(PYTHON), str(SCRIPT), "--site", site_id, "--pop-size", str(POP_SIZE), "--n-gen", str(N_GEN)]
        procs[site_id] = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        print(f"launched {site_id} (pid {procs[site_id].pid})")

    for site_id, proc in procs.items():
        rc = proc.wait()
        print(f"{site_id} finished with exit code {rc}")


if __name__ == "__main__":
    main()
