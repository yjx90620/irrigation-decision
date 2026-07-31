"""Run NSGA-II for every site on loam soil (same settings as the first
hebei_central run, for cross-site comparability), skipping any that
already have a saved Pareto front."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))

from config import SITES
from optimize_nsga2 import run

OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

if __name__ == "__main__":
    for site_id in SITES:
        out_path = OUT_DIR / f"pareto_front_{site_id}_loam.csv"
        if out_path.exists():
            print(f"skip {site_id}, already have {out_path.name}")
            continue
        print(f"optimizing {site_id} / loam ...")
        front = run(site_id, "loam", pop_size=40, n_gen=25)
        front.to_csv(out_path, index=False)
        print(f"  saved {len(front)} policies -> {out_path.name}")
