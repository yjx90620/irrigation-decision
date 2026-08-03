"""Run NSGA-II for every site on loam soil (same settings as the first
hebei_central run, for cross-site comparability), skipping any that
already have a saved Pareto front.

audit-v2 (P0-12/P0-13): DEPRECATED for final results - optimize_nsga2.py
optimizes the SINGLE-SEASON prototype environment (each crop in isolation,
no rotation/quota), whose outputs were archived to
data/processed/invalidated/legacy_v1/. This script now refuses to run
without --allow-legacy so it cannot regenerate prototype results into the
formal results directory; the rotation-era cross-season optimization
(run_cross_season_all_sites.py / optimize_cross_season.py) is the
current pipeline.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))

from config import SITES
from optimize_nsga2 import run

OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-legacy", action="store_true",
                        help="audit-v2 P0-12: regenerate single-season prototype Pareto fronts (dev record only)")
    args = parser.parse_args()
    if not args.allow_legacy:
        raise SystemExit(
            "batch_optimize_loam.py regenerates SINGLE-SEASON PROTOTYPE results, archived to "
            "data/processed/invalidated/legacy_v1/ and invalidated by audit-v2 P0-12. "
            "Pass --allow-legacy for development record only."
        )
    for site_id in SITES:
        out_path = OUT_DIR / f"pareto_front_{site_id}_loam.csv"
        if out_path.exists():
            print(f"skip {site_id}, already have {out_path.name}")
            continue
        print(f"optimizing {site_id} / loam ...")
        front = run(site_id, "loam", pop_size=40, n_gen=25)
        front.to_csv(out_path, index=False)
        print(f"  saved {len(front)} policies -> {out_path.name}")
