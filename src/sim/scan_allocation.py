"""1-D scan of the wheat/maize annual-quota split (alpha) per site and year.

Motivation: the first joint-vs-fixed-split comparison found joint
optimization *losing* to a control that pins alpha=0.5. A diagnostic
showed why - alpha matters a lot mechanically (total yield at Hebei swung
13.68 / 15.56 / 14.54 t/ha across alpha = 0.2 / 0.5 / 0.8 with SMT held
fixed, and the MaxIrrSeason caps do bind), but 0.5 happens to sit near the
optimum *at that site*, so the "control" was accidentally a strong
baseline while the joint optimizer had to search a 9-dim space on the same
evaluation budget.

That makes the more fundamental question this: does the optimal split vary
enough across sites and years that no single fixed split can serve? If it
does, allocation flexibility is worth having and the comparison should be
against the *best* fixed split rather than an arbitrary one. If it
doesn't, the honest conclusion is that a fixed rule suffices - which is
itself a useful, publishable result.

This scan is cheap (1-D, no evolutionary search) and answers that directly.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))

import numpy as np
import pandas as pd
from aquacrop import IrrigationManagement

from config import SITES
from cropping_systems import is_double_crop
from rotation import run_rotation_series

ANNUAL_QUOTA_MM = 450.0
ALPHAS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
SMT_LEVELS = {"conservative": [40] * 4, "moderate": [60] * 4, "aggressive": [80] * 4}
YEARS = list(range(2011, 2021))  # 10 years, includes wet and dry

OUT_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "allocation_scan.csv"


def scan_site(site_id, soil_key="loam"):
    rows = []
    for smt_label, smt in SMT_LEVELS.items():
        for alpha in ALPHAS:
            wheat_cap = ANNUAL_QUOTA_MM * alpha
            maize_cap = ANNUAL_QUOTA_MM * (1 - alpha)
            df = run_rotation_series(
                site_id, soil_key, YEARS,
                lambda: IrrigationManagement(irrigation_method=1, SMT=list(smt), MaxIrrSeason=wheat_cap),
                lambda: IrrigationManagement(irrigation_method=1, SMT=list(smt), MaxIrrSeason=maize_cap),
            )
            per_year = df.groupby("year").agg(
                total_yield=("dry_yield_t_ha", "sum"), total_irr=("irrigation_mm", "sum")
            )
            by_crop = df.groupby("crop")[["dry_yield_t_ha", "irrigation_mm"]].mean()
            rows.append({
                "site_id": site_id, "soil": soil_key, "smt_level": smt_label, "alpha": alpha,
                "mean_total_yield": per_year["total_yield"].mean(),
                "std_total_yield": per_year["total_yield"].std(),
                "mean_total_irr": per_year["total_irr"].mean(),
                "wheat_yield": by_crop.loc["wheat", "dry_yield_t_ha"],
                "maize_yield": by_crop.loc["maize", "dry_yield_t_ha"],
                "wheat_irr": by_crop.loc["wheat", "irrigation_mm"],
                "maize_irr": by_crop.loc["maize", "irrigation_mm"],
            })
            print(f"  {site_id} {smt_label} alpha={alpha:.1f} -> yield={rows[-1]['mean_total_yield']:.2f}")
    return pd.DataFrame(rows)


def main():
    existing = pd.read_csv(OUT_PATH) if OUT_PATH.exists() else pd.DataFrame()
    done = set(zip(existing["site_id"], existing["smt_level"])) if not existing.empty else set()

    frames = [existing] if not existing.empty else []
    for site_id in SITES:
        # alpha is the wheat/maize split of the annual quota, so it only has
        # meaning where both crops exist. Ningxia is single spring maize
        # (it lacks the growing degree days for winter wheat - see
        # cropping_systems.py), so there is nothing to allocate there.
        if not is_double_crop(site_id):
            print(f"skip {site_id}: single-crop site, no wheat/maize split to scan")
            continue
        if all((site_id, lvl) in done for lvl in SMT_LEVELS):
            print(f"skip {site_id}, already scanned")
            continue
        frames.append(scan_site(site_id))
        pd.concat(frames, ignore_index=True).to_csv(OUT_PATH, index=False)
    print(f"saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
