"""Stage-wise marginal water value (研究方案 4.10 分析二 / paper 1 对比4):
how much system yield one more increment of irrigation buys, broken down
by which of AquaCrop's 4 growth stages it's applied in, and by crop
(wheat vs maize, or just maize at Ningxia).

Method (P1-2, audit-v2): this is an SMT-parameter LOCAL SENSITIVITY
measure, not a fixed-increment water experiment - raising one stage's
SMT by DELTA_PP percentage points asks the scheduler to be more generous
there, which usually (not always) raises actual applied irrigation.
MWV = delta(system yield) / delta(irrigation applied) is reported ONLY
when delta irrigation > 0; otherwise it is NaN (a threshold change that
did not move water has no marginal water value, and fabricating one
would be wrong). The baseline/perturbed irrigation and yield are saved
alongside so the ratio is traceable.

A high MWV means that stage is where additional water pays off most; a
low or negative MWV means water applied there is largely going to waste
(deep percolation, runoff, or simply not the growth phase driving yield).

As with optimize_cross_season.py/optimize_nsga2.py, AquaCrop's 4 SMT
slots follow its own internal growth-stage split, not exactly the
苗期/拔节/抽雄吐丝/灌浆 phenological stages named in 研究方案 4.6 - so this
reports "stage 1-4" rather than claiming a precise phenological label.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))

import pandas as pd
from aquacrop import IrrigationManagement

from config import SITES
from cropping_systems import is_double_crop
from rotation import run_rotation_series

BASELINE_SMT = [50, 50, 50, 50]
DELTA_PP = 15  # percentage points to raise one stage by
# audit-v3 (5.1): MWV runs on the validation window (selection-time
# analysis), never on the final-test years.
from temporal_split import SPLIT

EVAL_YEARS = list(SPLIT.validation_years)

OUT_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "marginal_water_value.csv"


def _smt_with_bump(stage_idx):
    smt = list(BASELINE_SMT)
    smt[stage_idx] = min(smt[stage_idx] + DELTA_PP, 100)
    return smt


def _irr(smt):
    return lambda: IrrigationManagement(irrigation_method=1, SMT=list(smt))


def _system_totals(df):
    per_year = df.groupby("year").agg(yield_=("dry_yield_t_ha", "sum"), irr=("irrigation_mm", "sum"))
    return per_year["yield_"].mean(), per_year["irr"].mean()


def compute_mwv_for_site(site_id, soil_key="loam"):
    double_crop = is_double_crop(site_id)
    rows = []

    baseline_df = run_rotation_series(
        site_id, soil_key, EVAL_YEARS, _irr(BASELINE_SMT), _irr(BASELINE_SMT)
    )
    base_yield, base_irr = _system_totals(baseline_df)

    crops_to_test = ["wheat", "maize"] if double_crop else ["maize"]
    for crop in crops_to_test:
        for stage_idx in range(4):
            bumped_smt = _smt_with_bump(stage_idx)
            wheat_irr = _irr(bumped_smt) if crop == "wheat" else _irr(BASELINE_SMT)
            maize_irr = _irr(bumped_smt) if crop == "maize" else _irr(BASELINE_SMT)

            df = run_rotation_series(site_id, soil_key, EVAL_YEARS, wheat_irr, maize_irr)
            bumped_yield, bumped_irr = _system_totals(df)

            d_yield = bumped_yield - base_yield
            d_irr = bumped_irr - base_irr
            # P1-2 (audit-v2): raising an SMT threshold does NOT guarantee
            # more water actually gets applied (the scheduler may already
            # be above the new threshold, or the season may not need more) -
            # when delta irrigation is not positive there is no marginal
            # water value to report, only a NaN, never a fabricated ratio.
            mwv = d_yield / d_irr if d_irr > 1e-6 else float("nan")

            rows.append({
                "site_id": site_id, "crop": crop, "stage": stage_idx + 1,
                "delta_yield_t_ha": d_yield, "delta_irrigation_mm": d_irr, "mwv_t_ha_per_mm": mwv,
                # P1-2 (audit-v2): the baseline/perturbed pairing, so the
                # number is traceable to the two runs it came from instead
                # of being a bare ratio.
                "smt_delta_pp": DELTA_PP,
                "baseline_irrigation_mm": base_irr, "perturbed_irrigation_mm": bumped_irr,
                "baseline_yield_t_ha": base_yield, "perturbed_yield_t_ha": bumped_yield,
            })
            print(f"  {site_id} {crop} stage{stage_idx + 1}: MWV={mwv:.4f} t/ha per mm")

    return pd.DataFrame(rows)


def main():
    existing = pd.read_csv(OUT_PATH) if OUT_PATH.exists() else pd.DataFrame()
    # audit-v2 (P0-13): a site whose MWV rows are missing or NaN (e.g. the
    # delta-irrigation <= 0 cases) is not "done" and must be recomputed -
    # file existence alone never counts as completion.
    if not existing.empty and {"site_id", "crop", "stage", "mwv_t_ha_per_mm"}.issubset(existing.columns):
        complete = existing[existing["mwv_t_ha_per_mm"].notna()]
        done = set(complete["site_id"])
    else:
        done = set()

    rows = [] if existing.empty else [existing]
    for site_id in SITES:
        if site_id in done:
            print(f"skip {site_id}, already computed")
            continue
        print(f"=== {site_id} ===")
        rows.append(compute_mwv_for_site(site_id))
        pd.concat(rows, ignore_index=True).to_csv(OUT_PATH, index=False)

    print(f"saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
