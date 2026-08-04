"""audit-v3 (5.3): fixed-dose stage-wise marginal yield response - the
paper's 对比4 replacement.

The old MWV perturbed an SMT threshold and read the resulting irrigation
delta; the audit correctly showed a threshold bump does not guarantee
more water gets applied (delta_irr <= 0 entries, tiny denominators).
This version implements the audit's preferred design:

1. A deterministic baseline schedule (threshold_policy, safety off) is
   rolled out in the RL env and the per-decision-day applied mm recorded.
2. For each (site, crop, stage, delta, year): a single fixed dose is
   added/subtracted on one pre-defined pulse date inside that stage
   (calendar stage boundaries per crop; pulse = stage midpoint).
3. -delta / 0 / +delta are each replayed through the env with the
   schedule, and the ACTUAL delivered irrigation (irr_cum delta) is used
   for the response: marginal = (yield_plus - yield_minus) /
   (irr_plus - irr_minus), only when delta_irr > epsilon, else
   status='undefined_no_actual_delta'.
4. Full key set (site, crop, stage, delta, year) with explicit statuses;
   resume requires the complete key set, never "some rows exist".

Stages (calendar, pre-defined; AquaCrop's own 4 internal stages are not
exposed with stable dates, so the paper states the calendar rule):
  wheat:    [10/10-02/28], [03/01-04/14], [04/15-05/14], [05/15-harvest]
  maize:    [06/15-07/14], [07/15-08/09], [08/10-08/29], [08/30-harvest]
  spring:   [04/25-05/31], [06/01-07/09], [07/10-08/09], [08/10-harvest]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rl"))

import numpy as np
import pandas as pd

from config import SITES
from experiment_config import PRIMARY_CONFIG
from rotation_env import RotationIrrigationEnv, threshold_policy
from temporal_split import SPLIT

YEARS = list(SPLIT.validation_years)
DELTAS = [5.0, 10.0, 15.0]
EPSILON = 1.0  # delta_irr must exceed 1mm to define a marginal response
OUT_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "marginal_response_v2.csv"

# calendar stage boundaries per crop (month/day) - see module docstring
STAGE_RANGES = {
    "wheat": [("10/10", "02/28"), ("03/01", "04/14"), ("04/15", "05/14"), ("05/15", "06/25")],
    "maize": [("06/15", "07/14"), ("07/15", "08/09"), ("08/10", "08/29"), ("08/30", "10/05")],
    "spring_maize": [("04/25", "05/31"), ("06/01", "07/09"), ("07/10", "08/09"), ("08/10", "09/30")],
}


def _stage_midpoint(crop, stage_idx, year):
    start, end = STAGE_RANGES[crop][stage_idx]
    s = pd.Timestamp(f"{year}-{start.replace('/', '-')}")
    e = pd.Timestamp(f"{year}-{end.replace('/', '-')}")
    return (s + (e - s) / 2).date()


def _schedule(site_id, year):
    """Roll out threshold_policy (safety OFF, so the recorded schedule is
    exactly what the policy asked) and return {date: applied_mm}."""
    env = RotationIrrigationEnv(site_id, "loam", year, config=PRIMARY_CONFIG, safety_enabled=False)
    state = env.reset()
    schedule = {}
    while not env.done:
        date = pd.Timestamp(env.model._clock_struct.step_start_time).date()
        action = threshold_policy(state)
        state, reward, done, info = env.step(action)
        schedule[date] = float(action)
    return schedule, env


def _run_schedule(site_id, year, schedule, label):
    """Replay a schedule through the env (safety off); return
    (yield, actual_irrigation)."""
    env = RotationIrrigationEnv(site_id, "loam", year, config=PRIMARY_CONFIG, safety_enabled=False)
    state = env.reset()
    applied_sum = 0.0
    while not env.done:
        date = pd.Timestamp(env.model._clock_struct.step_start_time).date()
        action = schedule.get(date, 0.0)
        state, reward, done, info = env.step(float(action))
        applied_sum += info["actual_model_irrigation_mm"]
    return info["total_yield_t_ha"], applied_sum


def central_marginal_response(yield_minus, yield_plus, irr_minus, irr_plus, epsilon=EPSILON):
    delta_irr = irr_plus - irr_minus
    if delta_irr <= epsilon:
        return None, delta_irr
    return (yield_plus - yield_minus) / delta_irr, delta_irr


def compute_mwv_for_site(site_id):
    rows = []
    baseline, _ = _schedule(site_id, YEARS[0])  # schedule shape only (dates stable per crop calendar)
    for year in YEARS:
        schedule, _ = _schedule(site_id, year)
        for crop in STAGE_RANGES:
            if crop == "wheat" and not _double_crop(site_id):
                continue
            if crop == "spring_maize" and _double_crop(site_id):
                continue
            for stage_idx in range(4):
                pulse = _stage_midpoint(crop, stage_idx, year)
                for delta in DELTAS:
                    minus = dict(schedule)
                    plus = dict(schedule)
                    minus[pulse] = max(0.0, minus.get(pulse, 0.0) - delta)
                    plus[pulse] = plus.get(pulse, 0.0) + delta
                    y_minus, i_minus = _run_schedule(site_id, year, minus, "minus")
                    y_base, i_base = _run_schedule(site_id, year, schedule, "base")
                    y_plus, i_plus = _run_schedule(site_id, year, plus, "plus")
                    mwv, d_irr = central_marginal_response(y_minus, y_plus, i_minus, i_plus)
                    rows.append({
                        "site_id": site_id, "crop": crop, "stage": stage_idx + 1,
                        "delta_mm": delta, "year": year, "pulse_date": str(pulse),
                        "yield_minus": y_minus, "yield_baseline": y_base, "yield_plus": y_plus,
                        "irrigation_minus": i_minus, "irrigation_baseline": i_base,
                        "irrigation_plus": i_plus,
                        "delta_irrigation_mm": d_irr,
                        "marginal_response": mwv,
                        "status": "success" if mwv is not None else "undefined_no_actual_delta",
                    })
    return pd.DataFrame(rows)


def _double_crop(site_id):
    from cropping_systems import is_double_crop

    return is_double_crop(site_id)


def main():
    frames = []
    for site_id in SITES:
        print(f"=== {site_id} ===")
        frames.append(compute_mwv_for_site(site_id))
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"saved -> {OUT_PATH} ({len(df)} rows; "
          f"undefined={int((df['status'] == 'undefined_no_actual_delta').sum())})")


if __name__ == "__main__":
    main()
