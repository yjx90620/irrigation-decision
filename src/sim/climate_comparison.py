"""audit-v3 对比5: how optimal irrigation strategies behave under the
CMIP6 delta-change future climate (2031-2050 signal applied to observed
weather sequencing) vs the historical record. This is the experiment
论文一 对比5 has been waiting on since the CMIP6 data arrived.

Two parts:

A. STRATEGY ROBUSTNESS - evaluate a small strategy set under historical
   vs future weather on the validation window (2011-2017, same window as
   the alpha scan; final-test years stay untouched):
   double-crop sites: scan-best (smt_level, alpha) / uniform threshold
   (SMT=50, alpha=0.5) / full irrigation (SMT=100, alpha=0.5)
   ningxia: threshold (SMT=50) / full (SMT=100) / rainfed
   -> data/processed/climate_robustness.csv (mean yield + irrigation per
   site x strategy x climate, plus the future-minus-historical delta)

B. OPTIMUM MOVEMENT - re-run the alpha scan (smt_level x alpha grid) under
   FUTURE weather only, for the double-crop sites, so we can see whether
   the best fixed split moves under climate change (the scan's headline is
   'optimum flips by site aridity' - does the climate signal shift it?).
   Subprocess-per-combo with a timeout, same pattern as scan_allocation.py
   (pathological AquaCrop states happen under extreme policies).
   -> data/processed/allocation_scan_future_{site}.csv +
      best_fixed_alpha_future.csv

Weather: build_future_weather() applies per-model-first ensemble deltas
(docs: climate_scenario.py) to the observed series; the perturbed record
keeps the observed day-to-day sequencing, so the evaluation window of the
perturbed series is the climate-changed counterpart of the same calendar
years.
"""

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))

import pandas as pd
from aquacrop import IrrigationManagement

from climate_scenario import build_future_weather, compute_deltas
from config import SITES
from cropping_systems import is_double_crop
from rotation import run_rotation_years_independent
from scan_allocation import ALPHAS, SMT_LEVELS
from temporal_split import SPLIT

YEARS = list(SPLIT.validation_years)
OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
PYTHON = sys.executable
COMBO_TIMEOUT_S = 90


def future_weather_path(site_id: str) -> Path:
    return OUT_DIR / f"future_weather_{site_id}.csv"


def build_future_weather_cached(site_id: str) -> pd.DataFrame:
    path = future_weather_path(site_id)
    if path.exists():
        return pd.read_csv(path, parse_dates=["Date"])
    df = build_future_weather(site_id, compute_deltas(site_id))
    df.to_csv(path, index=False)
    print(f"cached future weather -> {path}", flush=True)
    return df


def _smt_irr(smt, alpha):
    def wheat():
        return IrrigationManagement(irrigation_method=1, SMT=list(smt), MaxIrrSeason=450.0 * alpha)

    def maize():
        return IrrigationManagement(irrigation_method=1, SMT=list(smt), MaxIrrSeason=450.0 * (1 - alpha))

    return wheat, maize


def _system_means(df: pd.DataFrame):
    per_year = df.groupby("year").agg(total_yield=("dry_yield_t_ha", "sum"), total_irr=("irrigation_mm", "sum"))
    return per_year["total_yield"].mean(), per_year["total_irr"].mean()


def _best_scan_combo(site_id: str):
    scan = pd.read_csv(OUT_DIR / "allocation_scan.csv")
    sub = scan[scan["site_id"] == site_id]
    best = sub.loc[sub["mean_total_yield"].idxmax()]
    return best["smt_level"], float(best["alpha"])


def robustness_rows(site_id: str, weather_df, label: str) -> dict:
    rows = {}
    if is_double_crop(site_id):
        smt_best, alpha_best = _best_scan_combo(site_id)
        policies = {
            "scan_best": (SMT_LEVELS[smt_best], alpha_best),
            "uniform_threshold": ([50] * 4, 0.5),
            "full_irrigation": ([100] * 4, 0.5),
        }
    else:
        policies = {
            "threshold": ([50] * 4, 0.5),
            "full_irrigation": ([100] * 4, 0.5),
        }
    for name, (smt, alpha) in policies.items():
        wheat, maize = _smt_irr(smt, alpha)
        df = run_rotation_years_independent(site_id, "loam", YEARS, wheat, maize, weather_df=weather_df)
        y, i = _system_means(df)
        rows[name] = (y, i)
    return rows


def part_a():
    frames = []
    for site_id in SITES:
        hist = None  # run_rotation_years_independent loads observed weather
        fut = build_future_weather_cached(site_id)
        for climate, wdf in (("historical", hist), ("future", fut)):
            for name, (y, i) in robustness_rows(site_id, wdf, climate).items():
                frames.append({
                    "site_id": site_id, "strategy": name, "climate": climate,
                    "mean_yield_t_ha": y, "mean_irrigation_mm": i,
                })
                print(f"  {site_id} {name} {climate}: yield={y:.2f} irr={i:.0f}", flush=True)
    df = pd.DataFrame(frames)
    pivot = df.pivot_table(index=["site_id", "strategy"], columns="climate",
                           values=["mean_yield_t_ha", "mean_irrigation_mm"])
    pivot.columns = [f"{a}_{b}" for a, b in pivot.columns]
    pivot["delta_yield_t_ha"] = pivot["mean_yield_t_ha_future"] - pivot["mean_yield_t_ha_historical"]
    pivot["delta_irrigation_mm"] = pivot["mean_irrigation_mm_future"] - pivot["mean_irrigation_mm_historical"]
    pivot = pivot.reset_index()
    pivot.to_csv(OUT_DIR / "climate_robustness.csv", index=False)
    print(f"saved -> {OUT_DIR / 'climate_robustness.csv'}", flush=True)
    return pivot


def _run_combo_subprocess(site_id: str, smt_label: str, alpha: float, weather_path: Path) -> dict:
    cmd = [str(PYTHON), str(Path(__file__).resolve()),
           "--combo", site_id, smt_label, str(alpha), str(weather_path)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=COMBO_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return {"site_id": site_id, "smt_level": smt_label, "alpha": alpha,
                "mean_total_yield": float("nan"), "failure_reason": "timeout"}
    last_line = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
    if result.returncode == 0 and last_line.startswith("{"):
        return {"site_id": site_id, "smt_level": smt_label, "alpha": alpha, **json.loads(last_line)}
    return {"site_id": site_id, "smt_level": smt_label, "alpha": alpha,
            "mean_total_yield": float("nan"), "failure_reason": f"exit_code={result.returncode}"}


def part_b():
    rows = []
    for site_id in SITES:
        if not is_double_crop(site_id):
            continue
        weather_path = future_weather_path(site_id)
        for smt_label in SMT_LEVELS:
            for alpha in ALPHAS:
                rows.append(_run_combo_subprocess(site_id, smt_label, alpha, weather_path))
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "allocation_scan_future.csv", index=False)
    print(f"saved -> {OUT_DIR / 'allocation_scan_future.csv'}", flush=True)

    avg = df.groupby(["site_id", "alpha"])["mean_total_yield"].mean().reset_index()
    best = avg.loc[avg.groupby("site_id")["mean_total_yield"].idxmax(), ["site_id", "alpha"]]
    best = best.rename(columns={"alpha": "best_fixed_alpha_future"})
    best.to_csv(OUT_DIR / "best_fixed_alpha_future.csv", index=False)
    print(best.to_string(index=False), flush=True)
    print(f"saved -> {OUT_DIR / 'best_fixed_alpha_future.csv'}", flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--combo", nargs=4, default=None,
                        help="single-combo subprocess mode: site smt_label alpha weather_file")
    parser.add_argument("--parts", nargs="+", default=["a", "b"], choices=["a", "b"])
    args = parser.parse_args()

    if args.combo is not None:
        site_id, smt_label, alpha, weather_file = args.combo
        wdf = pd.read_csv(weather_file, parse_dates=["Date"])
        wheat = lambda: IrrigationManagement(
            irrigation_method=1, SMT=list(SMT_LEVELS[smt_label]), MaxIrrSeason=450.0 * float(alpha))
        maize = lambda: IrrigationManagement(
            irrigation_method=1, SMT=list(SMT_LEVELS[smt_label]), MaxIrrSeason=450.0 * (1 - float(alpha)))
        df = run_rotation_years_independent(site_id, "loam", YEARS, wheat, maize, weather_df=wdf)
        per_year = df.groupby("year").agg(total_yield=("dry_yield_t_ha", "sum"), total_irr=("irrigation_mm", "sum"))
        print(json.dumps({
            "mean_total_yield": per_year["total_yield"].mean(),
            "std_total_yield": per_year["total_yield"].std(),
            "mean_total_irr": per_year["total_irr"].mean(),
        }))
        raise SystemExit(0)

    for part in args.parts:
        if part == "a":
            part_a()
        else:
            part_b()
