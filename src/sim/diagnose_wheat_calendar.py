"""Diagnostic for P0-1 (see docs/审计修复计划.md): find every year where
winter wheat's actual simulated harvest date falls on or after
MAIZE_PLANTING (06/15), for each double-crop site's current cultivar
parameters (cropping_systems.py's wheat_params_for()).

Uses full/generous irrigation (SMT=100 throughout) deliberately: AquaCrop
accelerates senescence/maturity under water stress, so a stressed run
would tend to harvest *earlier*, not later - full irrigation is the
worst-case (latest-maturing) test for calendar safety margin, which is
what P0-1's fix needs to be safe against.

Not wired into the rotation engine - this is a one-off calibration tool,
run manually while tuning cropping_systems.py's Maturity/Senescence/
HIstart, then re-run to confirm the fix before moving on.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))

import pandas as pd
from aquacrop import AquaCropModel, Crop, InitialWaterContent, IrrigationManagement

from cropping_systems import CROPPING_SYSTEMS, is_double_crop, wheat_params_for
from rotation import WHEAT_HARVEST, WHEAT_PLANTING
from soils import get_soil
from weather import load_site_weather

MAIZE_PLANTING_MD = (6, 15)  # month, day - rotation.py's MAIZE_PLANTING="06/15"


def wheat_harvest_date(site_id: str, year: int, soil_key: str = "loam"):
    weather_df = load_site_weather(site_id)
    crop = Crop("WheatGDD", planting_date=WHEAT_PLANTING, harvest_date=WHEAT_HARVEST, **wheat_params_for(site_id))
    model = AquaCropModel(
        sim_start_time=f"{year - 1}/{WHEAT_PLANTING}",
        sim_end_time=f"{year}/{WHEAT_HARVEST}",
        weather_df=weather_df,
        soil=get_soil(soil_key),
        crop=crop,
        initial_water_content=InitialWaterContent(),  # default field capacity - irrelevant here, full irrigation anyway
        irrigation_management=IrrigationManagement(irrigation_method=1, SMT=[100, 100, 100, 100]),
    )
    model.run_model(till_termination=True)
    results = model.get_simulation_results()
    return pd.Timestamp(str(results["Harvest Date (YYYY/MM/DD)"].iloc[0])[:10])


def diagnose(site_id: str, years, soil_key: str = "loam") -> pd.DataFrame:
    rows = []
    for year in years:
        harvest = wheat_harvest_date(site_id, year, soil_key)
        cutoff = pd.Timestamp(year, *MAIZE_PLANTING_MD)
        rows.append({
            "site_id": site_id, "year": year, "harvest_date": harvest.date(),
            "days_before_maize_planting": (cutoff - harvest).days,
            "violates": harvest >= cutoff,
        })
    return pd.DataFrame(rows)


def main():
    years = range(1982, 2026)
    for site_id in CROPPING_SYSTEMS:
        if not is_double_crop(site_id):
            continue
        df = diagnose(site_id, years)
        n_viol = df["violates"].sum()
        margin = df["days_before_maize_planting"]
        print(f"\n=== {site_id} (Maturity={wheat_params_for(site_id)['Maturity']}) ===")
        print(f"  violations (harvest >= 06/15): {n_viol}/{len(df)}")
        print(f"  margin days: min={margin.min()}, p5={margin.quantile(0.05):.1f}, median={margin.median()}")
        if n_viol:
            print(df[df.violates][["year", "harvest_date", "days_before_maize_planting"]].to_string(index=False))


if __name__ == "__main__":
    main()
