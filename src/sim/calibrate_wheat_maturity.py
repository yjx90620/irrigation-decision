"""Search for the shortest-season Maturity (GDD) per double-crop site that
still leaves every 1982-2025 year's simulated wheat harvest safely before
MAIZE_PLANTING (06/15) - the calibration step P0-1's fix (方案A, see
docs/审计修复计划.md) needs before rotation.py's date-order assertion can
pass for every year.

Senescence/HIstart are scaled with Maturity at the same ratios
cropping_systems.py's existing cultivars already use (Senescence ~0.727 x
Maturity, HIstart ~0.545 x Maturity for the standard cultivar; Beijing's
existing short cultivar uses the same ratios to two decimal places), so
this searches one parameter, not three independently.

Searches downward from the current Maturity in steps of 25 GDD until the
worst (latest-maturing) year over 1982-2025 clears a SAFETY_MARGIN_DAYS
buffer before 06/15 - not just zero violations, so a year added later
(more history, or the CMIP6 future period) doesn't immediately reopen the
bug at the boundary.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))

import pandas as pd
from aquacrop import AquaCropModel, Crop, InitialWaterContent, IrrigationManagement

from rotation import WHEAT_HARVEST, WHEAT_PLANTING
from soils import get_soil
from weather import load_site_weather

SENESCENCE_RATIO = 1600 / 2200
HISTART_RATIO = 1200 / 2200
SAFETY_MARGIN_DAYS = 7
STEP = 25
MAIZE_PLANTING_MD = (6, 15)


def worst_case_margin(site_id: str, maturity: int, years, soil_key: str = "loam") -> int:
    weather_df = load_site_weather(site_id)
    params = dict(
        Maturity=maturity, Senescence=round(maturity * SENESCENCE_RATIO), HIstart=round(maturity * HISTART_RATIO),
    )
    worst = None
    for year in years:
        crop = Crop("WheatGDD", planting_date=WHEAT_PLANTING, harvest_date=WHEAT_HARVEST, **params)
        model = AquaCropModel(
            sim_start_time=f"{year - 1}/{WHEAT_PLANTING}", sim_end_time=f"{year}/{WHEAT_HARVEST}",
            weather_df=weather_df, soil=get_soil(soil_key), crop=crop,
            initial_water_content=InitialWaterContent(),
            irrigation_management=IrrigationManagement(irrigation_method=1, SMT=[100, 100, 100, 100]),
        )
        model.run_model(till_termination=True)
        harvest = pd.Timestamp(str(model.get_simulation_results()["Harvest Date (YYYY/MM/DD)"].iloc[0])[:10])
        margin = (pd.Timestamp(year, *MAIZE_PLANTING_MD) - harvest).days
        if worst is None or margin < worst:
            worst = margin
    return worst


def calibrate(site_id: str, start_maturity: int, years=range(1982, 2026)):
    maturity = start_maturity
    while True:
        margin = worst_case_margin(site_id, maturity, years)
        print(f"  {site_id} Maturity={maturity}: worst-case margin = {margin} days")
        if margin >= SAFETY_MARGIN_DAYS:
            params = dict(
                Maturity=maturity, Senescence=round(maturity * SENESCENCE_RATIO),
                HIstart=round(maturity * HISTART_RATIO),
            )
            print(f"  -> settled: {params}, worst-case margin {margin} days")
            return params
        maturity -= STEP
        if maturity < 1000:
            raise RuntimeError(f"{site_id}: maturity search fell below 1000 GDD without clearing the margin")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--site", required=True)
    parser.add_argument("--start", type=int, default=2200)
    args = parser.parse_args()
    calibrate(args.site, args.start)
