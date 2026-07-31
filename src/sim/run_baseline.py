"""First end-to-end digital-farm sanity run: summer maize at one site under
rainfed / full-irrigation / soil-moisture-threshold baselines.

This validates that weather data + AquaCrop standard soil + default maize
parameters + AquaCropModel actually run together before scaling up to the
full site x year x soil x strategy experiment grid from docs/研究方案.md
section 4.5.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))

import pandas as pd
from aquacrop import AquaCropModel, Crop, InitialWaterContent, IrrigationManagement

from soils import get_soil
from weather import load_site_weather

SITE_ID = "hebei_central"
SIM_YEAR = 2020
PLANTING_DATE = "05/01"  # MM/DD, typical summer maize sowing window in the plan's region set
HARVEST_DATE = "09/15"


def run(irrigation_management: IrrigationManagement, label: str):
    weather_df = load_site_weather(SITE_ID)
    weather_df = weather_df[
        (weather_df["Date"] >= f"{SIM_YEAR}-01-01") & (weather_df["Date"] <= f"{SIM_YEAR}-12-31")
    ].reset_index(drop=True)

    soil = get_soil("loam")
    crop = Crop("Maize", planting_date=PLANTING_DATE, harvest_date=HARVEST_DATE)
    init_wc = InitialWaterContent(value=["FC"])  # start at field capacity

    model = AquaCropModel(
        sim_start_time=f"{SIM_YEAR}/01/01",
        sim_end_time=f"{SIM_YEAR}/12/31",
        weather_df=weather_df,
        soil=soil,
        crop=crop,
        initial_water_content=init_wc,
        irrigation_management=irrigation_management,
    )
    model.run_model(till_termination=True)
    out = model.get_simulation_results()
    yield_t_ha = out["Dry yield (tonne/ha)"].iloc[0]
    irr_mm = out["Seasonal irrigation (mm)"].iloc[0]
    print(f"{label:20s} yield={yield_t_ha:6.2f} t/ha  irrigation={irr_mm:7.1f} mm")


def main():
    print(f"site={SITE_ID} year={SIM_YEAR} soil=loam crop=Maize(default)")
    run(IrrigationManagement(irrigation_method=0), "rainfed")
    run(IrrigationManagement(irrigation_method=1, SMT=[100, 100, 100, 100]), "full_irrigation")
    run(IrrigationManagement(irrigation_method=1, SMT=[50, 50, 50, 50]), "threshold_50pct")


if __name__ == "__main__":
    main()
