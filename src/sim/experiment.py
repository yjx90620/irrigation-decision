"""Digital-farm baseline experiment grid: site x soil x year x strategy.

Builds the six-baseline comparison table from docs/研究方案.md section 4.8
(NSGA-II excluded - see strategies.py). Uses Open-Meteo weather (full
1981-2025 record) since AgERA5 currently only covers a validation sample.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))

import pandas as pd
from aquacrop import AquaCropModel, Crop, InitialWaterContent

from config import END_YEAR, SITES, START_YEAR
from soils import STANDARD_SOILS, get_soil
from strategies import STRATEGIES
from weather import load_site_weather

PLANTING_DATE = "05/01"
HARVEST_DATE = "09/15"
OUT_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "baseline_experiment_results.csv"


def run_one(weather_year_df, soil, strategy_label, sim_year):
    crop = Crop("Maize", planting_date=PLANTING_DATE, harvest_date=HARVEST_DATE)
    init_wc = InitialWaterContent(value=["FC"])
    irr_mngt = STRATEGIES[strategy_label]()
    model = AquaCropModel(
        sim_start_time=f"{sim_year}/01/01",
        sim_end_time=f"{sim_year}/12/31",
        weather_df=weather_year_df,
        soil=soil,
        crop=crop,
        initial_water_content=init_wc,
        irrigation_management=irr_mngt,
    )
    model.run_model(till_termination=True)
    res = model.get_simulation_results()
    flux = model.get_water_flux()
    return {
        "dry_yield_t_ha": res["Dry yield (tonne/ha)"].iloc[0],
        "irrigation_mm": res["Seasonal irrigation (mm)"].iloc[0],
        "irrigation_events": int((flux["IrrDay"] > 0).sum()),
        "deep_perc_mm": flux["DeepPerc"].sum(),
        "runoff_mm": flux["Runoff"].sum(),
        "et_mm": (flux["Es"] + flux["Tr"]).sum(),
    }


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    t_start = time.time()
    n_done = 0
    for site_id in SITES:
        weather_df = load_site_weather(site_id)
        for soil_key in STANDARD_SOILS:
            soil = get_soil(soil_key)
            for year in range(START_YEAR, END_YEAR + 1):
                year_df = weather_df[
                    (weather_df["Date"] >= f"{year}-01-01") & (weather_df["Date"] <= f"{year}-12-31")
                ].reset_index(drop=True)
                if len(year_df) < 300:
                    continue  # incomplete year, skip
                for strategy_label in STRATEGIES:
                    try:
                        metrics = run_one(year_df, soil, strategy_label, year)
                    except Exception as exc:
                        print(f"FAIL {site_id} {soil_key} {year} {strategy_label}: {exc}")
                        continue
                    rows.append(
                        {"site_id": site_id, "soil": soil_key, "year": year, "strategy": strategy_label, **metrics}
                    )
                    n_done += 1
            print(f"done {site_id} / {soil_key}  ({n_done} runs, {time.time()-t_start:.0f}s elapsed)")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_PATH, index=False)
    print(f"saved {len(df)} rows -> {OUT_PATH}")


if __name__ == "__main__":
    main()
