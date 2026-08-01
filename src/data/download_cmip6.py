"""Download CMIP6 HighResMIP climate projections (Open-Meteo Climate API).

Used for the climate-change comparisons in paper 1 (how optimal irrigation
strategy shifts) and paper 3 (temporal transfer: train on historical
climate, test on future). See docs/系统升级方案.md.

Downloads a 7-model ensemble for two windows:
  - historical 1991-2010: the models' own historical runs, needed as the
    reference for delta-change bias correction (comparing each model
    against itself removes its systematic bias)
  - future 2031-2050: mid-century. 2050 is the hard upper bound of the
    HighResMIP protocol these downscaled runs follow.

Note ET0 comes back as null from this API even though it's an accepted
parameter, so the radiation/wind/humidity inputs needed to compute it via
FAO-56 are downloaded instead - see src/sim/et0.py.

All 7 models are requested in a single call per site/period (the API
suffixes each variable with the model name), which keeps this to 10
requests total rather than 70.
"""

import time
from pathlib import Path

import pandas as pd
import requests

from config import SITES

CLIMATE_URL = "https://climate-api.open-meteo.com/v1/climate"
OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "cmip6"

MODELS = [
    "CMCC_CM2_VHR4",
    "FGOALS_f3_H",
    "HiRAM_SIT_HR",
    "MRI_AGCM3_2_S",
    "EC_Earth3P_HR",
    "MPI_ESM1_2_XR",
    "NICAM16_8S",
]

DAILY_VARS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "precipitation_sum",
    "shortwave_radiation_sum",
    "wind_speed_10m_mean",
    "relative_humidity_2m_mean",
]

PERIODS = {
    "historical": ("1991-01-01", "2010-12-31"),
    "future": ("2031-01-01", "2050-12-31"),
}


def fetch(site_id, lat, lon, start, end) -> pd.DataFrame:
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start, "end_date": end,
        "models": ",".join(MODELS),
        "daily": ",".join(DAILY_VARS),
    }
    for attempt in range(6):
        resp = requests.get(CLIMATE_URL, params=params, timeout=300)
        if resp.status_code == 429:
            wait = 30 * (attempt + 1)
            print(f"  rate limited, waiting {wait}s")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        break
    df = pd.DataFrame(resp.json()["daily"]).rename(columns={"time": "date"})
    df.insert(0, "site_id", site_id)
    return df


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for site_id, meta in SITES.items():
        for period, (start, end) in PERIODS.items():
            out_path = OUT_DIR / f"{site_id}_{period}_cmip6.csv"
            if out_path.exists():
                print(f"skip {site_id} {period}, already downloaded")
                continue
            print(f"downloading {site_id} {period} ({start}..{end}) x {len(MODELS)} models ...")
            df = fetch(site_id, meta["lat"], meta["lon"], start, end)
            df.to_csv(out_path, index=False)
            print(f"  saved {len(df)} rows, {len(df.columns)} cols -> {out_path.name}")
            time.sleep(5)


if __name__ == "__main__":
    main()
