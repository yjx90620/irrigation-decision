"""Download ERA5-Land daily soil moisture for all sites (Open-Meteo archive).

Used for two things (see docs/系统升级方案.md):
1. Realistic per-year initial soil water for the rotation simulations,
   replacing the "every season starts at field capacity" assumption that
   was inflating rainfed yields.
2. An observed-soil-moisture state input for the RL agent (a data
   dimension the prototype didn't have).
"""

import time
from pathlib import Path

import pandas as pd
import requests

from config import END_YEAR, SITES, START_YEAR

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "soil_moisture"
CHUNK_YEARS = 10

SOIL_MOISTURE_VARS = [
    "soil_moisture_0_to_7cm_mean",
    "soil_moisture_7_to_28cm_mean",
    "soil_moisture_28_to_100cm_mean",
    "soil_moisture_100_to_255cm_mean",
]


def year_chunks(start_year, end_year, size):
    y = start_year
    while y <= end_year:
        y2 = min(y + size - 1, end_year)
        yield y, y2
        y = y2 + 1


def fetch_site(site_id, lat, lon) -> pd.DataFrame:
    frames = []
    for y1, y2 in year_chunks(START_YEAR, END_YEAR, CHUNK_YEARS):
        params = {
            "latitude": lat, "longitude": lon,
            "start_date": f"{y1}-01-01", "end_date": f"{y2}-12-31",
            "daily": ",".join(SOIL_MOISTURE_VARS), "timezone": "auto",
        }
        for attempt in range(6):
            resp = requests.get(ARCHIVE_URL, params=params, timeout=90)
            if resp.status_code == 429:
                wait = 20 * (attempt + 1)
                print(f"  rate limited {site_id} {y1}-{y2}, waiting {wait}s")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            break
        df = pd.DataFrame(resp.json()["daily"])
        frames.append(df)
        time.sleep(4)
    full = pd.concat(frames, ignore_index=True).rename(columns={"time": "date"})
    full.insert(0, "site_id", site_id)
    return full


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for site_id, meta in SITES.items():
        out_path = OUT_DIR / f"{site_id}_{START_YEAR}_{END_YEAR}_soilmoisture.csv"
        if out_path.exists():
            print(f"skip {site_id}, already downloaded")
            continue
        print(f"downloading soil moisture for {site_id} ...")
        df = fetch_site(site_id, meta["lat"], meta["lon"])
        df.to_csv(out_path, index=False)
        print(f"  saved {len(df)} rows -> {out_path.name}")


if __name__ == "__main__":
    main()
