"""Download daily weather series for all sites from the Open-Meteo Archive API.

Open-Meteo's archive is ERA5/ERA5-Land reanalysis based, so it serves as the
tier-2 fallback weather source described in docs/研究方案.md while
AgERA5 (CDS) and NASA POWER stay unreachable from this network. Output is
also treated as the tier-3 "frozen" dataset once downloaded.
"""

import time
from pathlib import Path

import pandas as pd
import requests

from config import DAILY_VARIABLES, END_YEAR, SITES, START_YEAR

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "weather"
CHUNK_YEARS = 10  # request in ~10-year windows to keep responses small/reliable


def year_chunks(start_year: int, end_year: int, size: int):
    y = start_year
    while y <= end_year:
        y2 = min(y + size - 1, end_year)
        yield y, y2
        y = y2 + 1


def fetch_site(site_id: str, lat: float, lon: float) -> pd.DataFrame:
    frames = []
    for y1, y2 in year_chunks(START_YEAR, END_YEAR, CHUNK_YEARS):
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": f"{y1}-01-01",
            "end_date": f"{y2}-12-31",
            "daily": ",".join(DAILY_VARIABLES),
            "timezone": "auto",
        }
        for attempt in range(6):
            try:
                resp = requests.get(ARCHIVE_URL, params=params, timeout=60)
                if resp.status_code == 429:
                    wait = 20 * (attempt + 1)
                    print(f"  rate limited {site_id} {y1}-{y2}, waiting {wait}s")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                break
            except requests.RequestException as exc:
                if attempt == 5:
                    raise
                print(f"  retry {site_id} {y1}-{y2}: {exc}")
                time.sleep(10)
        payload = resp.json()
        df = pd.DataFrame(payload["daily"])
        frames.append(df)
        time.sleep(4)  # be polite to the free API's per-minute rate limit
    full = pd.concat(frames, ignore_index=True)
    full.rename(columns={"time": "date"}, inplace=True)
    full.insert(0, "site_id", site_id)
    return full


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for site_id, meta in SITES.items():
        out_path = OUT_DIR / f"{site_id}_{START_YEAR}_{END_YEAR}_openmeteo.csv"
        if out_path.exists():
            print(f"skip {site_id}, already downloaded -> {out_path.name}")
            continue
        print(f"downloading {site_id} ({meta['name_cn']}) {START_YEAR}-{END_YEAR} ...")
        df = fetch_site(site_id, meta["lat"], meta["lon"])
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"  saved {len(df)} rows -> {out_path.relative_to(out_path.parents[3])}")


if __name__ == "__main__":
    main()
