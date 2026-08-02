"""Backfill wind_speed_10m_mean into the existing weather CSVs.

et0.py's validate_against_observed() found a +0.31-0.50 mm/d ET0 bias
traced to using wind_speed_10m_max (the only wind variable originally
requested in config.DAILY_VARIABLES) as if it were representative daily
wind - FAO-56's aerodynamic term wants the day's mean, not its max. This
was deferred (task #15) because Open-Meteo's daily archive quota was
exhausted; only the single new variable is fetched here (not a full
re-download) to keep the request small, then it's merged into each
site's existing CSV as an added column.
"""

import time
from pathlib import Path

import pandas as pd
import requests

from config import END_YEAR, SITES, START_YEAR

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "weather"
CHUNK_YEARS = 10


def year_chunks(start_year: int, end_year: int, size: int):
    y = start_year
    while y <= end_year:
        y2 = min(y + size - 1, end_year)
        yield y, y2
        y = y2 + 1


def fetch_wind_mean(lat: float, lon: float) -> pd.DataFrame:
    frames = []
    for y1, y2 in year_chunks(START_YEAR, END_YEAR, CHUNK_YEARS):
        params = {
            "latitude": lat, "longitude": lon,
            "start_date": f"{y1}-01-01", "end_date": f"{y2}-12-31",
            "daily": "wind_speed_10m_mean", "timezone": "auto",
        }
        payload = None
        for attempt in range(6):
            try:
                resp = requests.get(ARCHIVE_URL, params=params, timeout=60)
            except requests.RequestException as exc:
                # transient SSL/connection errors get their own short
                # backoff and don't eat into the rate-limit attempt budget's
                # exponential wait - a fixed 10s retry is enough for those
                print(f"  connection error {y1}-{y2}: {exc}")
                time.sleep(10)
                continue
            body = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
            if resp.status_code == 429 or body.get("error"):
                reason = body.get("reason", f"HTTP {resp.status_code}")
                # Open-Meteo enforces separate hourly/minutely quotas (see
                # download_cmip6.py) - a short backoff can't clear an hourly
                # one, confirmed by 10 consecutive 20-200s waits all still
                # getting "Hourly API request limit exceeded" here.
                wait = 660 if "hour" in reason.lower() else 70 if "minut" in reason.lower() else 20 * (attempt + 1)
                print(f"  {reason} - waiting {wait}s (attempt {attempt + 1}/6)")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            payload = resp.json()
            break
        if payload is None:
            raise RuntimeError(f"gave up on {y1}-{y2} after 6 attempts - re-run later to resume")
        frames.append(pd.DataFrame(payload["daily"]))
        time.sleep(4)
    full = pd.concat(frames, ignore_index=True)
    full.rename(columns={"time": "date"}, inplace=True)
    return full


def main():
    for site_id, meta in SITES.items():
        out_path = OUT_DIR / f"{site_id}_{START_YEAR}_{END_YEAR}_openmeteo.csv"
        df = pd.read_csv(out_path)
        if "wind_speed_10m_mean" in df.columns:
            print(f"skip {site_id}, already has wind_speed_10m_mean")
            continue
        print(f"backfilling wind_speed_10m_mean for {site_id} ({meta['name_cn']}) ...")
        wind = fetch_wind_mean(meta["lat"], meta["lon"])
        merged = df.merge(wind, on="date", how="left")
        assert merged["wind_speed_10m_mean"].notna().all(), f"{site_id}: unmerged rows after backfill"
        merged.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"  saved -> {out_path.name}")


if __name__ == "__main__":
    main()
