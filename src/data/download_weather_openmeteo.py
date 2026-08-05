"""Download daily weather series for all sites from the Open-Meteo Archive API.

Open-Meteo's archive is ERA5/ERA5-Land reanalysis based, so it serves as the
tier-2 fallback weather source described in docs/研究方案.md while
AgERA5 (CDS) and NASA POWER stay unreachable from this network. Output is
also treated as the tier-3 "frozen" dataset once downloaded.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import requests

from config import DAILY_VARIABLES, END_YEAR, SITES, START_YEAR
from http_retry import request_with_retry

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
        # audit-v3 (4.5): request_with_retry returns a VALID response or
        # raises - the old loop could fall out with the last 429 and parse
        # its error body as data.
        def _do():
            return requests.get(ARCHIVE_URL, params=params, timeout=60)

        resp = request_with_retry(_do)
        payload = resp.json()
        df = pd.DataFrame(payload["daily"])
        frames.append(df)
        time.sleep(4)  # be polite to the free API's per-minute rate limit
    full = pd.concat(frames, ignore_index=True)
    full.rename(columns={"time": "date"}, inplace=True)
    full.insert(0, "site_id", site_id)
    return full


def main():
    import os
    import tempfile

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for site_id, meta in SITES.items():
        out_path = OUT_DIR / f"{site_id}_{START_YEAR}_{END_YEAR}_openmeteo.csv"
        # audit-v3 (4.5): existence is not validity - re-validate (rows +
        # full date range); a partial file is re-downloaded, never skipped.
        if out_path.exists():
            try:
                df = pd.read_csv(out_path)
                dates = pd.to_datetime(df["date"])
                expected = pd.date_range(f"{START_YEAR}-01-01", f"{END_YEAR}-12-31", freq="D")
                if len(dates) >= len(expected) and not dates.duplicated().any():
                    print(f"valid {site_id} ({len(df)} rows)")
                    continue
                print(f"INVALID {site_id} ({len(df)} rows) - re-downloading")
            except Exception as exc:
                print(f"INVALID {site_id}: {exc} - re-downloading")
        print(f"downloading {site_id} ({meta['name_cn']}) {START_YEAR}-{END_YEAR} ...")
        df = fetch_site(site_id, meta["lat"], meta["lon"])
        # atomic write: temp file + replace
        fd, tmp = tempfile.mkstemp(dir=str(OUT_DIR), prefix=out_path.name + ".", suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8-sig", newline="") as fh:
            df.to_csv(fh, index=False)
        os.replace(tmp, out_path)
        print(f"  saved {len(df)} rows -> {out_path.relative_to(out_path.parents[3])}")


if __name__ == "__main__":
    main()
