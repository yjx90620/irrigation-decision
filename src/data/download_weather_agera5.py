"""Download AgERA5 (CDS, tier-1 formal data) for a validation sample.

A full 1981-2025 x 5-site x N-variable AgERA5 pull is impractical here: CDS
serves one NetCDF file per day per request, and a single site/variable/year
request takes roughly 5-6 minutes end to end (queue + per-file processing).
A full 45-year pull per site/variable would take several hours BY ITSELF,
and this project needs 4 core variables x 5 sites, i.e. days of serial
wall-clock time.

Given that, AgERA5 here is used as a validation/calibration sample against
the Open-Meteo (ERA5-Land-based) series that already covers the full
1981-2025 record (see download_weather_openmeteo.py) and is the working
dataset for model development. This script pulls AGERA5_SAMPLE_YEARS for
all sites and the 4 variables AquaCrop actually needs, so the two sources
can be cross-checked. Extending SAMPLE_YEARS to the full range later is a
matter of raising the constant and re-running (it resumes automatically,
skipping any site/variable/year already downloaded).
"""

import shutil
import zipfile
from pathlib import Path

import cdsapi
import pandas as pd
import xarray as xr

from config import SITES

AGERA5_SAMPLE_YEARS = [2024]  # extend this list to pull more years later

CORE_VARIABLES = {
    "tmax": {"variable": "2m_temperature", "statistic": "24_hour_maximum"},
    "tmin": {"variable": "2m_temperature", "statistic": "24_hour_minimum"},
    "precip": {"variable": "precipitation_flux"},
    "et0": {"variable": "reference_evapotranspiration"},
}

OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "weather_agera5"
TMP_DIR = OUT_DIR / "_tmp"
MONTHS = [f"{m:02d}" for m in range(1, 13)]
DAYS = [f"{d:02d}" for d in range(1, 32)]
BOX_DEG = 0.3  # half-width of the request bounding box around each site


def site_area(lat: float, lon: float):
    return [lat + BOX_DEG, lon - BOX_DEG, lat - BOX_DEG, lon + BOX_DEG]  # N,W,S,E


def parse_zip_to_series(zip_path: Path, lat: float, lon: float) -> pd.DataFrame:
    rows = []
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            z.extract(name, TMP_DIR)
            nc_path = TMP_DIR / name
            ds = xr.open_dataset(nc_path)
            data_var = [v for v in ds.data_vars if v != "crs"][0]
            point = ds[data_var].sel(lat=lat, lon=lon, method="nearest")
            date = pd.Timestamp(ds["time"].values[0]).date()
            rows.append({"date": date, "value": float(point.values.squeeze())})
            ds.close()
            nc_path.unlink()
    return pd.DataFrame(rows)


def download_site_variable_year(client, site_id, lat, lon, var_key, var_params, year):
    out_csv = OUT_DIR / f"{site_id}_{var_key}_{year}.csv"
    if out_csv.exists():
        print(f"skip {site_id} {var_key} {year}, already downloaded")
        return
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = TMP_DIR / f"{site_id}_{var_key}_{year}.zip"
    params = {
        "version": "2_0",
        "year": str(year),
        "month": MONTHS,
        "day": DAYS,
        "area": site_area(lat, lon),
        "format": "zip",
        **var_params,
    }
    print(f"downloading {site_id} {var_key} {year} ...")
    client.retrieve("sis-agrometeorological-indicators", params, str(zip_path))
    df = parse_zip_to_series(zip_path, lat, lon)
    df.rename(columns={"value": var_key}, inplace=True)
    df.insert(0, "site_id", site_id)
    df.to_csv(out_csv, index=False)
    zip_path.unlink()
    print(f"  saved {len(df)} rows -> {out_csv.name}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    client = cdsapi.Client()
    for site_id, meta in SITES.items():
        for var_key, var_params in CORE_VARIABLES.items():
            for year in AGERA5_SAMPLE_YEARS:
                download_site_variable_year(
                    client, site_id, meta["lat"], meta["lon"], var_key, var_params, year
                )
    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR, ignore_errors=True)


if __name__ == "__main__":
    main()
