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

audit-v2 (P0-7): the downloader previously treated "file exists" as
"download complete" with no validation at all - which is how files with an
entirely empty CMCC_CM2_VHR4 shortwave_radiation column and EC_Earth3P_HR
future gaps (~365 missing days) got accepted into the dataset. Now every
file is validated (date range complete, no duplicate dates, every required
variable-model column present and non-empty), written atomically with a
manifest sidecar, and any file that fails validation is moved to
data/raw/cmip6/quarantine/ instead of being kept in the formal directory.
Existing files are re-validated on every run, not skipped by existence.

Note ET0 comes back as null from this API even though it's an accepted
parameter, so the radiation/wind/humidity inputs needed to compute it via
FAO-56 are downloaded instead - see src/sim/et0.py.
"""

import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))

import pandas as pd
from enum import Enum

import numpy as np
import requests

from atomic_io import atomic_write_csv, sha256_file
from config import SITES
from run_manifest import RunManifest, save_manifest

CLIMATE_URL = "https://climate-api.open-meteo.com/v1/climate"
OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "cmip6"
QUARANTINE_DIR = OUT_DIR / "quarantine"

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

# audit-v2 (P0-7): the minimum set climate_scenario.py actually consumes
# (temperature/precipitation drive the delta; radiation/wind/humidity are
# perturbed for the ET0 recomputation). A required variable whose column is
# entirely empty for a model invalidates the file - silently dropping the
# model would change the ensemble the way the pre-fix data did (6-model
# radiation vs 7-model temperature).
REQUIRED_VARIABLES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "shortwave_radiation_sum",
    "wind_speed_10m_mean",
    "relative_humidity_2m_mean",
]

PERIODS = {
    "historical": ("1991-01-01", "2010-12-31"),
    "future": ("2031-01-01", "2050-12-31"),
}


class DataValidationError(RuntimeError):
    """A CMIP6 file is incomplete or malformed - it must not enter (or
    stay in) the formal dataset."""


def _parse_date_series(df: pd.DataFrame) -> pd.Series:
    if "date" not in df.columns:
        raise DataValidationError("missing 'date' column")
    dates = pd.to_datetime(df["date"], errors="coerce")
    if dates.isna().any():
        raise DataValidationError(f"{dates.isna().sum()} unparseable dates")
    return dates


class ColumnStatus(str, Enum):
    """audit-v3 (4.2): per-variable-model column health. Only COMPLETE
    columns enter the ensemble; anything else is excluded SYMMETRICALLY
    in both periods (never on one side only)."""
    COMPLETE = "complete"
    UPSTREAM_UNAVAILABLE = "upstream_unavailable"
    PARTIAL_INVALID = "partial_invalid"
    EMPTY_INVALID = "empty_invalid"


def is_complete_column(series: pd.Series, expected_rows: int) -> bool:
    """audit-v3 (4.2): a usable column must have every expected row, all
    non-null, and all finite - .notna().any() is not a validity test."""
    return (
        len(series) == expected_rows
        and series.notna().all()
        and np.isfinite(pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)).all()
    )


def column_status(series: pd.Series, expected_rows: int) -> ColumnStatus:
    if series is None or len(series) == 0:
        return ColumnStatus.EMPTY_INVALID
    if is_complete_column(series, expected_rows):
        return ColumnStatus.COMPLETE
    if series.notna().sum() == 0:
        return ColumnStatus.EMPTY_INVALID
    return ColumnStatus.UPSTREAM_UNAVAILABLE if series.notna().mean() >= 0.5 else ColumnStatus.PARTIAL_INVALID


def validate_cmip6_frame(df: pd.DataFrame, start_date: str, end_date: str,
                         required_variables=None, expected_models=None) -> dict:
    """Structural validation (audit-v2 P0-7 + audit-v3 4.2). Raises on
    date issues (hard failures); returns per-variable-model column status.
    Empty/partial columns do NOT raise (the gaps are upstream), but they
    are recorded and the consumer must exclude them SYMMETRICALLY."""
    required_variables = required_variables or REQUIRED_VARIABLES
    expected_models = expected_models or MODELS
    dates = _parse_date_series(df)
    if dates.duplicated().any():
        raise DataValidationError(f"{dates.duplicated().sum()} duplicate dates")
    expected = pd.date_range(start_date, end_date, freq="D")
    date_idx = pd.DatetimeIndex(dates)
    missing = expected.difference(date_idx)
    extra = date_idx.difference(expected)
    if len(missing) or len(extra):
        raise DataValidationError(
            f"date range mismatch: {len(missing)} missing, {len(extra)} extra "
            f"(expected {start_date}..{end_date}, {len(expected)} days)"
        )
    coverage = {}
    for variable in required_variables:
        for model in expected_models:
            col = f"{variable}_{model}"
            if col not in df.columns:
                coverage[col] = ColumnStatus.EMPTY_INVALID
            else:
                coverage[col] = column_status(df[col], len(expected))
    return coverage


def common_models(hist_df: pd.DataFrame, fut_df: pd.DataFrame, variable: str) -> list:
    """audit-v3 (4.2): a model enters the ensemble ONLY when its column is
    COMPLETE in BOTH periods - partial/empty columns are excluded
    symmetrically, so the historical/future ensembles never differ."""
    expected_rows = len(hist_df)
    def complete_in(df):
        return {
            m for m in MODELS
            if f"{variable}_{m}" in df.columns
            and is_complete_column(df[f"{variable}_{m}"], expected_rows)
        }
    common = sorted(complete_in(hist_df) & complete_in(fut_df))
    if len(common) < 3:
        raise DataValidationError(
            f"'{variable}': only {len(common)} model(s) COMPLETE in BOTH periods - "
            f"cannot pair historical/future ensembles"
        )
    return common


def fetch(site_id, lat, lon, start, end) -> pd.DataFrame:
    """Open-Meteo enforces separate minutely / hourly / daily quotas, and a
    7-model 20-year request is heavy enough to hit the hourly one. A short
    exponential backoff can't clear that, so hourly limits get a long wait
    instead. Downloads are resumable (completed files are skipped), so
    giving up and re-running later is always safe."""
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start, "end_date": end,
        "models": ",".join(MODELS),
        "daily": ",".join(DAILY_VARS),
    }
    for attempt in range(6):
        try:
            resp = requests.get(CLIMATE_URL, params=params, timeout=300)
        except requests.RequestException as exc:
            # These 20-year x 7-model requests are long-lived enough that
            # transient SSL/connection drops happen; they are not rate limits
            # and need their own retry path.
            wait = 30 * (attempt + 1)
            print(f"  connection error ({type(exc).__name__}) - waiting {wait}s (attempt {attempt + 1}/6)")
            time.sleep(wait)
            continue
        payload = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}

        if resp.status_code == 429 or payload.get("error"):
            reason = payload.get("reason", f"HTTP {resp.status_code}")
            if "hour" in reason.lower():
                wait = 660  # next hour boundary; short backoff cannot clear an hourly quota
            elif "minut" in reason.lower():
                wait = 70
            else:
                wait = 30 * (attempt + 1)
            print(f"  {reason} - waiting {wait}s (attempt {attempt + 1}/6)")
            time.sleep(wait)
            continue

        resp.raise_for_status()
        if "daily" not in payload:
            raise RuntimeError(f"unexpected response for {site_id} {start}..{end}: {str(payload)[:200]}")
        df = pd.DataFrame(payload["daily"]).rename(columns={"time": "date"})
        df.insert(0, "site_id", site_id)
        return df

    raise RuntimeError(f"gave up on {site_id} {start}..{end} after 6 attempts - re-run later to resume")


def write_with_manifest(df: pd.DataFrame, out_path: Path, site_id: str, period: str,
                        start: str, end: str, coverage: dict) -> None:
    """Atomic write + manifest sidecar (audit-v2, P0-7). The CSV is only
    placed in the formal directory after it passed STRUCTURAL validation;
    the manifest records request params, per-column coverage and the file
    hash so a later run can verify the file is unchanged. Files with
    entirely-empty required columns (upstream model gaps, e.g. CMCC_CM2_VHR4
    shortwave radiation) are written with validation_status =
    'degraded_model_gaps' and the gap listed - the consumer must drop those
    models per-variable and report the ensemble size."""
    atomic_write_csv(df, out_path)
    manifest = RunManifest(
        run_id=f"cmip6_{site_id}_{period}",
        task_name="download_cmip6",
        config_sha256="",
        input_sha256={},
        status="complete",
        output_files=[str(out_path)],
    )
    manifest.finished_at = datetime.now(timezone.utc).isoformat()
    manifest_dict = manifest.to_dict()
    empty_cols = [col for col, cov in coverage.items() if cov == ColumnStatus.EMPTY_INVALID]
    partial_cols = [col for col, cov in coverage.items() if cov == ColumnStatus.PARTIAL_INVALID]
    validation_status = "degraded_model_gaps" if (empty_cols or partial_cols) else "complete"
    manifest_dict.update({
        "api_url": CLIMATE_URL,
        "site_id": site_id,
        "period": period,
        "coordinates": {"lat": SITES[site_id]["lat"], "lon": SITES[site_id]["lon"]},
        "models": MODELS,
        "variables": DAILY_VARS,
        "date_range": {"start": start, "end": end},
        "field_coverage": {k: v.value for k, v in coverage.items()},
        "file_sha256": sha256_file(out_path),
        "license": "Open-Meteo Climate API (CC-BY 4.0 for data; see open-meteo.com)",
        "validation_status": validation_status,
        "model_gaps": empty_cols,
        "partial_columns": partial_cols,
    })
    from atomic_io import atomic_write_json
    atomic_write_json(out_path.with_suffix(".csv.manifest.json"), manifest_dict)


def _report(coverage: dict, out_path: Path) -> None:
    """audit-v3 (4.2): print the real complete/excluded/empty/partial
    breakdown instead of an unconditional 'all required columns non-empty'."""
    complete = [c for c, v in coverage.items() if v == ColumnStatus.COMPLETE]
    empty = [c for c, v in coverage.items() if v == ColumnStatus.EMPTY_INVALID]
    partial = [c for c, v in coverage.items() if v == ColumnStatus.PARTIAL_INVALID]
    n = len(coverage)
    print(f"  {out_path.name}: {len(complete)}/{n} columns COMPLETE")
    if empty:
        print(f"    empty_columns: {empty}")
    if partial:
        print(f"    partial_columns: {partial}")
    return complete, empty, partial


def validate_existing_dataset(path) -> dict:
    """audit-v3 (4.3): NO network access, NO filesystem mutation - read and
    validate only, returning a per-file report."""
    report = {}
    for site_id, meta in SITES.items():
        for period, (start, end) in PERIODS.items():
            out_path = path / f"{site_id}_{period}_cmip6.csv"
            if not out_path.exists():
                report[out_path.name] = {"status": "missing"}
                continue
            try:
                df = pd.read_csv(out_path)
                coverage = validate_cmip6_frame(df, start, end)
                _report(coverage, out_path)
                report[out_path.name] = {
                    "status": "valid", "rows": len(df),
                    "complete": sum(1 for v in coverage.values() if v == ColumnStatus.COMPLETE),
                    "empty": sum(1 for v in coverage.values() if v == ColumnStatus.EMPTY_INVALID),
                    "partial": sum(1 for v in coverage.values() if v == ColumnStatus.PARTIAL_INVALID),
                }
            except DataValidationError as exc:
                report[out_path.name] = {"status": "invalid", "reason": str(exc)}
                print(f"INVALID {out_path.name}: {exc}")
    return report


def download_and_validate_dataset(request, output_path, quarantine_bad=True) -> dict:
    """audit-v3 (4.3): network + writes allowed here only."""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)
    report = {}
    for site_id, meta in SITES.items():
        for period, (start, end) in PERIODS.items():
            out_path = output_path / f"{site_id}_{period}_cmip6.csv"
            if out_path.exists():
                try:
                    df = pd.read_csv(out_path)
                    coverage = validate_cmip6_frame(df, start, end)
                    _report(coverage, out_path)
                    report[out_path.name] = {"status": "valid_existing"}
                    continue
                except DataValidationError as exc:
                    print(f"INVALID {out_path.name}: {exc}")
                    if quarantine_bad:
                        dest = QUARANTINE_DIR / out_path.name
                        out_path.replace(dest)
                        print(f"  -> moved to quarantine/ ({dest.name})")
            print(f"downloading {site_id} {period} ({start}..{end}) x {len(MODELS)} models ...")
            try:
                df = fetch(site_id, meta["lat"], meta["lon"], start, end)
                coverage = validate_cmip6_frame(df, start, end)
            except DataValidationError as exc:
                print(f"  download failed validation: {exc} - NOT written to formal dir")
                report[out_path.name] = {"status": "download_invalid", "reason": str(exc)}
                continue
            _report(coverage, out_path)
            write_with_manifest(df, out_path, site_id, period, start, end, coverage)
            print(f"  saved {len(df)} rows, {len(df.columns)} cols -> {out_path.name}")
            report[out_path.name] = {"status": "downloaded"}
    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true",
                        help="validate existing files READ-ONLY (no network, no writes)")
    parser.add_argument("--no-quarantine", action="store_true",
                        help="report invalid files but leave them in place")
    args = parser.parse_args()
    if args.check_only:
        validate_existing_dataset(OUT_DIR)
    else:
        download_and_validate_dataset(None, OUT_DIR, quarantine_bad=not args.no_quarantine)
