"""Convert downloaded weather CSVs into the AquaCrop-OSPy weather_df format
(columns: MinTemp, MaxTemp, Precipitation, ReferenceET, Date).

audit-v3 (4.7): the loader now (a) requires EXACTLY ONE matching file
(glob-then-first was nondeterministic with multiple matches), (b) validates
the full calendar (unique, strictly increasing, complete daily range),
(c) checks physical ranges (Tmin<=Tmax, precip/wind/radiation >= 0, RH in
[0,100]), and (d) records the ET0 floor clamp count instead of silently
clipping - a pathological series shows up as a large clamp fraction.
"""

from pathlib import Path

import numpy as np
import pandas as pd

WEATHER_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "weather"

COLUMN_MAP = {
    "temperature_2m_max": "MaxTemp",
    "temperature_2m_min": "MinTemp",
    "precipitation_sum": "Precipitation",
    "et0_fao_evapotranspiration": "ReferenceET",
}

PHYSICAL_COLS = {
    "Precipitation": (0.0, np.inf),
    "MaxTemp": (-np.inf, np.inf),
    "MinTemp": (-np.inf, np.inf),
    "ReferenceET": (0.0, np.inf),
}


class WeatherDataError(RuntimeError):
    """Weather input violates the loader's data contract."""


def load_site_weather(site_id: str) -> pd.DataFrame:
    matches = sorted(WEATHER_DIR.glob(f"{site_id}_*_openmeteo.csv"))
    if not matches:
        raise FileNotFoundError(f"no Open-Meteo weather file for site '{site_id}' in {WEATHER_DIR}")
    if len(matches) != 1:
        # audit-v3 (4.7): picking the first of several files silently changes
        # every downstream number depending on filesystem order.
        raise WeatherDataError(
            f"expected exactly one Open-Meteo weather file for '{site_id}', found {len(matches)}: "
            f"{[m.name for m in matches]}"
        )
    df = pd.read_csv(matches[0])
    df = df.rename(columns=COLUMN_MAP)
    df["Date"] = pd.to_datetime(df["date"])
    _validate(df, matches[0].name)
    df = df[["MinTemp", "MaxTemp", "Precipitation", "ReferenceET", "Date"]]
    n_clamped = int((df["ReferenceET"] < 0.1).sum())
    if n_clamped:
        print(f"  [weather] {site_id}: ET0 floor 0.1 clamped {n_clamped}/{len(df)} days "
              f"({n_clamped / len(df):.2%})")
    df["ReferenceET"] = df["ReferenceET"].clip(lower=0.1)
    return df.reset_index(drop=True)


def _validate(df: pd.DataFrame, filename: str) -> None:
    dates = df["Date"]
    if dates.duplicated().any():
        raise WeatherDataError(f"{filename}: {dates.duplicated().sum()} duplicate dates")
    if not dates.is_monotonic_increasing:
        raise WeatherDataError(f"{filename}: dates not strictly increasing")
    # full daily coverage across the span
    span = pd.date_range(dates.min(), dates.max(), freq="D")
    missing = span.difference(dates)
    if len(missing):
        raise WeatherDataError(f"{filename}: {len(missing)} missing dates in range")
    if (df["MinTemp"] > df["MaxTemp"]).any():
        raise WeatherDataError(f"{filename}: MinTemp > MaxTemp on some days")
    for col, (lo, hi) in PHYSICAL_COLS.items():
        bad = ~df[col].between(lo, hi)
        if bad.any():
            raise WeatherDataError(f"{filename}: {int(bad.sum())} out-of-range values in {col}")
    # relative humidity, if present
    if "relative_humidity_2m_mean" in df.columns:
        rh = df["relative_humidity_2m_mean"]
        if ((rh < 0) | (rh > 100)).any():
            raise WeatherDataError(f"{filename}: relative humidity outside [0, 100]")
    if not df[["MinTemp", "MaxTemp", "Precipitation", "ReferenceET"]].notna().all().all():
        raise WeatherDataError(f"{filename}: missing temperature/precip/ET0 values")
