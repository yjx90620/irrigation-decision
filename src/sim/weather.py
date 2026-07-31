"""Convert downloaded weather CSVs into the AquaCrop-OSPy weather_df format
(columns: MinTemp, MaxTemp, Precipitation, ReferenceET, Date)."""

from pathlib import Path

import pandas as pd

WEATHER_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "weather"

COLUMN_MAP = {
    "temperature_2m_max": "MaxTemp",
    "temperature_2m_min": "MinTemp",
    "precipitation_sum": "Precipitation",
    "et0_fao_evapotranspiration": "ReferenceET",
}


def load_site_weather(site_id: str) -> pd.DataFrame:
    matches = list(WEATHER_DIR.glob(f"{site_id}_*_openmeteo.csv"))
    if not matches:
        raise FileNotFoundError(f"no Open-Meteo weather file for site '{site_id}' in {WEATHER_DIR}")
    df = pd.read_csv(matches[0])
    df = df.rename(columns=COLUMN_MAP)
    df["Date"] = pd.to_datetime(df["date"])
    # AquaCropModel indexes weather_df.values positionally (MinTemp, MaxTemp,
    # Precipitation, ReferenceET, Date) - column order here must match that,
    # mirroring aquacrop.utils.prepare_weather's output order.
    df = df[["MinTemp", "MaxTemp", "Precipitation", "ReferenceET", "Date"]]
    df["ReferenceET"] = df["ReferenceET"].clip(lower=0.1)
    return df.reset_index(drop=True)
