"""Cross-check the Open-Meteo working dataset against official AgERA5 for
the 2024 validation sample (see download_weather_agera5.py docstring for
why AgERA5 isn't pulled for the full 1981-2025 record)."""

from pathlib import Path

import pandas as pd

AGERA5_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "weather_agera5"
OPENMETEO_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "weather"
OUT_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "agera5_vs_openmeteo_validation.csv"

SITES = ["beijing_plain", "hebei_central", "henan_north", "shaanxi_guanzhong", "ningxia_irrigation"]
VARS = ["tmax", "tmin", "precip", "et0"]
OPENMETEO_COLS = {
    "tmax": "temperature_2m_max",
    "tmin": "temperature_2m_min",
    "precip": "precipitation_sum",
    "et0": "et0_fao_evapotranspiration",
}


def load_agera5(site_id: str) -> pd.DataFrame:
    df = None
    for var in VARS:
        v = pd.read_csv(AGERA5_DIR / f"{site_id}_{var}_2024.csv")[["date", var]]
        df = v if df is None else df.merge(v, on="date")
    df["tmax"] -= 273.15
    df["tmin"] -= 273.15
    return df


def load_openmeteo_2024(site_id: str) -> pd.DataFrame:
    matches = list(OPENMETEO_DIR.glob(f"{site_id}_*_openmeteo.csv"))
    df = pd.read_csv(matches[0])
    df = df[(df["date"] >= "2024-01-01") & (df["date"] <= "2024-12-31")]
    return df


def main():
    rows = []
    for site_id in SITES:
        agera5 = load_agera5(site_id)
        om = load_openmeteo_2024(site_id)
        merged = agera5.merge(om, on="date", suffixes=("_agera5", "_om"))
        for var in VARS:
            a = merged[var]
            o = merged[OPENMETEO_COLS[var]]
            bias = (o - a).mean()
            rmse = ((o - a) ** 2).mean() ** 0.5
            corr = a.corr(o)
            rows.append(
                {"site_id": site_id, "variable": var, "mean_bias_om_minus_agera5": bias, "rmse": rmse, "corr": corr}
            )
    result = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT_PATH, index=False)
    print(result.round(3).to_string(index=False))
    print(f"\nsaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
