"""Build bias-corrected future weather series from CMIP6 projections, in
the AquaCrop input format, for paper 1's climate comparison and paper 3's
temporal-transfer experiment.

Method: delta change (a.k.a. the perturbation method). Rather than feeding
raw model output into AquaCrop - which would inherit each model's
systematic bias - this takes each model's *change signal* between its own
historical run (1991-2010) and its future run (2031-2050), then applies
that signal to the observed series. Comparing a model against itself
cancels most of its bias, and the result keeps the observed record's
day-to-day weather sequencing.

    temperature:   additive delta   (future_mean - historical_mean)
    precipitation: multiplicative   (future_mean / historical_mean), so a
                   dry site cannot be handed a negative rainfall total

Deltas are computed per calendar month, because the monsoon's seasonal
timing matters far more here than an annual mean would capture - and are
averaged across the 7-model ensemble rather than trusting any one model.

ET0 is then *recomputed* from the perturbed temperatures via FAO-56 rather
than being perturbed directly, so it stays physically consistent with the
temperatures the crop model actually sees.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))

import numpy as np
import pandas as pd

from config import SITES
from et0 import penman_monteith_et0
from weather import load_site_weather

CMIP6_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "cmip6"

MODELS = [
    "CMCC_CM2_VHR4", "FGOALS_f3_H", "HiRAM_SIT_HR", "MRI_AGCM3_2_S",
    "EC_Earth3P_HR", "MPI_ESM1_2_XR", "NICAM16_8S",
]


def _load_period(site_id: str, period: str) -> pd.DataFrame:
    path = CMIP6_DIR / f"{site_id}_{period}_cmip6.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run src/data/download_cmip6.py")
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month
    return df


def _ensemble_monthly_mean(df: pd.DataFrame, variable: str) -> pd.Series:
    """Average the per-model columns for `variable`, then take monthly means.
    Models with no data for a site are skipped rather than poisoning the
    ensemble with NaN."""
    cols = [f"{variable}_{m}" for m in MODELS if f"{variable}_{m}" in df.columns]
    usable = [c for c in cols if df[c].notna().any()]
    if not usable:
        raise ValueError(f"no usable model columns for {variable}")
    ensemble = df[usable].mean(axis=1)
    return ensemble.groupby(df["month"]).mean()


def compute_deltas(site_id: str) -> pd.DataFrame:
    hist, fut = _load_period(site_id, "historical"), _load_period(site_id, "future")
    deltas = pd.DataFrame(index=range(1, 13))

    for var in ["temperature_2m_max", "temperature_2m_min"]:
        deltas[f"d_{var}"] = _ensemble_monthly_mean(fut, var) - _ensemble_monthly_mean(hist, var)

    hist_p = _ensemble_monthly_mean(hist, "precipitation_sum")
    fut_p = _ensemble_monthly_mean(fut, "precipitation_sum")
    # ratio, clipped: an ensemble ratio far outside this range in a dry month
    # reflects a near-zero denominator rather than a credible signal
    deltas["r_precipitation"] = (fut_p / hist_p.replace(0, np.nan)).clip(0.3, 3.0).fillna(1.0)
    return deltas


def build_future_weather(site_id: str, deltas: pd.DataFrame = None) -> pd.DataFrame:
    """Observed weather with the climate change signal applied, in the
    column layout src/sim/weather.py produces (so it drops straight into
    the rotation engine)."""
    if deltas is None:
        deltas = compute_deltas(site_id)

    obs = load_site_weather(site_id).copy()
    month = obs["Date"].dt.month

    obs["MaxTemp"] = obs["MaxTemp"] + month.map(deltas["d_temperature_2m_max"]).values
    obs["MinTemp"] = obs["MinTemp"] + month.map(deltas["d_temperature_2m_min"]).values
    obs["Precipitation"] = obs["Precipitation"] * month.map(deltas["r_precipitation"]).values

    raw = pd.read_csv(next((Path(__file__).resolve().parents[2] / "data" / "raw" / "weather").glob(f"{site_id}_*_openmeteo.csv")))
    obs["ReferenceET"] = np.maximum(
        penman_monteith_et0(
            tmax=obs["MaxTemp"].values,
            tmin=obs["MinTemp"].values,
            rs=raw["shortwave_radiation_sum"].values,
            wind10=raw["wind_speed_10m_mean"].values / 3.6,
            rh_mean=raw["relative_humidity_2m_mean"].values,
            lat_deg=SITES[site_id]["lat"],
            doy=obs["Date"].dt.dayofyear.values,
        ),
        0.1,
    )
    return obs


if __name__ == "__main__":
    for site_id in SITES:
        try:
            d = compute_deltas(site_id)
            gs = d.loc[5:9]  # main growing-season months
            print(
                f"{site_id:22s} 生育期增温 {gs['d_temperature_2m_max'].mean():+.2f}°C  "
                f"降水变化 {(gs['r_precipitation'].mean() - 1) * 100:+.1f}%"
            )
        except FileNotFoundError:
            print(f"{site_id:22s} (CMIP6 data not downloaded yet)")
