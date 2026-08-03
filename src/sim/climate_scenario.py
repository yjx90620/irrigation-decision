"""Build bias-corrected future weather series from CMIP6 projections, in
the AquaCrop input format, for paper 1's climate comparison and paper 3's
temporal-transfer experiment.

Method: delta change (a.k.a. the perturbation method). Rather than feeding
raw model output into AquaCrop - which would inherit each model's
systematic bias - this takes each model's *own* change signal between its
own historical run (1991-2010) and its own future run (2031-2050), then
averages those per-model signals across the 7-model ensemble, then applies
the ensemble-mean signal to the observed series. Comparing a model against
itself cancels most of its bias, and the result keeps the observed
record's day-to-day weather sequencing.

    temperature, humidity:  additive       (future_mean - historical_mean)
    precipitation, radiation, wind: multiplicative (future_mean / historical_mean),
                   so none of these strictly-positive quantities can be
                   handed a negative value

P1-2a (docs/审计修复计划.md): the previous version averaged each variable
across models *first*, then took one difference/ratio between the two
resulting series - "ensemble mean of the ratios" and "ratio of the
ensemble means" are not the same computation (they're only equal for the
additive/temperature case, which is linear), and the mismatch actually
contradicted this module's own docstring, which always described the
per-model-first method. Fixed by computing each model's own monthly
delta/ratio first, then averaging *those* across models - also now
restricted to the models present in *both* periods (a model with data in
only one period no longer silently drops out of just one side's average).

Deltas are computed per calendar month, because the monsoon's seasonal
timing matters far more here than an annual mean would capture.

P1-2b: previously only temperature was perturbed for the future ET0
calculation - radiation, wind, and relative humidity kept using the
*historical* observed values even though CMIP6's own future-period
projections for all three were already downloaded and sitting unused.
Now perturbed the same delta-change way as temperature/precipitation.

ET0 is *recomputed* from the perturbed inputs via FAO-56 rather than
having Open-Meteo's et0_fao_evapotranspiration column perturbed directly,
so it stays physically consistent with the actual perturbed weather.
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


class DataValidationError(RuntimeError):
    """Input weather/CMIP6 data is incomplete or misaligned - fail loudly
    rather than build a future series from silently wrong values
    (audit-v2, P0-8)."""

MODELS = [
    "CMCC_CM2_VHR4", "FGOALS_f3_H", "HiRAM_SIT_HR", "MRI_AGCM3_2_S",
    "EC_Earth3P_HR", "MPI_ESM1_2_XR", "NICAM16_8S",
]

# (cmip6 column prefix, output delta name, mode) - mode is "additive" or
# "multiplicative". Humidity is additive (a percentage-point shift, not a
# ratio) but gets clipped to [0,100] when applied, not here.
VARIABLES = [
    ("temperature_2m_max", "d_temperature_2m_max", "additive"),
    ("temperature_2m_min", "d_temperature_2m_min", "additive"),
    ("precipitation_sum", "r_precipitation", "multiplicative"),
    ("shortwave_radiation_sum", "r_radiation", "multiplicative"),
    ("wind_speed_10m_mean", "r_wind", "multiplicative"),
    ("relative_humidity_2m_mean", "d_humidity", "additive"),
]


def _load_period(site_id: str, period: str) -> pd.DataFrame:
    path = CMIP6_DIR / f"{site_id}_{period}_cmip6.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run src/data/download_cmip6.py")
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month
    return df


def _per_model_monthly_means(df: pd.DataFrame, variable: str) -> dict:
    """{model_name: monthly-mean Series}, models with no data for this
    site/variable excluded rather than poisoning anything with NaN."""
    out = {}
    for model in MODELS:
        col = f"{variable}_{model}"
        if col in df.columns and df[col].notna().any():
            out[model] = df.groupby(df["month"])[col].mean()
    return out


def _ensemble_delta(hist: pd.DataFrame, fut: pd.DataFrame, variable: str, mode: str) -> tuple:
    """Per-model delta/ratio first, then averaged across models (P1-2a) -
    restricted to models with usable data in *both* periods. Returns
    (ensemble_mean, ensemble_std, n_models) - std so model disagreement
    stays visible instead of only ever reporting the mean."""
    hist_by_model = _per_model_monthly_means(hist, variable)
    fut_by_model = _per_model_monthly_means(fut, variable)
    common = sorted(set(hist_by_model) & set(fut_by_model))
    if not common:
        raise ValueError(f"no models with usable {variable} data in both periods")

    if mode == "additive":
        per_model = pd.DataFrame({m: fut_by_model[m] - hist_by_model[m] for m in common})
    else:
        # ratio, clipped: an ensemble ratio far outside this range in a dry/
        # low month reflects a near-zero denominator rather than a credible signal
        per_model = pd.DataFrame({
            m: (fut_by_model[m] / hist_by_model[m].replace(0, np.nan)).clip(0.3, 3.0).fillna(1.0) for m in common
        })
    return per_model.mean(axis=1), per_model.std(axis=1), len(common)


def compute_deltas(site_id: str) -> pd.DataFrame:
    hist, fut = _load_period(site_id, "historical"), _load_period(site_id, "future")
    deltas = pd.DataFrame(index=range(1, 13))
    for cmip6_var, out_name, mode in VARIABLES:
        mean, std, n_models = _ensemble_delta(hist, fut, cmip6_var, mode)
        deltas[out_name] = mean
        deltas[f"{out_name}_std"] = std
        deltas.attrs[f"{out_name}_n_models"] = n_models
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

    # P0-8 (audit-v2): the radiation/wind/humidity drivers for the future
    # ET0 were previously taken from the raw Open-Meteo file by ARRAY
    # POSITION (`.values`), silently assuming the raw rows align with
    # load_site_weather()'s rows - if the raw file ever covers a different
    # date span or has a gap, every ET0 value downstream would be computed
    # from misaligned weather without any error. Merge on date instead and
    # fail loudly on any unaligned day.
    raw = pd.read_csv(next((Path(__file__).resolve().parents[2] / "data" / "raw" / "weather").glob(
        f"{site_id}_*_openmeteo.csv")))
    raw["date"] = pd.to_datetime(raw["date"])
    merged = obs.reset_index().merge(
        raw[["date", "shortwave_radiation_sum", "wind_speed_10m_mean", "relative_humidity_2m_mean"]],
        left_on="Date", right_on="date", how="left", validate="one_to_one",
    )
    required_drivers = ["shortwave_radiation_sum", "wind_speed_10m_mean", "relative_humidity_2m_mean"]
    if merged[required_drivers].isna().any().any():
        raise DataValidationError(
            f"{site_id}: observed weather and raw Open-Meteo file do not align on "
            f"{int(merged[required_drivers].isna().any(axis=1).sum())} day(s) - cannot build a "
            f"physically consistent future weather series"
        )
    future_radiation = merged["shortwave_radiation_sum"].values * month.map(deltas["r_radiation"]).values
    future_wind = merged["wind_speed_10m_mean"].values * month.map(deltas["r_wind"]).values
    future_humidity = np.clip(
        merged["relative_humidity_2m_mean"].values + month.map(deltas["d_humidity"]).values, 0.0, 100.0
    )

    obs["ReferenceET"] = np.maximum(
        penman_monteith_et0(
            tmax=obs["MaxTemp"].values,
            tmin=obs["MinTemp"].values,
            rs=future_radiation,
            wind10=future_wind / 3.6,
            rh_mean=future_humidity,
            lat_deg=SITES[site_id]["lat"],
            doy=obs["Date"].dt.dayofyear.values,
            elevation_m=SITES[site_id]["elevation_m"],  # P1-2c
        ),
        0.1,
    )
    return obs


if __name__ == "__main__":
    for site_id in SITES:
        try:
            d = compute_deltas(site_id)
            gs = d.loc[5:9]  # main growing-season months
            # P0-7 (audit-v2): report the per-variable ensemble size so a
            # model silently dropping out (e.g. CMCC_CM2_VHR4 has no
            # shortwave radiation in the current files) is visible, not
            # hidden inside an unlabeled average.
            n_models = {v: d.attrs.get(f"{v}_n_models", "?") for v in
                        ["d_temperature_2m_max", "r_precipitation", "r_radiation", "r_wind", "d_humidity"]}
            print(
                f"{site_id:22s} 生育期增温 {gs['d_temperature_2m_max'].mean():+.2f}"
                f"(+/-{gs['d_temperature_2m_max_std'].mean():.2f})°C  "
                f"降水变化 {(gs['r_precipitation'].mean() - 1) * 100:+.1f}"
                f"(+/-{gs['r_precipitation_std'].mean()*100:.1f})%  "
                f"n_models={n_models}"
            )
        except FileNotFoundError:
            print(f"{site_id:22s} (CMIP6 data not downloaded yet)")
        except DataValidationError as exc:
            print(f"{site_id:22s} INVALID CMIP6 data: {exc}")
