"""FAO-56 Penman-Monteith reference evapotranspiration.

Needed because the CMIP6 climate API returns null for ET0, so future
climate series have to have it computed from temperature/radiation/wind/
humidity. Implemented per FAO Irrigation and Drainage Paper 56, eq. 6.

validate_against_observed() checks this implementation by recomputing ET0
from the observed weather inputs and comparing against the ET0 Open-Meteo
already provides for the same days - if the two agree, the same code can
be trusted on the perturbed future series.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))

import numpy as np
import pandas as pd

SOLAR_CONSTANT = 0.0820  # MJ m-2 min-1
ALBEDO = 0.23  # FAO-56 reference grass
STEFAN_BOLTZMANN = 4.903e-9  # MJ K-4 m-2 day-1


def _saturation_vapour_pressure(t_c):
    return 0.6108 * np.exp(17.27 * t_c / (t_c + 237.3))


def _extraterrestrial_radiation(lat_deg, doy):
    """FAO-56 eq. 21."""
    lat = np.radians(lat_deg)
    dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)
    decl = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)
    # clip guards against |x|>1 at high latitudes / polar day-night
    ws = np.arccos(np.clip(-np.tan(lat) * np.tan(decl), -1, 1))
    return (
        24 * 60 / np.pi * SOLAR_CONSTANT * dr
        * (ws * np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.sin(ws))
    )


def penman_monteith_et0(tmax, tmin, rs, wind10, rh_mean, lat_deg, doy, elevation_m=50.0):
    """Daily ET0 (mm). rs = incoming shortwave (MJ m-2 d-1), wind10 = 10m
    wind speed (m/s), rh_mean = mean relative humidity (%)."""
    tmean = (tmax + tmin) / 2
    # 10m -> 2m wind, FAO-56 eq. 47
    u2 = wind10 * 4.87 / np.log(67.8 * 10 - 5.42)

    es = (_saturation_vapour_pressure(tmax) + _saturation_vapour_pressure(tmin)) / 2
    ea = es * rh_mean / 100
    delta = 4098 * _saturation_vapour_pressure(tmean) / (tmean + 237.3) ** 2

    pressure = 101.3 * ((293 - 0.0065 * elevation_m) / 293) ** 5.26
    gamma = 0.000665 * pressure

    ra = _extraterrestrial_radiation(lat_deg, doy)
    rso = (0.75 + 2e-5 * elevation_m) * ra  # clear-sky radiation
    rns = (1 - ALBEDO) * rs
    with np.errstate(divide="ignore", invalid="ignore"):
        cloud_factor = np.where(rso > 0, np.clip(rs / rso, 0.0, 1.0), 0.0)
    rnl = (
        STEFAN_BOLTZMANN
        * (((tmax + 273.16) ** 4 + (tmin + 273.16) ** 4) / 2)
        * (0.34 - 0.14 * np.sqrt(np.maximum(ea, 0)))
        * (1.35 * cloud_factor - 0.35)
    )
    rn = rns - rnl

    numerator = 0.408 * delta * rn + gamma * 900 / (tmean + 273) * u2 * (es - ea)
    denominator = delta + gamma * (1 + 0.34 * u2)
    return np.maximum(numerator / denominator, 0.0)


def validate_against_observed(site_id="hebei_central", n_days=3650):
    """Recompute ET0 from observed inputs and compare with Open-Meteo's own
    ET0 for the same days."""
    from config import SITES

    weather_dir = Path(__file__).resolve().parents[2] / "data" / "raw" / "weather"
    path = next(weather_dir.glob(f"{site_id}_*_openmeteo.csv"))
    df = pd.read_csv(path).tail(n_days).copy()
    df["date"] = pd.to_datetime(df["date"])

    meta = SITES[site_id]
    computed = penman_monteith_et0(
        tmax=df["temperature_2m_max"].values,
        tmin=df["temperature_2m_min"].values,
        rs=df["shortwave_radiation_sum"].values,
        wind10=df["wind_speed_10m_max"].values / 3.6,  # km/h -> m/s
        rh_mean=df["relative_humidity_2m_mean"].values,
        lat_deg=meta["lat"],
        doy=df["date"].dt.dayofyear.values,
    )
    reference = df["et0_fao_evapotranspiration"].values
    bias = np.nanmean(computed - reference)
    rmse = np.sqrt(np.nanmean((computed - reference) ** 2))
    corr = np.corrcoef(computed[~np.isnan(reference)], reference[~np.isnan(reference)])[0, 1]
    return {"bias": bias, "rmse": rmse, "corr": corr, "n": len(df)}


if __name__ == "__main__":
    for site in ["hebei_central", "ningxia_irrigation", "shaanxi_guanzhong"]:
        stats = validate_against_observed(site)
        print(f"{site:22s} bias={stats['bias']:+.3f} mm/d  rmse={stats['rmse']:.3f}  r={stats['corr']:.4f}")
