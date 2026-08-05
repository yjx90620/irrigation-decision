"""FAO-56 Penman-Monteith reference evapotranspiration.

Needed because the CMIP6 climate API returns null for ET0, so future
climate series have to have it computed from temperature/radiation/wind/
humidity. Implemented per FAO Irrigation and Drainage Paper 56, eq. 6.

validate_against_observed() checks this implementation by recomputing ET0
from the observed weather inputs and comparing against the ET0 Open-Meteo
already provides for the same days - if the two agree, the same code can
be trusted on the perturbed future series.

Originally used wind_speed_10m_max (the only wind variable in the initial
download) as if it were the day's representative wind, which inflated the
aerodynamic term and gave a +0.31-0.50 mm/d bias across sites. Backfilled
wind_speed_10m_mean (src/data/download_wind_mean.py) and switched to it
here and in climate_scenario.py; bias dropped to -0.10 to -0.22 mm/d.

Also defaulted elevation_m=50 for every site regardless of actual
elevation (P1-2c, docs/审计修复计划.md) - most consequential at
ningxia_irrigation (1112m) and shaanxi_guanzhong (472m), both far from
the North China Plain sites' ~20-80m. Now uses each site's real
elevation (config.py's SITES[...]["elevation_m"], from Open-Meteo's own
elevation model); bias is now -0.13 to -0.20 mm/d, with the largest
improvement at the two elevated sites as expected.

audit-v2 (P1-7), remaining documented assumptions and limitations:
- actual vapour pressure uses ea = es(Tmean) x rh_mean/100 (FAO-56
  approximation with mean relative humidity); a dew-point or RHmin/RHmax
  formulation would be more faithful but the input data only provides
  mean RH. The -0.13..-0.20 mm/d bias against Open-Meteo's ET0 is the
  practical measure of the error this introduces.
- ET0 is floored at 0 (no negative reference ET); the fraction of days
  actually clamped is reported by validate_against_observed() so a
  pathological series (e.g. grossly misaligned inputs) shows up as a
  large clamp fraction instead of silently producing zero ET0 days.
- validation threshold: |bias| < 0.3 mm/d and RMSE < 0.8 mm/d against
  Open-Meteo's own et0_fao_evapotranspiration is treated as acceptable
  for the delta-change scenarios (observed biases are ~0.13-0.20 mm/d).
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


def penman_monteith_et0(tmax, tmin, rs, wind10, rh_mean, lat_deg, doy, elevation_m=50.0,
                        rh_max=None, rh_min=None):
    """Daily ET0 (mm). rs = incoming shortwave (MJ m-2 d-1), wind10 = 10m
    wind speed (m/s), rh_mean = mean relative humidity (%).

    audit-v3 (4.8): actual vapour pressure uses FAO-56 eq. 17 when
    RHmax/RHmin are available (the proper formulation), falling back to
    the documented ea = es(Tmean) x rh_mean/100 approximation otherwise -
    the approximation's deviation from eq. 17 is the main source of the
    remaining bias against reference implementations."""
    tmean = (tmax + tmin) / 2
    # 10m -> 2m wind, FAO-56 eq. 47
    u2 = wind10 * 4.87 / np.log(67.8 * 10 - 5.42)

    es = (_saturation_vapour_pressure(tmax) + _saturation_vapour_pressure(tmin)) / 2
    if rh_max is not None and rh_min is not None:
        # FAO-56 eq. 17: ea = [e(Tmin).RHmax + e(Tmax).RHmin] / 2
        ea = (_saturation_vapour_pressure(tmin) * rh_max / 100
              + _saturation_vapour_pressure(tmax) * rh_min / 100) / 2
    else:
        ea = es * rh_mean / 100
    delta = 4098 * _saturation_vapour_pressure(tmean) / (tmean + 237.3) ** 2

    pressure = 101.3 * ((293 - 0.0065 * elevation_m) / 293) ** 5.26
    gamma = 0.000665 * pressure

    ra = _extraterrestrial_radiation(lat_deg, doy)
    rso = (0.75 + 2e-5 * elevation_m) * ra  # clear-sky radiation
    rns = (1 - ALBEDO) * rs
    # FAO-56 eq. 39 cloud factor: Rs/Rso clipped to [0, 1] (FAO: the ratio
    # must be <= 1; values >1 indicate measurement/reflection artifacts).
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
        wind10=df["wind_speed_10m_mean"].values / 3.6,  # km/h -> m/s
        rh_mean=df["relative_humidity_2m_mean"].values,
        lat_deg=meta["lat"],
        doy=df["date"].dt.dayofyear.values,
        elevation_m=meta["elevation_m"],  # P1-2c, docs/审计修复计划.md
    )
    reference = df["et0_fao_evapotranspiration"].values
    bias = np.nanmean(computed - reference)
    rmse = np.sqrt(np.nanmean((computed - reference) ** 2))
    corr = np.corrcoef(computed[~np.isnan(reference)], reference[~np.isnan(reference)])[0, 1]
    # P1-7 (audit-v2): report how many days the 0-floor actually clamped -
    # a healthy series clamps ~0 days; a large fraction means the inputs
    # are pathological and the zero-ET0 days would silently corrupt results.
    n_clamped = int(np.sum(computed == 0.0))
    return {
        "bias": bias, "rmse": rmse, "corr": corr, "n": len(df),
        "n_days_clamped_at_zero": n_clamped,
        "clamp_fraction": n_clamped / len(df),
    }


if __name__ == "__main__":
    for site in ["hebei_central", "ningxia_irrigation", "shaanxi_guanzhong"]:
        stats = validate_against_observed(site)
        print(
            f"{site:22s} bias={stats['bias']:+.3f} mm/d  rmse={stats['rmse']:.3f}  r={stats['corr']:.4f}  "
            f"clamped={stats['n_days_clamped_at_zero']}/{stats['n']} ({stats['clamp_fraction']:.3%})"
        )
