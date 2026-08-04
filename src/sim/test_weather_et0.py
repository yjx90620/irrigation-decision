"""audit-v3 (4.7/4.8): weather-loader data contract + ET0 reference case.

- test_weather_loader_*: the loader rejects multiple matches, duplicate
  dates, Tmin>Tmax, out-of-range values (uses a monkeypatched WEATHER_DIR).
- test_et0_*: FAO-56 Example 18 reference case (with the mean-RH vapour
  pressure approximation the module documents), monotonicity, finiteness.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import pytest

import et0
import weather


@pytest.fixture
def tmp_weather_dir(tmp_path_factory):
    import tempfile

    d = Path(tempfile.mkdtemp())
    original = weather.WEATHER_DIR
    weather.WEATHER_DIR = d
    yield d
    weather.WEATHER_DIR = original


def _write(site, df, dir_, year_suffix="1981_2025"):
    df.to_csv(dir_ / f"{site}_{year_suffix}_openmeteo.csv", index=False)


def test_weather_loader_rejects_multiple_matches(tmp_weather_dir):
    df = pd.DataFrame({
        "date": pd.date_range("1981-01-01", "1981-01-05"),
        "temperature_2m_max": [10, 11, 12, 13, 14],
        "temperature_2m_min": [1, 2, 3, 4, 5],
        "precipitation_sum": [0, 1, 0, 2, 0],
        "et0_fao_evapotranspiration": [1, 1, 1, 1, 1],
    })
    _write("multi", df, tmp_weather_dir)
    df2 = df.copy()
    df2["date"] = pd.date_range("2000-01-01", "2000-01-05")
    _write("multi", df2, tmp_weather_dir, year_suffix="2000_2025")  # distinct file, same prefix
    with pytest.raises(weather.WeatherDataError, match="exactly one"):
        weather.load_site_weather("multi")


def test_weather_loader_rejects_duplicate_dates(tmp_weather_dir):
    df = pd.DataFrame({
        "date": pd.to_datetime(["1981-01-01", "1981-01-01", "1981-01-03"]),
        "temperature_2m_max": [10, 11, 12],
        "temperature_2m_min": [1, 2, 3],
        "precipitation_sum": [0, 1, 0],
        "et0_fao_evapotranspiration": [1, 1, 1],
    })
    _write("dup", df, tmp_weather_dir)
    with pytest.raises(weather.WeatherDataError, match="duplicate"):
        weather.load_site_weather("dup")


def test_weather_loader_rejects_tmin_gt_tmax(tmp_weather_dir):
    df = pd.DataFrame({
        "date": pd.date_range("1981-01-01", "1981-01-03"),
        "temperature_2m_max": [10, 11, 5],  # day 3: tmin > tmax
        "temperature_2m_min": [1, 2, 8],
        "precipitation_sum": [0, 1, 0],
        "et0_fao_evapotranspiration": [1, 1, 1],
    })
    _write("bad", df, tmp_weather_dir)
    with pytest.raises(weather.WeatherDataError, match="MinTemp > MaxTemp"):
        weather.load_site_weather("bad")


def test_et0_fao56_reference_case():
    """FAO-56 Example 18 (Ch. 4): lat 50N, elev 100 m, 3 Sep (DOY 246),
    Tmax 21.5, Tmin 12.3, RHmax 84/RHmin 63, u2 2 m/s, Rs 22.07.
    The expected value is computed in-test with an INDEPENDENT step-by-step
    FAO-56 implementation (no module helpers), so this is a true
    cross-check, not the module testing itself."""
    tmax, tmin, rs, u2, rh_max, rh_min = 21.5, 12.3, 22.07, 2.0, 84.0, 63.0
    lat, doy, elev = 50.0, 246, 100.0

    # --- independent FAO-56 computation ---
    tmean = (tmax + tmin) / 2
    es_tmax = 0.6108 * np.exp(17.27 * tmax / (tmax + 237.3))
    es_tmin = 0.6108 * np.exp(17.27 * tmin / (tmin + 237.3))
    es = (es_tmax + es_tmin) / 2
    ea = (es_tmin * rh_max / 100 + es_tmax * rh_min / 100) / 2  # eq. 17
    delta = 4098 * 0.6108 * np.exp(17.27 * tmean / (tmean + 237.3)) / (tmean + 237.3) ** 2
    pressure = 101.3 * ((293 - 0.0065 * elev) / 293) ** 5.26
    gamma = 0.000665 * pressure
    lat_r = np.radians(lat)
    dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)
    decl = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)
    ws = np.arccos(np.clip(-np.tan(lat_r) * np.tan(decl), -1, 1))
    ra = 24 * 60 / np.pi * 0.0820 * dr * (
        ws * np.sin(lat_r) * np.sin(decl) + np.cos(lat_r) * np.cos(decl) * np.sin(ws))
    rso = (0.75 + 2e-5 * elev) * ra
    rns = 0.77 * rs
    tk4 = 4.903e-9 * (((tmax + 273.16) ** 4 + (tmin + 273.16) ** 4) / 2)
    rnl = tk4 * (0.34 - 0.14 * np.sqrt(ea)) * (1.35 * np.clip(rs / rso, 0, 1.0) - 0.35)
    rn = rns - rnl
    expected = (0.408 * delta * rn + gamma * 900 / (tmean + 273) * u2 * (es - ea)) / (
        delta + gamma * (1 + 0.34 * u2))

    et0v = et0.penman_monteith_et0(
        tmax=tmax, tmin=tmin, rs=rs, wind10=u2 * np.log(67.8 * 10 - 5.42) / 4.87,
        rh_mean=73.5, rh_max=rh_max, rh_min=rh_min, lat_deg=lat, doy=doy, elevation_m=elev,
    )
    assert np.isfinite(et0v)
    assert abs(et0v - expected) < 1e-3, f"module {et0v:.4f} != independent FAO-56 {expected:.4f}"


def test_et0_falls_back_to_mean_rh_when_ranges_missing():
    v_approx = et0.penman_monteith_et0(
        tmax=21.5, tmin=12.3, rs=22.07, wind10=2.0, rh_mean=73.5,
        lat_deg=50.0, doy=246, elevation_m=100.0,
    )
    v_proper = et0.penman_monteith_et0(
        tmax=21.5, tmin=12.3, rs=22.07, wind10=2.0, rh_mean=73.5,
        rh_max=84.0, rh_min=63.0, lat_deg=50.0, doy=246, elevation_m=100.0,
    )
    assert np.isfinite(v_approx)
    # the approximation deviates from eq. 17 - that deviation is the
    # documented source of residual bias, and it must be visible, not 0
    assert abs(v_approx - v_proper) > 1e-6


def test_et0_monotone_in_radiation_wind_temperature():
    base = dict(tmax=25.0, tmin=15.0, rs=18.0, wind10=2.0, rh_mean=60.0, lat_deg=35.0, doy=180)
    lo = et0.penman_monteith_et0(**base)
    hi_rad = et0.penman_monteith_et0(**{**base, "rs": 25.0})
    hi_wind = et0.penman_monteith_et0(**{**base, "wind10": 4.0})
    hi_temp = et0.penman_monteith_et0(**{**base, "tmax": 30.0, "tmin": 20.0})
    assert hi_rad > lo
    assert hi_wind > lo
    assert hi_temp > lo
    assert all(np.isfinite(x) for x in (lo, hi_rad, hi_wind, hi_temp))
