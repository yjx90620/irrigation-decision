"""Site and date-range configuration for the irrigation-decision digital farm."""

# elevation_m: from Open-Meteo's own elevation model for each site's
# coordinates (matches what the archive API already uses internally for
# its own et0_fao_evapotranspiration) - added for P1-2c (docs/审计修复计划.md):
# src/sim/et0.py's penman_monteith_et0() previously defaulted to 50m for
# every site regardless of actual elevation, most consequentially wrong
# for ningxia_irrigation (1112m) and shaanxi_guanzhong (472m), both far
# from the North China Plain sites' ~20-80m.
SITES = {
    "beijing_plain": {
        "name_cn": "北京平原",
        "lat": 39.80,
        "lon": 116.60,
        "elevation_m": 23.0,
    },
    "hebei_central": {
        "name_cn": "河北中部(石家庄)",
        "lat": 38.03,
        "lon": 114.51,
        "elevation_m": 76.0,
    },
    "henan_north": {
        "name_cn": "河南北部(新乡)",
        "lat": 35.30,
        "lon": 113.90,
        "elevation_m": 77.0,
    },
    "shaanxi_guanzhong": {
        "name_cn": "陕西关中(杨凌)",
        "lat": 34.30,
        "lon": 108.24,
        "elevation_m": 472.0,
    },
    "ningxia_irrigation": {
        "name_cn": "宁夏灌区(银川)",
        "lat": 38.47,
        "lon": 106.27,
        "elevation_m": 1112.0,
    },
}

START_YEAR = 1981
END_YEAR = 2025

DAILY_VARIABLES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "et0_fao_evapotranspiration",
    "shortwave_radiation_sum",
    "wind_speed_10m_max",
    "wind_speed_10m_mean",
    "relative_humidity_2m_mean",
]
