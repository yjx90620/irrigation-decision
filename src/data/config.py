"""Site and date-range configuration for the irrigation-decision digital farm."""

SITES = {
    "beijing_plain": {
        "name_cn": "北京平原",
        "lat": 39.80,
        "lon": 116.60,
    },
    "hebei_central": {
        "name_cn": "河北中部(石家庄)",
        "lat": 38.03,
        "lon": 114.51,
    },
    "henan_north": {
        "name_cn": "河南北部(新乡)",
        "lat": 35.30,
        "lon": 113.90,
    },
    "shaanxi_guanzhong": {
        "name_cn": "陕西关中(杨凌)",
        "lat": 34.30,
        "lon": 108.24,
    },
    "ningxia_irrigation": {
        "name_cn": "宁夏灌区(银川)",
        "lat": 38.47,
        "lon": 106.27,
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
