"""Map ERA5-Land soil moisture onto AquaCrop's soil profile to give each
simulated season a realistic starting water content.

ERA5-Land's volumetric water contents cannot be transplanted into AquaCrop
directly: the two models use different soil hydraulic parameterizations, so
the same m3/m3 number means a different degree of wetness in each. Instead
this converts ERA5-Land values to a *relative available water fraction*
using ERA5's own wilting-point/field-capacity constants, then maps that
fraction onto AquaCrop's th_wp..th_fc range for the configured soil. That
preserves "how wet is this profile, in plant-available terms" rather than
a raw number that would land outside AquaCrop's valid range.

ERA5-Land layer depths (0-7, 7-28, 28-100, 100-255 cm) are resampled onto
AquaCrop's 12 x 0.1 m compartments by depth overlap.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))

import numpy as np
import pandas as pd
from aquacrop import InitialWaterContent

from soils import get_soil

SM_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "soil_moisture"

# ERA5-Land HTESSEL hydraulic constants for its medium soil texture class,
# used only to convert ERA5 values to a relative wetness fraction.
ERA5_WILTING_POINT = 0.151
ERA5_FIELD_CAPACITY = 0.346

ERA5_LAYERS = [
    ("soil_moisture_0_to_7cm_mean", 0.00, 0.07),
    ("soil_moisture_7_to_28cm_mean", 0.07, 0.28),
    ("soil_moisture_28_to_100cm_mean", 0.28, 1.00),
    ("soil_moisture_100_to_255cm_mean", 1.00, 2.55),
]

_CACHE: dict = {}


def _load_site_sm(site_id: str) -> pd.DataFrame:
    if site_id not in _CACHE:
        matches = list(SM_DIR.glob(f"{site_id}_*_soilmoisture.csv"))
        if not matches:
            raise FileNotFoundError(
                f"no soil moisture file for '{site_id}' - run src/data/download_soil_moisture.py first"
            )
        df = pd.read_csv(matches[0])
        df["date"] = pd.to_datetime(df["date"])
        _CACHE[site_id] = df.set_index("date")
    return _CACHE[site_id]


def initial_water_content(site_id: str, soil_key: str, date: str) -> InitialWaterContent:
    """Build an AquaCrop InitialWaterContent for `date` from observed ERA5-Land
    soil moisture at `site_id`."""
    sm = _load_site_sm(site_id)
    ts = pd.Timestamp(date)
    if ts not in sm.index:
        raise KeyError(f"no soil moisture record for {site_id} on {date}")
    row = sm.loc[ts]

    profile = get_soil(soil_key).profile
    th_values = []
    for _, comp in profile.iterrows():
        top, bottom = comp["z_top"], comp["zBot"]
        # depth-weighted average of the ERA5 layers overlapping this compartment
        weighted, total_w = 0.0, 0.0
        for var, l_top, l_bot in ERA5_LAYERS:
            overlap = max(0.0, min(bottom, l_bot) - max(top, l_top))
            if overlap > 0:
                weighted += row[var] * overlap
                total_w += overlap
        era5_theta = weighted / total_w if total_w > 0 else row[ERA5_LAYERS[-1][0]]

        rel = (era5_theta - ERA5_WILTING_POINT) / (ERA5_FIELD_CAPACITY - ERA5_WILTING_POINT)
        rel = float(np.clip(rel, 0.0, 1.2))  # allow slightly above FC, cap below saturation
        th = comp["th_wp"] + rel * (comp["th_fc"] - comp["th_wp"])
        th_values.append(float(np.clip(th, comp["th_wp"], comp["th_s"])))

    return InitialWaterContent(
        wc_type="Num", method="Layer", depth_layer=list(range(1, len(th_values) + 1)), value=th_values
    )
