"""Climate-soil environmental fingerprint per site (研究方案 6.4).

Only the climate block actually differentiates sites right now: soil is
still the same 3 standard AquaCrop profiles everywhere (SoilGrids access
is blocked, see src/data/README.md), and management (planting date,
seasonal water cap, decision interval) is currently a fixed project-wide
setting rather than something calibrated per site. Sand/Clay fraction is
left out entirely rather than faked - AquaCrop-OSPy's built-in soil
classes expose hydraulic properties (th_fc, th_wp, th_s) but not texture
fractions.

GDD uses AquaCrop's own growing_degree_day() with the Maize crop's actual
Tbase/Tupp (8/30 degC, GDDmethod=3) so this stays consistent with what
the simulations in src/sim and src/rl actually use, rather than assuming
a generic threshold.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim"))

import numpy as np
import pandas as pd
from aquacrop import Crop
from aquacrop.solution.growing_degree_day import growing_degree_day

from config import SITES
from soils import get_soil
from weather import load_site_weather

GROWING_SEASON_START = "05-01"
GROWING_SEASON_END = "09-15"
HOT_DAY_THRESHOLD_C = 33.0


def _precipitation_concentration_index(daily: pd.DataFrame) -> float:
    """Oliver's PCI: 100 * sum(monthly_precip^2) / (sum(monthly_precip))^2,
    over the full calendar year (not just the growing season) - standard
    climatology definition of how concentrated rainfall is across months."""
    monthly = daily.groupby(daily["Date"].dt.month)["Precipitation"].sum()
    return float(100 * (monthly**2).sum() / (monthly.sum() ** 2))


def _max_consecutive_dry_days(precip: pd.Series, dry_threshold_mm: float = 1.0) -> float:
    is_dry = (precip < dry_threshold_mm).astype(int)
    groups = (is_dry != is_dry.shift()).cumsum()
    run_lengths = is_dry.groupby(groups).transform("size") * is_dry
    return float(run_lengths.max())


def compute_site_fingerprint(site_id: str, crop: Crop) -> dict:
    weather = load_site_weather(site_id)
    weather["year"] = weather["Date"].dt.year

    season_mask = (weather["Date"].dt.strftime("%m-%d") >= GROWING_SEASON_START) & (
        weather["Date"].dt.strftime("%m-%d") <= GROWING_SEASON_END
    )
    season = weather[season_mask]

    per_year_precip = season.groupby("year")["Precipitation"].sum()
    per_year_et0 = season.groupby("year")["ReferenceET"].sum()
    per_year_dry_spell = season.groupby("year")["Precipitation"].apply(_max_consecutive_dry_days)
    per_year_hot_days = season.groupby("year").apply(lambda g: (g["MaxTemp"] > HOT_DAY_THRESHOLD_C).sum(), include_groups=False)
    per_year_gdd = season.groupby("year").apply(
        lambda g: sum(
            growing_degree_day(crop.GDDmethod, crop.Tupp, crop.Tbase, row.MaxTemp, row.MinTemp)
            for row in g.itertuples()
        ),
        include_groups=False,
    )
    pci = weather.groupby("year").apply(_precipitation_concentration_index, include_groups=False).mean()

    P = per_year_precip.mean()
    ET0 = per_year_et0.mean()

    return {
        "site_id": site_id,
        "P_mm": P,
        "ET0_mm": ET0,
        "aridity_index": P / ET0,
        "CV_P": per_year_precip.std() / per_year_precip.mean(),
        "max_dry_spell_days": per_year_dry_spell.mean(),
        "hot_days": per_year_hot_days.mean(),
        "GDD": per_year_gdd.mean(),
        "PCI": pci,
    }


def build_fingerprints() -> pd.DataFrame:
    crop = Crop("Maize", planting_date="05/01", harvest_date="09/15")
    rows = [compute_site_fingerprint(site_id, crop) for site_id in SITES]
    df = pd.DataFrame(rows).set_index("site_id")

    # Soil block: constant across sites for now (same standard profile
    # everywhere) - included so the fingerprint schema matches 研究方案 6.4,
    # but it won't contribute to inter-site distance until real per-site
    # soil data is available.
    soil = get_soil("loam")
    top_layer = soil.profile.iloc[0]
    df["AWC"] = top_layer["th_fc"] - top_layer["th_wp"]

    return df


if __name__ == "__main__":
    df = build_fingerprints()
    out_path = Path(__file__).resolve().parents[2] / "data" / "processed" / "environmental_fingerprints.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path)
    print(df.round(3).to_string())
    print(f"\nsaved -> {out_path}")
