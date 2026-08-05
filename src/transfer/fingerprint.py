"""Climate-soil environmental fingerprint per site (研究方案 6.4).

P1-1 (docs/审计修复计划.md): two issues fixed here.

1. The old version used one hardcoded single-season maize window
   (05-01~09-15) for every site, which represents neither the winter
   wheat -> summer maize double-cropping sites nor Ningxia's spring
   maize system. Now uses the *same* calendar boundaries the actual
   rotation model runs on (rotation.py's WHEAT_PLANTING/MAIZE_PLANTING/
   MAIZE_HARVEST) split into a "cool season" (10/10-06/15, wheat's
   growing window at double-crop sites) and "warm season" (06/15-10/05,
   maize's window everywhere including Ningxia's actual spring-maize
   season, which overlaps this window closely). Computed the same way
   for every site regardless of what's actually planted there - it's a
   calendar-based climate split, not a claim about what grows where -
   plus an explicit cropping_system flag, so a downstream distance/
   similarity computation can use both without conflating them.

2. The old version used the full 1981-2025 weather record, including
   years that other experiments (task_sensitivity.py, RL's TEST_YEARS,
   etc.) hold out as an unseen test period - meaning a "target-domain-
   independent" transfer risk predictor was quietly built partly from
   the same period it would later be evaluated against. Restricted to
   TRAIN_YEARS (matches src/rl/residual_gym_env.py's training split).

audit-v3 (6.5): on top of the cool/warm season blocks, the fingerprint
is now also crop-stage-resolved - the same 4 calendar stages per crop
that marginal_water_value_v2.py measures yield response on (CROP_STAGES
in src/sim/cropping_systems.py is the single source), again computed
for EVERY site regardless of what grows there. So the feature set is
unified (identical columns for all sites) AND aligned with the
decision-relevant stages of the rotation calendar.

Only the climate block actually differentiates sites right now: soil is
still the same 3 standard AquaCrop profiles everywhere (SoilGrids access
is blocked, see src/data/README.md), and the seasonal water cap/decision
interval are fixed project-wide settings rather than something calibrated
per site. Sand/Clay fraction is left out entirely rather than faked -
AquaCrop-OSPy's built-in soil classes expose hydraulic properties
(th_fc, th_wp, th_s) but not texture fractions.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rl"))

import pandas as pd
from aquacrop import Crop
from aquacrop.solution.growing_degree_day import growing_degree_day

from config import SITES
from cropping_systems import CROP_STAGES, is_double_crop
from residual_gym_env import TRAIN_YEARS
from rotation import MAIZE_HARVEST, MAIZE_PLANTING, WHEAT_PLANTING
from soils import get_soil
from weather import load_site_weather

HOT_DAY_THRESHOLD_C = 33.0

COOL_SEASON = (WHEAT_PLANTING, MAIZE_PLANTING)  # 10/10 - 06/15, wraps year end
WARM_SEASON = (MAIZE_PLANTING, MAIZE_HARVEST)  # 06/15 - 10/05

WHEAT_REF_CROP = Crop("WheatGDD", planting_date="10/10", harvest_date="06/25", Maturity=1, Senescence=1, HIstart=1)
MAIZE_REF_CROP = Crop("Maize", planting_date="06/15", harvest_date="10/05")

# audit-v3 (6.5): crop-stage climate features - the same 4 calendar stages
# per crop that marginal_water_value_v2.py measures yield response on
# (CROP_STAGES in cropping_systems.py is the single source), computed for
# EVERY site regardless of what is actually planted, exactly like the
# cool/warm season blocks: a calendar-based split, not a claim about what
# grows where. This makes the fingerprint crop-stage-resolved (task-aligned
# with where water decisions matter) while staying unified across sites.
STAGE_CROPS = ("wheat", "maize")


def _precipitation_concentration_index(daily: pd.DataFrame) -> float:
    """Oliver's PCI: 100 * sum(monthly_precip^2) / (sum(monthly_precip))^2,
    over the full calendar year - standard climatology definition of how
    concentrated rainfall is across months."""
    monthly = daily.groupby(daily["Date"].dt.month)["Precipitation"].sum()
    return float(100 * (monthly**2).sum() / (monthly.sum() ** 2))


def _max_consecutive_dry_days(precip: pd.Series, dry_threshold_mm: float = 1.0) -> float:
    is_dry = (precip < dry_threshold_mm).astype(int)
    groups = (is_dry != is_dry.shift()).cumsum()
    run_lengths = is_dry.groupby(groups).transform("size") * is_dry
    return float(run_lengths.max())


def _season_mask(dates: pd.Series, start_md: str, end_md: str) -> pd.Series:
    md = dates.dt.strftime("%m-%d")
    if start_md <= end_md:
        return (md >= start_md) & (md <= end_md)
    return (md >= start_md) | (md <= end_md)  # wraps across year end (cool season)


def _season_stats(weather: pd.DataFrame, start_md_slash: str, end_md_slash: str, ref_crop: Crop, prefix: str) -> dict:
    start_md, end_md = start_md_slash.replace("/", "-"), end_md_slash.replace("/", "-")
    season = weather[_season_mask(weather["Date"], start_md, end_md)]
    # cool season wraps the year boundary (Oct of year Y-1 -> Jun of year
    # Y) - group by the *ending* calendar year so a full season lands in
    # one group instead of splitting across two.
    group_year = season["Date"].dt.year + (season["Date"].dt.strftime("%m-%d") >= "07-01").astype(int)

    per_year_precip = season.groupby(group_year)["Precipitation"].sum()
    per_year_et0 = season.groupby(group_year)["ReferenceET"].sum()
    per_year_dry_spell = season.groupby(group_year)["Precipitation"].apply(_max_consecutive_dry_days)
    per_year_hot_days = season.groupby(group_year).apply(
        lambda g: (g["MaxTemp"] > HOT_DAY_THRESHOLD_C).sum(), include_groups=False
    )
    per_year_gdd = season.groupby(group_year).apply(
        lambda g: sum(
            growing_degree_day(ref_crop.GDDmethod, ref_crop.Tupp, ref_crop.Tbase, row.MaxTemp, row.MinTemp)
            for row in g.itertuples()
        ),
        include_groups=False,
    )
    return {
        f"{prefix}_precip_mm": per_year_precip.mean(),
        f"{prefix}_et0_mm": per_year_et0.mean(),
        f"{prefix}_gdd": per_year_gdd.mean(),
        f"{prefix}_hot_days": per_year_hot_days.mean(),
        f"{prefix}_dry_spell_days": per_year_dry_spell.mean(),
    }


def compute_site_fingerprint(site_id: str) -> dict:
    weather = load_site_weather(site_id)
    weather = weather[weather["Date"].dt.year.isin(TRAIN_YEARS)].copy()

    row = {"site_id": site_id, "cropping_system_double_crop": int(is_double_crop(site_id))}
    row.update(_season_stats(weather, *COOL_SEASON, WHEAT_REF_CROP, "cool_season"))
    row.update(_season_stats(weather, *WARM_SEASON, MAIZE_REF_CROP, "warm_season"))

    # audit-v3 (6.5): crop-stage block - per-calendar-stage climate features
    # for wheat and maize, computed for every site (see STAGE_CROPS note).
    for crop, ref_crop in (("wheat", WHEAT_REF_CROP), ("maize", MAIZE_REF_CROP)):
        for stage_idx, (start, end) in enumerate(CROP_STAGES[crop], start=1):
            row.update(_season_stats(weather, start, end, ref_crop, f"{crop}_stage{stage_idx}"))

    weather["year"] = weather["Date"].dt.year
    per_year_precip = weather.groupby("year")["Precipitation"].sum()
    per_year_et0 = weather.groupby("year")["ReferenceET"].sum()
    row["P_mm"] = per_year_precip.mean()
    row["ET0_mm"] = per_year_et0.mean()
    row["aridity_index"] = row["P_mm"] / row["ET0_mm"]
    row["CV_P"] = per_year_precip.std() / per_year_precip.mean()
    row["PCI"] = weather.groupby("year").apply(_precipitation_concentration_index, include_groups=False).mean()
    return row


def build_fingerprints() -> pd.DataFrame:
    rows = [compute_site_fingerprint(site_id) for site_id in SITES]
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
