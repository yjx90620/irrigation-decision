"""P0-1 acceptance tests (docs/审计修复计划.md): the rotation engine's
wheat->maize handoff must never hand maize a soil-moisture state from a
date that hasn't happened yet in its own simulated timeline.

Deliberately slow (runs real AquaCropModel seasons across the full
1982-2025 history for every double-crop site) - these are acceptance
tests for a calibration, not fast unit tests, and are meant to be run
after any change to cropping_systems.py's wheat_params_for() or to
rotation.py's fixed calendar constants.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))

import pandas as pd
import pytest
from aquacrop import IrrigationManagement

from cropping_systems import CROPPING_SYSTEMS, is_double_crop
from rotation import (
    MAIZE_HARVEST, MAIZE_PLANTING, RotationCalendarError, run_rotation_series,
    run_rotation_years_independent,
)

YEARS = list(range(1982, 2026))
DOUBLE_CROP_SITES = [s for s in CROPPING_SYSTEMS if is_double_crop(s)]


def moderate_irr():
    return IrrigationManagement(irrigation_method=1, SMT=[60, 60, 60, 60])


def full_irr():
    return IrrigationManagement(irrigation_method=1, SMT=[100, 100, 100, 100])


@pytest.mark.parametrize("site_id", DOUBLE_CROP_SITES)
def test_wheat_harvest_precedes_maize_planting(site_id):
    """Every historical year must clear the calendar without
    RotationCalendarError - if this fails, cropping_systems.py's cultivar
    for this site needs recalibrating (see calibrate_wheat_maturity.py)."""
    df = run_rotation_years_independent(site_id, "loam", YEARS, moderate_irr, moderate_irr)
    wheat = df[df["crop"] == "wheat"]
    assert (wheat["gap_days_to_maize_planting"] >= 0).all()


@pytest.mark.parametrize("site_id", DOUBLE_CROP_SITES)
def test_dates_are_strictly_monotonic(site_id):
    """A continuous multi-year run (run_rotation_series) must not regress:
    each year's wheat sowing is after the previous year's maize harvest."""
    years = YEARS[:6]  # a handful of consecutive years is enough to catch a regression
    df = run_rotation_series(site_id, "loam", years, moderate_irr, moderate_irr)
    harvest_dates = pd.to_datetime(df["harvest_date"])
    assert harvest_dates.is_monotonic_increasing, (
        f"{site_id}: harvest dates are not strictly increasing across consecutive years:\n{df[['year','crop','harvest_date']]}"
    )


def test_calendar_violation_raises_not_silently_continues():
    """Sanity check that the guard is load-bearing: reintroducing the
    pre-calibration cultivar (Maturity=2200, empirically confirmed to
    violate the 06/15 cutoff in 21/44 historical years at hebei_central -
    see docs/审计修复计划.md) must raise RotationCalendarError on at
    least one of those known-bad years, not silently run maize forward
    with a stale/future soil profile. Maturity is capped by how much GDD
    the Oct-Jun window can actually accumulate (~2478 at this site) -
    anything higher makes AquaCrop itself refuse to build a crop
    calendar, which would test a different failure mode, not this one."""
    import cropping_systems

    site_id = "hebei_central"
    known_bad_year = 2013  # harvested 2013-06-22 under the old Maturity=2200 cultivar
    original = cropping_systems.CROPPING_SYSTEMS[site_id]["wheat_params"]
    cropping_systems.CROPPING_SYSTEMS[site_id]["wheat_params"] = dict(Maturity=2200, Senescence=1600, HIstart=1200)
    try:
        with pytest.raises(RotationCalendarError):
            run_rotation_years_independent(site_id, "loam", [known_bad_year], moderate_irr, moderate_irr)
    finally:
        cropping_systems.CROPPING_SYSTEMS[site_id]["wheat_params"] = original


# --- audit-v2 (P0-2): maize must actually mature inside its window --------

@pytest.mark.parametrize("site_id", DOUBLE_CROP_SITES)
def test_summer_maize_matures_naturally(site_id):
    """P0-2 (audit-v2): the short-season summer-maize cultivar must reach
    maturity inside the 06/15-10/05 window under full irrigation -
    AquaCrop's crop_mature flag must fire and harvest must land strictly
    before 10/05. The old stock 132-day cultivar was always truncated at
    the window end (measured 0/20 seasons matured naturally)."""
    years = [1982, 1995, 2010, 2018, 2024]
    df = run_rotation_years_independent(site_id, "loam", years, full_irr, full_irr)
    maize = df[df["crop"] == "maize"]
    assert len(maize) == len(years)
    assert maize["matured_naturally"].all(), f"{site_id}: maize seasons truncated at window end"
    harvest = pd.to_datetime(maize["harvest_date"])
    window_end = pd.to_datetime(maize["year"].astype(str) + "-10-05")
    assert (harvest < window_end).all(), f"{site_id}: maize harvest not strictly before 10/05"


def test_ningxia_spring_maize_matures_naturally():
    """P0-2 (audit-v2): Ningxia's spring maize (stock 132-day cultivar in
    the 04/25-09/30 window) must also mature naturally."""
    years = [1982, 1995, 2010, 2018, 2024]
    df = run_rotation_years_independent("ningxia_irrigation", "loam", years, full_irr, full_irr)
    maize = df[df["crop"] == "spring_maize"]
    assert len(maize) == len(years)
    assert maize["matured_naturally"].all(), "Ningxia spring maize truncated at window end"
    harvest = pd.to_datetime(maize["harvest_date"])
    window_end = pd.to_datetime(maize["year"].astype(str) + "-09-30")
    assert (harvest < window_end).all()


def test_build_crop_selects_site_cultivar():
    """P0-2 (audit-v2): build_crop is the single cultivar entry point -
    double-crop sites get the 105-day summer maize, Ningxia keeps the
    stock 132-day spring maize; constructing crops with hardcoded dates
    anywhere else is what let the wrong cultivar into the window."""
    from cropping_systems import build_crop

    hebei_maize = build_crop("hebei_central", "maize")
    assert hebei_maize.MaturityCD == 105
    nx_maize = build_crop("ningxia_irrigation", "spring_maize")
    assert nx_maize.MaturityCD == 132


# --- audit-v2 (P0-1): fallow periods are simulated, not jumped -------------

def test_nonconsecutive_years_are_not_chained():
    """P0-1 (audit-v2): run_rotation_series must refuse non-consecutive
    year lists - chaining across an un-simulated gap year hands a season a
    year-stale profile (and previously hung AquaCrop at Beijing)."""
    with pytest.raises(ValueError, match="consecutive"):
        run_rotation_series("beijing_plain", "loam", [2011, 2013], moderate_irr, moderate_irr)


def test_offseason_fallow_water_balance_conserves_water():
    """P0-1 (audit-v2): the fallow days between wheat's ACTUAL harvest and
    maize planting are simulated as bare-soil balance - the maize season's
    window starts the day after wheat harvest with off_season=True, and
    advancing it to the planting day (the env's _advance_to_planting path)
    must move the profile. Water cannot be created during fallow: total
    profile water may rise by at most the precipitation that actually fell
    in the gap (rainfed - no irrigation occurs)."""
    from aquacrop import AquaCropModel, IrrigationManagement

    from cropping_systems import build_crop
    from rotation import _wc_from_profile
    from soils import get_soil
    from weather import load_site_weather

    site_id, year = "hebei_central", 2015
    df = run_rotation_years_independent(site_id, "loam", [year], moderate_irr, moderate_irr)
    hw = pd.Timestamp(df[df["crop"] == "wheat"].iloc[0]["harvest_date"])
    planting = pd.Timestamp(f"{year}-{MAIZE_PLANTING.replace('/', '-')}")
    assert (planting - hw).days > 0  # a real gap exists this year

    soil = get_soil("loam")
    dz = soil.profile["dz"].tolist()
    wc_before = [0.30] * len(dz)
    weather = load_site_weather(site_id)
    gap_rain = float(weather[(weather["Date"] > hw) & (weather["Date"] <= planting)]["Precipitation"].sum())

    model = AquaCropModel(
        sim_start_time=(hw + pd.Timedelta(days=1)).strftime("%Y/%m/%d"),
        sim_end_time=f"{year}/{MAIZE_HARVEST}",  # window continues past planting (AquaCrop needs it)
        weather_df=weather, soil=soil, crop=build_crop(site_id, "maize"),
        initial_water_content=_wc_from_profile(wc_before),
        irrigation_management=IrrigationManagement(irrigation_method=0),  # rainfed - no fallow irrigation
        off_season=True,
    )
    model._initialize()
    while pd.Timestamp(model._clock_struct.step_start_time) < planting:
        model.run_model(num_steps=1, initialize_model=False)
    th_after = model._init_cond.th

    before_mm = sum(w * d for w, d in zip(wc_before, dz)) * 1000
    after_mm = sum(w * d for w, d in zip(th_after, dz)) * 1000
    assert after_mm <= before_mm + gap_rain + 1e-6, (
        f"fallow created water: profile gained {after_mm - before_mm:.2f} mm "
        f"with only {gap_rain:.1f} mm of rain"
    )
    # the fallow actually moved water (bare-soil evaporation/redistribution)
    assert abs(after_mm - before_mm) > 1e-6
