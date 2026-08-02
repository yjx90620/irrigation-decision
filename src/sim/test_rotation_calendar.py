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
    MAIZE_PLANTING, RotationCalendarError, run_rotation_series, run_rotation_years_independent,
)

YEARS = list(range(1982, 2026))
DOUBLE_CROP_SITES = [s for s in CROPPING_SYSTEMS if is_double_crop(s)]


def moderate_irr():
    return IrrigationManagement(irrigation_method=1, SMT=[60, 60, 60, 60])


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
