"""Winter wheat - summer maize double-cropping rotation engine.

AquaCrop-OSPy cannot simulate crop rotations in a single run (the rotation
code in aquacrop.initialize.read_model_parameters is commented out with
"The model does not allow rotations now"), so this chains separate runs
and hands the soil water profile from each crop to the next via
InitialWaterContent(wc_type="Num", method="Layer").

Why this matters (see docs/系统升级方案.md): with every season restarting
at field capacity, 4 of 5 sites reached 93-99% of full-irrigation yield
with zero irrigation - there was almost no irrigation signal to study.
Under a real rotation the wheat crop draws the profile down near wilting
point by June, so the following maize starts water-limited: measured
3.58 t/ha (after rainfed wheat) vs 7.41 (after irrigated wheat).

Winter wheat phenology is calibrated for the North China Plain rather
than using AquaCrop's stock parameters, which don't fit: calendar-day
`Wheat` matures late April (vs the real early-June harvest) with yield
pinned regardless of irrigation, and stock `WheatGDD` needs 2400 GDD
which an Oct-Jun window just barely misses (2398 measured). See
WHEAT_PARAMS below.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))

import pandas as pd
from aquacrop import AquaCropModel, Crop, InitialWaterContent

from cropping_systems import (
    SPRING_MAIZE_HARVEST, SPRING_MAIZE_PLANTING, is_double_crop, wheat_params_for,
)
from soil_moisture_init import initial_water_content as observed_initial_wc
from soils import get_soil
from weather import load_site_weather

# Winter wheat is GDD-based (Tbase=0) and its maturity requirement is set
# per site by cropping_systems.py, because sites differ in whether they can
# finish a winter wheat cycle at all - Ningxia cannot (1833 GDD vs the 2200
# a standard cultivar needs) and Beijing only can with a shorter-season
# cultivar. See cropping_systems.py for the measurements.
WHEAT_PLANTING = "10/10"
WHEAT_HARVEST = "06/25"  # simulation window end; actual maturity is GDD-driven

MAIZE_PLANTING = "06/15"  # after wheat harvest, standard NCP double-cropping
MAIZE_HARVEST = "10/05"


class RotationCalendarError(Exception):
    """Raised when a rotation year's simulated crop calendar violates date
    order (P0-1, docs/审计修复计划.md) - concretely, wheat's actual
    (GDD-driven) harvest date lands on or after maize's fixed planting
    date, which would otherwise hand maize an initial soil-moisture state
    from days that haven't happened yet in its own simulated timeline.

    Not caught anywhere by design: this means the cultivar calibration in
    cropping_systems.py is wrong for this (site, year), and the fix is
    recalibrating it (see calibrate_wheat_maturity.py), not silently
    continuing with a physically inconsistent handoff."""


def _wc_from_profile(th) -> InitialWaterContent:
    return InitialWaterContent(
        wc_type="Num", method="Layer", depth_layer=list(range(1, len(th) + 1)), value=list(th)
    )


def run_season(weather_df, soil_key, crop, sim_start, sim_end, irrigation_management, initial_wc):
    model = AquaCropModel(
        sim_start_time=sim_start,
        sim_end_time=sim_end,
        weather_df=weather_df,
        soil=get_soil(soil_key),
        crop=crop,
        initial_water_content=initial_wc,
        irrigation_management=irrigation_management,
    )
    model.run_model(till_termination=True)
    results = model.get_simulation_results()
    flux = model.get_water_flux()
    metrics = {
        "dry_yield_t_ha": results["Dry yield (tonne/ha)"].iloc[0],
        "irrigation_mm": results["Seasonal irrigation (mm)"].iloc[0],
        "irrigation_events": int((flux["IrrDay"] > 0).sum()),
        "deep_perc_mm": flux["DeepPerc"].sum(),
        "runoff_mm": flux["Runoff"].sum(),
        "et_mm": (flux["Es"] + flux["Tr"]).sum(),
        "harvest_date": str(results["Harvest Date (YYYY/MM/DD)"].iloc[0])[:10],
    }
    return metrics, model._init_cond.th


def run_rotation_year(site_id, weather_df, soil_key, year, wheat_irr, maize_irr, initial_wc):
    """One cropping cycle for `year`. At double-cropping sites that's winter
    wheat (sown Oct of year-1) followed by summer maize; at single-crop sites
    (Ningxia, which lacks the growing degree days for winter wheat) it's one
    spring maize season. Returns (rows, end_of_year_soil_profile)."""
    if not is_double_crop(site_id):
        maize_crop = Crop("Maize", planting_date=SPRING_MAIZE_PLANTING, harvest_date=SPRING_MAIZE_HARVEST)
        maize_metrics, th_end = run_season(
            weather_df, soil_key, maize_crop,
            f"{year}/{SPRING_MAIZE_PLANTING}", f"{year}/{SPRING_MAIZE_HARVEST}",
            maize_irr, initial_wc,
        )
        return [{"year": year, "crop": "spring_maize", **maize_metrics}], th_end

    wheat_crop = Crop(
        "WheatGDD", planting_date=WHEAT_PLANTING, harvest_date=WHEAT_HARVEST, **wheat_params_for(site_id)
    )
    wheat_metrics, th_after_wheat = run_season(
        weather_df, soil_key, wheat_crop,
        f"{year - 1}/{WHEAT_PLANTING}", f"{year}/{WHEAT_HARVEST}",
        wheat_irr, initial_wc,
    )

    # P0-1 date-order check (docs/审计修复计划.md): wheat's actual
    # GDD-driven harvest must precede maize's fixed planting date, or
    # maize would be initialized with a soil-moisture state from days
    # that haven't happened yet in its own simulated timeline.
    maize_planting_date = pd.Timestamp(f"{year}-{MAIZE_PLANTING.replace('/', '-')}")
    wheat_harvest_date = pd.Timestamp(wheat_metrics["harvest_date"])
    gap_days = (maize_planting_date - wheat_harvest_date).days
    if gap_days < 0:
        raise RotationCalendarError(
            f"{site_id} {year}: wheat harvested {wheat_harvest_date.date()}, on/after maize's fixed planting "
            f"date {maize_planting_date.date()} ({-gap_days} day(s) late) - recalibrate wheat_params_for('"
            f"{site_id}') in cropping_systems.py (see calibrate_wheat_maturity.py)"
        )
    wheat_metrics["gap_days_to_maize_planting"] = gap_days

    maize_crop = Crop("Maize", planting_date=MAIZE_PLANTING, harvest_date=MAIZE_HARVEST)
    maize_metrics, th_after_maize = run_season(
        weather_df, soil_key, maize_crop,
        f"{year}/{MAIZE_PLANTING}", f"{year}/{MAIZE_HARVEST}",
        maize_irr, _wc_from_profile(th_after_wheat),
    )

    rows = [
        {"year": year, "crop": "wheat", **wheat_metrics},
        {"year": year, "crop": "maize", **maize_metrics},
    ]
    return rows, th_after_maize


def run_rotation_series(site_id, soil_key, years, wheat_irr_factory, maize_irr_factory, initial_wc=None):
    """Multi-year continuous rotation - soil water carries across the whole
    series, not just within a year, so a dry year's depletion propagates
    forward the way it does in a real field.

    initial_wc defaults to the ERA5-Land observed profile on the first
    wheat sowing date rather than field capacity: starting at FC hands the
    first season ~150-200mm of free water and is a large part of why the
    single-season prototype had almost no irrigation signal. Observed
    autumn wetness varies a lot between years (Hebei 2018-10-10 sat at 7%
    of plant-available capacity vs 65% in 2020), and that variation is
    itself part of what an irrigation policy has to respond to."""
    weather_df = load_site_weather(site_id)
    if initial_wc is None:
        if is_double_crop(site_id):
            first_sowing = f"{years[0] - 1}-{WHEAT_PLANTING.replace('/', '-')}"
        else:
            first_sowing = f"{years[0]}-{SPRING_MAIZE_PLANTING.replace('/', '-')}"
        initial_wc = observed_initial_wc(site_id, soil_key, first_sowing)

    all_rows = []
    carry_wc = initial_wc
    for year in years:
        rows, th_end = run_rotation_year(
            site_id, weather_df, soil_key, year, wheat_irr_factory(), maize_irr_factory(), carry_wc
        )
        for r in rows:
            r.update({"site_id": site_id, "soil": soil_key})
        all_rows.extend(rows)
        carry_wc = _wc_from_profile(th_end)

    return pd.DataFrame(all_rows)


def run_rotation_years_independent(site_id, soil_key, years, wheat_irr_factory, maize_irr_factory):
    """Same per-year simulation as run_rotation_series, but each year in
    `years` starts fresh from ITS OWN observed autumn soil moisture instead
    of carrying the profile over from the previous entry in the list.

    Use this (not run_rotation_series) whenever `years` is a non-consecutive
    sample (e.g. scan_allocation.py's [2011,2013,2015,2017,2018,2020]) rather
    than a genuine continuous multi-year run. Chaining across a gap year
    that was never simulated doesn't make physical sense - it hands a season
    a year-stale soil profile as if no time had passed - and empirically it
    is not just wrong but pathological: [2011, 2013] with 2012 skipped hung
    AquaCrop indefinitely at Beijing (diagnosed while debugging
    scan_allocation.py's timeouts), while [2011, 2012] (consecutive, no
    gap) and [2011] alone both ran in under 2 seconds. Independent-year
    evaluation sidesteps the issue entirely rather than working around it
    with a timeout, and is also the more defensible experiment design here:
    scan_allocation.py's question is "does the optimal split vary by year
    *type*", which independent samples answer more cleanly than one
    particular stitched-together trajectory would anyway.
    """
    weather_df = load_site_weather(site_id)
    all_rows = []
    for year in years:
        if is_double_crop(site_id):
            first_sowing = f"{year - 1}-{WHEAT_PLANTING.replace('/', '-')}"
        else:
            first_sowing = f"{year}-{SPRING_MAIZE_PLANTING.replace('/', '-')}"
        initial_wc = observed_initial_wc(site_id, soil_key, first_sowing)

        rows, _ = run_rotation_year(
            site_id, weather_df, soil_key, year, wheat_irr_factory(), maize_irr_factory(), initial_wc
        )
        for r in rows:
            r.update({"site_id": site_id, "soil": soil_key})
        all_rows.extend(rows)

    return pd.DataFrame(all_rows)
