"""Per-site cropping systems.

A single nationwide "winter wheat -> summer maize" assumption is wrong:
measured growing-degree-day accumulation over an Oct 10 - Jun 25 window
(AquaCrop's GDD method 3, Tbase=0, 1982-2022) shows

    henan_north          2844 GDD   100% of years >= 2200
    shaanxi_guanzhong    2515       100%
    hebei_central        2478       100%
    beijing_plain        2283        73%   <- marginal
    ningxia_irrigation   1833         0%   <- winter wheat not viable

which reproduces the real northern limit of winter wheat in China (it runs
roughly through the Beijing area) and the fact that Ningxia is a
spring-sown region, not a wheat-maize double-cropping one.

So the cropping system varies by site:
  - NCP double cropping (winter wheat -> summer maize): Henan, Shaanxi
    Guanzhong, Hebei, and Beijing with shorter-season cultivars.
  - Ningxia: single spring maize, its actual system.

This heterogeneity is not a nuisance to be smoothed over - it makes the
transfer study (paper 3) sharper, since transferring across *different
cropping systems* is a harder and more realistic test than transferring
across climates alone.

Cultivar calibration (P0-1, docs/审计修复计划.md): "clears Maturity GDD
by 06/25" (the wheat simulation window's outer bound) is not the same
thing as "harvests before 06/15" (maize's fixed planting date) - a crop
can need the full window and mature anywhere up to 06/25, which used to
let wheat's actual GDD-driven harvest land *after* maize's already-begun
simulated timeline (measured: hebei_central harvested 06-22 in 2013,
seven days into maize's season) - a genuine date-order violation, not
just a missed-day-of-water-balance rounding error. calibrate_wheat_
maturity.py searches, per site, for the shortest-season Maturity (with
Senescence/HIstart scaled at the same ratios as before) that leaves a
>=7-day margin before 06/15 in *every* year of 1982-2025 under full
irrigation (the worst case for late maturity - AquaCrop accelerates
senescence under stress, so a deficit-irrigated run matures no later).
Earlier-maturing cultivars at the margin is what growers in this region
actually do (the previous single-site version of this argument, applied
to Beijing only, is now generalized). Henan's GDD headroom is wide enough
that 2200 already clears every year with no changes needed.
"""

WHEAT_MATURITY_STANDARD = 2200  # henan_north only now - see per-site calibration below

# Fixed rotation calendar (MM/DD), the single source of truth - previously
# defined in rotation.py, which caused the dates to be re-declared wherever
# a crop was built; moved here (audit-v2) so crop construction via
# build_crop() and the engine share one definition.
WHEAT_PLANTING = "10/10"  # winter wheat sown in the autumn of year-1
WHEAT_HARVEST = "06/25"  # wheat simulation window end; actual maturity is GDD-driven
MAIZE_PLANTING = "06/15"  # after wheat harvest, standard NCP double-cropping
MAIZE_HARVEST = "10/05"

# audit-v3 (3.3): wheat's ACTUAL harvest must precede maize planting by at
# least this many days - an equal-date handoff would hand maize a soil
# state from a day wheat itself hadn't finished yet.
MIN_HANDOFF_GAP_DAYS = 1

DOUBLE_CROP = "wheat_maize"
SINGLE_SPRING_MAIZE = "spring_maize"

CROPPING_SYSTEMS = {
    # calibrate_wheat_maturity.py --site hebei_central: worst-case margin 7 days at Maturity=1825
    "hebei_central": {
        "system": DOUBLE_CROP,
        "wheat_params": dict(Maturity=1825, Senescence=1327, HIstart=995),
    },
    "henan_north": {
        "system": DOUBLE_CROP,
        "wheat_params": dict(Maturity=WHEAT_MATURITY_STANDARD, Senescence=1600, HIstart=1200),
    },
    # calibrate_wheat_maturity.py --site shaanxi_guanzhong: worst-case margin 7 days at Maturity=1850
    "shaanxi_guanzhong": {
        "system": DOUBLE_CROP,
        "wheat_params": dict(Maturity=1850, Senescence=1345, HIstart=1009),
    },
    # calibrate_wheat_maturity.py --site beijing_plain --start 2000: worst-case margin 7 days at
    # Maturity=1600 - needed a much bigger cut than the others (was already the shortest-season
    # cultivar at 2000 and still had 06/25-window violations up to 10 days late)
    "beijing_plain": {
        "system": DOUBLE_CROP,
        "wheat_params": dict(Maturity=1600, Senescence=1164, HIstart=873),
    },
    "ningxia_irrigation": {
        "system": SINGLE_SPRING_MAIZE,
        "wheat_params": None,
    },
}

# Spring maize (Ningxia): sown late April once soils warm, harvested late
# September - the standard single-crop calendar for the irrigation district.
SPRING_MAIZE_PLANTING = "04/25"
SPRING_MAIZE_HARVEST = "09/30"

# Summer maize (NCP double-crop sites): sown 06/15 after wheat, harvested by
# 10/05 (112-day window). AquaCrop's stock Maize cultivar matures at
# MaturityCD=132 days - 20 days longer than this window, so under full
# irrigation it never reached maturity and every season was silently
# truncated at the window end (P0-2, audit-v2; measured 0/20 seasons
# matured naturally, harvest date pinned at 10/05). Real North China Plain
# summer maize cultivars (e.g. 郑单958-class ~103-110 days from sowing to
# maturity) fit this window, so a 105-day calendar cultivar is used here
# instead, with Senescence/HIstart/MaxRooting/YldForm scaled at the same
# ratios as the stock cultivar (105/132 = 0.7955). This is a literature-
# based scenario parameter, NOT a site-calibrated cultivar - see
# docs/methods/crop_calibration.md for sources and the validation record
# (data/calibration/crop_parameter_sources.csv).
SUMMER_MAIZE_PARAMS = dict(
    MaturityCD=105, SenescenceCD=85, HIstartCD=53, MaxRootingCD=86, YldFormCD=49, EmergenceCD=5,
)

# Ningxia spring maize: the stock 132-day cultivar fits its long season
# (04/25 - 09/30, 158 days; matures ~09/04 under full irrigation) and is
# used unmodified.
SPRING_MAIZE_PARAMS = dict()


def system_for(site_id: str) -> str:
    return CROPPING_SYSTEMS[site_id]["system"]


def wheat_params_for(site_id: str) -> dict:
    params = CROPPING_SYSTEMS[site_id]["wheat_params"]
    if params is None:
        raise ValueError(f"{site_id} does not grow winter wheat (insufficient growing degree days)")
    return params


def maize_params_for(site_id: str) -> dict:
    """P0-2 (audit-v2): one unified place deciding which maize cultivar a
    site grows - double-crop sites get the short-season summer maize,
    Ningxia keeps the stock full-season spring maize."""
    if is_double_crop(site_id):
        return dict(SUMMER_MAIZE_PARAMS)
    return dict(SPRING_MAIZE_PARAMS)


def build_crop(site_id: str, crop_name: str) -> "Crop":
    """P0-2 (audit-v2): the single entry point for building crop objects -
    no caller may construct Crop("Maize", ...) with its own hardcoded
    dates/params anymore (previously scattered across rotation.py and
    rotation_env.py, which is how the 132-day stock cultivar slipped into
    a 112-day window)."""
    from aquacrop import Crop

    if crop_name == "wheat":
        return Crop(
            "WheatGDD", planting_date=WHEAT_PLANTING, harvest_date=WHEAT_HARVEST, **wheat_params_for(site_id)
        )
    if crop_name == "maize":
        return Crop(
            "Maize", planting_date=MAIZE_PLANTING, harvest_date=MAIZE_HARVEST, **maize_params_for(site_id)
        )
    if crop_name == "spring_maize":
        return Crop(
            "Maize", planting_date=SPRING_MAIZE_PLANTING, harvest_date=SPRING_MAIZE_HARVEST,
            **maize_params_for(site_id),
        )
    raise ValueError(f"unknown crop '{crop_name}' for {site_id}")


def is_double_crop(site_id: str) -> bool:
    return system_for(site_id) == DOUBLE_CROP
