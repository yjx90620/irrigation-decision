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


def system_for(site_id: str) -> str:
    return CROPPING_SYSTEMS[site_id]["system"]


def wheat_params_for(site_id: str) -> dict:
    params = CROPPING_SYSTEMS[site_id]["wheat_params"]
    if params is None:
        raise ValueError(f"{site_id} does not grow winter wheat (insufficient growing degree days)")
    return params


def is_double_crop(site_id: str) -> bool:
    return system_for(site_id) == DOUBLE_CROP
