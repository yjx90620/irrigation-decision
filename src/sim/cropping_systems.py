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
    Guanzhong, Hebei, and Beijing with a shorter-season wheat cultivar
    (Maturity 2000 GDD, which Beijing clears in every year on record -
    using earlier-maturing cultivars at the margin is what growers
    actually do).
  - Ningxia: single spring maize, its actual system.

This heterogeneity is not a nuisance to be smoothed over - it makes the
transfer study (paper 3) sharper, since transferring across *different
cropping systems* is a harder and more realistic test than transferring
across climates alone.
"""

WHEAT_MATURITY_STANDARD = 2200
WHEAT_MATURITY_SHORT = 2000  # earlier cultivar for the northern margin

DOUBLE_CROP = "wheat_maize"
SINGLE_SPRING_MAIZE = "spring_maize"

CROPPING_SYSTEMS = {
    "hebei_central": {
        "system": DOUBLE_CROP,
        "wheat_params": dict(Maturity=WHEAT_MATURITY_STANDARD, Senescence=1600, HIstart=1200),
    },
    "henan_north": {
        "system": DOUBLE_CROP,
        "wheat_params": dict(Maturity=WHEAT_MATURITY_STANDARD, Senescence=1600, HIstart=1200),
    },
    "shaanxi_guanzhong": {
        "system": DOUBLE_CROP,
        "wheat_params": dict(Maturity=WHEAT_MATURITY_STANDARD, Senescence=1600, HIstart=1200),
    },
    "beijing_plain": {
        "system": DOUBLE_CROP,
        # shorter-season cultivar; site clears 2000 GDD in all years on record
        "wheat_params": dict(Maturity=WHEAT_MATURITY_SHORT, Senescence=1450, HIstart=1090),
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
