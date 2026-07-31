"""Standard soil profiles used until SoilGrids access is available.

AquaCrop-OSPy ships built-in parameterizations for these soil types
(confirmed against aquacrop==3.1.0: SandyLoam/Loam/ClayLoam all load with
12 layers). They stand in for 砂壤土/壤土/黏壤土 from docs/研究方案.md
section 4.4 until real SoilGrids-derived profiles replace them.
"""


from aquacrop.entities.soil import Soil

STANDARD_SOILS = {
    "sandy_loam": "SandyLoam",
    "loam": "Loam",
    "clay_loam": "ClayLoam",
}


def get_soil(key: str) -> Soil:
    return Soil(STANDARD_SOILS[key])
