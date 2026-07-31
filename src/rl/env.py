"""Preference-conditioned irrigation decision environment (研究方案 5.4-5.6).

Wraps AquaCropModel for day-by-day external control: irrigation_method=5
("constant depth") reads `param_struct.IrrMngt.depth` fresh every simulated
day, so mutating it between calls to `run_model(num_steps=..., initialize_model=False)`
lets an outside policy choose the depth applied each decision step - this
is what makes it usable as an RL environment instead of one of AquaCrop's
built-in auto-schedulers (verified against a manual day-by-day probe: a
15mm depth set on day 1 shows up as irr_cum=15 and stays there through two
subsequent zero-depth days).

The simulation window starts at the planting date rather than Jan 1: before
planting AquaCrop runs a "fallow" sub-model that ignores our IrrMngt object
entirely, which would make early actions silently no-ops.

Decision interval is 3 days (研究方案 5.4), action is a single irrigation
pulse applied on the first day of each window. Reward is a 4-vector
[yield_proxy, water, cost, risk] combined via the caller-supplied preference
weights - see combine_reward(). This is a first-pass reward design to
validate the environment mechanics; the yield_proxy uses the season's
transpiration ratio (1.0 = no water stress) as a per-step proxy since true
yield is only known at harvest, plus a final-day bonus from actual Dry
yield normalized against the site's full-irrigation ceiling from the
baseline experiment grid (data/processed/baseline_strategy_summary.csv).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim"))

import numpy as np
from aquacrop import AquaCropModel, Crop, InitialWaterContent, IrrigationManagement

from soils import get_soil
from weather import load_site_weather

ACTIONS_MM = [0, 10, 20, 30, 40]
DECISION_INTERVAL_DAYS = 3
PLANTING_DATE = "05/01"
HARVEST_DATE = "09/15"

COST_WATER = 1.0
COST_START = 5.0
MAX_IRR_SEASON_DEFAULT = 600.0  # mm, seasonal water budget cap

# Fallback yield ceiling (t/ha) used to normalize the terminal reward until
# every site/soil has a baseline_strategy_summary.csv entry to read from.
DEFAULT_YIELD_CEILING = 14.5


class IrrigationEnv:
    def __init__(self, site_id: str, soil_key: str, year: int, max_irr_season: float = MAX_IRR_SEASON_DEFAULT):
        self.site_id = site_id
        self.soil_key = soil_key
        self.year = year
        self.max_irr_season = max_irr_season
        self._weather_df = load_site_weather(site_id)
        self._year_df = self._weather_df[
            (self._weather_df["Date"] >= f"{year}-01-01") & (self._weather_df["Date"] <= f"{year}-12-31")
        ].reset_index(drop=True)

    def reset(self):
        soil = get_soil(self.soil_key)
        crop = Crop("Maize", planting_date=PLANTING_DATE, harvest_date=HARVEST_DATE)
        init_wc = InitialWaterContent(value=["FC"])
        irr_mngt = IrrigationManagement(irrigation_method=5, depth=0, MaxIrrSeason=self.max_irr_season)
        self.model = AquaCropModel(
            sim_start_time=f"{self.year}/{PLANTING_DATE}",
            sim_end_time=f"{self.year}/{HARVEST_DATE}",
            weather_df=self._year_df,
            soil=soil,
            crop=crop,
            initial_water_content=init_wc,
            irrigation_management=irr_mngt,
        )
        self.model._initialize()
        self.days_since_last_irr = 99
        self.last_irr_mm = 0.0
        self.done = False
        return self._get_state()

    def _forecast(self, horizon_days: int) -> dict:
        current_date = self.model._clock_struct.step_start_time
        mask = (self._year_df["Date"] > current_date) & (
            self._year_df["Date"] <= current_date + np.timedelta64(horizon_days, "D")
        )
        window = self._year_df[mask]
        return {
            "precip_sum": window["Precipitation"].sum(),
            "et0_sum": window["ReferenceET"].sum(),
            "hot_days": int((window["MaxTemp"] > 33).sum()),
        }

    def _get_state(self) -> dict:
        nc = self.model._init_cond
        fc3 = self._forecast(3)
        fc7 = self._forecast(7)
        return {
            "dap": nc.dap,
            "growth_stage": nc.growth_stage,
            "canopy_cover": nc.canopy_cover,
            "z_root": nc.z_root,
            "biomass": nc.biomass,
            "gdd_cum": nc.gdd_cum,
            "tr_ratio": nc.tr_ratio,
            "depletion_frac": (nc.depletion / nc.taw) if nc.taw > 0 else 0.0,
            "irr_cum": nc.irr_cum,
            "days_since_last_irr": self.days_since_last_irr,
            "last_irr_mm": self.last_irr_mm,
            "precip_next_3d": fc3["precip_sum"],
            "precip_next_7d": fc7["precip_sum"],
            "et0_next_7d": fc7["et0_sum"],
            "hot_days_next_7d": fc7["hot_days"],
            "remaining_water_budget": self.max_irr_season - nc.irr_cum,
        }

    def step(self, action_mm: float):
        assert not self.done, "call reset() before stepping a finished episode"
        remaining_budget = self.max_irr_season - self.model._init_cond.irr_cum
        applied = min(action_mm, max(remaining_budget, 0))

        self.model._param_struct.IrrMngt.depth = applied
        self.model.run_model(num_steps=1, initialize_model=False)
        stress_samples = [self.model._init_cond.tr_ratio]
        self.model._param_struct.IrrMngt.depth = 0
        for _ in range(DECISION_INTERVAL_DAYS - 1):
            if self.model._clock_struct.model_is_finished:
                break
            self.model.run_model(num_steps=1, initialize_model=False)
            stress_samples.append(self.model._init_cond.tr_ratio)

        if applied > 0:
            self.days_since_last_irr = 0
            self.last_irr_mm = applied
        else:
            self.days_since_last_irr += DECISION_INTERVAL_DAYS

        self.done = self.model._clock_struct.model_is_finished
        state = self._get_state()

        mean_stress = float(np.mean(stress_samples))
        reward = {
            "yield_proxy": mean_stress,  # 1.0 = no water stress this window
            "water": -applied,
            "cost": -(COST_WATER * applied + COST_START * (applied > 0)),
            "risk": -1.0 if mean_stress < 0.5 else 0.0,
        }
        info = {"applied_mm": applied}

        if self.done:
            res = self.model.get_simulation_results()
            info["dry_yield_t_ha"] = res["Dry yield (tonne/ha)"].iloc[0]
            info["seasonal_irrigation_mm"] = res["Seasonal irrigation (mm)"].iloc[0]
            reward["yield_proxy"] += info["dry_yield_t_ha"] / DEFAULT_YIELD_CEILING

        return state, reward, self.done, info


def combine_reward(reward: dict, weights: dict) -> float:
    """Scalarize the reward vector with preference weights (研究方案 5.6/5.7),
    weights keyed the same as the reward dict: yield_proxy/water/cost/risk."""
    return sum(weights[k] * reward[k] for k in reward)
