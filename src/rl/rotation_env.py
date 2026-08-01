"""Rotation-based irrigation decision environment: one episode covers a
full winter wheat -> summer maize cycle under a shared annual water quota.

Replaces env.py's single-season maize setup, which turned out to have
almost no irrigation signal (4 of 5 sites reached 93-99% of
full-irrigation yield with zero irrigation - see docs/系统升级方案.md).
Under the rotation the agent faces a real cross-season tradeoff: water
spent on wheat both produces wheat yield *and* depletes the profile the
following maize inherits, and both draw on one annual quota.

Each season is stepped day-by-day through AquaCrop's irrigation_method=5
(constant depth, re-read every simulated day), then the wheat season's
final soil profile is handed to the maize season - the same chaining
rotation.py uses for the non-RL simulations, so RL and optimization
results stay comparable.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim"))

import numpy as np
import pandas as pd
from aquacrop import AquaCropModel, Crop, InitialWaterContent, IrrigationManagement

from rotation import (
    MAIZE_HARVEST, MAIZE_PLANTING, WHEAT_HARVEST, WHEAT_PARAMS, WHEAT_PLANTING, _wc_from_profile,
)
from soil_moisture_init import initial_water_content as observed_initial_wc
from soils import get_soil
from weather import load_site_weather

ACTIONS_MM = [0, 10, 20, 30, 40]
DECISION_INTERVAL_DAYS = 3
ANNUAL_QUOTA_MM = 450.0

COST_WATER = 1.0
COST_START = 5.0

# Safety layer thresholds (研究方案 5.8), same intent as env.py's
SATURATION_DEPLETION_FRAC = 0.1
HEAVY_RAIN_MM_3D = 20.0
LATE_SEASON_DAYS_TO_HARVEST = 7

# Yield normalization per crop, so the two seasons contribute comparably to
# reward despite maize out-yielding wheat. Values are near the top of what
# the rotation produces under generous irrigation (measured: wheat ~6.5-7,
# maize ~8.8-9).
YIELD_REFERENCE = {"wheat": 7.0, "maize": 9.0}


def safety_filter(action_mm, state, days_to_harvest, remaining_quota):
    adjusted = action_mm
    if adjusted > 0 and state["depletion_frac"] < SATURATION_DEPLETION_FRAC:
        adjusted = 0.0
    if adjusted > 0 and state["precip_next_3d"] >= HEAVY_RAIN_MM_3D:
        adjusted = 0.0
    if adjusted > 0 and days_to_harvest <= LATE_SEASON_DAYS_TO_HARVEST:
        adjusted = 0.0
    adjusted = min(adjusted, max(remaining_quota, 0))
    return adjusted, adjusted != action_mm


class RotationIrrigationEnv:
    def __init__(self, site_id, soil_key, year, annual_quota=ANNUAL_QUOTA_MM):
        self.site_id = site_id
        self.soil_key = soil_key
        self.year = year
        self.annual_quota = annual_quota
        self._weather = load_site_weather(site_id)

    # --- season plumbing -------------------------------------------------
    def _start_season(self, crop_name, sim_start, sim_end, initial_wc):
        if crop_name == "wheat":
            crop = Crop("WheatGDD", planting_date=WHEAT_PLANTING, harvest_date=WHEAT_HARVEST, **WHEAT_PARAMS)
        else:
            crop = Crop("Maize", planting_date=MAIZE_PLANTING, harvest_date=MAIZE_HARVEST)
        model = AquaCropModel(
            sim_start_time=sim_start,
            sim_end_time=sim_end,
            weather_df=self._weather,
            soil=get_soil(self.soil_key),
            crop=crop,
            initial_water_content=initial_wc,
            irrigation_management=IrrigationManagement(irrigation_method=5, depth=0),
        )
        model._initialize()
        self.model = model
        self.current_crop = crop_name
        planting = pd.Timestamp(sim_start.replace("/", "-"))
        self.season_days = (pd.Timestamp(sim_end.replace("/", "-")) - planting).days

    def reset(self):
        first_sowing = f"{self.year - 1}-{WHEAT_PLANTING.replace('/', '-')}"
        initial_wc = observed_initial_wc(self.site_id, self.soil_key, first_sowing)
        self._start_season(
            "wheat", f"{self.year - 1}/{WHEAT_PLANTING}", f"{self.year}/{WHEAT_HARVEST}", initial_wc
        )
        self.quota_used = 0.0
        self.days_since_last_irr = 99
        self.last_irr_mm = 0.0
        self.done = False
        self.season_results = {}
        self._last_state = self._get_state()
        return self._last_state

    # --- observation -----------------------------------------------------
    def _forecast(self, horizon_days):
        current_date = self.model._clock_struct.step_start_time
        mask = (self._weather["Date"] > current_date) & (
            self._weather["Date"] <= current_date + np.timedelta64(horizon_days, "D")
        )
        window = self._weather[mask]
        return {
            "precip_sum": window["Precipitation"].sum(),
            "et0_sum": window["ReferenceET"].sum(),
            "hot_days": int((window["MaxTemp"] > 33).sum()),
        }

    def _get_state(self):
        nc = self.model._init_cond
        fc3, fc7 = self._forecast(3), self._forecast(7)
        return {
            "is_wheat": 1.0 if self.current_crop == "wheat" else 0.0,
            "dap": nc.dap,
            "growth_stage": nc.growth_stage,
            "canopy_cover": nc.canopy_cover,
            "z_root": nc.z_root,
            "biomass": nc.biomass,
            "gdd_cum": nc.gdd_cum,
            "tr_ratio": nc.tr_ratio,
            "depletion_frac": (nc.depletion / nc.taw) if nc.taw > 0 else 0.0,
            "season_irr_cum": nc.irr_cum,
            "days_since_last_irr": self.days_since_last_irr,
            "last_irr_mm": self.last_irr_mm,
            "precip_next_3d": fc3["precip_sum"],
            "precip_next_7d": fc7["precip_sum"],
            "et0_next_7d": fc7["et0_sum"],
            "hot_days_next_7d": fc7["hot_days"],
            "remaining_annual_quota": self.annual_quota - self.quota_used,
        }

    # --- stepping --------------------------------------------------------
    def _finish_season(self):
        results = self.model.get_simulation_results()
        flux = self.model.get_water_flux()
        crop = self.current_crop
        self.season_results[crop] = {
            "dry_yield_t_ha": results["Dry yield (tonne/ha)"].iloc[0],
            "irrigation_mm": results["Seasonal irrigation (mm)"].iloc[0],
            "deep_perc_mm": flux["DeepPerc"].sum(),
            "runoff_mm": flux["Runoff"].sum(),
        }
        return self.model._init_cond.th

    def step(self, action_mm):
        assert not self.done, "call reset() before stepping a finished episode"
        remaining_quota = self.annual_quota - self.quota_used
        days_to_harvest = self.season_days - self._last_state["dap"]
        applied, modified = safety_filter(action_mm, self._last_state, days_to_harvest, remaining_quota)

        self.model._param_struct.IrrMngt.depth = applied
        self.model.run_model(num_steps=1, initialize_model=False)
        stress = [self.model._init_cond.tr_ratio]
        self.model._param_struct.IrrMngt.depth = 0
        for _ in range(DECISION_INTERVAL_DAYS - 1):
            if self.model._clock_struct.model_is_finished:
                break
            self.model.run_model(num_steps=1, initialize_model=False)
            stress.append(self.model._init_cond.tr_ratio)

        self.quota_used += applied
        if applied > 0:
            self.days_since_last_irr, self.last_irr_mm = 0, applied
        else:
            self.days_since_last_irr += DECISION_INTERVAL_DAYS

        mean_stress = float(np.mean(stress))
        reward = {
            "yield_proxy": mean_stress,
            "water": -applied / max(ACTIONS_MM),
            "cost": -(COST_WATER * applied + COST_START * (applied > 0)) / (COST_WATER * max(ACTIONS_MM) + COST_START),
            "risk": -1.0 if mean_stress < 0.5 else 0.0,
        }
        info = {"applied_mm": applied, "raw_action_mm": action_mm, "action_modified": modified,
                "crop": self.current_crop}

        if self.model._clock_struct.model_is_finished:
            th_end = self._finish_season()
            crop = self.current_crop
            reward["yield_proxy"] += self.season_results[crop]["dry_yield_t_ha"] / YIELD_REFERENCE[crop]

            if crop == "wheat":
                # hand the depleted profile to maize and keep going
                self._start_season(
                    "maize", f"{self.year}/{MAIZE_PLANTING}", f"{self.year}/{MAIZE_HARVEST}",
                    _wc_from_profile(th_end),
                )
                self.days_since_last_irr, self.last_irr_mm = 99, 0.0
            else:
                self.done = True
                info["wheat"] = self.season_results["wheat"]
                info["maize"] = self.season_results["maize"]
                info["total_yield_t_ha"] = sum(v["dry_yield_t_ha"] for v in self.season_results.values())
                info["total_irrigation_mm"] = sum(v["irrigation_mm"] for v in self.season_results.values())

        state = self._get_state()
        self._last_state = state
        return state, reward, self.done, info


def combine_reward(reward, weights):
    return sum(weights[k] * reward[k] for k in reward)


def threshold_policy(state, weights=None, threshold=0.4, depth=20.0):
    """Rule baseline, also the base policy that residual RL corrects."""
    return depth if state["depletion_frac"] > threshold else 0.0
