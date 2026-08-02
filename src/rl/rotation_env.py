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

from cropping_systems import (
    SPRING_MAIZE_HARVEST, SPRING_MAIZE_PLANTING, is_double_crop, wheat_params_for,
)
from rotation import (
    MAIZE_HARVEST, MAIZE_PLANTING, WHEAT_HARVEST, WHEAT_PLANTING, _wc_from_profile,
)
from soil_moisture_init import initial_water_content as observed_initial_wc
from soils import get_soil
from weather import load_site_weather

ACTIONS_MM = [0, 10, 20, 30, 40]
DECISION_INTERVAL_DAYS = 3
ANNUAL_QUOTA_MM = 450.0

COST_WATER = 1.0
COST_START = 5.0

# P0-4 (docs/审计修复计划.md): the old reward added mean_stress every
# single step (~120 accumulations across a double-crop episode) but only
# added the actual yield outcome once or twice, so a policy optimizing
# this reward was mostly optimizing accumulated stress-proxy, not yield.
# Potential-based shaping (Ng, Harada & Russell 1999): F(s,a,s') = gamma *
# Phi(s') - Phi(s) for any potential function Phi provably does not change
# which policy is optimal, and telescopes to Phi(terminal) - Phi(initial)
# over a full episode regardless of how many steps it took - so this
# structurally fixes the "reward scales with episode length" problem
# instead of just picking a small coefficient and hoping it's small
# enough. Phi = tr_ratio (transpiration ratio, already in state, bounded
# [0,1]) is a reasonable stand-in for "how well-watered is the crop right
# now". SHAPING_GAMMA matches PPO's own discount (train_rotation_compare.py
# uses gamma=0.995) - the invariance proof requires the same gamma the
# policy is actually optimized under.
SHAPING_GAMMA = 0.995

# Safety layer thresholds (研究方案 5.8), same intent as env.py's
SATURATION_DEPLETION_FRAC = 0.1
HEAVY_RAIN_MM_3D = 20.0
LATE_SEASON_DAYS_TO_HARVEST = 7

# Soft floor: originally added after training hung for 20+ hours with
# zero checkpoints, as a workaround for what turned out to be a genuine
# unbounded loop in aquacrop-ospy's calculate_HIGC() (severe water stress
# can collapse a crop's calendar into a state where that function never
# converges - see docs/aquacrop_patches.md). That's now fixed at the
# actual root (patches/patch_aquacrop_higc.py), so this floor no longer
# needs to - and per P0-2 (docs/审计修复计划.md) must not - bypass the
# annual quota to do its job. It's now a *preference* applied only within
# whatever quota remains: step() clamps to remaining_quota unconditionally,
# so a depleted quota near season end can still leave the crop under
# critical stress (recorded as an emergency shortfall, not hidden).
CRITICAL_DEPLETION_FRAC = 0.85
CRITICAL_DEPLETION_MIN_MM = 20.0

# Yield normalization per crop, so seasons contribute comparably to reward
# despite maize out-yielding wheat. Values are near the top of what each
# system produces under generous irrigation (measured: wheat ~6.5-7,
# double-crop maize ~8.8-9, Ningxia spring maize ~14.4 - the single-crop
# season is much longer and yields far more).
YIELD_REFERENCE = {"wheat": 7.0, "maize": 9.0, "spring_maize": 14.5}


def safety_filter(action_mm, state, days_to_harvest, remaining_quota):
    """Returns (filtered_mm, triggered_rules). filtered_mm is NOT yet
    clamped to remaining_quota - callers (RotationIrrigationEnv.step())
    must do `applied = min(filtered_mm, max(remaining_quota, 0.0))`
    themselves and are responsible for the quota being an actual hard cap
    (P0-2, docs/审计修复计划.md). This function no longer clamps to quota
    itself so that "the safety filter wanted X mm but quota only allowed Y"
    is an observable distinction, not silently merged into one number."""
    adjusted = action_mm
    triggered = []
    if adjusted > 0 and state["depletion_frac"] < SATURATION_DEPLETION_FRAC:
        adjusted = 0.0
        triggered.append("saturation")
    if adjusted > 0 and state["precip_next_3d"] >= HEAVY_RAIN_MM_3D:
        adjusted = 0.0
        triggered.append("heavy_rain_forecast")
    if adjusted > 0 and days_to_harvest <= LATE_SEASON_DAYS_TO_HARVEST:
        adjusted = 0.0
        triggered.append("late_season")

    # Critical-depletion floor: a *preference* for more water when the
    # crop is under severe stress, applied within whatever quota is left
    # (never bypasses it - see CRITICAL_DEPLETION_FRAC's comment). Still
    # overrides the agronomic rules above (saturation/rain/late-season),
    # since "severely water-stressed" trumps those heuristics regardless
    # of quota.
    if state["depletion_frac"] >= CRITICAL_DEPLETION_FRAC:
        if adjusted < CRITICAL_DEPLETION_MIN_MM:
            triggered.append("critical_depletion")
        adjusted = max(adjusted, CRITICAL_DEPLETION_MIN_MM)

    return adjusted, triggered


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
            crop = Crop(
                "WheatGDD", planting_date=WHEAT_PLANTING, harvest_date=WHEAT_HARVEST,
                **wheat_params_for(self.site_id),
            )
        elif crop_name == "spring_maize":
            crop = Crop("Maize", planting_date=SPRING_MAIZE_PLANTING, harvest_date=SPRING_MAIZE_HARVEST)
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
        self.double_crop = is_double_crop(self.site_id)
        if self.double_crop:
            first_sowing = f"{self.year - 1}-{WHEAT_PLANTING.replace('/', '-')}"
            initial_wc = observed_initial_wc(self.site_id, self.soil_key, first_sowing)
            self._start_season(
                "wheat", f"{self.year - 1}/{WHEAT_PLANTING}", f"{self.year}/{WHEAT_HARVEST}", initial_wc
            )
        else:
            first_sowing = f"{self.year}-{SPRING_MAIZE_PLANTING.replace('/', '-')}"
            initial_wc = observed_initial_wc(self.site_id, self.soil_key, first_sowing)
            self._start_season(
                "spring_maize", f"{self.year}/{SPRING_MAIZE_PLANTING}",
                f"{self.year}/{SPRING_MAIZE_HARVEST}", initial_wc,
            )
        self.quota_used = 0.0
        self.days_since_last_irr = 99
        self.last_irr_mm = 0.0
        self.done = False
        self.season_results = {}
        self._pending_yield_bonus = 0.0
        self._last_state = self._get_state()
        self._potential = self._last_state["tr_ratio"]
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
            "harvest_date": str(results["Harvest Date (YYYY/MM/DD)"].iloc[0])[:10],
        }
        return self.model._init_cond.th

    def _check_wheat_maize_handoff(self):
        # P0-1 date-order check (docs/审计修复计划.md), mirrors
        # rotation.py's run_rotation_year - wheat's actual GDD-driven
        # harvest must precede maize's fixed planting date, or maize would
        # start from a soil-moisture state that hasn't happened yet in its
        # own simulated timeline.
        maize_planting_date = pd.Timestamp(f"{self.year}-{MAIZE_PLANTING.replace('/', '-')}")
        wheat_harvest_date = pd.Timestamp(self.season_results["wheat"]["harvest_date"])
        gap_days = (maize_planting_date - wheat_harvest_date).days
        if gap_days < 0:
            from rotation import RotationCalendarError

            raise RotationCalendarError(
                f"{self.site_id} {self.year}: wheat harvested {wheat_harvest_date.date()}, on/after maize's "
                f"fixed planting date {maize_planting_date.date()} ({-gap_days} day(s) late) - recalibrate "
                f"wheat_params_for('{self.site_id}') in cropping_systems.py"
            )

    def step(self, action_mm):
        assert not self.done, "call reset() before stepping a finished episode"
        remaining_quota = self.annual_quota - self.quota_used
        days_to_harvest = self.season_days - self._last_state["dap"]
        filtered_mm, triggered_rules = safety_filter(action_mm, self._last_state, days_to_harvest, remaining_quota)
        # P0-2 (docs/审计修复计划.md): the annual quota is a hard cap,
        # unconditionally - nothing (including the critical-depletion
        # floor above) may push actual model irrigation past what's left.
        applied = min(filtered_mm, max(remaining_quota, 0.0))
        emergency_shortfall_mm = (
            max(0.0, CRITICAL_DEPLETION_MIN_MM - applied) if "critical_depletion" in triggered_rules else 0.0
        )

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
        assert self.quota_used <= self.annual_quota + 1e-6, (
            f"quota_used {self.quota_used} exceeded annual_quota {self.annual_quota} - P0-2 hard-cap violated"
        )
        if applied > 0:
            self.days_since_last_irr, self.last_irr_mm = 0, applied
        else:
            self.days_since_last_irr += DECISION_INTERVAL_DAYS

        mean_stress = float(np.mean(stress))  # kept only for the risk indicator below
        new_potential = float(self.model._init_cond.tr_ratio)
        shaping_reward = SHAPING_GAMMA * new_potential - self._potential
        self._potential = new_potential
        reward = {
            "yield_proxy": shaping_reward,
            "water": -applied / max(ACTIONS_MM),
            "cost": -(COST_WATER * applied + COST_START * (applied > 0)) / (COST_WATER * max(ACTIONS_MM) + COST_START),
            "risk": -1.0 if mean_stress < 0.5 else 0.0,
        }
        info = {
            "raw_action_mm": action_mm,
            "filtered_action_mm": filtered_mm,
            "actual_model_irrigation_mm": applied,
            "applied_mm": applied,  # kept for backward compatibility with existing callers
            "action_modified": applied != action_mm,
            "safety_rule_triggered": ",".join(triggered_rules) if triggered_rules else "",
            "quota_used_mm": self.quota_used,
            "quota_violation_mm": max(0.0, self.quota_used - self.annual_quota),
            "emergency_shortfall_mm": emergency_shortfall_mm,
            "crop": self.current_crop,
        }

        if self.model._clock_struct.model_is_finished:
            th_end = self._finish_season()
            crop = self.current_crop
            # P0-4 (docs/审计修复计划.md): yield is banked here, not added
            # to reward yet - paid out as a single system-level terminal
            # reward when the whole rotation year ends (see `self.done`
            # branch below), not once per crop, so a double-crop episode
            # doesn't get 2x the terminal signal a single-crop one gets.
            self._pending_yield_bonus += self.season_results[crop]["dry_yield_t_ha"] / YIELD_REFERENCE[crop]

            if crop == "wheat":
                self._check_wheat_maize_handoff()
                # hand the depleted profile to maize and keep going
                self._start_season(
                    "maize", f"{self.year}/{MAIZE_PLANTING}", f"{self.year}/{MAIZE_HARVEST}",
                    _wc_from_profile(th_end),
                )
                self.days_since_last_irr, self.last_irr_mm = 99, 0.0
                # reset the shaping potential's baseline at the crop switch -
                # otherwise the first maize step's shaping term would jump
                # purely because maize's tr_ratio dynamics start from a
                # different baseline than wheat's, not because of anything
                # the policy did.
                self._potential = float(self.model._init_cond.tr_ratio)
            else:
                # end of the last (or only) season of the cycle
                self.done = True
                reward["yield_proxy"] += self._pending_yield_bonus
                for name, res in self.season_results.items():
                    info[name] = res
                info["total_yield_t_ha"] = sum(v["dry_yield_t_ha"] for v in self.season_results.values())
                info["total_irrigation_mm"] = sum(v["irrigation_mm"] for v in self.season_results.values())

        state = self._get_state()
        self._last_state = state
        info["site_id"] = self.site_id
        return state, reward, self.done, info


def combine_reward(reward, weights):
    return sum(weights[k] * reward[k] for k in reward)


def threshold_policy(state, weights=None, threshold=0.4, depth=20.0):
    """Rule baseline, also the base policy that residual RL corrects."""
    return depth if state["depletion_frac"] > threshold else 0.0


def quota_reserving_policy(
    state, weights=None, threshold=0.4, depth=20.0, wheat_reserve_frac=0.6, annual_quota=ANNUAL_QUOTA_MM,
):
    """P0-5 (docs/审计修复计划.md): a stronger baseline than threshold_policy
    - same depletion-threshold trigger, but wheat additionally stops
    irrigating once it has used more than `wheat_reserve_frac` of the
    *annual* quota, reserving the rest for maize instead of burning the
    whole budget on wheat and leaving maize nothing (the specific failure
    mode threshold_policy has - see this module's evaluate() results).
    Comparing RL against only the weaker threshold_policy risks crediting
    RL for beating a strawman rather than a rule that already encodes the
    obvious fix."""
    if state["depletion_frac"] <= threshold:
        return 0.0
    if state["is_wheat"] and state["remaining_annual_quota"] <= (1 - wheat_reserve_frac) * annual_quota:
        return 0.0
    return depth
