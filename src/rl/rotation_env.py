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
from aquacrop import AquaCropModel, InitialWaterContent, IrrigationManagement

from cropping_systems import (
    MAIZE_HARVEST, MAIZE_PLANTING, SPRING_MAIZE_HARVEST, SPRING_MAIZE_PLANTING,
    WHEAT_HARVEST, WHEAT_PLANTING, build_crop, is_double_crop,
)
from rotation import RotationCalendarError, _wc_from_profile
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
# system produces under generous irrigation. Measured with the audit-v2
# cultivar fixes (P0-2): wheat ~6.5-7, double-crop summer maize now
# ~10.3-11.6 (the old 9.0 reference came from the truncated 132-day stock
# cultivar, which never reached maturity inside the 112-day window),
# Ningxia spring maize ~13.5-14.8.
YIELD_REFERENCE = {"wheat": 7.0, "maize": 11.5, "spring_maize": 14.5}


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
    def __init__(self, site_id, soil_key, year, annual_quota=ANNUAL_QUOTA_MM, water_norm="per_action"):
        """water_norm (audit-v2 P0-9): the per-step water/cost penalty's
        normalizer. "per_action" (= max(ACTIONS_MM), the historical default)
        makes the water signal dominate the return ~10:1 over the terminal
        yield bonus (measured: -3.15 vs +0.32 on hebei 2019), driving the
        policy to minimize water at the expense of yield. "per_quota"
        (=/annual_quota) shrinks the per-step penalty ~11x so yield and
        water balance at ~1.7:1 - the audit's suggested renormalization.
        The reward scale is an experiment axis; it must be consistent
        between an arm's training and its evaluation."""
        self.site_id = site_id
        self.soil_key = soil_key
        self.year = year
        self.annual_quota = annual_quota
        self.water_norm = water_norm
        self._weather = load_site_weather(site_id)

    # --- season plumbing -------------------------------------------------
    def _start_season(self, crop_name, sim_start, sim_end, initial_wc, off_season=False):
        # P0-2 (audit-v2): crops are built by cropping_systems.build_crop(),
        # the single place that knows each site's cultivar - the old inline
        # Crop("WheatGDD"/"Maize") construction here is how the wrong
        # 132-day stock maize slipped into the 112-day summer window.
        crop = build_crop(self.site_id, crop_name)
        model = AquaCropModel(
            sim_start_time=sim_start,
            sim_end_time=sim_end,
            weather_df=self._weather,
            soil=get_soil(self.soil_key),
            crop=crop,
            initial_water_content=initial_wc,
            irrigation_management=IrrigationManagement(irrigation_method=5, depth=0),
            off_season=off_season,
        )
        model._initialize()
        self.model = model
        self.current_crop = crop_name
        planting = pd.Timestamp(sim_start.replace("/", "-"))
        self.season_days = (pd.Timestamp(sim_end.replace("/", "-")) - planting).days

    def _advance_to_planting(self):
        """P0-1 (audit-v2): fast-forward an off-season window from its
        start (the day after wheat's actual harvest) to the maize planting
        day, simulating the fallow days' bare-soil water balance with zero
        irrigation. The policy only decides from planting onward - the
        fallow itself is not a decision period, but its effect on the
        profile maize inherits IS simulated (previously skipped entirely)."""
        planting = pd.Timestamp(f"{self.year}-{MAIZE_PLANTING.replace('/', '-')}")
        self.model._param_struct.IrrMngt.depth = 0
        while pd.Timestamp(self.model._clock_struct.step_start_time) < planting:
            if self.model._clock_struct.model_is_finished:
                raise RotationCalendarError(
                    f"{self.site_id} {self.year}: model finished before maize planting during fallow "
                    f"advance - check the summer-maize cultivar (cropping_systems.SUMMER_MAIZE_PARAMS)"
                )
            self.model.run_model(num_steps=1, initialize_model=False)

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
        # P0-3 (audit-v2): AquaCrop applies at most what the soil profile
        # can actually hold on the decision day - requesting 40mm delivered
        # a constant 25mm across every step (measured) - so quota and the
        # water/cost rewards must account for the ACTUAL applied amount
        # (irr_cum delta), not the requested depth, or training over-counts
        # water use by ~60% and the quota narrative describes a water
        # volume that was never applied.
        irr_before = float(self.model._init_cond.irr_cum)
        self.model.run_model(num_steps=1, initialize_model=False)
        actual_applied = max(0.0, float(self.model._init_cond.irr_cum) - irr_before)
        stress = [self.model._init_cond.tr_ratio]
        self.model._param_struct.IrrMngt.depth = 0
        for _ in range(DECISION_INTERVAL_DAYS - 1):
            if self.model._clock_struct.model_is_finished:
                break
            self.model.run_model(num_steps=1, initialize_model=False)
            stress.append(self.model._init_cond.tr_ratio)

        self.quota_used += actual_applied
        assert self.quota_used <= self.annual_quota + 1e-6, (
            f"quota_used {self.quota_used} exceeded annual_quota {self.annual_quota} - P0-2 hard-cap violated"
        )
        if actual_applied > 0:
            self.days_since_last_irr, self.last_irr_mm = 0, actual_applied
        else:
            self.days_since_last_irr += DECISION_INTERVAL_DAYS

        mean_stress = float(np.mean(stress))  # kept only for the risk indicator below
        new_potential = float(self.model._init_cond.tr_ratio)
        shaping_reward = SHAPING_GAMMA * new_potential - self._potential
        self._potential = new_potential
        # audit-v2 P0-9: water_norm scales the per-step water/cost penalty
        # (see __init__ docstring) - "per_quota" rebalances it against the
        # terminal yield bonus instead of letting it dominate ~10:1.
        water_denom = self.annual_quota if self.water_norm == "per_quota" else max(ACTIONS_MM)
        reward = {
            "yield_proxy": shaping_reward,
            "water": -actual_applied / water_denom,
            "cost": -(COST_WATER * actual_applied + COST_START * (actual_applied > 0))
            / (COST_WATER * water_denom + COST_START),
            "risk": -1.0 if mean_stress < 0.5 else 0.0,
        }
        info = {
            "raw_action_mm": action_mm,
            "filtered_action_mm": filtered_mm,
            # P0-3 (audit-v2): the model-truth applied amount (irr_cum
            # delta), which is what quota/rewards now account for.
            "actual_model_irrigation_mm": actual_applied,
            "applied_mm": applied,  # requested-after-safety-filter, kept for backward compatibility
            "action_modified": actual_applied != action_mm,
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
                # P0-1 (audit-v2): hand the depleted profile to maize via an
                # off-season window starting the day after wheat's ACTUAL
                # harvest - the fallow days before maize planting (up to a
                # month of rain/ET, previously skipped) are simulated by
                # _advance_to_planting before the policy takes over.
                wheat_harvest = pd.Timestamp(self.season_results["wheat"]["harvest_date"])
                maize_start = (wheat_harvest + pd.Timedelta(days=1)).strftime("%Y/%m/%d")
                self._start_season(
                    "maize", maize_start, f"{self.year}/{MAIZE_HARVEST}",
                    _wc_from_profile(th_end), off_season=True,
                )
                self._advance_to_planting()
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
