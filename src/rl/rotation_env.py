"""Winter wheat -> summer maize rotation RL environment (audit-v2/v3).

Wraps AquaCropModel for day-by-day external control (irrigation_method=5),
annual 450mm hard quota, rule-based safety layer, preference-conditioned
4-term reward.

audit-v3 (round 3, pre-submission) changes in this file:
- 2.1: every reward/shaping/water-normalization knob comes from a single
  RLExperimentConfig (gamma, shaping_mode, water_normalizer) - the env no
  longer hardcodes SHAPING_GAMMA; PPO and the shaping use the SAME gamma,
  so a gamma ablation is single-factor.
- 3.1: potential shaping is now strict PBRS: Phi(terminal)=0 (absorbing
  state), NO crop-switch potential reset (the wheat->maize handoff is an
  ordinary MDP transition), and the telescoping identity holds exactly.
  shaping_mode="none" is the no-shaping ablation.
- 3.2: the terminal yield bonus uses SYSTEM-level normalization
  (sum of crop yields / sum of per-crop references) so double-crop and
  single-crop episodes have comparable reward magnitude.
- 3.3: wheat->maize handoff requires gap >= 1 day (MIN_HANDOFF_GAP_DAYS).
- 3.4: days_to_harvest is computed from calendar dates
  (harvest_window_end - current_model_date), not season_length - DAP, so
  the fallow days between wheat harvest and maize planting no longer push
  the late-season safety filter later.
- 3.5: emergency_shortfall_mm is computed from the ACTUAL applied
  irrigation (irr_cum delta), not the requested depth.
- 3.6: action modification reasons are split
  (safety_modified / quota_clipped / delivery_shortfall + per-rule flags);
  the paper's intervention rate must use safety_modified, not a mixed flag.
- 3.7: safety_enabled=False lets a raw (unsafety-filtered) agent be
  evaluated separately from agent+safety.
- 3.8: the "risk" reward term is renamed acute_stress (a 3-day mean
  tr_ratio<0.5 binary indicator) - it is NOT interannual downside risk.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim"))

import numpy as np
import pandas as pd
from aquacrop import AquaCropModel, InitialWaterContent, IrrigationManagement

from cropping_systems import (
    MAIZE_HARVEST, MAIZE_PLANTING, MIN_HANDOFF_GAP_DAYS, SPRING_MAIZE_HARVEST,
    SPRING_MAIZE_PLANTING, WHEAT_HARVEST, WHEAT_PLANTING, build_crop, is_double_crop,
)
from experiment_config import PRIMARY_CONFIG, RLExperimentConfig
from rotation import RotationCalendarError, _wc_from_profile
from soil_moisture_init import initial_water_content as observed_initial_wc
from soils import get_soil
from weather import load_site_weather

ANNUAL_QUOTA_MM = 450.0
DECISION_INTERVAL_DAYS = 3
ACTIONS_MM = [0.0, 10.0, 20.0, 30.0, 40.0]
MAX_ACTION_MM = max(ACTIONS_MM)

# Safety layer thresholds (研究方案 5.8)
SATURATION_DEPLETION_FRAC = 0.1
HEAVY_RAIN_MM_3D = 20.0
LATE_SEASON_DAYS_TO_HARVEST = 7

# audit-v3 (3.3): the wheat->maize calendar handoff must leave at least
# one full day of gap - a same-day handoff would hand maize a soil state
# from a day wheat itself hadn't finished yet.
MIN_HANDOFF_GAP_DAYS = 1

CRITICAL_DEPLETION_FRAC = 0.85
CRITICAL_DEPLETION_MIN_MM = 20.0

# audit-v3 (3.8): cost model - water resource penalty lives in the
# "water" term; "cost" covers water-fee + per-event startup (documented
# overlap, reported in the paper; see 3.9 方案A).
COST_WATER = 0.1  # CNY per mm
COST_START = 0.5  # CNY per irrigation event start

# Yield normalization per crop (near the top of what each system produces
# under generous irrigation; audit-v2 P0-2 cultivar-fixed measurements).
YIELD_REFERENCE = {"wheat": 7.0, "maize": 11.5, "spring_maize": 14.5}


def safety_filter(action_mm, state, days_to_harvest, remaining_quota):
    """Returns (filtered_mm, triggered_rules). filtered_mm is NOT yet
    clamped to remaining_quota - step() applies the hard cap itself, so
    "safety wanted X but quota allowed Y" stays an observable distinction."""
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

    # Critical-depletion floor: a preference within whatever quota is left
    # (never bypasses it - step() clamps unconditionally).
    if state["depletion_frac"] >= CRITICAL_DEPLETION_FRAC:
        if adjusted < CRITICAL_DEPLETION_MIN_MM:
            triggered.append("critical_depletion")
        adjusted = max(adjusted, CRITICAL_DEPLETION_MIN_MM)

    # Round tiny residuals to avoid spuriously flagged modifications.
    adjusted = float(np.round(adjusted, 3))
    return adjusted, triggered


class RotationIrrigationEnv:
    def __init__(self, site_id, soil_key, year, annual_quota=ANNUAL_QUOTA_MM,
                 config: RLExperimentConfig = PRIMARY_CONFIG, safety_enabled: bool = True):
        """audit-v3 (2.1/3.7): all reward knobs come from `config`; the
        safety layer can be disabled for the raw-agent evaluation (3.7)."""
        self.site_id = site_id
        self.soil_key = soil_key
        self.year = year
        self.annual_quota = annual_quota
        self.config = config
        self.config.validate()
        self.safety_enabled = safety_enabled
        self._weather = load_site_weather(site_id)

    # --- season plumbing -------------------------------------------------
    def _start_season(self, crop_name, sim_start, sim_end, initial_wc, off_season=False):
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
        # audit-v3 (3.4): track the crop's ACTUAL planting date (the window
        # may start before it, e.g. maize's fallow bridge) and the fixed
        # harvest-window end; days_to_harvest is derived from calendar
        # dates, never from season_length - DAP (which would count the
        # fallow gap as crop days and delay the late-season filter).
        if crop_name == "wheat":
            self.crop_planting_date = pd.Timestamp(f"{self.year - 1}-{WHEAT_PLANTING.replace('/', '-')}")
        elif crop_name == "spring_maize":
            self.crop_planting_date = pd.Timestamp(f"{self.year}-{SPRING_MAIZE_PLANTING.replace('/', '-')}")
        else:
            self.crop_planting_date = pd.Timestamp(f"{self.year}-{MAIZE_PLANTING.replace('/', '-')}")
        self.harvest_window_end_date = pd.Timestamp(sim_end.replace("/", "-"))
        self.season_days = (self.harvest_window_end_date - self.crop_planting_date).days

    def _advance_to_planting(self):
        """P0-1 (audit-v2): fast-forward an off-season window from its
        start (day after wheat's actual harvest) to the maize planting day,
        simulating fallow bare-soil water balance with zero irrigation."""
        planting = self.crop_planting_date
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
        self._last_state = self._get_state()
        # audit-v3 (3.1): PBRS potential starts at the initial state's
        # potential; Phi(terminal)=0, no resets anywhere in between.
        self._potential = float(self.model._init_cond.tr_ratio)
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
        """audit-v3 (3.3): maize planting must be >= MIN_HANDOFF_GAP_DAYS
        after wheat's actual harvest - equal dates are rejected."""
        maize_planting_date = pd.Timestamp(f"{self.year}-{MAIZE_PLANTING.replace('/', '-')}")
        wheat_harvest_date = pd.Timestamp(self.season_results["wheat"]["harvest_date"])
        gap_days = (maize_planting_date - wheat_harvest_date).days
        if gap_days < MIN_HANDOFF_GAP_DAYS:
            raise RotationCalendarError(
                f"{self.site_id} {self.year}: wheat harvested {wheat_harvest_date.date()}, maize plants "
                f"{maize_planting_date.date()} - handoff gap {gap_days} day(s) < "
                f"{MIN_HANDOFF_GAP_DAYS} (recalibrate wheat_params_for('{self.site_id}'))"
            )

    def _water_denominator(self) -> float:
        if self.config.water_normalizer == "annual_quota":
            # guard a degenerate zero-quota setup (used by tests) so the
            # reward never divides by zero
            return max(self.annual_quota, 1e-6)
        return MAX_ACTION_MM

    def step(self, action_mm):
        assert not self.done, "call reset() before stepping a finished episode"
        remaining_quota = self.annual_quota - self.quota_used
        # audit-v3 (3.4): calendar-based days to harvest - the fallow days
        # between wheat harvest and maize planting must NOT shift the
        # late-season safety filter.
        current_date = pd.Timestamp(self.model._clock_struct.step_start_time)
        days_to_harvest = max(0, (self.harvest_window_end_date - current_date).days)

        if self.safety_enabled:
            filtered_mm, triggered_rules = safety_filter(
                action_mm, self._last_state, days_to_harvest, remaining_quota
            )
        else:
            # audit-v3 (3.7): raw-agent evaluation - no safety filtering.
            filtered_mm, triggered_rules = float(action_mm), []
        # annual quota is a hard cap, unconditionally
        applied = min(filtered_mm, max(remaining_quota, 0.0))

        self.model._param_struct.IrrMngt.depth = applied
        # P0-3 (audit-v2): account for ACTUAL applied (irr_cum delta), not
        # the requested depth.
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
            f"quota_used {self.quota_used} exceeded annual_quota {self.annual_quota} - hard-cap violated"
        )
        if actual_applied > 0:
            self.days_since_last_irr, self.last_irr_mm = 0, actual_applied
        else:
            self.days_since_last_irr += DECISION_INTERVAL_DAYS

        # audit-v3 (3.5): emergency shortfall is based on the ACTUAL
        # delivered irrigation, not the requested/clipped depth.
        emergency_shortfall_mm = (
            max(0.0, CRITICAL_DEPLETION_MIN_MM - actual_applied)
            if "critical_depletion" in triggered_rules else 0.0
        )

        # season/switch handling must complete BEFORE the shaping term so
        # phi_next reflects the true next state (or the absorbing terminal
        # state) - audit-v3 (3.1).
        yield_bonus = 0.0
        if self.model._clock_struct.model_is_finished:
            th_end = self._finish_season()
            crop = self.current_crop
            # audit-v3 (3.2 FIX): the yield signal is paid when the crop is
            # harvested - per-crop yield/reference with the discount factor
            # of THAT step - not lumped once at episode end (where gamma^T
            # halves it and the per-step water penalty dominates, collapsing
            # the policy into water-minimization; see the retrain learning
            # curves). The PBRS shaping is untouched: Phi(terminal)=0 still
            # holds and the bonus is a separate non-shaping reward term.
            yield_bonus = self.season_results[crop]["dry_yield_t_ha"] / YIELD_REFERENCE[crop]
            if crop == "wheat":
                self._check_wheat_maize_handoff()
                wheat_harvest = pd.Timestamp(self.season_results["wheat"]["harvest_date"])
                maize_start = (wheat_harvest + pd.Timedelta(days=1)).strftime("%Y/%m/%d")
                self._start_season(
                    "maize", maize_start, f"{self.year}/{MAIZE_HARVEST}",
                    _wc_from_profile(th_end), off_season=True,
                )
                self._advance_to_planting()
                self.days_since_last_irr, self.last_irr_mm = 99, 0.0
            else:
                self.done = True

        # audit-v3 (3.1): strict PBRS - Phi(terminal)=0 (absorbing state),
        # no potential resets anywhere, gamma from the shared config.
        if self.config.shaping_mode == "potential":
            phi_next = 0.0 if self.done else float(self.model._init_cond.tr_ratio)
            shaping_reward = self.config.gamma * phi_next - self._potential
            self._potential = phi_next
        else:
            shaping_reward = 0.0

        mean_stress = float(np.mean(stress))
        water_denom = self._water_denominator()
        reward = {
            # yield_proxy = strict-PBRS shaping (telescopes to a constant,
            # zero policy gradient) + the per-harvest yield bonus (the
            # actual yield signal).
            "yield_proxy": shaping_reward + yield_bonus,
            "water": -actual_applied / water_denom,
            "cost": -(COST_WATER * actual_applied + COST_START * (actual_applied > 0))
            / (COST_WATER * water_denom + COST_START),
            # audit-v3 (3.8): acute-stress indicator, NOT interannual risk
            "acute_stress": -1.0 if mean_stress < 0.5 else 0.0,
        }

        # audit-v3 (3.6): split the action-modification reasons.
        tolerance = 1e-6
        safety_modified = not np.isclose(filtered_mm, float(action_mm), atol=tolerance)
        quota_clipped = not np.isclose(applied, filtered_mm, atol=tolerance)
        delivery_shortfall_mm = max(0.0, applied - actual_applied)
        delivery_modified = delivery_shortfall_mm > tolerance
        any_modified = safety_modified or quota_clipped or delivery_modified
        rule_flags = {r: (r in triggered_rules) for r in
                      ("saturation", "heavy_rain_forecast", "late_season", "critical_depletion")}

        info = {
            "raw_action_mm": float(action_mm),
            "filtered_action_mm": filtered_mm,
            "actual_model_irrigation_mm": actual_applied,
            "applied_mm": applied,
            "action_modified": any_modified,  # deprecated aggregate; use the split fields
            "safety_modified": safety_modified,
            "quota_clipped": quota_clipped,
            "delivery_shortfall_mm": delivery_shortfall_mm,
            "delivery_modified": delivery_modified,
            **{f"modified_by_{r}": v for r, v in rule_flags.items()},
            "modified_by_quota": quota_clipped,
            "safety_rule_triggered": ",".join(triggered_rules) if triggered_rules else "",
            "quota_used_mm": self.quota_used,
            "quota_violation_mm": max(0.0, self.quota_used - self.annual_quota),
            "emergency_shortfall_mm": emergency_shortfall_mm,
            "crop": self.current_crop,
        }

        if self.done:
            # audit-v3 (3.2): per-harvest bonuses already paid the yield
            # signal; this block only reports the system-level aggregates.
            system_yield = sum(v["dry_yield_t_ha"] for v in self.season_results.values())
            info["total_yield_t_ha"] = system_yield
            info["total_irrigation_mm"] = sum(v["irrigation_mm"] for v in self.season_results.values())

        for name, res in self.season_results.items():
            info[name] = res

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
    """P0-5: stronger baseline - same trigger, but wheat stops once it has
    used more than wheat_reserve_frac of the annual quota, reserving the
    rest for maize."""
    if state["depletion_frac"] <= threshold:
        return 0.0
    if state["is_wheat"] and state["remaining_annual_quota"] <= (1 - wheat_reserve_frac) * annual_quota:
        return 0.0
    return depth
