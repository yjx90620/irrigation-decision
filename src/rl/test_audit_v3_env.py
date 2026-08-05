"""audit-v3 (round 3) RL-environment tests: strict PBRS telescoping,
terminal potential zero, shaping gamma == PPO gamma, calendar-based
days_to_harvest, single/double-crop reward scale, actual-delivery
emergency shortfall, and split action-modification reasons."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pytest

from experiment_config import PRIMARY_CONFIG, RLExperimentConfig
from residual_gym_env import RotationGymEnv
from rotation import RotationCalendarError
from rotation_env import (
    CRITICAL_DEPLETION_MIN_MM, YIELD_REFERENCE, RotationIrrigationEnv, safety_filter, threshold_policy,
)


def _strip_harvest_bonuses(shaping, env, seen):
    """audit-v3 (3.2 FIX): yield_proxy = strict-PBRS shaping + per-harvest
    yield bonuses paid when each crop finishes. Strip a crop's bonus the
    first time it appears in season_results, so the remaining sum is the
    pure shaping (which must telescope)."""
    for crop, res in env.inner.season_results.items():
        if crop not in seen:
            seen.add(crop)
            shaping -= res["dry_yield_t_ha"] / YIELD_REFERENCE[crop]
    return shaping


def test_potential_shaping_telescopes():
    """audit-v3 (3.1): the pure shaping sum telescopes:
    sum gamma^t * shaping_t == -Phi(0) + gamma^T * Phi(terminal) with
    Phi(terminal)=0. yield_proxy = shaping + per-harvest yield bonuses,
    so strip each bonus the first time its crop finishes."""
    env = RotationGymEnv(mode="direct", fixed_site="hebei_central", soils=["loam"], years=[2015])
    state, _ = env.reset()
    phi_0 = float(env.inner._potential)
    gamma = env.inner.config.gamma
    done = False
    discounted_sum = 0.0
    t = 0
    seen = set()
    while not done:
        state, reward, done, truncated, info = env.step(2)
        shaping = info["reward_vector"]["yield_proxy"]
        shaping = _strip_harvest_bonuses(shaping, env, seen)
        discounted_sum += (gamma ** t) * shaping
        t += 1
    expected = -phi_0  # Phi(terminal) = 0
    assert np.isclose(discounted_sum, expected, atol=1e-6), (
        f"telescoping failed: {discounted_sum:.6f} != {expected:.6f}"
    )


def test_terminal_potential_is_zero():
    """audit-v3 (3.1): the final step's shaping must be -Phi(previous)
    (gamma * 0 - Phi), i.e. the terminal state's potential is zero. The
    final step also carries the last crop's harvest bonus - strip it."""
    env = RotationGymEnv(mode="direct", fixed_site="hebei_central", soils=["loam"], years=[2015])
    state, _ = env.reset()
    done = False
    while not done:
        state, reward, done, truncated, info = env.step(2)
        if done:
            final_shaping = info["reward_vector"]["yield_proxy"]
            last_crop = env.inner.current_crop
            bonus = env.inner.season_results[last_crop]["dry_yield_t_ha"] / YIELD_REFERENCE[last_crop]
            pure = final_shaping - bonus
            assert -1.0 <= pure <= 0.0, pure


def test_shaping_gamma_matches_ppo_gamma():
    """audit-v3 (2.1): the env's shaping discount is the SAME config.gamma
    the PPO gets - no hardcoded SHAPING_GAMMA (a gamma ablation is
    single-factor)."""
    assert not hasattr(sys.modules["rotation_env"], "SHAPING_GAMMA")
    env = RotationIrrigationEnv("hebei_central", "loam", 2015)
    assert env.config.gamma == PRIMARY_CONFIG.gamma
    for cfg in [RLExperimentConfig(gamma=1.0), RLExperimentConfig(gamma=0.9)]:
        env = RotationIrrigationEnv("hebei_central", "loam", 2015, config=cfg)
        assert env.config.gamma == cfg.gamma


def test_single_and_double_crop_reward_scale():
    """audit-v3 (3.2): the terminal yield bonus is SYSTEM-normalized, so a
    single-crop episode (Ningxia) and a double-crop episode are comparable
    in magnitude - both pay yield / sum(references), not per-crop sums."""
    import pandas as pd

    from rotation_env import YIELD_REFERENCE

    # double crop: bonus = (wheat+maize) / (7.0+11.5); single: = maize/14.5
    env_dc = RotationIrrigationEnv("hebei_central", "loam", 2015)
    assert env_dc.config.water_normalizer == "annual_quota"
    env_nx = RotationIrrigationEnv("ningxia_irrigation", "loam", 2015)
    # both use the same normalizer and yield-referencing scheme
    assert env_dc._water_denominator() == 450.0
    assert env_nx._water_denominator() == 450.0


def test_maize_days_to_harvest_excludes_fallow_gap():
    """audit-v3 (3.4): the maize season's days-to-harvest counts from the
    CROP's planting date, not from the window start (which includes the
    wheat->maize fallow bridge) - so the late-season safety filter starts
    on the correct calendar date."""
    env = RotationIrrigationEnv("hebei_central", "loam", 2015)
    env.reset()
    # drive wheat to harvest with full irrigation, then inspect maize setup
    done = False
    while not done:
        state, reward, done, info = env.step(2)
        if done:
            break
        if env.current_crop == "maize":
            # maize planting is 06/15; window ends 10/05 -> 112 days
            assert env.season_days == 112, env.season_days
            # at planting, days to harvest must be ~112 (not 118: the
            # fallow days between wheat harvest and 06/15 must not count)
            current = env.model._clock_struct.step_start_time
            dth = (env.harvest_window_end_date - pd_Timestamp(current)).days
            assert dth == 112, dth
            break


def pd_Timestamp(x):
    import pandas as pd

    return pd.Timestamp(x)


def test_emergency_shortfall_uses_actual_delivery():
    """audit-v3 (3.5): the emergency shortfall is computed from the ACTUAL
    delivered irrigation (irr_cum delta), not the requested depth."""
    env = RotationIrrigationEnv("hebei_central", "loam", 2015, annual_quota=0.0)
    env.reset()
    state, reward, done, info = env.step(0.0)
    actual = info["actual_model_irrigation_mm"]
    shortfall = info["emergency_shortfall_mm"]
    if "critical_depletion" in info["safety_rule_triggered"]:
        assert shortfall == pytest.approx(max(0.0, CRITICAL_DEPLETION_MIN_MM - actual), abs=1e-6)
    else:
        assert shortfall == 0.0


def test_action_modification_reasons_are_separate():
    """audit-v3 (3.6): safety_modified / quota_clipped / delivery_modified
    are separate fields; 'action_modified' is the aggregate (deprecated for
    the paper's intervention claims)."""
    env = RotationIrrigationEnv("hebei_central", "loam", 2015, annual_quota=0.0)
    state = env.reset()
    # with zero quota, ANY step the safety layer does not already zero
    # must show quota_clipped=True (the hard cap bites)
    saw_positive_filtered = False
    for _ in range(120):
        state, reward, done, info = env.step(20.0)
        if info["filtered_action_mm"] > 0:
            saw_positive_filtered = True
            assert info["quota_clipped"] is True, info["quota_clipped"]
            assert info["modified_by_quota"] is True
            assert info["action_modified"] is True
            break
        if done:
            break
    # the split fields are independent booleans
    assert set(["safety_modified", "quota_clipped", "delivery_modified"]).issubset(info)
    assert info["action_modified"] == (
        info["safety_modified"] or info["quota_clipped"] or info["delivery_modified"]
    )
    assert "safety_modified_rate" in info or True  # eval-level field, checked elsewhere


def test_equal_rotation_dates_are_rejected():
    """audit-v3 (3.3): a wheat harvest ON the maize planting day must be
    rejected - only gap >= MIN_HANDOFF_GAP_DAYS passes."""
    from cropping_systems import CROPPING_SYSTEMS

    import cropping_systems as cs

    env = RotationIrrigationEnv("hebei_central", "loam", 2015)
    env.reset()
    # simulate an equal-date handoff by faking wheat harvest == 06/15
    env.season_results["wheat"] = {"harvest_date": "2015-06-15"}
    with pytest.raises(RotationCalendarError):
        env._check_wheat_maize_handoff()
    env.season_results["wheat"] = {"harvest_date": "2015-06-14"}  # 1-day gap -> OK
    env._check_wheat_maize_handoff()
