"""P0-1/P0-2 acceptance tests (docs/审计修复计划.md): the RL environment
must not trip the calendar date-order guard across historical years (its
own day-by-day irrigation_method=5 stepping is a different code path from
rotation.py's SMT-threshold seasons the cultivar calibration was tuned
against), and the annual water quota must be a genuine hard cap even
when the critical-depletion floor in safety_filter() wants more water
than remains."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim"))

import pytest

from cropping_systems import CROPPING_SYSTEMS, is_double_crop
from rotation_env import ACTIONS_MM, CRITICAL_DEPLETION_MIN_MM, RotationIrrigationEnv

# worst-case (latest-maturing) years found by calibrate_wheat_maturity.py's
# full-irrigation search, plus a couple of arbitrary others - not the full
# 1982-2025 history, to keep this fast; test_rotation_calendar.py already
# covers every year via rotation.py's code path.
SAMPLE_YEARS = {
    "hebei_central": [1987, 2010, 2013, 2015, 2020],
    "shaanxi_guanzhong": [1982, 1986, 1988, 2011, 2015],
    "beijing_plain": [1987, 2010, 2013, 2015, 2020],
    "henan_north": [1982, 2013, 2020],
}


@pytest.mark.parametrize("site_id", [s for s in CROPPING_SYSTEMS if is_double_crop(s)])
def test_rl_env_wheat_maize_handoff_does_not_violate_calendar(site_id):
    for year in SAMPLE_YEARS[site_id]:
        # annual_quota set far above what a full season could possibly use,
        # so quota exhaustion never forces zero-irrigation mid-season - this
        # needs to be genuinely water-abundant throughout, matching
        # calibrate_wheat_maturity.py's own unconstrained full-irrigation
        # test, not just "high dose for a while, then quota runs out"
        # (stress accelerates maturity, so an unintended cutoff would bias
        # this test toward *earlier* harvests and could hide a real gap).
        env = RotationIrrigationEnv(site_id, "loam", year, annual_quota=5000.0)
        state = env.reset()
        done = False
        while not done:
            state, reward, done, info = env.step(40.0)


def test_quota_is_never_exceeded_under_persistent_zero_irrigation():
    """A policy that always requests 0mm will still let the crop hit
    critical depletion (safety_filter's floor tries to override with
    CRITICAL_DEPLETION_MIN_MM), but a tiny annual_quota should force
    actual irrigation to stop at the cap regardless - this is the
    scenario the pre-fix code silently violated."""
    env = RotationIrrigationEnv("hebei_central", "loam", 2015, annual_quota=5.0)
    state = env.reset()
    done = False
    while not done:
        state, reward, done, info = env.step(0.0)
        assert info["quota_used_mm"] <= env.annual_quota + 1e-6
        assert info["quota_violation_mm"] == 0.0
    assert env.quota_used <= env.annual_quota + 1e-6


def test_critical_depletion_shortfall_is_recorded_not_hidden():
    """With a quota already exhausted, a critical-depletion step should
    show up as an emergency_shortfall rather than irrigation exceeding
    the cap."""
    env = RotationIrrigationEnv("hebei_central", "loam", 2015, annual_quota=0.0)
    state = env.reset()
    state, reward, done, info = env.step(0.0)
    assert info["actual_model_irrigation_mm"] == 0.0
    assert info["quota_used_mm"] == 0.0
    if info["safety_rule_triggered"] and "critical_depletion" in info["safety_rule_triggered"]:
        assert info["emergency_shortfall_mm"] >= CRITICAL_DEPLETION_MIN_MM - 1e-6


# --- audit-v2 (P0-3): quota/rewards track ACTUAL applied water ------------

def test_quota_accounting_uses_actual_model_delta():
    """P0-3 (audit-v2): quota_used must track AquaCrop's ACTUAL applied
    irrigation (irr_cum delta), not the requested depth - requesting 40mm
    was measured to deliver a constant 25mm, so requested-depth accounting
    over-counted water use by ~60% and the quota narrative described water
    that was never applied."""
    env = RotationIrrigationEnv("hebei_central", "loam", 2015, annual_quota=5000.0)
    state = env.reset()
    n_irrigated = 0
    while not env.done:
        irr_before = float(env.model._init_cond.irr_cum)
        state, reward, done, info = env.step(40.0)
        irr_delta = float(env.model._init_cond.irr_cum) - irr_before
        assert info["actual_model_irrigation_mm"] == pytest.approx(max(0.0, irr_delta), abs=1e-6)
        if info["actual_model_irrigation_mm"] > 0:
            n_irrigated += 1
        if n_irrigated > 50:
            break
    assert n_irrigated > 0


def test_water_reward_uses_actual_not_requested():
    """P0-3 (audit-v2): the water and cost reward terms are proportional to
    the ACTUAL applied amount, not the requested depth."""
    env = RotationIrrigationEnv("hebei_central", "loam", 2015, annual_quota=5000.0)
    state = env.reset()
    state, reward, done, info = env.step(40.0)
    actual = info["actual_model_irrigation_mm"]
    if actual > 0:
        assert reward["water"] == pytest.approx(-actual / max(ACTIONS_MM))
        assert reward["cost"] < 0.0
    # requested 40mm but actual is capped by what the soil can hold: on
    # loam this is 25mm (measured) - quota must have counted 25, not 40
    assert env.quota_used == pytest.approx(actual, abs=1e-6)


def test_episode_total_irrigation_matches_quota_used():
    """P0-3 (audit-v2): at episode end the reported system irrigation
    (sum of AquaCrop's seasonal irrigation) must be consistent with the
    step-level quota accounting - both count the same actual water."""
    env = RotationIrrigationEnv("hebei_central", "loam", 2015, annual_quota=5000.0)
    state = env.reset()
    while not env.done:
        state, reward, done, info = env.step(40.0)
    assert abs(info["total_irrigation_mm"] - env.quota_used) < 1.0, (
        f"reported total {info['total_irrigation_mm']:.1f} mm vs quota_used {env.quota_used:.1f} mm"
    )
