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
from rotation_env import CRITICAL_DEPLETION_MIN_MM, RotationIrrigationEnv

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
