"""audit-v3 (6.2/6.3): two-factor validation report machinery - stratified
bootstrap CI and leave-one-out degradation, tested on synthetic data so
the test does not depend on (re)generated transfer results. The real
report is produced by scripts/validate_two_factor.py against
leave_one_out_rotation_summary_wq.csv + two_factor_risk.csv."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "validate_two_factor.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("validate_two_factor", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _synthetic():
    """5 sites: one strong linear driver plus a fixed outlier site in its
    own stratum (mirrors the real double-crop / Ningxia setup)."""
    sites = ["a", "b", "c", "d", "ningxia"]
    gap = np.array([1.0, 2.0, 3.0, 4.0, 10.0])
    risk = np.array([1.1, 2.1, 2.9, 4.2, 10.5])
    dist = np.array([3.0, 1.0, 4.0, 2.0, 9.0])
    df = pd.DataFrame({
        "target_site": sites, "gap": gap, "two_factor_risk": risk,
        "nearest_source_distance": dist, "task_sensitivity": risk,
        "stratum": ["double_crop"] * 4 + ["spring_maize"],
    })
    return df


def test_bootstrap_ci_shape_and_reproducibility():
    mod = _load_module()
    m = _synthetic()
    lo, hi, sd, n = mod.bootstrap_ci(m, "two_factor", seed=123)
    assert lo <= hi
    assert n == 5000  # strong linear signal -> no degenerate resamples
    lo2, hi2, sd2, n2 = mod.bootstrap_ci(m, "two_factor", seed=123)
    assert (lo, hi) == (lo2, hi2)


def test_bootstrap_ci_contains_point_estimate():
    mod = _load_module()
    m = _synthetic()
    point = mod.pearson(m["gap"].values, m["two_factor_risk"].values)
    lo, hi, _, _ = mod.bootstrap_ci(m, "two_factor", seed=123)
    assert lo <= point <= hi


def test_stratified_bootstrap_keeps_outlier_site():
    # with a size-1 stratum the outlier is ALWAYS present, so the
    # correlation can never drop to zero even when resamples are noisy
    mod = _load_module()
    m = _synthetic()
    rng = np.random.default_rng(7)
    samples = [mod._bootstrap_sample(rng, m, stratified=True) for _ in range(50)]
    assert all(set(s["target_site"]).issuperset({"ningxia"}) for s in samples)
    unstrat = [mod._bootstrap_sample(rng, m, stratified=False) for _ in range(50)]
    assert any("ningxia" not in set(s["target_site"]) for s in unstrat)


def test_leave_one_out_contains_full_sample_row():
    mod = _load_module()
    m = _synthetic()
    loo = mod.leave_one_out(m)
    assert set(loo["dropped_site"]) == {"a", "b", "c", "d", "ningxia", "none"}
    full = loo[loo["dropped_site"] == "none"].iloc[0]
    assert np.isclose(full["corr_two_factor"],
                      mod.pearson(m["gap"].values, m["two_factor_risk"].values))
