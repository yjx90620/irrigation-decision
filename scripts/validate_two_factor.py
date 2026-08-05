"""audit-v3 (6.2/6.3): two-factor risk validation report - correlation of
predicted risk with the ACTUAL transfer yield gap, vs distance alone
(paper-3 core claim), WITH uncertainty.

The n=5 leave-one-out correlation (0.97 two-factor vs 0.70 distance) is
directional evidence at best - the audit's point (6.3) is that a
correlation on 5 sites carries no meaningful precision, so this script:

1. Pearson point estimates for distance-only, sensitivity-only, and
   two-factor risk vs the observed yield gap (Spearman for two-factor).
2. Stratified bootstrap 95% CIs (strata = cropping system: 4 double-crop
   sites + 1 spring-maize site, resampled WITHIN stratum; the size-1
   Ningxia stratum is deterministic, which is itself worth reporting).
   An unstratified bootstrap is also included as a robustness check.
3. Leave-one-out degradation table (drop each site, recompute both
   correlations) so a paper can show the result does not hinge on any
   single site.

Everything is saved to data/processed/two_factor_validation_report.csv
(long format) + two_factor_leave_one_out.csv, with a fixed bootstrap seed
so the numbers are reproducible.

Usage: python scripts/validate_two_factor.py
(reads data/processed/leave_one_out_rotation_summary_wq.csv and
 two_factor_risk.csv - i.e. it must run after the transfer re-run)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "sim"))

import numpy as np
import pandas as pd

from cropping_systems import is_double_crop

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
SUMMARY_PATH = DATA_DIR / "leave_one_out_rotation_summary_wq.csv"
RISK_PATH = DATA_DIR / "two_factor_risk.csv"
OUT_REPORT = DATA_DIR / "two_factor_validation_report.csv"
OUT_LOO = DATA_DIR / "two_factor_leave_one_out.csv"

BOOTSTRAP_ITER = 5000
BOOTSTRAP_SEED = 20260804  # fixed so the CI is reproducible
CONF_LEVEL = 0.95


def load() -> pd.DataFrame:
    summary = pd.read_csv(SUMMARY_PATH)
    risk = pd.read_csv(RISK_PATH)
    risk_col = "site_id" if "site_id" in risk.columns else "target_site"
    m = summary.merge(risk, left_on="target_site", right_on=risk_col)
    m["gap"] = m["yield_gap_vs_rule"]
    m["stratum"] = np.where(m["target_site"].map(is_double_crop), "double_crop", "spring_maize")
    return m


def pearson(x, y) -> float:
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y) -> float:
    xr = pd.Series(x).rank().values
    yr = pd.Series(y).rank().values
    return pearson(xr, yr)


def _bootstrap_sample(rng, m, stratified):
    if stratified:
        strata = m["stratum"].values
        labels = list(dict.fromkeys(strata))
        chosen = np.concatenate(
            [rng.choice(np.flatnonzero(strata == lab), size=int((strata == lab).sum()), replace=True)
             for lab in labels]
        )
    else:
        chosen = rng.choice(np.arange(len(m)), size=len(m), replace=True)
    return m.iloc[chosen]


def bootstrap_ci(m: pd.DataFrame, metric, stratified=True, n_iter=BOOTSTRAP_ITER, seed=BOOTSTRAP_SEED):
    """Percentile CI for the correlation `metric` under site resampling.
    Stratified = resample with replacement WITHIN cropping-system stratum
    (the size-1 Ningxia stratum is always present, so the stratified CI
    only reflects sampling variation among the double-crop sites)."""
    rng = np.random.default_rng(seed)

    def _value(df):
        gap = df["gap"].values
        if metric == "two_factor":
            return pearson(gap, df["two_factor_risk"].values)
        if metric == "distance":
            return pearson(gap, df["nearest_source_distance"].values)
        if metric == "sensitivity":
            return pearson(gap, df["task_sensitivity"].values)
        if metric == "diff_two_factor_minus_distance":
            return pearson(gap, df["two_factor_risk"].values) - pearson(
                gap, df["nearest_source_distance"].values
            )
        raise ValueError(f"unknown metric {metric}")

    boot = np.array([_value(_bootstrap_sample(rng, m, stratified)) for _ in range(n_iter)])
    # degenerate resamples (e.g. all-identical gap draws) yield NaN - drop
    # them and report how many survived so the CI is reproducible.
    boot = boot[np.isfinite(boot)]
    lo = np.percentile(boot, 100 * (1 - CONF_LEVEL) / 2)
    hi = np.percentile(boot, 100 * (1 + CONF_LEVEL) / 2)
    return float(lo), float(hi), float(boot.std()), int(len(boot))


def leave_one_out(m: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for site in m["target_site"]:
        keep = m[m["target_site"] != site]
        rows.append({
            "dropped_site": site,
            "corr_two_factor": pearson(keep["gap"].values, keep["two_factor_risk"].values),
            "corr_distance": pearson(keep["gap"].values, keep["nearest_source_distance"].values),
        })
    # full-sample row for reference
    rows.append({
        "dropped_site": "none",
        "corr_two_factor": pearson(m["gap"].values, m["two_factor_risk"].values),
        "corr_distance": pearson(m["gap"].values, m["nearest_source_distance"].values),
    })
    return pd.DataFrame(rows)


def main():
    m = load()
    n = len(m)

    def _row(metric, point, method, stratified):
        lo, hi, sd, n_valid = bootstrap_ci(m, metric, stratified=stratified)
        return {
            "metric": metric, "point_estimate": point, "bootstrap_ci_low": lo,
            "bootstrap_ci_high": hi, "bootstrap_std": sd,
            "bootstrap_kind": "stratified" if stratified else "unstratified",
            "correlation_method": method, "n_sites": n, "n_bootstrap": n_valid,
            "bootstrap_seed": BOOTSTRAP_SEED,
        }

    gap = m["gap"].values
    rows = []
    for stratified in (True, False):
        rows.append(_row("two_factor", pearson(gap, m["two_factor_risk"].values), "pearson", stratified))
        rows.append(_row("distance", pearson(gap, m["nearest_source_distance"].values), "pearson", stratified))
        rows.append(_row("sensitivity", pearson(gap, m["task_sensitivity"].values), "pearson", stratified))
        rows.append(_row("two_factor", spearman(gap, m["two_factor_risk"].values), "spearman", stratified))
        rows.append(_row(
            "diff_two_factor_minus_distance",
            pearson(gap, m["two_factor_risk"].values) - pearson(gap, m["nearest_source_distance"].values),
            "pearson_diff", stratified,
        ))

    report = pd.DataFrame(rows)
    report.to_csv(OUT_REPORT, index=False)

    loo = leave_one_out(m)
    loo.to_csv(OUT_LOO, index=False)

    print(m[["target_site", "stratum", "nearest_source_distance", "task_sensitivity", "two_factor_risk", "gap"]]
          .round(3).to_string(index=False))
    print()
    print("=== point estimates (n=5, directional only) ===")
    for _, r in report.iterrows():
        if r["bootstrap_kind"] == "stratified" and r["correlation_method"] in ("pearson", "pearson_diff"):
            print(f"  {r['metric']:38s} {r['correlation_method']:12s} "
                  f"{r['point_estimate']:+.3f}  [95% CI {r['bootstrap_ci_low']:+.3f}, {r['bootstrap_ci_high']:+.3f}]")
    print(f"\nSpearman two_factor: {spearman(gap, m['two_factor_risk'].values):+.3f}")
    print(f"\nsaved -> {OUT_REPORT}")
    print(f"saved -> {OUT_LOO}")


if __name__ == "__main__":
    main()
