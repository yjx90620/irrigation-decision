"""audit-v3 (7.1): single source of truth for the headline numbers the
three papers quote - data/processed/paper_claims.csv.

Every claim the papers state (yield gaps, water savings, correlations,
stage values, ...) is computed HERE from the data/processed CSVs, never
typed by hand into a paper. The paper rewrite phase reads this file; a
`--check` run recomputes every value from the current CSVs and prints
any drift, so after a re-run a stale claim is caught instead of silently
surviving in prose.

Usage:
    python scripts/collect_paper_claims.py            # (re)generate paper_claims.csv
    python scripts/collect_paper_claims.py --check    # verify values still match data

Claims whose source file is missing (e.g. marginal_response_v2.csv before
the re-run) are recorded with value=MISSING so the papers cannot quote
them yet.
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "sim"))

import pandas as pd

from cropping_systems import is_double_crop

DATA = Path(__file__).resolve().parents[1] / "data" / "processed"
OUT_PATH = DATA / "paper_claims.csv"

DOUBLE_CROP_SITES = [s for s in ("beijing_plain", "hebei_central", "henan_north", "shaanxi_guanzhong")
                     if is_double_crop(s)]


def _mean_by(df, policy, preference, metric):
    g = df[(df["policy"] == policy) & (df["preference"] == preference)].groupby("site_id")[metric].mean()
    return g


def claim_rl_water_saving(policy="ppo_direct_wq"):
    df = pd.read_csv(DATA / "ppo_rotation_wq_comparison.csv")
    rl = _mean_by(df, policy, "balanced", "total_irrigation_mm")
    rule = _mean_by(df, "quota_reserving_rule", "balanced", "total_irrigation_mm")
    pct = (rule.loc[DOUBLE_CROP_SITES] - rl.loc[DOUBLE_CROP_SITES]) / rule.loc[DOUBLE_CROP_SITES] * 100
    return f"{pct.min():.1f}% - {pct.max():.1f}%", "per-site pct across 4 double-crop sites"


def claim_rl_yield_gap(policy="ppo_direct_wq"):
    df = pd.read_csv(DATA / "ppo_rotation_wq_comparison.csv")
    rl = _mean_by(df, policy, "balanced", "total_yield_t_ha")
    rule = _mean_by(df, "quota_reserving_rule", "balanced", "total_yield_t_ha")
    gap = rule.loc[DOUBLE_CROP_SITES] - rl.loc[DOUBLE_CROP_SITES]
    return f"{gap.min():.2f} - {gap.max():.2f} t/ha", "rule minus RL, 4 double-crop sites"


def claim_rl_water_use(policy="ppo_direct_wq"):
    df = pd.read_csv(DATA / "ppo_rotation_wq_comparison.csv")
    rl = _mean_by(df, policy, "balanced", "total_irrigation_mm")
    return f"{rl.loc[DOUBLE_CROP_SITES].min():.0f} - {rl.loc[DOUBLE_CROP_SITES].max():.0f} mm", "4 double-crop sites"


def claim_preference_water_range(policy="ppo_direct_wq"):
    df = pd.read_csv(DATA / "ppo_rotation_wq_comparison.csv")
    g = df[df["policy"] == policy].groupby(["site_id", "preference"])["total_irrigation_mm"].mean()
    return f"{g.min():.0f} - {g.max():.0f} mm", "across all 7 preferences x 5 sites"


def _corr_rows():
    rep = pd.read_csv(DATA / "two_factor_validation_report.csv")
    return rep


def claim_corr(metric, method="pearson", kind="stratified"):
    rep = _corr_rows()
    row = rep[(rep["metric"] == metric) & (rep["correlation_method"] == method)
              & (rep["bootstrap_kind"] == kind)].iloc[0]
    return f"{row['point_estimate']:.3f}", f"95% CI [{row['bootstrap_ci_low']:.3f}, {row['bootstrap_ci_high']:.3f}]"


def claim_corr_diff():
    # the diff row carries correlation_method='pearson_diff' in the report
    return claim_corr("diff_two_factor_minus_distance", method="pearson_diff")


def claim_loo_ningxia_drop():
    loo = pd.read_csv(DATA / "two_factor_leave_one_out.csv")
    row = loo[loo["dropped_site"] == "ningxia_irrigation"].iloc[0]
    return f"{row['corr_two_factor']:.3f}", "two-factor corr with Ningxia dropped"


def claim_task_sensitivity_range():
    t = pd.read_csv(DATA / "task_sensitivity.csv")
    spread = t["sensitivity_across_years_std"].max() if "sensitivity_across_years_std" in t else float("nan")
    note = f"spread: std up to {spread:.2f}" if spread == spread else "spread unavailable (pre-audit-v3-6.6 data)"
    return f"{t['sensitivity'].min():.2f} - {t['sensitivity'].max():.2f}", note


def claim_transfer_gap_ningxia():
    s = pd.read_csv(DATA / "leave_one_out_rotation_summary_wq.csv")
    gap = s[s["target_site"] == "ningxia_irrigation"]["yield_gap_vs_rule"]
    return f"{gap.mean():.2f} t/ha", "mean over evaluation years"


def claim_alpha_optima():
    # audit-v3 (7.1): single source = best_fixed_alpha.csv, derived by
    # scan_allocation.py (alpha maximizing mean yield averaged over the
    # SMT dimension).
    b = pd.read_csv(DATA / "best_fixed_alpha.csv").set_index("site_id")["best_fixed_alpha"]
    return b.round(2).to_dict(), "best fixed alpha per site (scan-derived, SMT-averaged)"


def claim_algo_hv():
    hv = pd.read_csv(DATA / "algorithm_comparison_hebei_central_hv_summary.csv")
    mean = hv.groupby("algorithm")["hv_mean"].mean().round(4)
    return mean.to_dict(), "mean hypervolume across seeds, hebei_central"


def claim_mwv_top_stage():
    path = DATA / "marginal_response_v2.csv"
    if not path.exists():
        return "MISSING", "marginal_response_v2.csv not generated yet (re-run phase)"
    df = pd.read_csv(path)
    ok = df[df["status"] == "success"]
    best = ok.loc[ok["marginal_response"].idxmax()]
    undefined = (df["status"] == "undefined_no_actual_delta").sum()
    return (
        f"{best['crop']} stage{best['stage']} {best['marginal_response']:.3f} t/ha/mm",
        f"undefined_no_actual_delta count: {undefined}",
    )


CLAIMS = [
    ("paper2_water_saving_pct_direct", "论文二", "对比三", "%",
     "data/processed/ppo_rotation_wq_comparison.csv", "retrain + evaluate (re-run phase)", claim_rl_water_saving),
    ("paper2_yield_gap_direct", "论文二", "对比三", "t/ha",
     "data/processed/ppo_rotation_wq_comparison.csv", "retrain + evaluate (re-run phase)", claim_rl_yield_gap),
    ("paper2_water_use_direct", "论文二", "对比三", "mm",
     "data/processed/ppo_rotation_wq_comparison.csv", "retrain + evaluate (re-run phase)", claim_rl_water_use),
    ("paper2_preference_water_range", "论文二", "对比三", "mm",
     "data/processed/ppo_rotation_wq_comparison.csv", "retrain + evaluate (re-run phase)", claim_preference_water_range),
    ("paper3_corr_two_factor", "论文三", "对比四", "r",
     "data/processed/two_factor_validation_report.csv", "python scripts/validate_two_factor.py",
     lambda: claim_corr("two_factor")),
    ("paper3_corr_distance", "论文三", "对比四", "r",
     "data/processed/two_factor_validation_report.csv", "python scripts/validate_two_factor.py",
     lambda: claim_corr("distance")),
    ("paper3_corr_diff_ci", "论文三", "对比四", "r-diff",
     "data/processed/two_factor_validation_report.csv", "python scripts/validate_two_factor.py",
     claim_corr_diff),
    ("paper3_loo_ningxia_drop", "论文三", "对比四", "r",
     "data/processed/two_factor_leave_one_out.csv", "python scripts/validate_two_factor.py",
     claim_loo_ningxia_drop),
    ("paper3_task_sensitivity_range", "论文三", "对比四", "0-1",
     "data/processed/task_sensitivity.csv", "python src/transfer/task_sensitivity.py",
     claim_task_sensitivity_range),
    ("paper3_transfer_gap_ningxia", "论文三", "对比四", "t/ha",
     "data/processed/leave_one_out_rotation_summary_wq.csv",
     "python -m src.transfer.leave_one_out_rotation", claim_transfer_gap_ningxia),
    ("paper1_alpha_optima", "论文一", "对比1b", "alpha",
     "data/processed/allocation_scan.csv", "python src/sim/scan_allocation.py", claim_alpha_optima),
    ("paper1_algo_hv", "论文一", "对比3", "hypervolume",
     "data/processed/algorithm_comparison_hebei_central_hv_summary.csv",
     "python src/sim/optimize_algorithm_comparison.py", claim_algo_hv),
    ("paper1_mwv_top_stage", "论文一", "对比4", "t/ha/mm",
     "data/processed/marginal_response_v2.csv", "python src/sim/marginal_water_value_v2.py",
     claim_mwv_top_stage),
]


def collect() -> pd.DataFrame:
    rows = []
    for key, paper, section, unit, source, regen, fn in CLAIMS:
        try:
            value, note = fn()
        except FileNotFoundError as e:
            value, note = "MISSING", f"source not generated yet: {e.filename}"
        except Exception as e:  # noqa: BLE001 - report drift, don't die
            value, note = "ERROR", str(e)
        rows.append({
            "claim_key": key, "paper": paper, "section": section,
            "value": value, "unit": unit, "value_note": note,
            "source_file": source, "regenerate_command": regen,
            "collected_at_utc": datetime.now(timezone.utc).isoformat(timespec="minutes"),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true",
                        help="recompute values from data files and report drift vs the saved registry")
    args = parser.parse_args()

    fresh = collect()
    if args.check:
        if not OUT_PATH.exists():
            print(f"no registry at {OUT_PATH}; run without --check to create it")
            return 1
        old = pd.read_csv(OUT_PATH)
        merged = old.merge(fresh, on="claim_key", suffixes=("_registry", "_data"))
        n_drift = 0
        for _, r in merged.iterrows():
            if str(r["value_registry"]) != str(r["value_data"]):
                n_drift += 1
                print(f"DRIFT {r['claim_key']}: registry={r['value_registry']} data={r['value_data']}")
        print(f"checked {len(merged)} claims, {n_drift} drifted")
        return 1 if n_drift else 0

    fresh.to_csv(OUT_PATH, index=False)
    print(f"saved -> {OUT_PATH} ({len(fresh)} claims)")
    for _, r in fresh.iterrows():
        print(f"  {r['claim_key']}: {r['value']} {r['unit']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
