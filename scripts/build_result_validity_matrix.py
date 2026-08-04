"""audit-v3: build docs/RESULT_VALIDITY_MATRIX.csv - per-artifact validity
status, so no result is accidentally cited without knowing which generation
it belongs to. Round-3 state: everything whose code path changed in the
audit-v3 phases (RL env rewrite, temporal split, MWV v2, transfer arms,
crop-stage fingerprint, sensitivity uncertainty) is marked
regenerate_needed with its regeneration command.

After the full re-run, rebuild with `--mark-valid` to flip the rotation-era
regenerate_needed rows to valid_regenerated (the re-run log then documents
what was actually produced)."""
import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "figures"

AUDIT_V3 = "audit-v3"

# (glob or exact name, category, status, reason, regeneration_command)
def rules(mark_valid: bool) -> list:
    v = "valid_regenerated" if mark_valid else "regenerate_needed"
    rl_retrain = "retrain + evaluate (re-run phase)"
    return [
        # --- legacy single-season prototype (archived) ---
        ("baseline_experiment_results.csv", "legacy", "invalid_archived",
         "single-season prototype (see invalidated/legacy_v1/)", "n/a - do not regenerate"),
        ("baseline_strategy_summary.csv", "legacy", "invalid_archived",
         "single-season prototype (see invalidated/legacy_v1/)", "n/a - do not regenerate"),
        ("pareto_front_*.csv", "legacy", "invalid_archived",
         "single-season prototype NSGA-II fronts", "n/a - batch_optimize_loam.py requires --allow-legacy"),
        ("rl_vs_baselines_comparison.csv", "legacy", "invalid_archived",
         "single-season prototype RL comparison", "n/a - fig5 requires --allow-legacy"),
        ("leave_one_out_transfer.csv", "legacy", "invalid_archived",
         "single-season prototype transfer", "n/a - fig6 requires --allow-legacy"),
        ("instance_weighted_transfer.csv", "legacy", "invalid_archived",
         "single-season prototype transfer", "n/a"),
        ("ppo_irrigation*.zip", "legacy", "invalid_archived",
         "single-season prototype PPO models", "n/a - train_ppo.py is prototype-only"),
        ("ppo_learning_curve.csv", "legacy", "invalid_archived",
         "single-season prototype learning curve", "n/a - fig4 requires --allow-legacy"),
        ("transfer_source_excl_*", "legacy", "invalid_archived",
         "single-season prototype leave-one-out source models", "n/a"),
        ("transfer_finetuned_*", "legacy", "invalid_archived",
         "single-season prototype finetuned models", "n/a"),
        ("transfer_weighted_excl_*", "legacy", "invalid_archived",
         "single-season prototype instance-weighted models", "n/a"),
        # --- rotation-era, superseded by the wq primary arm ---
        ("leave_one_out_rotation_transfer.csv", "rotation_era", "invalid_superseded",
         "per_action water-normalizer arm; replaced by the per_quota (wq) primary arm (audit-v2 P0-9)",
         "n/a - wq arm is the paper-3 primary"),
        ("leave_one_out_rotation_summary.csv", "rotation_era", "invalid_superseded",
         "derived from the per_action transfer; replaced by summary_wq", "n/a - wq arm is the paper-3 primary"),
        ("marginal_water_value.csv", "rotation_era", "invalid_superseded",
         "SMT-threshold MWV (P1-2 design); replaced by fixed-dose marginal_response_v2.csv (audit-v3 5.3)",
         "n/a - use marginal_water_value_v2.py"),
        # --- rotation-era, audit-v3 code changes -> regenerate ---
        ("task_sensitivity.csv", "rotation_era", v,
         f"{AUDIT_V3} 6.6: per-year sensitivity spread columns added", "python src/transfer/task_sensitivity.py"),
        ("allocation_scan.csv", "rotation_era", v,
         f"{AUDIT_V3} 5.1: scans on the shared validation window (SPLIT)", "python src/sim/scan_allocation.py"),
        ("allocation_scan_*.csv", "rotation_era", v,
         f"{AUDIT_V3} 5.1: scans on the shared validation window (SPLIT)", "python src/sim/scan_allocation.py"),
        ("rotation_policy_comparison.csv", "rotation_era", v,
         f"{AUDIT_V3} 2.x: rotation_env rewrite (strict PBRS, system-level yield, calendar days_to_harvest)", rl_retrain),
        ("ppo_rotation_*_final.zip", "rotation_era", v,
         f"{AUDIT_V3} 2.x: env rewrite invalidates every trained policy", rl_retrain),
        ("ppo_rotation_*_final_vecnormalize.pkl", "rotation_era", v,
         "paired with the (to-be-retrained) models", rl_retrain),
        ("ppo_rotation_*_site_transitions.csv", "rotation_era", v,
         "training diagnostic of the (to-be-retrained) runs", rl_retrain),
        ("ppo_rotation_learning_curve.csv", "rotation_era", v,
         "reconstructed from the (to-be-retrained) checkpoints", rl_retrain),
        ("ppo_rotation_gamma1_comparison.csv", "rotation_era", v,
         f"{AUDIT_V3} 2.x: env rewrite invalidates the gamma ablation", rl_retrain),
        ("ppo_rotation_wq_comparison.csv", "rotation_era", v,
         f"{AUDIT_V3} 2.x: paper-2 primary arm, env rewrite invalidates it", rl_retrain),
        ("transfer_rot_source_excl_*", "rotation_era", v,
         f"{AUDIT_V3} 2.x: source policies retrained on the new env", rl_retrain),
        ("transfer_rot_finetuned_*", "rotation_era", v,
         f"{AUDIT_V3} 2.x: finetuned policies retrained on the new env", rl_retrain),
        ("cross_season_pareto_*.csv", "rotation_era", v,
         f"{AUDIT_V3} 5.1: optimization on the shared validation window (SPLIT)", "python src/sim/run_cross_season_all_sites.py"),
        ("marginal_response_v2.csv", "rotation_era", v,
         f"{AUDIT_V3} 5.3: fixed-dose stage-wise marginal response (new)", "python src/sim/marginal_water_value_v2.py"),
        ("algorithm_comparison_*.csv", "rotation_era", v,
         f"{AUDIT_V3} 5.1: CrossSeasonProblem now evaluates on the validation window",
         "python src/sim/optimize_algorithm_comparison.py"),
        ("leave_one_out_rotation_transfer_wq.csv", "rotation_era", v,
         f"{AUDIT_V3} 2.x env + 6.1: scratch/full_target_expert arms added", "python -m src.transfer.leave_one_out_rotation"),
        ("leave_one_out_rotation_summary_wq.csv", "rotation_era", v,
         "derived from the wq transfer", "python -m src.transfer.leave_one_out_rotation"),
        ("environmental_fingerprints.csv", "rotation_era", v,
         f"{AUDIT_V3} 6.5: crop-stage climate features added", "python src/transfer/fingerprint.py"),
        ("site_distance_matrix.csv", "rotation_era", v,
         "derived from the crop-stage fingerprints", "python src/transfer/similarity_analysis.py"),
        ("site_pca.csv", "rotation_era", v,
         "derived from the crop-stage fingerprints", "python src/transfer/similarity_analysis.py"),
        ("site_clusters.csv", "rotation_era", v,
         "derived from the crop-stage fingerprints", "python src/transfer/similarity_analysis.py"),
        ("two_factor_risk.csv", "rotation_era", v,
         "depends on the regenerated fingerprints + transfer", "python src/transfer/two_factor_predictor.py"),
        ("two_factor_validation_report.csv", "rotation_era", v,
         f"{AUDIT_V3} 6.2/6.3: bootstrap CI report, derived from transfer + risk",
         "python scripts/validate_two_factor.py"),
        ("two_factor_leave_one_out.csv", "rotation_era", v,
         f"{AUDIT_V3} 6.3: leave-one-out degradation table",
         "python scripts/validate_two_factor.py"),
        # --- still valid (code path untouched by audit-v3) ---
        ("agera5_vs_openmeteo_validation.csv", "other", "valid", "cross-product consistency check", "n/a"),
        ("ppo_smoke_test.zip", "other", "valid", "CI smoke test artifact, not a result", "n/a"),
        # --- raw CMIP6 data ---
        ("data/raw/cmip6/*.csv", "raw_data", "valid_regenerated",
         "10/10 files re-downloaded (2026-08-04); degraded manifests recorded per file "
         "(CMCC_CM2_VHR4 radiation is an upstream API gap)",
         "python src/data/download_cmip6.py"),
    ]


FIGURES_RULES = [
    ("fig3_environmental_fingerprint.png", "fig_environmental_fingerprint.py"),
    ("fig7_allocation_scan.png", "fig_allocation_scan.py"),
    ("fig8_algorithm_comparison.png", "fig_algorithm_comparison.py"),
    ("fig10_rotation_policy_comparison.png", "fig_rotation_policy_comparison.py"),
    ("fig_transfer_rotation.png", "fig_transfer_rotation.py"),
    ("fig_rl_training_curves_rotation.png", "fig_rotation_training_curves.py"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mark-valid", action="store_true",
                        help="flip rotation-era regenerate_needed rows to valid_regenerated "
                             "(use AFTER the full re-run)")
    args = parser.parse_args()
    status_label = "valid" if args.mark_valid else "regenerate_needed"

    rows = []
    seen = set()
    for pattern, category, status, reason, regen in rules(args.mark_valid):
        if pattern.startswith("data/raw/"):
            # raw-data patterns are directory globs ("data/raw/cmip6/*.csv"):
            # list the files inside (plus manifests, which record the
            # per-file degraded status).
            base = ROOT / pattern.split("*")[0]
            files = sorted(base.glob("*.csv")) + sorted(base.glob("*.csv.manifest.json"))
        else:
            base = PROCESSED / pattern.split("*")[0]
            files = sorted(base.parent.glob(base.name + "*"))
        for f in files:
            rel = str(f.relative_to(ROOT)).replace("\\", "/")
            if rel in seen or not f.is_file():
                continue
            seen.add(rel)
            rows.append({
                "artifact": rel, "category": category, "status": status,
                "reason": reason, "regeneration_command": regen,
            })

    # figures: every listed figure is regenerate_needed until the re-run
    # has produced its inputs (then --mark-valid flips existing ones).
    for name, script in FIGURES_RULES:
        rel = f"figures/{name}"
        rows.append({
            "artifact": rel, "category": "figure", "status": status_label,
            "reason": f"{AUDIT_V3}: inputs regenerated in the re-run",
            "regeneration_command": f"python src/viz/{script}",
        })

    out = ROOT / "docs" / "RESULT_VALIDITY_MATRIX.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} artifacts -> {out} (status label: {status_label})")


if __name__ == "__main__":
    main()
