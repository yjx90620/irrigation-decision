"""audit-v2: build docs/RESULT_VALIDITY_MATRIX.csv - per-artifact validity
status after the round-2 fixes, so no result is accidentally cited without
knowing which generation it belongs to."""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "figures"

# (glob or exact name, category, status, reason, regeneration_command)
RULES = [
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
    # --- rotation-era results, regenerated with audit-v2 fixes ---
    ("task_sensitivity.csv", "rotation_era", "valid_regenerated",
     "regenerated with P0-1/P0-2 physics (audit-v2)", "n/a"),
    ("allocation_scan.csv", "rotation_era", "valid_regenerated",
     "regenerated with P0-1/P0-2 physics (audit-v2)", "n/a"),
    ("allocation_scan_*.csv", "rotation_era", "valid_regenerated",
     "regenerated with P0-1/P0-2 physics (audit-v2)", "n/a"),
    ("rotation_policy_comparison.csv", "rotation_era", "valid_regenerated",
     "regenerated with P0-1/P0-2/P0-3 fixes + P1-4 preference axis (audit-v2)", "n/a"),
    ("ppo_rotation_*_final.zip", "rotation_era", "valid_regenerated",
     "retrained with P0-1/P0-2/P0-3 fixes, 3 seeds (audit-v2)", "n/a"),
    ("ppo_rotation_*_final_vecnormalize.pkl", "rotation_era", "valid_regenerated",
     "paired with regenerated models", "n/a"),
    ("ppo_rotation_*_site_transitions.csv", "rotation_era", "valid_regenerated",
     "training diagnostic of the regenerated runs", "n/a"),
    ("ppo_rotation_learning_curve.csv", "rotation_era", "valid_regenerated",
     "reconstructed from the regenerated ~100k-step checkpoints", "n/a"),
    # --- still pending regeneration (audit-v2) ---
    ("cross_season_pareto_*.csv", "rotation_era", "regenerate_needed",
     "P0-1 fallow bridge + P0-2 maize cultivar changed rotation physics", "python src/sim/run_cross_season_all_sites.py"),
    ("marginal_water_value.csv", "rotation_era", "regenerate_needed",
     "P0-1/P0-2 changed rotation physics; P1-2 adds baseline/perturbed pairing columns",
     "python src/sim/marginal_water_value.py"),
    ("algorithm_comparison_*.csv", "rotation_era", "regenerate_needed",
     "P0-1/P0-2 changed rotation physics", "python src/sim/optimize_algorithm_comparison.py"),
    ("leave_one_out_rotation_transfer.csv", "rotation_era", "regenerate_needed",
     "transfer in progress (beijing fold done)", "python src/transfer/leave_one_out_rotation.py"),
    ("leave_one_out_rotation_summary.csv", "rotation_era", "regenerate_needed",
     "derived from leave_one_out_rotation_transfer.csv", "python src/transfer/leave_one_out_rotation.py"),
    ("two_factor_risk.csv", "rotation_era", "regenerate_needed",
     "built on pre-fix transfer + fingerprint inputs", "python src/transfer/two_factor_predictor.py"),
    ("transfer_rot_*", "rotation_era", "regenerate_needed",
     "rotation env changed (P0-1/P0-2/P0-3)", "python src/transfer/leave_one_out_rotation.py"),
    # --- still valid (code untouched in audit-v2) ---
    ("environmental_fingerprints.csv", "rotation_era", "valid",
     "fingerprint code unchanged in audit-v2 (cool/warm seasons, TRAIN_YEARS)", "n/a"),
    ("site_distance_matrix.csv", "rotation_era", "valid",
     "derived from environmental_fingerprints.csv", "n/a"),
    ("site_pca.csv", "rotation_era", "valid", "derived from fingerprints", "n/a"),
    ("site_clusters.csv", "rotation_era", "valid", "derived from fingerprints", "n/a"),
    ("agera5_vs_openmeteo_validation.csv", "other", "valid", "cross-product consistency check", "n/a"),
    ("ppo_smoke_test.zip", "other", "valid", "CI smoke test artifact, not a result", "n/a"),
    # --- raw CMIP6 data ---
    ("data/raw/cmip6/*.csv", "raw_data", "valid_regenerated",
     "8/10 re-downloaded with degraded manifests (CMCC_CM2_VHR4 radiation is an upstream API gap, "
     "recorded per file); ningxia 2 files pending daily-quota reset (2026-08-04)",
     "python src/data/download_cmip6.py"),
]


def main():
    rows = []
    seen = set()
    for pattern, category, status, reason, regen in RULES:
        if pattern.startswith("data/raw/"):
            base = ROOT / pattern.split("*")[0]
        else:
            base = PROCESSED / pattern.split("*")[0]
        files = sorted(base.parent.glob(base.name + "*"))
        for f in files:
            rel = str(f.relative_to(ROOT)).replace("\\", "/")
            if rel in seen:
                continue
            seen.add(rel)
            rows.append({
                "artifact": rel, "category": category, "status": status,
                "reason": reason, "regeneration_command": regen,
            })
    # figures
    figure_paths = list(FIGURES.glob("fig[7-9]*.png"))
    figure_paths.append(FIGURES / "fig10_rotation_policy_comparison.png")
    for p in figure_paths:
        if p.exists():
            rel = str(p.relative_to(ROOT)).replace("\\", "/")
            rows.append({"artifact": rel, "category": "figure", "status": "regenerate_needed",
                         "reason": "built from pre-fix rotation results",
                         "regeneration_command": "rerun after results regenerated"})
    out = ROOT / "docs" / "RESULT_VALIDITY_MATRIX.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} artifacts -> {out}")


if __name__ == "__main__":
    main()
