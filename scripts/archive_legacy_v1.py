"""audit-v2 P0-12: archive the OLD single-season prototype artifacts
(never migrated in round 1 - they sat at data/processed/ and figures/
root alongside the rotation-era results, indistinguishable by location).

Moves each file with `git mv` (history preserved), records
original path / new path / sha256 / generating script / reason in
migration_manifest.csv, and drops a PROTOTYPE marker. These results are
prototype-only: the single-season env started each crop near field
capacity with no rotation/quota, which the system upgrade (docs/系统升级方案.md)
and the round-2 audit fixes (P0-1/P0-2) showed to be not valid for final
scientific claims.
"""
import csv
import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "figures"
DEST_DATA = PROCESSED / "invalidated" / "legacy_v1"
DEST_FIG = FIGURES / "invalidated" / "legacy_v1"

REASON = (
    "single-season prototype: each crop simulated in isolation starting near field capacity, "
    "no rotation chain / annual quota / fallow bridge; invalidated by the system upgrade "
    "(docs/系统升级方案.md) and audit-v2 fixes (P0-1/P0-2); prototype only, not for final claims"
)

GENERATORS = {
    "baseline_experiment_results.csv": "src/sim/experiment.py",
    "baseline_strategy_summary.csv": "src/sim/experiment.py",
    "pareto_front_": "src/sim/optimize_nsga2.py",
    "rl_vs_baselines_comparison.csv": "src/rl/train_ppo.py + src/rl/evaluate_policy.py",
    "leave_one_out_transfer.csv": "src/transfer/leave_one_out.py",
    "instance_weighted_transfer.csv": "src/transfer/instance_weighted.py",
    "ppo_irrigation_final.zip": "src/rl/train_ppo.py",
    "ppo_irrigation_v2_final.zip": "src/rl/train_ppo.py",
    "ppo_irrigation_v2_final_vecnormalize.pkl": "src/rl/train_ppo.py",
    "ppo_learning_curve.csv": "src/rl/reconstruct_learning_curve.py",
    "ppo_irrigation": "src/rl/train_ppo.py",  # checkpoints + vecnormalize
    "transfer_source_excl_": "src/transfer/leave_one_out.py",
    "transfer_finetuned_": "src/transfer/leave_one_out.py",
    "transfer_weighted_excl_": "src/transfer/instance_weighted.py",
    "fig1_": "src/viz/fig_baseline_overview.py",
    "fig2_": "src/viz/fig_pareto_fronts.py",
    "fig3_": "src/viz/fig_environmental_fingerprint.py",
    "fig4_": "src/viz/fig_rl_training_curve.py",
    "fig5_": "src/viz/fig_rl_vs_baselines.py",
    "fig6_": "src/viz/fig_transfer_leave_one_out.py",
}


def generator_for(name: str) -> str:
    for prefix, gen in GENERATORS.items():
        if name.startswith(prefix):
            return gen
    return "unknown"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def candidates() -> list:
    """(source, dest_dir) pairs for every single-season artifact still at
    the processed/figures root."""
    items = []
    for name in [
        "baseline_experiment_results.csv", "baseline_strategy_summary.csv",
        "rl_vs_baselines_comparison.csv", "leave_one_out_transfer.csv",
        "instance_weighted_transfer.csv",
        "ppo_irrigation_final.zip", "ppo_irrigation_v2_final.zip",
        "ppo_irrigation_v2_final_vecnormalize.pkl", "ppo_learning_curve.csv",
    ]:
        p = PROCESSED / name
        if p.exists():
            items.append((p, DEST_DATA))
    for p in PROCESSED.glob("pareto_front_*.csv"):
        items.append((p, DEST_DATA))
    for p in PROCESSED.glob("transfer_source_excl_*"):
        items.append((p, DEST_DATA))
    for p in PROCESSED.glob("transfer_finetuned_*"):
        items.append((p, DEST_DATA))
    for p in PROCESSED.glob("transfer_weighted_excl_*"):
        items.append((p, DEST_DATA))
    ckpt_dir = PROCESSED / "ppo_checkpoints"
    for p in list(ckpt_dir.glob("ppo_irrigation*.zip")) + list(ckpt_dir.glob("ppo_irrigation_v2*.pkl")):
        items.append((p, DEST_DATA / "ppo_checkpoints"))
    for p in FIGURES.glob("fig[1-6]_*.png"):
        items.append((p, DEST_FIG))
    return items


def main():
    DEST_DATA.mkdir(parents=True, exist_ok=True)
    DEST_FIG.mkdir(parents=True, exist_ok=True)
    (DEST_DATA / "ppo_checkpoints").mkdir(parents=True, exist_ok=True)

    rows = []
    for src, dest_dir in candidates():
        rel = src.relative_to(ROOT)
        dest = dest_dir / src.name
        h = sha256(src)
        subprocess.run(["git", "mv", str(src), str(dest)], check=True, cwd=ROOT)
        rows.append({
            "original_path": str(rel).replace("\\", "/"),
            "new_path": str(dest.relative_to(ROOT)).replace("\\", "/"),
            "sha256": h,
            "generating_script": generator_for(src.name),
            "reason": REASON,
            "usable_for_paper": "no",
        })
        print(f"moved {src.name} -> {dest.relative_to(ROOT)}")

    manifest = DEST_DATA / "migration_manifest.csv"
    with open(manifest, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else [
            "original_path", "new_path", "sha256", "generating_script", "reason", "usable_for_paper"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n{len(rows)} files archived; manifest -> {manifest}")


if __name__ == "__main__":
    main()
