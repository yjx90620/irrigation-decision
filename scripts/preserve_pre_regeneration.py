"""audit-v2 regeneration prep: move the PRE-regeneration rotation-era
results (all marked regenerate_needed in docs/RESULT_VALIDITY_MATRIX.csv)
into data/processed/invalidated/audit_v2_pre/ with git mv, so the fresh
runs never silently overwrite the pre-fix record."""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
DEST = PROCESSED / "invalidated" / "audit_v2_pre"

PATTERNS = [
    "cross_season_pareto_*.csv",
    "allocation_scan*.csv",
    "marginal_water_value.csv",
    "algorithm_comparison_*.csv",
    "task_sensitivity.csv",
    "rotation_policy_comparison.csv",
    "leave_one_out_rotation_transfer.csv",
    "leave_one_out_rotation_summary.csv",
    "two_factor_risk.csv",
    "ppo_rotation_*_final.zip",
    "ppo_rotation_*_final_vecnormalize.pkl",
    "ppo_rotation_*_site_transitions.csv",
    "transfer_rot_*",
]


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / "ppo_checkpoints").mkdir(parents=True, exist_ok=True)
    moved = []
    for pattern in PATTERNS:
        for src in sorted(PROCESSED.glob(pattern)):
            dest = DEST / src.name
            subprocess.run(["git", "mv", str(src), str(dest)], check=True, cwd=ROOT)
            moved.append((src.name, dest))
            print(f"moved {src.name} -> invalidated/audit_v2_pre/")
    ckpt_dest = DEST / "ppo_checkpoints"
    for src in sorted((PROCESSED / "ppo_checkpoints").glob("ppo_rotation_*")):
        subprocess.run(["git", "mv", str(src), str(ckpt_dest / src.name)], check=True, cwd=ROOT)
        moved.append((src.name, ckpt_dest))
        print(f"moved ppo_checkpoints/{src.name} -> invalidated/audit_v2_pre/ppo_checkpoints/")
    # figures fig7-10
    fig_dest = ROOT / "figures" / "invalidated" / "audit_v2_pre"
    fig_dest.mkdir(parents=True, exist_ok=True)
    for src in sorted((ROOT / "figures").glob("fig[7-9]*.png")) + [
        ROOT / "figures" / "fig10_rotation_policy_comparison.png"
    ]:
        if src.exists():
            subprocess.run(["git", "mv", str(src), str(fig_dest / src.name)], check=True, cwd=ROOT)
            moved.append((src.name, fig_dest))
            print(f"moved figures/{src.name} -> figures/invalidated/audit_v2_pre/")
    print(f"\n{len(moved)} artifacts preserved under invalidated/audit_v2_pre/")


if __name__ == "__main__":
    main()
