"""audit-v3 re-run prep: remove stale pre-audit-v3 result artifacts whose
resume checks (content-level 'done' detection) would otherwise wrongly
skip regeneration or whose rows would contaminate fresh output:

- old-env PPO checkpoints (144 files) - the learning-curve reconstructor
  scans the whole dir, old-env checkpoints would mix with new ones
- alpha scan / cross-season / algorithm-comparison CSVs - resume logic
  sees the old columns and 'done'
- wq transfer results - 3-condition rows from the old env would be
  appended to the new 5-condition rows
- old wq-named models/eval files (the new primary arm drops the '_wq'
  suffix, so these would be orphaned stale artifacts)
- old-env transfer models

All are git-tracked, so nothing is lost irrecoverably.
"""
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

GROUPS = [
    "ppo_checkpoints/*",
    "allocation_scan*.csv",
    "cross_season_pareto*.csv",
    "algorithm_comparison_hebei_central*.csv",
    "leave_one_out_rotation_transfer_wq.csv",
    "leave_one_out_rotation_summary_wq.csv",
    "ppo_rotation_learning_curve.csv",
    "ppo_rotation_gamma1_comparison.csv",
    "ppo_rotation_wq_comparison.csv",
    "rotation_policy_comparison.csv",
    "ppo_rotation_*_wq*",
    "transfer_rot_*",
]


def main():
    n = 0
    for group in GROUPS:
        for p in PROCESSED.glob(group):
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
            else:
                p.unlink()
            n += 1
            print(f"removed {p.relative_to(ROOT)}")
    print(f"removed {n} stale artifacts")


if __name__ == "__main__":
    main()
