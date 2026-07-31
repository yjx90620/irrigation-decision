"""Reconstruct a PPO learning curve from saved checkpoints (研究方案 5.9 图表
准备). train_ppo.py doesn't log a training-reward CSV directly (SB3's
built-in logger only prints to console), so this evaluates every
checkpoint in data/processed/ppo_checkpoints/ against a fixed validation
scenario instead - same idea, and it works after the fact on whatever
checkpoints exist, without needing to have planned ahead or rerun
training. Safe to re-run any time (skips checkpoints already in the
output CSV).
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from evaluate_policy import load_ppo_policy, run_episode

CHECKPOINT_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "ppo_checkpoints"
OUT_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "ppo_learning_curve.csv"

# Fixed scenarios spanning the aridity gradient, evaluated at a fixed
# balanced preference so points are comparable across checkpoints.
EVAL_SCENARIOS = [
    ("hebei_central", "loam", 2015),
    ("ningxia_irrigation", "loam", 2015),
    ("shaanxi_guanzhong", "loam", 2015),
]
BALANCED_WEIGHTS = {"yield_proxy": 0.4, "water": 0.3, "cost": 0.2, "risk": 0.1}


def parse_checkpoint_name(path: Path) -> tuple:
    """Different training runs (train_ppo.py's RUN_NAME) reuse the same
    checkpoint step counts, so "timesteps alone" is not a unique key -
    e.g. both a stuck run and its VecNormalize-fixed rerun have a
    "..._300000_steps.zip". Must dedupe/group by (run, steps), not steps
    alone, or one run's already-evaluated rows silently mask another run's
    checkpoints at the same step count."""
    match = re.match(r"^(.+)_(\d+)_steps$", path.stem)
    run, steps = match.groups()
    return run, int(steps)


def main():
    existing = pd.read_csv(OUT_PATH) if OUT_PATH.exists() else pd.DataFrame()
    if not existing.empty and "run" not in existing.columns:
        raise RuntimeError(
            f"{OUT_PATH} predates per-run tracking and can't be deduped safely - "
            "delete it and rerun to regenerate from scratch."
        )
    done_keys = set(zip(existing["run"], existing["timesteps"])) if not existing.empty else set()

    rows = []
    checkpoints = sorted(CHECKPOINT_DIR.glob("*.zip"), key=lambda p: parse_checkpoint_name(p)[1])
    for ckpt in checkpoints:
        run, steps = parse_checkpoint_name(ckpt)
        if (run, steps) in done_keys:
            continue
        policy_fn = load_ppo_policy(str(ckpt))
        for site_id, soil_key, year in EVAL_SCENARIOS:
            result = run_episode(site_id, soil_key, year, BALANCED_WEIGHTS, policy_fn)
            result["run"] = run
            result["timesteps"] = steps
            rows.append(result)
        print(f"evaluated {run} checkpoint at {steps} steps")

    if rows:
        new_df = pd.DataFrame(rows)
        combined = pd.concat([existing, new_df], ignore_index=True) if not existing.empty else new_df
        combined.to_csv(OUT_PATH, index=False)
        print(f"saved -> {OUT_PATH} ({len(combined)} rows)")
    else:
        print("no new checkpoints to evaluate")


if __name__ == "__main__":
    main()
