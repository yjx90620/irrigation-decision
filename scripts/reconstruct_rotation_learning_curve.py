"""Rotation-era PPO learning-curve reconstruction (audit-v2, regen).

The legacy reconstruct_learning_curve.py evaluates checkpoints in the
SINGLE-SEASON env with the legacy loader - it cannot be used for the
rotation checkpoints (different observation space, and the checkpoint
models are paired with their own VecNormalize stats). This version:
  - scans data/processed/ppo_checkpoints/ppo_rotation_*_steps.zip
  - pairs each with its own {run}_vecnormalize_{steps}_steps.pkl
    (same training run - the strict_pairing name check in load_policy
    only understands _final.zip names, so it is bypassed deliberately)
  - evaluates in RotationIrrigationEnv under the balanced preference on
    the same 3-site aridity-gradient scenarios the legacy script used
  - dedupes by (run, timesteps), appending to ppo_rotation_learning_curve.csv

Output feeds the paper-2 training-diagnosis figure. Safe to re-run.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "rl"))
sys.path.insert(0, str(ROOT / "src" / "data"))
sys.path.insert(0, str(ROOT / "src" / "sim"))

import pandas as pd

from train_rotation_compare import BALANCED_WEIGHTS, evaluate, load_policy

CHECKPOINT_DIR = ROOT / "data" / "processed" / "ppo_checkpoints"
OUT_PATH = ROOT / "data" / "processed" / "ppo_rotation_learning_curve.csv"

EVAL_SITES = ["hebei_central", "ningxia_irrigation", "shaanxi_guanzhong"]
EVAL_YEAR = 2015

CHECKPOINT_RE = re.compile(r"^(ppo_rotation_\w+)_(\d+)_steps$")


def main():
    existing = pd.read_csv(OUT_PATH) if OUT_PATH.exists() else pd.DataFrame()
    done_keys = set(zip(existing["policy"], existing["timesteps"])) if not existing.empty else set()

    rows = []
    checkpoints = sorted(
        (p for p in CHECKPOINT_DIR.glob("ppo_rotation_*_steps.zip") if "vecnormalize" not in p.name),
        key=lambda p: int(CHECKPOINT_RE.match(p.stem).group(2)),
    )
    for ckpt in checkpoints:
        match = CHECKPOINT_RE.match(ckpt.stem)
        run, steps = match.group(1), int(match.group(2))
        if (run, steps) in done_keys:
            continue
        vecnorm = CHECKPOINT_DIR / f"{run}_vecnormalize_{steps}_steps.pkl"
        if not vecnorm.exists():
            print(f"WARNING: no paired vecnormalize for {ckpt.name} - skipping")
            continue
        mode = "residual" if "residual" in run else "direct"
        policy_fn = load_policy(ckpt, vecnorm, mode, strict_pairing=False)
        chunk = evaluate(policy_fn, label=run, sites=EVAL_SITES, years=[EVAL_YEAR],
                         preference_sets={"balanced": BALANCED_WEIGHTS})
        chunk["timesteps"] = steps
        rows.append(chunk)
        print(f"evaluated {run} at {steps} steps", flush=True)

    if rows:
        new_df = pd.concat(rows, ignore_index=True)
        combined = pd.concat([existing, new_df], ignore_index=True) if not existing.empty else new_df
        combined.to_csv(OUT_PATH, index=False)
        print(f"saved -> {OUT_PATH} ({len(combined)} rows)", flush=True)
    else:
        print("no new checkpoints to evaluate", flush=True)


if __name__ == "__main__":
    main()
