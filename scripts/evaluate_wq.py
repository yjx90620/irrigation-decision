"""audit-v2 P0-9: evaluate the reward-rebalanced (per_quota) models under
the same preference sets as the primary arm, writing
ppo_rotation_wq_comparison.csv. The policy behavior (yield/irrigation)
is what the reward rebalance changes; the returns are computed under the
per_quota reward (consistent with the arm's training)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "rl"))
sys.path.insert(0, str(ROOT / "src" / "data"))
sys.path.insert(0, str(ROOT / "src" / "sim"))

import pandas as pd

from train_rotation_compare import PREFERENCE_EVAL_SETS, evaluate, load_policy

OUT_DIR = ROOT / "data" / "processed"
SEEDS = [0, 1, 2]


def main():
    frames = []
    for mode in ["direct", "residual"]:
        for seed in SEEDS:
            suffix = "" if seed == 0 else f"_seed{seed}"
            model_path = OUT_DIR / f"ppo_rotation_{mode}_wq{suffix}_final.zip"
            vecnorm_path = OUT_DIR / f"ppo_rotation_{mode}_wq{suffix}_final_vecnormalize.pkl"
            if not model_path.exists():
                print(f"WARNING: {model_path.name} missing - run regen_train_ppo_wq.py first")
                continue
            eval_df = evaluate(
                load_policy(model_path, vecnorm_path, mode), f"ppo_{mode}_wq",
                preference_sets=PREFERENCE_EVAL_SETS, water_norm="per_quota",
            )
            eval_df["seed"] = seed
            frames.append(eval_df)
            print(f"evaluated ppo_{mode}_wq seed={seed}", flush=True)
    if frames:
        combined = pd.concat(frames, ignore_index=True)
        out_path = OUT_DIR / "ppo_rotation_wq_comparison.csv"
        combined.to_csv(out_path, index=False)
        print(f"saved -> {out_path} ({len(combined)} rows)", flush=True)


if __name__ == "__main__":
    main()
