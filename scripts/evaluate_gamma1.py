"""audit-v2 P0-9: evaluate the gamma=1.0 ablation models under the same
preference sets as the gamma=0.995 arm, writing
ppo_rotation_gamma1_comparison.csv (same schema as the primary
rotation_policy_comparison.csv but with policy labels tagged _gamma1).

The gamma=0.995 policies collapsed toward water-minimizing behavior;
this table is the direct evidence of what the discounting actually cost
in yield - if gamma=1.0 closes the gap, the paper must report both arms
and discuss which reward matches the "annual yield/water/cost/risk"
objective the study actually declares.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "rl"))
sys.path.insert(0, str(ROOT / "src" / "data"))
sys.path.insert(0, str(ROOT / "src" / "sim"))

import pandas as pd

from train_rotation_compare import (
    PREFERENCE_EVAL_SETS, TEST_YEARS, evaluate, load_policy,
)

OUT_DIR = ROOT / "data" / "processed"
SEEDS = [0, 1, 2]


def main():
    frames = []
    for mode in ["direct", "residual"]:
        for seed in SEEDS:
            suffix = "" if seed == 0 else f"_seed{seed}"
            model_path = OUT_DIR / f"ppo_rotation_{mode}_gamma1{suffix}_final.zip"
            vecnorm_path = OUT_DIR / f"ppo_rotation_{mode}_gamma1{suffix}_final_vecnormalize.pkl"
            if not model_path.exists():
                print(f"WARNING: {model_path.name} missing - run regen_train_ppo_gamma1.py first")
                continue
            eval_df = evaluate(
                load_policy(model_path, vecnorm_path, mode), f"ppo_{mode}_gamma1",
                preference_sets=PREFERENCE_EVAL_SETS, gamma=1.0,
            )
            eval_df["seed"] = seed
            frames.append(eval_df)
            print(f"evaluated ppo_{mode}_gamma1 seed={seed}", flush=True)
    if frames:
        combined = pd.concat(frames, ignore_index=True)
        out_path = OUT_DIR / "ppo_rotation_gamma1_comparison.csv"
        combined.to_csv(out_path, index=False)
        print(f"saved -> {out_path} ({len(combined)} rows)", flush=True)


if __name__ == "__main__":
    main()
