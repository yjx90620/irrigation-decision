"""audit-v3 (2.1): evaluate a trained ARM (identified by its
RLExperimentConfig) under the preference sets, writing
ppo_rotation_{armtag}_comparison.csv. The arm's config drives the eval
env, so returns are computed under the same reward the arm trained on."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "rl"))
sys.path.insert(0, str(ROOT / "src" / "data"))
sys.path.insert(0, str(ROOT / "src" / "sim"))

import pandas as pd

from experiment_config import PRIMARY_CONFIG, RLExperimentConfig
from train_rotation_compare import PREFERENCE_EVAL_SETS, _arm_tag, evaluate, load_policy

OUT_DIR = ROOT / "data" / "processed"
SEEDS = [0, 1, 2]


def main(config: RLExperimentConfig = PRIMARY_CONFIG):
    tag = _arm_tag(config)
    frames = []
    for mode in ["direct", "residual"]:
        for seed in SEEDS:
            suffix = "" if seed == 0 else f"_seed{seed}"
            model_path = OUT_DIR / f"ppo_rotation_{mode}{tag}{suffix}_final.zip"
            vecnorm_path = OUT_DIR / f"ppo_rotation_{mode}{tag}{suffix}_final_vecnormalize.pkl"
            if not model_path.exists():
                print(f"WARNING: {model_path.name} missing - train it first")
                continue
            # the primary arm keeps the historical 'wq' label so fig10 /
            # paper-claims / papers keep reading ppo_direct_wq etc.
            label = f"ppo_{mode}_wq" if not tag else f"ppo_{mode}{tag}"
            eval_df = evaluate(
                load_policy(model_path, vecnorm_path, mode), label,
                preference_sets=PREFERENCE_EVAL_SETS, config=config,
            )
            eval_df["seed"] = seed
            frames.append(eval_df)
            print(f"evaluated {label} seed={seed}", flush=True)
    if frames:
        combined = pd.concat(frames, ignore_index=True)
        out_path = OUT_DIR / f"ppo_rotation_{tag or 'wq'}_comparison.csv"
        combined.to_csv(out_path, index=False)
        print(f"saved -> {out_path} ({len(combined)} rows)", flush=True)


if __name__ == "__main__":
    main(PRIMARY_CONFIG)
