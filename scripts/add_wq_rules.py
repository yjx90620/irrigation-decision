"""audit-v2: evaluate the two rule baselines under the per_quota reward
and append them to the wq comparison CSV (the rules' yield/irrigation are
reward-independent but their returns must match the arm's reward)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "rl"))
sys.path.insert(0, str(ROOT / "src" / "data"))
sys.path.insert(0, str(ROOT / "src" / "sim"))

import pandas as pd

from train_rotation_compare import BALANCED_WEIGHTS, evaluate, quota_reserving_policy, threshold_policy

OUT = ROOT / "data" / "processed" / "ppo_rotation_wq_comparison.csv"


def main():
    df = pd.read_csv(OUT)
    rules = pd.concat([
        evaluate(threshold_policy, "threshold_rule", preference_sets={"balanced": BALANCED_WEIGHTS},
                 water_norm="per_quota"),
        evaluate(quota_reserving_policy, "quota_reserving_rule", preference_sets={"balanced": BALANCED_WEIGHTS},
                 water_norm="per_quota"),
    ], ignore_index=True)
    combined = pd.concat([df, rules], ignore_index=True)
    combined.to_csv(OUT, index=False)
    print(f"appended {len(rules)} rule rows -> {OUT} ({len(combined)} rows)")


if __name__ == "__main__":
    main()
