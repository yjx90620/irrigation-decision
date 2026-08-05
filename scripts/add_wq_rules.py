"""audit-v2: evaluate the two rule baselines under the per_quota reward
and add them to the wq comparison CSV (the rules' yield/irrigation are
reward-independent but their returns must match the arm's reward).
audit-v3 (2.4): UPSERT on the formal key set - reruns replace, never
duplicate; writes are atomic and contract-validated."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "rl"))
sys.path.insert(0, str(ROOT / "src" / "data"))
sys.path.insert(0, str(ROOT / "src" / "sim"))
sys.path.insert(0, str(ROOT / "src" / "utils"))

import pandas as pd

from result_contract import ResultSpec, atomic_write_csv, validate_result_frame
from train_rotation_compare import BALANCED_WEIGHTS, evaluate, quota_reserving_policy, threshold_policy

OUT = ROOT / "data" / "processed" / "ppo_rotation_wq_comparison.csv"

KEY_COLUMNS = ("policy", "site_id", "year", "preference", "seed")
REQUIRED = KEY_COLUMNS + ("total_yield_t_ha", "total_irrigation_mm")


def main():
    df = pd.read_csv(OUT)
    rules = pd.concat([
        evaluate(threshold_policy, "threshold_rule", preference_sets={"balanced": BALANCED_WEIGHTS},
                 water_norm="per_quota"),
        evaluate(quota_reserving_policy, "quota_reserving_rule", preference_sets={"balanced": BALANCED_WEIGHTS},
                 water_norm="per_quota"),
    ], ignore_index=True)
    combined = pd.concat([df, rules], ignore_index=True)
    combined = combined.drop_duplicates(subset=list(KEY_COLUMNS), keep="last")
    expected_keys = frozenset(
        map(tuple, combined.loc[:, list(KEY_COLUMNS)].itertuples(index=False, name=None))
    )
    validate_result_frame(combined, ResultSpec(
        key_columns=KEY_COLUMNS, required_columns=REQUIRED, expected_keys=expected_keys,
        finite_columns=("total_yield_t_ha", "total_irrigation_mm"),
    ))
    atomic_write_csv(combined, OUT)
    print(f"upserted {len(rules)} rule rows -> {OUT} ({len(combined)} rows)")


if __name__ == "__main__":
    main()

