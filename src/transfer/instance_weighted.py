"""Instance-weighted transfer (研究方案 7.1): instead of sampling the 4
source sites uniformly during training, weight them by inverse
environmental distance to the target site, so source scenarios more
similar to the target get seen more often. Compares zero-shot performance
against leave_one_out.py's uniform-sampling zero-shot to test whether
similarity weighting actually helps - run that script first.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rl"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))

# audit-v3 (8): legacy data script guard - single-season prototype outputs
# are invalidated; refuse to run without --allow-legacy.
if __name__ == "__main__" and "--allow-legacy" not in sys.argv:
    raise SystemExit(
        "instance_weighted.py is a DEPRECATED single-season prototype (audit-v2 P0-12); its "
        "outputs are invalid for paper claims. Pass --allow-legacy to run it for "
        "development record only."
    )

import pandas as pd

from config import SITES
from evaluate_policy import evaluate_grid, load_ppo_policy
from train_utils import train_policy

TRAIN_YEARS = list(range(1981, 2011))
EVAL_YEARS = [2018, 2019, 2020, 2021, 2022]
SOURCE_STEPS = 100_000
BALANCED_WEIGHTS = [{"yield_proxy": 0.4, "water": 0.3, "cost": 0.2, "risk": 0.1}]

OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
DIST_PATH = OUT_DIR / "site_distance_matrix.csv"


def inverse_distance_weights(target: str, sources: list) -> dict:
    dist = pd.read_csv(DIST_PATH, index_col="site_id")
    d = dist.loc[target, sources]
    inv = 1.0 / (d + 1e-6)
    return (inv / inv.sum()).to_dict()


def run_fold(target_site: str) -> pd.DataFrame:
    source_sites = [s for s in SITES if s != target_site]
    weights = inverse_distance_weights(target_site, source_sites)
    print(f"=== fold: target={target_site}, weights={ {k: round(v,3) for k,v in weights.items()} } ===")

    model_path, _ = train_policy(
        sites=source_sites, soils=["loam"], years=TRAIN_YEARS,
        total_timesteps=SOURCE_STEPS, run_name=f"transfer_weighted_excl_{target_site}",
        site_weights=weights,
    )
    policy_fn = load_ppo_policy(str(model_path))
    result = evaluate_grid([target_site], ["loam"], EVAL_YEARS, BALANCED_WEIGHTS, policy_fn)
    result["condition"] = "zero_shot_weighted"
    result["target_site"] = target_site
    return result


def main():
    out_path = OUT_DIR / "instance_weighted_transfer.csv"
    existing = pd.read_csv(out_path) if out_path.exists() else pd.DataFrame()
    done_targets = set(existing["target_site"]) if not existing.empty else set()

    all_rows = [existing] if not existing.empty else []
    for target_site in SITES:
        if target_site in done_targets:
            print(f"skip {target_site}, already done")
            continue
        all_rows.append(run_fold(target_site))
        pd.concat(all_rows, ignore_index=True).to_csv(out_path, index=False)

    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
