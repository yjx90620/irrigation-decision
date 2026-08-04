"""DEPRECATED (audit-v2 P0-12): single-season prototype leave-one-out
transfer - results invalidated (see docs/AUDIT_FIX_LOG.md); the
rotation-era transfer is src/transfer/leave_one_out_rotation.py.

Original docstring:
Leave-one-site-out transfer experiment (研究方案 7.2 / 10.1).

For each target site: train a "source" policy on the other 4 sites only
(uniform sampling), evaluate it zero-shot on the target (no target data at
all), then fine-tune that same policy on target-site data only and
evaluate again. Compares both against the threshold-rule baseline
(site-agnostic, so it's a fair fixed reference across all 5 folds) and
records the target's environmental distance to its nearest source site
(from src/transfer/similarity_analysis.py) to test the prediction that
distance predicts transfer difficulty.

This is a first, low-fidelity pass (研究方案 7.2 explicitly lists a "from
scratch on target only" condition too, which this skips for now - see
docs/papers/论文三_跨区域动态迁移.md's finalization checklist) - training
budgets here are intentionally small (~1/10 of the main PPO run) so all 5
folds finish in one sitting; the finalized version should raise them.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rl"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))

# audit-v3 (8): legacy data script guard - single-season prototype outputs
# are invalidated; refuse to run without --allow-legacy.
if __name__ == "__main__" and "--allow-legacy" not in sys.argv:
    raise SystemExit(
        "leave_one_out.py is a DEPRECATED single-season prototype (audit-v2 P0-12); its "
        "outputs are invalid for paper claims. Pass --allow-legacy to run it for "
        "development record only."
    )

import pandas as pd

from config import SITES
from evaluate_policy import evaluate_grid, load_ppo_policy, threshold_policy
from train_utils import train_policy

TRAIN_YEARS = list(range(1981, 2011))
EVAL_YEARS = [2018, 2019, 2020, 2021, 2022]
SOURCE_STEPS = 100_000
FINETUNE_STEPS = 20_000
BALANCED_WEIGHTS = [{"yield_proxy": 0.4, "water": 0.3, "cost": 0.2, "risk": 0.1}]

OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
DIST_PATH = OUT_DIR / "site_distance_matrix.csv"


def nearest_source_distance(target: str, sources: list) -> float:
    dist = pd.read_csv(DIST_PATH, index_col="site_id")
    return dist.loc[target, sources].min()


def run_fold(target_site: str) -> list:
    source_sites = [s for s in SITES if s != target_site]
    rows = []

    print(f"=== fold: target={target_site}, sources={source_sites} ===")
    src_model, src_vecnorm = train_policy(
        sites=source_sites, soils=["loam"], years=TRAIN_YEARS,
        total_timesteps=SOURCE_STEPS, run_name=f"transfer_source_excl_{target_site}",
    )
    zero_shot_fn = load_ppo_policy(str(src_model))
    zero_shot = evaluate_grid([target_site], ["loam"], EVAL_YEARS, BALANCED_WEIGHTS, zero_shot_fn)
    zero_shot["condition"] = "zero_shot"

    ft_model, _ = train_policy(
        sites=[target_site], soils=["loam"], years=TRAIN_YEARS,
        total_timesteps=FINETUNE_STEPS, run_name=f"transfer_finetuned_{target_site}",
        base_model_path=src_model, base_vecnormalize_path=src_vecnorm,
    )
    finetuned_fn = load_ppo_policy(str(ft_model))
    finetuned = evaluate_grid([target_site], ["loam"], EVAL_YEARS, BALANCED_WEIGHTS, finetuned_fn)
    finetuned["condition"] = "finetuned"

    rule = evaluate_grid([target_site], ["loam"], EVAL_YEARS, BALANCED_WEIGHTS, threshold_policy)
    rule["condition"] = "threshold_rule"

    combined = pd.concat([zero_shot, finetuned, rule], ignore_index=True)
    combined["target_site"] = target_site
    combined["nearest_source_distance"] = nearest_source_distance(target_site, source_sites)
    rows.append(combined)
    return rows


def main():
    out_path = OUT_DIR / "leave_one_out_transfer.csv"
    existing = pd.read_csv(out_path) if out_path.exists() else pd.DataFrame()
    done_targets = set(existing["target_site"]) if not existing.empty else set()

    all_rows = [existing] if not existing.empty else []
    for target_site in SITES:
        if target_site in done_targets:
            print(f"skip {target_site}, already done")
            continue
        all_rows.extend(run_fold(target_site))
        pd.concat(all_rows, ignore_index=True).to_csv(out_path, index=False)  # save incrementally

    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
