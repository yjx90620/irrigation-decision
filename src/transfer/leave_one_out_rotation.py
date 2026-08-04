"""Leave-one-site-out transfer on the rotation environment (paper 3's main
transfer experiment, upgraded from the single-season prototype).

For each target site: train a residual-PPO source policy on the other 4
sites (which may mix cropping systems - e.g. training on 3 double-crop
sites + Ningxia's spring maize when Ningxia isn't the target - a harder
and more realistic transfer test than the prototype's climate-only
version), evaluate it zero-shot on the target, fine-tune on target-only
data, and compare both against the site-agnostic threshold rule.

Residual mode is used throughout (not direct), since that's what paper 2
establishes as the stronger policy class on this environment - see
docs/papers/论文二.

Feeds src/transfer/two_factor_predictor.py's validate_against_observed_gap().
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rl"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))

import pandas as pd

from config import SITES
from rotation_env import RotationIrrigationEnv, combine_reward, threshold_policy
from train_rotation_compare import BALANCED_WEIGHTS, GAMMA, load_policy
from train_rotation_utils import train_rotation_policy

TEST_YEARS = [2018, 2019, 2020, 2021, 2022]
SOURCE_STEPS = 200_000
FINETUNE_STEPS = 50_000

OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
DIST_PATH = OUT_DIR / "site_distance_matrix.csv"


def evaluate(policy_fn, label, site_id, years=TEST_YEARS, water_norm="per_action"):
    rows = []
    for year in years:
        env = RotationIrrigationEnv(site_id, "loam", year, water_norm=water_norm)
        state = env.reset()
        done, n_steps, n_mod = False, 0, 0
        undiscounted_return, discounted_return = 0.0, 0.0
        while not done:
            state, reward, done, info = env.step(policy_fn(state, BALANCED_WEIGHTS))
            r = combine_reward(reward, BALANCED_WEIGHTS)
            undiscounted_return += r
            discounted_return += (GAMMA ** n_steps) * r
            n_steps += 1
            n_mod += int(info["action_modified"])
        rows.append({
            "condition": label, "target_site": site_id, "year": year,
            "total_yield_t_ha": info["total_yield_t_ha"],
            "total_irrigation_mm": info["total_irrigation_mm"],
            "action_modified_rate": n_mod / n_steps,
            # P0-4c (docs/审计修复计划.md): PPO optimizes the discounted
            # return, not the flat sum - report both, don't call either
            # one "scalar_return" as if it were unambiguous.
            "undiscounted_return": undiscounted_return,
            "discounted_return": discounted_return,
        })
    return pd.DataFrame(rows)


def run_fold(target_site, water_norm="per_action"):
    source_sites = [s for s in SITES if s != target_site]
    print(f"=== fold: target={target_site}, sources={source_sites} (water_norm={water_norm}) ===")

    src_model, src_vecnorm = train_rotation_policy(
        source_sites, "residual", SOURCE_STEPS, f"transfer_rot_source_excl_{target_site}", checkpoint_every=None,
        water_norm=water_norm,
    )
    zero_shot_fn = load_policy(src_model, src_vecnorm, "residual")
    zero_shot = evaluate(zero_shot_fn, "zero_shot", target_site, water_norm=water_norm)

    ft_model, ft_vecnorm = train_rotation_policy(
        [target_site], "residual", FINETUNE_STEPS, f"transfer_rot_finetuned_{target_site}",
        base_model_path=src_model, base_vecnormalize_path=src_vecnorm, water_norm=water_norm,
    )
    # P0-6a (docs/审计修复计划.md): evaluate with the fine-tuned
    # VecNormalize stats, not the source domain's - continuing training
    # with norm_obs=True keeps updating the running mean/var, so
    # evaluating against the stale source stats would feed the
    # fine-tuned model observations normalized on a different
    # distribution than the one its weights were actually tuned against.
    finetuned_fn = load_policy(ft_model, ft_vecnorm, "residual")
    finetuned = evaluate(finetuned_fn, "finetuned", target_site, water_norm=water_norm)

    rule = evaluate(threshold_policy, "threshold_rule", target_site, water_norm=water_norm)

    dist = pd.read_csv(DIST_PATH, index_col="site_id")
    nearest_distance = dist.loc[target_site, source_sites].min()

    combined = pd.concat([zero_shot, finetuned, rule], ignore_index=True)
    combined["nearest_source_distance"] = nearest_distance
    return combined


def main(water_norm="per_action"):
    suffix = "" if water_norm == "per_action" else "_wq"
    out_path = OUT_DIR / f"leave_one_out_rotation_transfer{suffix}.csv"
    existing = pd.read_csv(out_path) if out_path.exists() else pd.DataFrame()
    # audit-v2 (P0-13): a target is only "done" when its fold rows are
    # complete - every condition (zero_shot/finetuned/threshold_rule) x
    # every TEST_YEAR present with finite yield/irrigation. A target whose
    # fold crashed partway must be re-run, not skipped because its name
    # appears in the file.
    if not existing.empty:
        required = {"target_site", "condition", "year", "total_yield_t_ha", "total_irrigation_mm"}
        if required.issubset(existing.columns):
            complete = existing[
                existing[["total_yield_t_ha", "total_irrigation_mm"]].notna().all(axis=1)
            ]
            expected_rows = 3 * len(TEST_YEARS)  # 3 conditions x 5 years per target
            counts = complete.groupby("target_site").size()
            done_targets = set(counts[counts >= expected_rows].index)
        else:
            done_targets = set()
    else:
        done_targets = set()

    frames = [existing[existing["target_site"].isin(done_targets)]] if not existing.empty else []
    for target_site in SITES:
        if target_site in done_targets:
            print(f"skip {target_site}, already done")
            continue
        frames.append(run_fold(target_site, water_norm=water_norm))
        pd.concat(frames, ignore_index=True).to_csv(out_path, index=False)

    print(f"saved -> {out_path}")

    # also emit the yield_gap_vs_rule column two_factor_predictor.py expects
    df = pd.concat(frames, ignore_index=True)
    summary = df.groupby(["target_site", "condition"])["total_yield_t_ha"].mean().unstack()
    summary["yield_gap_vs_rule"] = summary["threshold_rule"] - summary["zero_shot"]
    summary.to_csv(OUT_DIR / f"leave_one_out_rotation_summary{suffix}.csv")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--water-norm", default="per_action", choices=["per_action", "per_quota"],
                        help="audit-v2 P0-9: reward-normalization arm; per_quota writes "
                             "leave_one_out_rotation_transfer_wq.csv")
    args = parser.parse_args()
    main(water_norm=args.water_norm)
