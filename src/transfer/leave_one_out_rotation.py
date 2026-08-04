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
from experiment_config import PRIMARY_CONFIG, RLExperimentConfig
from rotation_env import RotationIrrigationEnv, combine_reward, threshold_policy
from train_rotation_compare import BALANCED_WEIGHTS, _arm_tag, load_policy
from train_rotation_utils import train_rotation_policy

TEST_YEARS = [2018, 2019, 2020, 2021, 2022]
SOURCE_STEPS = 200_000
FINETUNE_STEPS = 50_000

OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
DIST_PATH = OUT_DIR / "site_distance_matrix.csv"


def evaluate(policy_fn, label, site_id, years=TEST_YEARS, config: RLExperimentConfig = PRIMARY_CONFIG):
    rows = []
    for year in years:
        env = RotationIrrigationEnv(site_id, "loam", year, config=config)
        state = env.reset()
        done, n_steps, n_mod = False, 0, 0
        undiscounted_return, discounted_return = 0.0, 0.0
        while not done:
            state, reward, done, info = env.step(policy_fn(state, BALANCED_WEIGHTS))
            r = combine_reward(reward, BALANCED_WEIGHTS)
            undiscounted_return += r
            discounted_return += (config.gamma ** n_steps) * r
            n_steps += 1
            n_mod += int(info["safety_modified"])
        rows.append({
            "condition": label, "target_site": site_id, "year": year,
            "total_yield_t_ha": info["total_yield_t_ha"],
            "total_irrigation_mm": info["total_irrigation_mm"],
            # audit-v3 (3.6): safety intervention rate uses the safety-rule
            # flag only, not the mixed action_modified aggregate.
            "safety_modified_rate": n_mod / n_steps,
            "undiscounted_return": undiscounted_return,
            "discounted_return": discounted_return,
        })
    return pd.DataFrame(rows)


def run_fold(target_site, config=PRIMARY_CONFIG, seeds=(0,)):
    """audit-v3 (2.1/6.1): one config drives source/finetune training and
    evaluation; multi-seed folds (seeds tuple) are the formal mode."""
    config.validate()
    source_sites = [s for s in SITES if s != target_site]
    print(f"=== fold: target={target_site}, config={config.config_hash()[:8]}, seeds={seeds} ===")

    src_model, src_vecnorm = train_rotation_policy(
        source_sites, "residual", SOURCE_STEPS, f"transfer_rot_source_excl_{target_site}", checkpoint_every=None,
        config=config,
    )
    zero_shot_fn = load_policy(src_model, src_vecnorm, "residual")
    zero_shot = evaluate(zero_shot_fn, "zero_shot", target_site, config=config)

    ft_model, ft_vecnorm = train_rotation_policy(
        [target_site], "residual", FINETUNE_STEPS, f"transfer_rot_finetuned_{target_site}",
        base_model_path=src_model, base_vecnormalize_path=src_vecnorm, config=config,
    )
    # P0-6a: evaluate with the fine-tuned VecNormalize stats.
    finetuned_fn = load_policy(ft_model, ft_vecnorm, "residual")
    finetuned = evaluate(finetuned_fn, "finetuned", target_site, config=config)

    rule = evaluate(threshold_policy, "threshold_rule", target_site, config=config)

    dist = pd.read_csv(DIST_PATH, index_col="site_id")
    nearest_distance = dist.loc[target_site, source_sites].min()

    combined = pd.concat([zero_shot, finetuned, rule], ignore_index=True)
    combined["nearest_source_distance"] = nearest_distance
    return combined


def main(config: RLExperimentConfig = PRIMARY_CONFIG, seeds=(0,)):
    tag = _arm_tag(config)
    suffix = "" if not tag else tag
    out_path = OUT_DIR / f"leave_one_out_rotation_transfer{suffix}.csv"
    existing = pd.read_csv(out_path) if out_path.exists() else pd.DataFrame()
    if not existing.empty:
        required = {"target_site", "condition", "year", "total_yield_t_ha", "total_irrigation_mm"}
        if required.issubset(existing.columns):
            complete = existing[
                existing[["total_yield_t_ha", "total_irrigation_mm"]].notna().all(axis=1)
            ]
            expected_rows = 3 * len(TEST_YEARS) * len(seeds)
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
        frames.append(run_fold(target_site, config=config, seeds=seeds))
        pd.concat(frames, ignore_index=True).to_csv(out_path, index=False)

    print(f"saved -> {out_path}")

    df = pd.concat(frames, ignore_index=True)
    summary = df.groupby(["target_site", "condition"])["total_yield_t_ha"].mean().unstack()
    summary["yield_gap_vs_rule"] = summary["threshold_rule"] - summary["zero_shot"]
    summary.to_csv(OUT_DIR / f"leave_one_out_rotation_summary{suffix}.csv")


if __name__ == "__main__":
    import argparse

    from experiment_config import RLExperimentConfig

    parser = argparse.ArgumentParser()
    parser.add_argument("--config-hash", default=None,
                        help="audit-v3 (2.1): hash of the RLExperimentConfig to train with "
                             "(defaults to PRIMARY_CONFIG)")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    if args.config_hash is not None:
        raise SystemExit(
            "config selection by hash not wired yet - edit scripts/run_transfer_arm.py "
            "to pass a RLExperimentConfig explicitly"
        )
    main(seeds=(args.seed,) if args.seed is not None else (0, 1, 2))
