"""Train direct-action vs residual PPO on the rotation environment and
evaluate both against the rule baseline (paper 2's core comparison).

Both variants get identical training budgets, network, and hyperparameters
so the only difference is action semantics. VecNormalize is on for both -
without it the single-season prototype's policy got stuck outputting a
constant action for a full million steps (see src/rl/README.md).

Evaluation uses held-out years and reports system totals (wheat + maize
yield under one annual quota), which is where the rule baseline visibly
fails: it exhausts the quota on wheat and starves maize.
"""

import functools
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from config import SITES
from experiment_config import PRIMARY_CONFIG, RLExperimentConfig
from residual_gym_env import (
    PREFERENCE_KEYS, RESIDUAL_DELTAS, STATE_KEYS, TRAIN_YEARS, RotationGymEnv,
)
from rotation_env import (
    ACTIONS_MM, RotationIrrigationEnv, combine_reward, quota_reserving_policy, threshold_policy,
)
from soils import STANDARD_SOILS

TEST_YEARS = [2018, 2019, 2020, 2021, 2022]
# audit-v3 (3.8): "risk" -> "acute_stress" (a 3-day mean tr_ratio<0.5
# binary indicator, NOT interannual downside risk).
BALANCED_WEIGHTS = {"yield_proxy": 0.4, "water": 0.3, "cost": 0.2, "acute_stress": 0.1}

# P1-4 (audit-v2): preference-conditioning evaluation sets.
PREFERENCE_EVAL_SETS = {
    "balanced": BALANCED_WEIGHTS,
    "yield_max": {"yield_proxy": 0.7, "water": 0.1, "cost": 0.1, "acute_stress": 0.1},
    "water_min": {"yield_proxy": 0.1, "water": 0.7, "cost": 0.1, "acute_stress": 0.1},
    "cost_min": {"yield_proxy": 0.1, "water": 0.1, "cost": 0.7, "acute_stress": 0.1},
    "stress_min": {"yield_proxy": 0.1, "water": 0.1, "cost": 0.1, "acute_stress": 0.7},
    "yield_vs_water": {"yield_proxy": 0.5, "water": 0.4, "cost": 0.05, "acute_stress": 0.05},
    "water_vs_cost": {"yield_proxy": 0.1, "water": 0.45, "cost": 0.45, "acute_stress": 0.0},
}

WORKERS_PER_SITE = 2
TOTAL_TIMESTEPS = 400_000

OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

TRAIN_SITES = list(SITES)


def _arm_tag(config: RLExperimentConfig) -> str:
    """audit-v3 (2.1): run-name suffix derived from the config so each arm
    (gamma/water-normalizer/shaping ablation) never clobbers another."""
    tag = ""
    if config.gamma != PRIMARY_CONFIG.gamma:
        tag += f"_gamma{int(config.gamma)}"
    if config.water_normalizer != PRIMARY_CONFIG.water_normalizer:
        tag += "_wq" if config.water_normalizer == "annual_quota" else "_peraction"
    if config.shaping_mode != PRIMARY_CONFIG.shaping_mode:
        tag += "_noshaping"
    return tag


def _make_env(rank, mode, sites, seed=0, config: RLExperimentConfig = PRIMARY_CONFIG):
    # same (seed, rank) -> same per-worker RNG stream regardless of mode,
    # so direct/residual trained with the same seed see identical
    # site/year/soil/preference sequences (P0-5)
    return RotationGymEnv(
        mode=mode, sites=list(sites), soils=["loam"], years=TRAIN_YEARS,
        fixed_site=sites[rank % len(sites)], seed=1000 * seed + rank, config=config,
    )


class SiteTransitionLogger(BaseCallback):
    """P0-4 (docs/审计修复计划.md): reports per-site episode/transition
    counts the training actually saw, so an uneven sampling ratio would
    be visible instead of assumed away. With fixed_site pinning this
    should land close to WORKERS_PER_SITE / (WORKERS_PER_SITE * n_sites)
    per site for both counts - if it doesn't, something upstream (an env
    crashing/restarting more on one site, say) is still skewing things."""

    def __init__(self, out_path, log_every=50_000, verbose=0):
        super().__init__(verbose)
        self.out_path = out_path
        self.log_every = log_every
        self.transitions = Counter()
        self.episodes = Counter()
        self._last_log = 0

    def _on_step(self) -> bool:
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            site_id = info.get("site_id")
            if site_id is None:
                continue
            self.transitions[site_id] += 1
            if done:
                self.episodes[site_id] += 1
        if self.num_timesteps - self._last_log >= self.log_every:
            self._last_log = self.num_timesteps
            total = sum(self.transitions.values())
            share = {k: f"{v / total:.1%}" for k, v in sorted(self.transitions.items())}
            print(f"  [site transitions @ {self.num_timesteps}] {share}", flush=True)
        return True

    def _on_training_end(self) -> None:
        pd.DataFrame({
            "site_id": list(self.transitions),
            "transitions": [self.transitions[s] for s in self.transitions],
            "episodes": [self.episodes.get(s, 0) for s in self.transitions],
        }).to_csv(self.out_path, index=False)


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """audit-v3 (3.11): explicit seeding of every RNG the training touches."""
    import os
    import random

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


def train(mode, config: RLExperimentConfig = PRIMARY_CONFIG, total_timesteps=TOTAL_TIMESTEPS,
          workers_per_site=WORKERS_PER_SITE, sites=TRAIN_SITES, seed=None):
    config.validate()
    seed = config.seed if seed is None else seed
    # P0-5 + audit-v3 (3.11): seed drives SB3's RNG AND every worker's
    # site-year-soil-preference draw, so direct/residual with the same
    # seed see identical scenario sequences.
    seed_everything(seed, deterministic=config.deterministic_torch)
    n_envs = workers_per_site * len(sites)
    tag = _arm_tag(config)
    run_name = f"ppo_rotation_{mode}{tag}" if seed == 0 else f"ppo_rotation_{mode}{tag}_seed{seed}"
    env_fns = [functools.partial(_make_env, rank, mode, sites, seed=seed, config=config)
               for rank in range(n_envs)]
    vec_env = SubprocVecEnv(env_fns)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    model = PPO(
        "MlpPolicy", vec_env, verbose=1, n_steps=512, batch_size=256, n_epochs=10,
        learning_rate=3e-4, gamma=config.gamma, ent_coef=0.01, seed=seed,
        device=config.device,
    )
    checkpoint_dir = OUT_DIR / "ppo_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    callback = CallbackList([
        CheckpointCallback(
            save_freq=max(100_000 // n_envs, 1), save_path=str(checkpoint_dir),
            name_prefix=run_name, save_vecnormalize=True,
        ),
        SiteTransitionLogger(OUT_DIR / f"{run_name}_site_transitions.csv"),
    ])
    model.learn(total_timesteps=total_timesteps, callback=callback)
    actual_timesteps = model.num_timesteps
    model_path = OUT_DIR / f"{run_name}_final.zip"
    vecnorm_path = OUT_DIR / f"{run_name}_final_vecnormalize.pkl"
    model.save(str(model_path))
    vec_env.save(str(vecnorm_path))
    vec_env.close()
    # audit-v3 (2.1): save the FULL config (requested + actual timesteps)
    # next to the model so every run is self-describing.
    config.with_updates(requested_timesteps=total_timesteps, actual_timesteps=actual_timesteps).to_json(
        OUT_DIR / f"{run_name}_config.json"
    )
    return model_path, vecnorm_path


def load_policy(model_path, vecnorm_path, mode, strict_pairing=True):
    from stable_baselines3.common.vec_env import DummyVecEnv

    model_path, vecnorm_path = Path(model_path), Path(vecnorm_path)
    if strict_pairing:
        # P0-6a (docs/审计修复计划.md): leave_one_out_rotation.py's
        # fine-tuning evaluation once loaded the *source* domain's
        # VecNormalize stats instead of the fine-tuned model's own -
        # normalization stats keep updating during training, so a
        # mismatched pair feeds observations normalized against the
        # wrong distribution. {run_name}_final.zip and
        # {run_name}_final_vecnormalize.pkl share a run_name by
        # train()'s/train_rotation_policy()'s own naming convention;
        # this catches an accidental mismatch instead of silently
        # evaluating with stale stats. Pass strict_pairing=False for a
        # deliberate cross-run load (e.g. an explicit ablation).
        run_name = model_path.name.removesuffix("_final.zip")
        expected_vecnorm = model_path.with_name(f"{run_name}_final_vecnormalize.pkl")
        assert vecnorm_path == expected_vecnorm, (
            f"model {model_path.name} and vecnormalize {vecnorm_path.name} don't look like they came from the "
            f"same training run (expected {expected_vecnorm.name}) - pass strict_pairing=False if intentional"
        )
    model = PPO.load(str(model_path))
    dummy = DummyVecEnv([lambda: RotationGymEnv(mode=mode, sites=["hebei_central"], soils=["loam"], years=[2019])])
    obs_rms = VecNormalize.load(str(vecnorm_path), dummy).obs_rms

    def policy_fn(state, weights):
        obs = np.array(
            [float(state[k]) for k in STATE_KEYS] + [weights[k] for k in PREFERENCE_KEYS], dtype=np.float32
        )
        obs = np.clip((obs - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8), -10.0, 10.0).astype(np.float32)
        idx, _ = model.predict(obs, deterministic=True)
        if mode == "direct":
            return ACTIONS_MM[int(idx)]
        return float(np.clip(threshold_policy(state, weights) + RESIDUAL_DELTAS[int(idx)], 0.0, max(ACTIONS_MM)))

    return policy_fn


def evaluate(policy_fn, label, sites=None, years=None, preference_sets=None,
             config: RLExperimentConfig = PRIMARY_CONFIG, safety_enabled: bool = True):
    """P1-4 (audit-v2) preference-conditioning evaluation + audit-v3
    (3.6/3.7) intervention accounting: rows carry the SPLIT modification
    reasons (safety_modified_rate, quota_clipped_rate,
    delivery_shortfall_mean), and safety_enabled=False rolls out the raw
    (unsafety-filtered) agent so agent+safety performance is never
    attributed to the raw policy."""
    preference_sets = preference_sets or {"balanced": BALANCED_WEIGHTS}
    rows = []
    for site_id in (sites or list(SITES)):
        for year in (years or TEST_YEARS):
            for pref_name, weights in preference_sets.items():
                env = RotationIrrigationEnv(site_id, "loam", year, config=config,
                                            safety_enabled=safety_enabled)
                state = env.reset()
                done, n_steps = False, 0
                n_safety, n_quota, delivery_shortfall_sum = 0, 0, 0.0
                undiscounted_return, discounted_return = 0.0, 0.0
                while not done:
                    state, reward, done, info = env.step(policy_fn(state, weights))
                    r = combine_reward(reward, weights)
                    undiscounted_return += r
                    discounted_return += (config.gamma ** n_steps) * r
                    n_steps += 1
                    n_safety += int(info["safety_modified"])
                    n_quota += int(info["quota_clipped"])
                    delivery_shortfall_sum += info["delivery_shortfall_mm"]
                row = {
                    "policy": label, "site_id": site_id, "year": year,
                    "preference": pref_name,
                    "safety_enabled": safety_enabled,
                    "total_yield_t_ha": info["total_yield_t_ha"],
                    "total_irrigation_mm": info["total_irrigation_mm"],
                    # audit-v3 (3.6): the intervention rate is the SAFETY
                    # rule rate only - quota clips and delivery shortfalls
                    # are separate, not folded into "safety intervention".
                    "safety_modified_rate": n_safety / n_steps,
                    "quota_clipped_rate": n_quota / n_steps,
                    "delivery_shortfall_mean_mm": delivery_shortfall_sum / n_steps,
                    "undiscounted_return": undiscounted_return,
                    "discounted_return": discounted_return,
                }
                for crop in ("wheat", "maize", "spring_maize"):
                    if crop in info:
                        row[f"{crop}_yield"] = info[crop]["dry_yield_t_ha"]
                        row[f"{crop}_irr"] = info[crop]["irrigation_mm"]
                rows.append(row)
    return pd.DataFrame(rows)


def main(n_seeds=3, seeds=None, config: RLExperimentConfig = PRIMARY_CONFIG):
    """Primary-arm comparison: rules + PPO (config) x seeds, each PPO also
    evaluated as the RAW agent (safety layer disabled) so agent+safety is
    separable (audit-v3 3.7). Ablation arms (gamma/water-normalizer/
    shaping) are trained by their own launchers and evaluated by their own
    scripts against the same preference sets."""
    seeds = seeds or list(range(n_seeds))
    frames = [
        evaluate(threshold_policy, "threshold_rule", preference_sets={"balanced": BALANCED_WEIGHTS},
                 config=config),
        evaluate(quota_reserving_policy, "quota_reserving_rule", preference_sets={"balanced": BALANCED_WEIGHTS},
                 config=config),
    ]
    print("rule baselines done")

    tag = _arm_tag(config)
    for mode in ["direct", "residual"]:
        for seed in seeds:
            suffix = "" if seed == 0 else f"_seed{seed}"
            model_path = OUT_DIR / f"ppo_rotation_{mode}{tag}{suffix}_final.zip"
            vecnorm_path = OUT_DIR / f"ppo_rotation_{mode}{tag}{suffix}_final_vecnormalize.pkl"
            if not model_path.exists():
                print(f"=== training {mode} seed={seed} ({tag or 'primary'}) ===")
                model_path, vecnorm_path = train(mode, config=config, seed=seed)
            policy_fn = load_policy(model_path, vecnorm_path, mode)
            eval_df = evaluate(policy_fn, f"ppo_{mode}", preference_sets=PREFERENCE_EVAL_SETS, config=config)
            # audit-v3 (3.7): raw-agent rollouts with the safety layer off
            raw_df = evaluate(policy_fn, f"ppo_{mode}_raw", preference_sets={"balanced": BALANCED_WEIGHTS},
                              config=config, safety_enabled=False)
            eval_df["seed"] = seed
            raw_df["seed"] = seed
            frames.append(eval_df)
            frames.append(raw_df)
            print(f"{mode} seed={seed} evaluated (safety + raw)")

    for frame in frames[:2]:
        frame["seed"] = None  # rule baselines are deterministic, no seed axis
    out_path = OUT_DIR / "rotation_policy_comparison.csv"
    pd.concat(frames, ignore_index=True).to_csv(out_path, index=False)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=3,
                        help="audit-v3 (3.11): formal runs require >= 3 seeds; 1 is not the default")
    args = parser.parse_args()
    if args.n_seeds < 1:
        raise SystemExit("--n-seeds must be >= 1")
    main(n_seeds=args.n_seeds)
