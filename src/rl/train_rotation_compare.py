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
from residual_gym_env import (
    PREFERENCE_KEYS, RESIDUAL_DELTAS, STATE_KEYS, TRAIN_YEARS, RotationGymEnv,
)
from rotation_env import ACTIONS_MM, RotationIrrigationEnv, combine_reward, threshold_policy
from soils import STANDARD_SOILS

TEST_YEARS = [2018, 2019, 2020, 2021, 2022]
BALANCED_WEIGHTS = {"yield_proxy": 0.4, "water": 0.3, "cost": 0.2, "risk": 0.1}
GAMMA = 0.995  # must match PPO's own gamma below, for discounted_return to mean anything

# P0-4 (docs/审计修复计划.md): WORKERS_PER_SITE * len(TRAIN_SITES) parallel
# envs, one fixed site per worker (RotationGymEnv's fixed_site), cycling
# evenly - not the old uniform-per-episode sampling, which skewed
# transition counts toward double-crop sites since their episodes run
# ~2.3x longer (~120 decision steps) than Ningxia's single-crop ones
# (~53). N_ENVS=8 with 5 sites couldn't divide evenly; this can.
WORKERS_PER_SITE = 2
TOTAL_TIMESTEPS = 400_000

OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

TRAIN_SITES = list(SITES)


def _make_env(rank, mode, sites):
    return RotationGymEnv(
        mode=mode, sites=list(sites), soils=["loam"], years=TRAIN_YEARS, fixed_site=sites[rank % len(sites)],
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


def train(mode, total_timesteps=TOTAL_TIMESTEPS, workers_per_site=WORKERS_PER_SITE, sites=TRAIN_SITES):
    n_envs = workers_per_site * len(sites)
    run_name = f"ppo_rotation_{mode}"
    env_fns = [functools.partial(_make_env, rank, mode, sites) for rank in range(n_envs)]
    vec_env = SubprocVecEnv(env_fns)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    model = PPO(
        "MlpPolicy", vec_env, verbose=1, n_steps=512, batch_size=256, n_epochs=10,
        learning_rate=3e-4, gamma=GAMMA, ent_coef=0.01,
        device="cpu",  # measured: GPU gives ~1.12x here and SB3 warns against it for MlpPolicy
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
    model_path = OUT_DIR / f"{run_name}_final.zip"
    vecnorm_path = OUT_DIR / f"{run_name}_final_vecnormalize.pkl"
    model.save(str(model_path))
    vec_env.save(str(vecnorm_path))
    vec_env.close()
    return model_path, vecnorm_path


def load_policy(model_path, vecnorm_path, mode):
    from stable_baselines3.common.vec_env import DummyVecEnv

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


def evaluate(policy_fn, label, sites=None, years=None):
    rows = []
    for site_id in (sites or list(SITES)):
        for year in (years or TEST_YEARS):
            env = RotationIrrigationEnv(site_id, "loam", year)
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
            # Single-crop sites (Ningxia) have no "wheat" key, so report
            # per-crop columns only for the crops that site actually grows.
            row = {
                "policy": label, "site_id": site_id, "year": year,
                "total_yield_t_ha": info["total_yield_t_ha"],
                "total_irrigation_mm": info["total_irrigation_mm"],
                "action_modified_rate": n_mod / n_steps,
                # P0-4c (docs/审计修复计划.md): PPO optimizes the discounted
                # return (gamma=0.995), not the flat sum - both are reported
                # so neither gets mistaken for "what the model optimizes".
                "undiscounted_return": undiscounted_return,
                "discounted_return": discounted_return,
            }
            for crop in ("wheat", "maize", "spring_maize"):
                if crop in info:
                    row[f"{crop}_yield"] = info[crop]["dry_yield_t_ha"]
                    row[f"{crop}_irr"] = info[crop]["irrigation_mm"]
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    frames = [evaluate(threshold_policy, "threshold_rule")]
    print("rule baseline done")

    for mode in ["direct", "residual"]:
        model_path = OUT_DIR / f"ppo_rotation_{mode}_final.zip"
        vecnorm_path = OUT_DIR / f"ppo_rotation_{mode}_final_vecnormalize.pkl"
        if not model_path.exists():
            print(f"=== training {mode} ===")
            model_path, vecnorm_path = train(mode)
        frames.append(evaluate(load_policy(model_path, vecnorm_path, mode), f"ppo_{mode}"))
        print(f"{mode} evaluated")

    out_path = OUT_DIR / "rotation_policy_comparison.csv"
    pd.concat(frames, ignore_index=True).to_csv(out_path, index=False)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
