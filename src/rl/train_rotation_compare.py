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

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from config import SITES
from residual_gym_env import (
    PREFERENCE_KEYS, RESIDUAL_DELTAS, STATE_KEYS, TRAIN_YEARS, RotationGymEnv,
)
from rotation_env import ACTIONS_MM, RotationIrrigationEnv, combine_reward, threshold_policy
from soils import STANDARD_SOILS

TEST_YEARS = [2018, 2019, 2020, 2021, 2022]
BALANCED_WEIGHTS = {"yield_proxy": 0.4, "water": 0.3, "cost": 0.2, "risk": 0.1}
N_ENVS = 8
TOTAL_TIMESTEPS = 400_000

OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


# Module-level factories, not closures: SubprocVecEnv pickles these to the
# worker processes, and Windows' spawn start method can't pickle a closure.
def make_direct_env():
    return RotationGymEnv(mode="direct", sites=list(SITES), soils=["loam"], years=TRAIN_YEARS)


def make_residual_env():
    return RotationGymEnv(mode="residual", sites=list(SITES), soils=["loam"], years=TRAIN_YEARS)


ENV_FACTORIES = {"direct": make_direct_env, "residual": make_residual_env}


def train(mode, total_timesteps=TOTAL_TIMESTEPS, n_envs=N_ENVS):
    run_name = f"ppo_rotation_{mode}"
    vec_env = make_vec_env(ENV_FACTORIES[mode], n_envs=n_envs, vec_env_cls=SubprocVecEnv)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    model = PPO(
        "MlpPolicy", vec_env, verbose=1, n_steps=512, batch_size=256, n_epochs=10,
        learning_rate=3e-4, gamma=0.995, ent_coef=0.01,
        device="cpu",  # measured: GPU gives ~1.12x here and SB3 warns against it for MlpPolicy
    )
    checkpoint_dir = OUT_DIR / "ppo_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.learn(
        total_timesteps=total_timesteps,
        callback=CheckpointCallback(
            save_freq=max(100_000 // n_envs, 1), save_path=str(checkpoint_dir),
            name_prefix=run_name, save_vecnormalize=True,
        ),
    )
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
            done, n_steps, n_mod, ret = False, 0, 0, 0.0
            while not done:
                state, reward, done, info = env.step(policy_fn(state, BALANCED_WEIGHTS))
                ret += combine_reward(reward, BALANCED_WEIGHTS)
                n_steps += 1
                n_mod += int(info["action_modified"])
            # Single-crop sites (Ningxia) have no "wheat" key, so report
            # per-crop columns only for the crops that site actually grows.
            row = {
                "policy": label, "site_id": site_id, "year": year,
                "total_yield_t_ha": info["total_yield_t_ha"],
                "total_irrigation_mm": info["total_irrigation_mm"],
                "action_modified_rate": n_mod / n_steps,
                "scalar_return": ret,
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
