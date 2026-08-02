"""Generic PPO training for RotationGymEnv with an arbitrary site subset -
train_rotation_compare.py's train() hardcodes all 5 sites (its module-level
env factories exist only so SubprocVecEnv's Windows spawn workers can
pickle them - closures can't be pickled there). This does the same thing
but for any site subset, using functools.partial over a module-level
factory instead of a per-call closure, which spawn CAN pickle since
partial's target is a plain module-level function and its bound args
(site list, mode string) are simple picklable values.

Used by src/transfer/leave_one_out_rotation.py so a leave-one-site-out
source model can be trained on an arbitrary 4-of-5 subset without writing
a new hardcoded factory per fold.
"""

import functools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from residual_gym_env import TRAIN_YEARS, RotationGymEnv

OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


def _make_env(rank, sites, mode, soils=("loam",), years=TRAIN_YEARS):
    # P0-4 (docs/审计修复计划.md): one fixed site per worker rank, not
    # domain-randomized per episode - see residual_gym_env.py's
    # RotationGymEnv.fixed_site docstring for why (episode-length-driven
    # transition-count skew across cropping systems).
    return RotationGymEnv(
        mode=mode, sites=list(sites), soils=list(soils), years=list(years), fixed_site=sites[rank % len(sites)],
    )


def train_rotation_policy(
    sites, mode, total_timesteps, run_name, workers_per_site=2, checkpoint_every=None,
    base_model_path=None, base_vecnormalize_path=None, ent_coef=0.01,
):
    n_envs = workers_per_site * len(sites)
    env_fns = [functools.partial(_make_env, rank, sites, mode) for rank in range(n_envs)]
    vec_env = SubprocVecEnv(env_fns)

    if base_model_path and base_vecnormalize_path:
        vec_env = VecNormalize.load(str(base_vecnormalize_path), vec_env)
        vec_env.training = True
        model = PPO.load(str(base_model_path), env=vec_env)
    else:
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
        model = PPO(
            "MlpPolicy", vec_env, verbose=0, n_steps=512, batch_size=256, n_epochs=10,
            learning_rate=3e-4, gamma=0.995, ent_coef=ent_coef, device="cpu",
        )

    callback = None
    if checkpoint_every:
        ckpt_dir = OUT_DIR / "ppo_checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        callback = CheckpointCallback(
            save_freq=max(checkpoint_every // n_envs, 1), save_path=str(ckpt_dir),
            name_prefix=run_name, save_vecnormalize=True,
        )

    model.learn(total_timesteps=total_timesteps, callback=callback, reset_num_timesteps=base_model_path is None)

    model_path = OUT_DIR / f"{run_name}_final.zip"
    vecnorm_path = OUT_DIR / f"{run_name}_final_vecnormalize.pkl"
    model.save(str(model_path))
    vec_env.save(str(vecnorm_path))
    vec_env.close()
    return model_path, vecnorm_path
