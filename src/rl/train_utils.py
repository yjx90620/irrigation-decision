"""Reusable PPO training routine, factored out of train_ppo.py so
src/transfer/ can reuse the exact same training procedure (same
VecNormalize/ent_coef fixes) for source-domain and fine-tuning runs
instead of duplicating it.
"""

from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from gym_env import GymIrrigationEnv

OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


def train_policy(
    sites,
    soils,
    years,
    total_timesteps,
    run_name,
    n_envs=8,
    site_weights=None,
    checkpoint_every=None,
    base_model_path=None,
    base_vecnormalize_path=None,
    ent_coef=0.01,
):
    """Train (or fine-tune, if base_model_path is given) a PPO policy.
    Returns (model_path, vecnormalize_path)."""

    def make_env():
        return GymIrrigationEnv(sites=list(sites), soils=list(soils), years=list(years), site_weights=site_weights)

    vec_env = make_vec_env(make_env, n_envs=n_envs, vec_env_cls=SubprocVecEnv)
    if base_model_path and base_vecnormalize_path:
        vec_env = VecNormalize.load(str(base_vecnormalize_path), vec_env)
        vec_env.training = True
        model = PPO.load(str(base_model_path), env=vec_env)
    else:
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
        model = PPO(
            "MlpPolicy", vec_env, verbose=0, n_steps=512, batch_size=256, n_epochs=10,
            learning_rate=3e-4, gamma=0.995, ent_coef=ent_coef,
        )

    callback = None
    if checkpoint_every:
        checkpoint_dir = OUT_DIR / "ppo_checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        callback = CheckpointCallback(
            save_freq=max(checkpoint_every // n_envs, 1), save_path=str(checkpoint_dir),
            name_prefix=run_name, save_vecnormalize=True,
        )

    model.learn(total_timesteps=total_timesteps, callback=callback, reset_num_timesteps=base_model_path is None)

    model_path = OUT_DIR / f"{run_name}_final.zip"
    vecnorm_path = OUT_DIR / f"{run_name}_final_vecnormalize.pkl"
    model.save(str(model_path))
    vec_env.save(str(vecnorm_path))
    vec_env.close()
    return model_path, vecnorm_path
