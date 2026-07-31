"""Real (not smoke-test) PPO training run for the preference-conditioned
irrigation policy (研究方案 5.9/5.10).

Domain-randomizes over all 5 sites x 3 soils x the 1981-2010 train-year
split; validation (2011-2017) and test (2018-2025) years are held out for
evaluate_policy.py, never seen during training. Parallelized across
SubprocVecEnv workers since each env step is CPU-bound (AquaCrop is not
GPU-accelerated) and this machine has plenty of cores to spare.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from config import SITES
from soils import STANDARD_SOILS

from gym_env import GymIrrigationEnv

TRAIN_YEARS = list(range(1981, 2011))
N_ENVS = 8
TOTAL_TIMESTEPS = 1_000_000

OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
CHECKPOINT_DIR = OUT_DIR / "ppo_checkpoints"


def make_env():
    return GymIrrigationEnv(sites=list(SITES), soils=list(STANDARD_SOILS), years=TRAIN_YEARS)


def main():
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    vec_env = make_vec_env(make_env, n_envs=N_ENVS, vec_env_cls=SubprocVecEnv)

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        n_steps=512,
        batch_size=256,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.995,
    )
    checkpoint_cb = CheckpointCallback(
        save_freq=max(50_000 // N_ENVS, 1), save_path=str(CHECKPOINT_DIR), name_prefix="ppo_irrigation"
    )
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=checkpoint_cb, progress_bar=False)

    final_path = OUT_DIR / "ppo_irrigation_final.zip"
    model.save(str(final_path))
    print(f"saved final model -> {final_path}")


if __name__ == "__main__":
    main()
