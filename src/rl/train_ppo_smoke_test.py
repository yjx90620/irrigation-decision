"""Smoke test: confirm PPO actually trains against GymIrrigationEnv end to
end (loss decreases, no crashes) - not a real training run. A proper run
needs far more timesteps across the full site x soil x year grid; see
docs/研究方案.md 5.9/5.10 for the intended algorithm/ablation set and
train/val/test year split.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from gym_env import GymIrrigationEnv

TRAIN_YEARS = list(range(1981, 2011))  # matches 研究方案 5.10 train split


def make_env():
    return GymIrrigationEnv(sites=["hebei_central"], soils=["loam"], years=TRAIN_YEARS)


def main():
    vec_env = make_vec_env(make_env, n_envs=1)
    model = PPO("MlpPolicy", vec_env, verbose=1, n_steps=256, batch_size=64)
    model.learn(total_timesteps=5000)
    out_path = Path(__file__).resolve().parents[2] / "data" / "processed" / "ppo_smoke_test.zip"
    model.save(str(out_path))
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
