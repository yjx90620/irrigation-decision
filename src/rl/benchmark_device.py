"""Fair CPU vs GPU throughput benchmark for this project's PPO setup.

AquaCrop env stepping is plain numpy on CPU with no GPU path - the
question is whether moving the (small) policy/value MLP to the RTX 3090
actually speeds up the *overall* rollout+update loop, or whether env
rollout collection dominates wall time enough that GPU transfer overhead
makes it a wash or worse.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from gym_env import GymIrrigationEnv

N_ENVS = 8
BENCH_TIMESTEPS = 20_000


def make_env():
    return GymIrrigationEnv(sites=["hebei_central"], soils=["loam"], years=list(range(2010, 2020)))


def run(device: str) -> float:
    vec_env = make_vec_env(make_env, n_envs=N_ENVS, vec_env_cls=SubprocVecEnv)
    model = PPO("MlpPolicy", vec_env, verbose=0, n_steps=512, batch_size=256, n_epochs=10, device=device)
    t0 = time.time()
    model.learn(total_timesteps=BENCH_TIMESTEPS)
    elapsed = time.time() - t0
    vec_env.close()
    fps = BENCH_TIMESTEPS / elapsed
    print(f"device={device:5s} elapsed={elapsed:6.1f}s fps={fps:7.1f}")
    return fps


if __name__ == "__main__":
    fps_cpu = run("cpu")
    fps_gpu = run("cuda")
    print(f"\nspeedup (gpu/cpu): {fps_gpu / fps_cpu:.2f}x")
