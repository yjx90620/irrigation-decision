"""Real (not smoke-test) PPO training run for the preference-conditioned
irrigation policy (研究方案 5.9/5.10).

Domain-randomizes over all 5 sites x 3 soils x the 1981-2010 train-year
split; validation (2011-2017) and test (2018-2025) years are held out for
evaluate_policy.py, never seen during training. Parallelized across
SubprocVecEnv workers since each env step is CPU-bound (AquaCrop is not
GPU-accelerated) and this machine has plenty of cores to spare.

Observation normalization (VecNormalize) matters here more than usual:
state features span ~600x in scale (tr_ratio/depletion_frac are 0-1,
biomass/gdd_cum/remaining_water_budget are in the hundreds), and a first
attempt without it got stuck at "never irrigate" even in scenarios with
severe, unambiguous water stress (checked across 12 checkpoints spanning
50k-600k steps via reconstruct_learning_curve.py - identical zero action
throughout, despite entropy_loss staying healthy in the raw training log,
meaning the *policy mode* was stuck even though it hadn't fully collapsed).
The working theory: without normalization, gradients from the
large-magnitude-but-only-weakly-informative features (biomass, gdd_cum -
both roughly just proxies for calendar time) drown out the small-scale
but decision-critical ones (depletion_frac, tr_ratio) early in training,
before the network ever learns to condition on them.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from config import SITES
from soils import STANDARD_SOILS

from gym_env import GymIrrigationEnv

TRAIN_YEARS = list(range(1981, 2011))
N_ENVS = 8
TOTAL_TIMESTEPS = 1_000_000
RUN_NAME = "ppo_irrigation"  # change this per run so parallel/rerun outputs don't collide

OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
CHECKPOINT_DIR = OUT_DIR / "ppo_checkpoints"


def make_env():
    return GymIrrigationEnv(sites=list(SITES), soils=list(STANDARD_SOILS), years=TRAIN_YEARS)


def main():
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    vec_env = make_vec_env(make_env, n_envs=N_ENVS, vec_env_cls=SubprocVecEnv)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        n_steps=512,
        batch_size=256,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.995,
        ent_coef=0.01,  # SB3 defaults to 0.0; see module docstring on why this
        # alone wasn't enough without also normalizing observations.
    )
    checkpoint_cb = CheckpointCallback(
        save_freq=max(50_000 // N_ENVS, 1),
        save_path=str(CHECKPOINT_DIR),
        name_prefix=RUN_NAME,
        save_vecnormalize=True,  # each checkpoint gets a matching *_vecnormalize_*.pkl
    )
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=checkpoint_cb, progress_bar=False)

    final_path = OUT_DIR / f"{RUN_NAME}_final.zip"
    model.save(str(final_path))
    vec_env.save(str(OUT_DIR / f"{RUN_NAME}_final_vecnormalize.pkl"))
    print(f"saved final model -> {final_path}")


if __name__ == "__main__":
    main()
