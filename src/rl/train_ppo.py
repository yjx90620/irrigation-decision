"""DEPRECATED (audit-v2 P0-12): single-season prototype PPO training -
results invalidated (see docs/AUDIT_FIX_LOG.md); the rotation-era
training is src/rl/train_rotation_compare.py.

Original docstring:
Real (not smoke-test) PPO training run for the preference-conditioned
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

The actual training loop lives in train_utils.train_policy() so
src/transfer/'s leave-one-out and instance-weighted experiments can reuse
the exact same procedure instead of duplicating it.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))

from config import SITES
from soils import STANDARD_SOILS
from train_utils import train_policy

TRAIN_YEARS = list(range(1981, 2011))
N_ENVS = 8
TOTAL_TIMESTEPS = 1_000_000
RUN_NAME = "ppo_irrigation"  # change this per run so parallel/rerun outputs don't collide


def main():
    model_path, vecnorm_path = train_policy(
        sites=SITES, soils=STANDARD_SOILS, years=TRAIN_YEARS,
        total_timesteps=TOTAL_TIMESTEPS, run_name=RUN_NAME, n_envs=N_ENVS,
        checkpoint_every=50_000,
    )
    print(f"saved final model -> {model_path}")


if __name__ == "__main__":
    main()
