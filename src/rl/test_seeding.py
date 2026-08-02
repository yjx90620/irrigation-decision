"""P0-6b acceptance tests (docs/审计修复计划.md): reset(seed=...) must
actually control the domain-randomization draw (site/soil/year/
preference), not silently no-op while gymnasium's unused self.np_random
gets seeded instead."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim"))

from residual_gym_env import RotationGymEnv


def _scenario(env):
    return (env.inner.site_id, env.inner.soil_key, env.inner.year, tuple(env.weights.values()))


def test_same_seed_reproduces_scenario():
    env = RotationGymEnv(mode="direct")
    env.reset(seed=42)
    first = _scenario(env)
    env.reset(seed=42)
    second = _scenario(env)
    assert first == second


def test_different_seeds_can_diverge():
    env = RotationGymEnv(mode="direct")
    scenarios = set()
    for seed in range(20):
        env.reset(seed=seed)
        scenarios.add(_scenario(env))
    assert len(scenarios) > 1, "20 different seeds produced the same scenario every time - seeding isn't taking effect"


def test_unseeded_reset_keeps_advancing_not_repeating():
    """reset(seed=None) (the normal training-time call) must not reset
    to the same scenario every time - that would defeat domain
    randomization entirely."""
    env = RotationGymEnv(mode="direct", seed=7)
    env.reset(seed=None)
    first = _scenario(env)
    seen_different = False
    for _ in range(20):
        env.reset(seed=None)
        if _scenario(env) != first:
            seen_different = True
            break
    assert seen_different, "20 consecutive unseeded resets produced identical scenarios"


def test_direct_and_residual_paired_seed_match_scenario_sequence():
    """P0-5: direct and residual trained with the same seed must see the
    same scenario sequence (train_rotation_compare.py's requirement)."""
    direct_env = RotationGymEnv(mode="direct", seed=123)
    residual_env = RotationGymEnv(mode="residual", seed=123)
    for _ in range(5):
        direct_env.reset()
        residual_env.reset()
        assert _scenario(direct_env) == _scenario(residual_env)
