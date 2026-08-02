"""P0-4 acceptance tests (docs/审计修复计划.md): per-site sampling balance
and the reward restructure (terminal-only yield payout, bounded dense
shaping instead of an unbounded per-step accumulation)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim"))

from residual_gym_env import RotationGymEnv


def test_fixed_site_pins_every_episode():
    env = RotationGymEnv(mode="direct", fixed_site="ningxia_irrigation", years=[2015, 2016, 2017])
    for _ in range(5):
        env.reset()
        assert env.inner.site_id == "ningxia_irrigation"


def test_yield_bonus_paid_once_per_episode_not_once_per_crop():
    """Double-crop episode: the yield_proxy component of the reward
    vector should only show a large jump (the terminal payout) on the
    final step, not on the wheat->maize crop switch too - the old code
    added a yield term at both transitions."""
    env = RotationGymEnv(mode="direct", fixed_site="hebei_central", years=[2015])
    state, _ = env.reset()
    done = False
    yield_components = []
    n_steps = 0
    while not done:
        state, reward, done, truncated, info = env.step(2)  # ACTIONS_MM[2] == 20mm, mid-range
        yield_components.append(info["reward_vector"]["yield_proxy"])
        n_steps += 1
        if n_steps > 400:  # guard against an infinite loop if done never flips
            raise AssertionError("episode did not terminate")

    # exactly one step (the last) should carry the terminal payout; every
    # other step's yield_proxy is bounded shaping, not a yield fraction
    # (yield fractions are O(0.1-1.0); shaping terms are much smaller in
    # magnitude since they're bounded differences of tr_ratio in [0,1])
    assert yield_components[-1] > 0.15, "final step should include the terminal yield payout"
    non_terminal = yield_components[:-1]
    assert all(abs(v) < 0.15 for v in non_terminal), (
        f"a non-terminal step's yield_proxy looks like it includes a yield payout: {non_terminal}"
    )
