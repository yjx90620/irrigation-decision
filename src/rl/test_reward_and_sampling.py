"""P0-4 acceptance tests (docs/审计修复计划.md): per-site sampling balance
and the reward restructure (terminal-only yield payout, bounded dense
shaping instead of an unbounded per-step accumulation)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim"))

from residual_gym_env import ACTIONS_MM, RESIDUAL_DELTAS, RotationGymEnv
from rotation_env import quota_reserving_policy, threshold_policy


def test_residual_deltas_do_not_collapse_under_clipping():
    """P0-5: threshold_policy (the base policy residual corrects) only
    ever outputs 0 or 20mm - both must yield 5 distinct clipped actions,
    not fewer, or the residual policy has strictly fewer effectively
    reachable actions than its action space size suggests."""
    lo, hi = 0.0, max(ACTIONS_MM)
    for base in (0.0, 20.0):
        clipped = [max(lo, min(hi, base + d)) for d in RESIDUAL_DELTAS]
        assert len(set(clipped)) == len(RESIDUAL_DELTAS), (
            f"base={base}: deltas {RESIDUAL_DELTAS} collapse to {clipped}"
        )


def test_quota_reserving_policy_stops_wheat_before_exhausting_quota():
    state = {"depletion_frac": 0.9, "is_wheat": 1.0, "remaining_annual_quota": 100.0}
    # default wheat_reserve_frac=0.6 on a 450mm quota reserves 180mm for
    # maize - 100mm remaining is below that, so wheat should NOT irrigate
    assert quota_reserving_policy(state) == 0.0
    state["remaining_annual_quota"] = 400.0
    assert quota_reserving_policy(state) == 20.0
    # maize is never subject to the wheat-only reserve check
    maize_state = {"depletion_frac": 0.9, "is_wheat": 0.0, "remaining_annual_quota": 5.0}
    assert quota_reserving_policy(maize_state) == 20.0


def test_fixed_site_pins_every_episode():
    env = RotationGymEnv(mode="direct", fixed_site="ningxia_irrigation", years=[2015, 2016, 2017])
    for _ in range(5):
        env.reset()
        assert env.inner.site_id == "ningxia_irrigation"


def test_yield_bonus_paid_once_per_episode_not_once_per_crop():
    """Double-crop episode: the yield_proxy component of the reward
    vector should only show a large jump (the terminal payout) on the
    final step, not on the wheat->maize crop switch too - the old code
    added a yield term at both transitions. Pure shaping noise (day-to-day
    tr_ratio swings, including the deliberate potential-reset at the crop
    switch) empirically stays under ~0.25 in magnitude; a real terminal
    yield payout for a non-failed season is well above that - so the
    count of steps exceeding a threshold in between is the robust check,
    not bounding every single step under a tight constant."""
    LARGE_JUMP_THRESHOLD = 0.3
    env = RotationGymEnv(mode="direct", fixed_site="hebei_central", soils=["loam"], years=[2015])
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

    assert yield_components[-1] > LARGE_JUMP_THRESHOLD, "final step should include the terminal yield payout"
    large_jumps = [v for v in yield_components[:-1] if abs(v) > LARGE_JUMP_THRESHOLD]
    assert not large_jumps, (
        f"a non-terminal step's yield_proxy looks like a second yield payout (old per-crop-payout bug): {large_jumps}"
    )
