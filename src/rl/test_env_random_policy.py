"""Smoke test: run one full episode with a random policy to confirm the
environment mechanics (state transitions, action application, episode
termination, terminal yield) work end to end."""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from env import ACTIONS_MM, IrrigationEnv, combine_reward

WEIGHTS = {"yield_proxy": 0.4, "water": 0.3, "cost": 0.2, "risk": 0.1}


def main():
    env = IrrigationEnv("hebei_central", "loam", 2020)
    state = env.reset()
    total_scalar_reward = 0.0
    n_steps = 0
    done = False
    while not done:
        action = random.choice(ACTIONS_MM)
        state, reward, done, info = env.step(action)
        total_scalar_reward += combine_reward(reward, WEIGHTS)
        n_steps += 1
    print(f"episode finished in {n_steps} decision steps")
    print(f"final dry yield: {info['dry_yield_t_ha']:.2f} t/ha")
    print(f"seasonal irrigation: {info['seasonal_irrigation_mm']:.1f} mm")
    print(f"scalarized return (random policy): {total_scalar_reward:.2f}")


if __name__ == "__main__":
    main()
