"""audit-v2 P0-9: quantify the reward-scale imbalance - how much of the
episode return comes from per-step water penalties vs the single terminal
yield bonus (and the potential-shaping term), for a trained policy."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "rl"))
sys.path.insert(0, str(ROOT / "src" / "data"))
sys.path.insert(0, str(ROOT / "src" / "sim"))

import numpy as np
import pandas as pd

from rotation_env import ACTIONS_MM, YIELD_REFERENCE, RotationIrrigationEnv, SHAPING_GAMMA
from train_rotation_compare import BALANCED_WEIGHTS, load_policy

OUT_DIR = ROOT / "data" / "processed"

model_path = OUT_DIR / "ppo_rotation_direct_final.zip"
vecnorm_path = OUT_DIR / "ppo_rotation_direct_final_vecnormalize.pkl"
policy_fn = load_policy(model_path, vecnorm_path, "direct")

env = RotationIrrigationEnv("hebei_central", "loam", 2019)
state = env.reset()
tot_water_penalty = 0.0
tot_shaping = 0.0
terminal_yield_bonus = 0.0
n_steps = 0
done = False
while not done:
    state, reward, done, info = env.step(policy_fn(state, BALANCED_WEIGHTS))
    w = BALANCED_WEIGHTS
    tot_water_penalty += w["water"] * reward["water"]
    tot_shaping += w["yield_proxy"] * reward["yield_proxy"]
    if done:
        # reconstruct the terminal yield bonus as the env pays it
        for crop in ("wheat", "maize", "spring_maize"):
            if crop in info:
                terminal_yield_bonus += info[crop]["dry_yield_t_ha"] / YIELD_REFERENCE[crop]
        terminal_yield_bonus *= w["yield_proxy"]
    n_steps += 1

print(f"hebei 2019, ppo_direct (gamma=0.995), {n_steps} steps:")
print(f"  weighted water penalty total : {tot_water_penalty:+.3f}  (per step mean {tot_water_penalty/n_steps:+.4f})")
print(f"  weighted shaping total       : {tot_shaping:+.3f}")
print(f"  weighted terminal yield bonus: {terminal_yield_bonus:+.3f}")
print(f"  gamma^T discount on terminal : {SHAPING_GAMMA**n_steps:.3f} (so discounted yield bonus ~ {SHAPING_GAMMA**n_steps * terminal_yield_bonus:+.3f})")
print(f"  total yield {info['total_yield_t_ha']:.2f} t/ha, irrigation {info['total_irrigation_mm']:.0f} mm")
