"""Evaluation harness (研究方案 5.12): run any policy across a site x soil x
year x preference grid and report metrics comparable to the baseline
experiment grid (data/processed/baseline_experiment_results.csv) and the
NSGA-II fronts, so RL policies can be judged against both.

A "policy" here is any callable `policy_fn(state: dict, weights: dict) ->
action_mm: float`. load_ppo_policy() adapts a saved stable-baselines3 model
to that interface; threshold_policy() is a simple non-RL baseline used to
sanity-check the harness itself.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from env import ACTIONS_MM, IrrigationEnv, combine_reward


def run_episode(site_id: str, soil_key: str, year: int, weights: dict, policy_fn) -> dict:
    env = IrrigationEnv(site_id, soil_key, year)
    state = env.reset()
    n_steps = 0
    n_modified = 0
    scalar_return = 0.0
    done = False
    while not done:
        action = policy_fn(state, weights)
        state, reward, done, info = env.step(action)
        scalar_return += combine_reward(reward, weights)
        n_steps += 1
        n_modified += int(info["action_modified"])

    return {
        "site_id": site_id,
        "soil": soil_key,
        "year": year,
        **{f"weight_{k}": v for k, v in weights.items()},
        "dry_yield_t_ha": info["dry_yield_t_ha"],
        "irrigation_mm": info["seasonal_irrigation_mm"],
        "n_decision_steps": n_steps,
        "action_modified_rate": n_modified / n_steps,
        "scalar_return": scalar_return,
    }


def evaluate_grid(sites, soils, years, weights_list, policy_fn) -> pd.DataFrame:
    rows = []
    for site_id in sites:
        for soil_key in soils:
            for year in years:
                for weights in weights_list:
                    rows.append(run_episode(site_id, soil_key, year, weights, policy_fn))
    return pd.DataFrame(rows)


def threshold_policy(state: dict, weights: dict) -> float:
    """Non-RL sanity baseline: irrigate 20mm once depletion exceeds 40% of TAW."""
    return 20.0 if state["depletion_frac"] > 0.4 else 0.0


def load_ppo_policy(model_path: str):
    from stable_baselines3 import PPO

    from gym_env import PREFERENCE_KEYS, STATE_KEYS

    model = PPO.load(model_path)

    def policy_fn(state: dict, weights: dict) -> float:
        obs = np.array([float(state[k]) for k in STATE_KEYS] + [weights[k] for k in PREFERENCE_KEYS], dtype=np.float32)
        action_idx, _ = model.predict(obs, deterministic=True)
        return ACTIONS_MM[int(action_idx)]

    return policy_fn


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=["threshold", "ppo"], default="threshold")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--sites", nargs="+", default=["hebei_central"])
    parser.add_argument("--soils", nargs="+", default=["loam"])
    parser.add_argument("--years", nargs="+", type=int, default=[2018, 2019, 2020])
    args = parser.parse_args()

    policy_fn = threshold_policy if args.policy == "threshold" else load_ppo_policy(args.model_path)
    balanced_weights = [{"yield_proxy": 0.4, "water": 0.3, "cost": 0.2, "risk": 0.1}]
    df = evaluate_grid(args.sites, args.soils, args.years, balanced_weights, policy_fn)
    print(df.to_string(index=False))
