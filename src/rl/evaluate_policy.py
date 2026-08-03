"""DEPRECATED (audit-v2 P0-12): single-season prototype evaluation
harness - its results were invalidated (see docs/AUDIT_FIX_LOG.md); the
rotation-era comparison lives in src/rl/train_rotation_compare.py.

Original docstring:
Evaluation harness (研究方案 5.12): run any policy across a site x soil x
year x preference grid and report metrics comparable to the baseline
experiment grid (data/processed/baseline_experiment_results.csv) and the
NSGA-II fronts, so RL policies can be judged against both.

A "policy" here is any callable `policy_fn(state: dict, weights: dict) ->
action_mm: float`. load_ppo_policy() adapts a saved stable-baselines3 model
to that interface; threshold_policy() is a simple non-RL baseline used to
sanity-check the harness itself.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from env import ACTIONS_MM, IrrigationEnv, combine_reward


def _find_vecnormalize_path(model_path: Path) -> Path | None:
    """Match train_ppo.py's save conventions: "{run}_final.zip" pairs with
    "{run}_final_vecnormalize.pkl"; CheckpointCallback's
    "{prefix}_{steps}_steps.zip" pairs with "{prefix}_vecnormalize_{steps}_steps.pkl".
    Returns None for models saved without VecNormalize (e.g. the pre-fix
    checkpoints, or ppo_smoke_test.zip) so callers can fall back gracefully."""
    if model_path.stem.endswith("_final"):
        candidate = model_path.with_name(model_path.stem + "_vecnormalize.pkl")
        return candidate if candidate.exists() else None

    match = re.match(r"(.+)_(\d+)_steps$", model_path.stem)
    if match:
        prefix, steps = match.groups()
        candidate = model_path.with_name(f"{prefix}_vecnormalize_{steps}_steps.pkl")
        return candidate if candidate.exists() else None

    return None


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
    from stable_baselines3.common.vec_env import VecNormalize
    from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv

    from gym_env import GymIrrigationEnv, PREFERENCE_KEYS, STATE_KEYS

    model = PPO.load(model_path)

    vecnorm_path = _find_vecnormalize_path(Path(model_path))
    obs_rms = None
    if vecnorm_path is not None:
        # DummyVecEnv here is just a container VecNormalize.load() requires -
        # it's never stepped, only used to read back obs_rms.mean/var.
        dummy_venv = DummyVecEnv([lambda: GymIrrigationEnv(sites=["hebei_central"], soils=["loam"], years=[2015])])
        obs_rms = VecNormalize.load(str(vecnorm_path), dummy_venv).obs_rms

    def policy_fn(state: dict, weights: dict) -> float:
        obs = np.array([float(state[k]) for k in STATE_KEYS] + [weights[k] for k in PREFERENCE_KEYS], dtype=np.float32)
        if obs_rms is not None:
            obs = np.clip((obs - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8), -10.0, 10.0).astype(np.float32)
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
