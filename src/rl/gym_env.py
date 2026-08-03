"""DEPRECATED (audit-v2 P0-12): single-season prototype Gym wrapper -
results invalidated (see docs/AUDIT_FIX_LOG.md); use
src/rl/residual_gym_env.py (rotation-era) instead.

Original docstring:
Gymnasium wrapper around IrrigationEnv for stable-baselines3 training.

Domain-randomizes site x soil x year x preference weights every episode
(研究方案 5.7: "训练过程中随机抽取不同偏好") so a single policy learns to
condition on the preference vector rather than overfitting one scenario.
Weather CSVs are cached at module level since reset() gets called once per
episode (~44 decision steps) and re-reading from disk every episode would
dominate training time.
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from config import END_YEAR, SITES, START_YEAR
from soils import STANDARD_SOILS

from env import ACTIONS_MM, IrrigationEnv, combine_reward

STATE_KEYS = [
    "dap",
    "growth_stage",
    "canopy_cover",
    "z_root",
    "biomass",
    "gdd_cum",
    "tr_ratio",
    "depletion_frac",
    "irr_cum",
    "days_since_last_irr",
    "last_irr_mm",
    "precip_next_3d",
    "precip_next_7d",
    "et0_next_7d",
    "hot_days_next_7d",
    "remaining_water_budget",
]
PREFERENCE_KEYS = ["yield_proxy", "water", "cost", "risk"]

_ENV_CACHE: dict[tuple[str, str, int], IrrigationEnv] = {}


def _get_env(site_id: str, soil_key: str, year: int) -> IrrigationEnv:
    key = (site_id, soil_key, year)
    if key not in _ENV_CACHE:
        _ENV_CACHE[key] = IrrigationEnv(site_id, soil_key, year)
    return _ENV_CACHE[key]


def _sample_preference(rng: random.Random) -> dict:
    raw = [rng.random() for _ in PREFERENCE_KEYS]
    total = sum(raw)
    return dict(zip(PREFERENCE_KEYS, [r / total for r in raw]))


class GymIrrigationEnv(gym.Env):
    def __init__(self, sites=None, soils=None, years=None, seed=None, site_weights=None):
        """site_weights: optional {site_id: weight} for non-uniform sampling
        (研究方案 7.1 实例加权迁移 - e.g. weighting source sites by inverse
        environmental distance to a transfer target). Sites not present in
        the dict get weight 0. Defaults to uniform sampling."""
        super().__init__()
        self.sites = sites or list(SITES)
        self.soils = soils or list(STANDARD_SOILS)
        self.years = years or list(range(START_YEAR, END_YEAR + 1))
        self._rng = random.Random(seed)
        self._site_weights = [site_weights.get(s, 0.0) for s in self.sites] if site_weights else None

        self.action_space = spaces.Discrete(len(ACTIONS_MM))
        obs_dim = len(STATE_KEYS) + len(PREFERENCE_KEYS)
        self.observation_space = spaces.Box(low=-1e6, high=1e6, shape=(obs_dim,), dtype=np.float32)

    def _encode(self, state: dict) -> np.ndarray:
        s = [float(state[k]) for k in STATE_KEYS]
        w = [self.weights[k] for k in PREFERENCE_KEYS]
        return np.array(s + w, dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        # Note: gymnasium's check_env() flags this as non-deterministic for a
        # fixed seed, because site/soil/year/preference draw from a RNG that
        # advances across episodes rather than resetting from `seed` each
        # call. That's intentional domain randomization (研究方案 5.7), not a
        # bug - a policy that only ever sees one fixed scenario per seed
        # wouldn't learn to condition on the preference vector.
        super().reset(seed=seed)
        # P0-6b (docs/审计修复计划.md): the note above is about *ordinary*
        # resets (seed=None) correctly continuing to advance self._rng - it
        # doesn't mean an explicitly-passed seed should be a no-op, which
        # it was before this fix (super().reset(seed=seed) only touches
        # gymnasium's unused self.np_random). Re-seeding only when the
        # caller actually asks for it keeps both properties true.
        if seed is not None:
            self._rng.seed(seed)
        if self._site_weights is not None:
            site_id = self._rng.choices(self.sites, weights=self._site_weights, k=1)[0]
        else:
            site_id = self._rng.choice(self.sites)
        soil_key = self._rng.choice(self.soils)
        year = self._rng.choice(self.years)
        self.weights = _sample_preference(self._rng)
        self.inner = _get_env(site_id, soil_key, year)
        state = self.inner.reset()
        return self._encode(state), {}

    def step(self, action_idx: int):
        action_mm = ACTIONS_MM[action_idx]
        state, reward_vec, done, info = self.inner.step(action_mm)
        reward = combine_reward(reward_vec, self.weights)
        obs = self._encode(state)
        info["reward_vector"] = reward_vec
        info["preference_weights"] = self.weights
        return obs, reward, done, False, info
