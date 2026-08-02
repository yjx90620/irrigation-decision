"""Gymnasium wrappers over the rotation environment, in two variants that
differ only in how the action is interpreted (paper 2's core comparison):

- "direct": the action IS the irrigation depth, as in a standard RL setup.
- "residual": the action is a *correction* to a rule-based policy's
  decision (研究方案 5.9 extension). The agent starts at rule-level
  performance by construction and only has to learn where the rule is
  wrong, rather than rediscovering irrigation scheduling from scratch.

Why residual is worth testing here specifically: the rule baseline on the
rotation environment fails in a very legible way - it burns the whole
annual quota on wheat and leaves maize nothing (measured: wheat 5.43 t/ha
on 450mm, maize 2.72 on 0mm). The correction needed ("stop irrigating
wheat once the quota is running low") is easy to express as a residual
and hard to reach from a random initialization.

Domain randomization over site/soil/year/preference is unchanged from the
single-season version.
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from config import SITES
from rotation_env import (
    ACTIONS_MM, ANNUAL_QUOTA_MM, RotationIrrigationEnv, combine_reward, threshold_policy,
)
from soils import STANDARD_SOILS

STATE_KEYS = [
    "is_wheat", "dap", "growth_stage", "canopy_cover", "z_root", "biomass", "gdd_cum",
    "tr_ratio", "depletion_frac", "season_irr_cum", "days_since_last_irr", "last_irr_mm",
    "precip_next_3d", "precip_next_7d", "et0_next_7d", "hot_days_next_7d",
    "remaining_annual_quota",
]
PREFERENCE_KEYS = ["yield_proxy", "water", "cost", "risk"]

# Residual corrections applied to the base policy's depth, clipped into the
# valid depth range afterwards.
RESIDUAL_DELTAS = [-20.0, -10.0, 0.0, +10.0, +20.0]

TRAIN_YEARS = list(range(1982, 2011))  # 1982 not 1981: a rotation year needs the prior autumn


def _sample_preference(rng):
    raw = [rng.random() for _ in PREFERENCE_KEYS]
    total = sum(raw)
    return dict(zip(PREFERENCE_KEYS, [r / total for r in raw]))


class RotationGymEnv(gym.Env):
    def __init__(self, mode="direct", sites=None, soils=None, years=None, seed=None,
                 annual_quota=ANNUAL_QUOTA_MM, base_policy=threshold_policy, fixed_site=None):
        """fixed_site (P0-4, docs/审计修复计划.md): when set, every reset()
        uses this site instead of sampling from `sites`. Domain
        randomization was previously uniform per-episode over sites, but
        double-crop episodes run ~2.3x longer (~120 decision steps) than
        Ningxia's single-crop ones (~53), so per-episode-uniform sampling
        skews per-site *transition* counts even though episode counts are
        balanced. Pinning one site per parallel worker (see
        train_rotation_compare.py's env factories) and cycling workers
        through all sites evenly is the fix - year/soil/preference stay
        randomized within that site."""
        super().__init__()
        assert mode in ("direct", "residual")
        self.mode = mode
        self.sites = sites or list(SITES)
        self.fixed_site = fixed_site
        self.soils = soils or list(STANDARD_SOILS)
        self.years = years or TRAIN_YEARS
        self.annual_quota = annual_quota
        self.base_policy = base_policy
        self._rng = random.Random(seed)

        n_actions = len(RESIDUAL_DELTAS) if mode == "residual" else len(ACTIONS_MM)
        self.action_space = spaces.Discrete(n_actions)
        self.observation_space = spaces.Box(
            low=-1e6, high=1e6, shape=(len(STATE_KEYS) + len(PREFERENCE_KEYS),), dtype=np.float32
        )

    def _encode(self, state):
        return np.array(
            [float(state[k]) for k in STATE_KEYS] + [self.weights[k] for k in PREFERENCE_KEYS],
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        site_id = self.fixed_site or self._rng.choice(self.sites)
        soil_key = self._rng.choice(self.soils)
        year = self._rng.choice(self.years)
        self.weights = _sample_preference(self._rng)
        self.inner = RotationIrrigationEnv(site_id, soil_key, year, annual_quota=self.annual_quota)
        self._state = self.inner.reset()
        return self._encode(self._state), {}

    def _resolve_action(self, action_idx):
        if self.mode == "direct":
            return ACTIONS_MM[action_idx]
        base = self.base_policy(self._state, self.weights)
        return float(np.clip(base + RESIDUAL_DELTAS[action_idx], 0.0, max(ACTIONS_MM)))

    def step(self, action_idx):
        action_mm = self._resolve_action(int(action_idx))
        state, reward_vec, done, info = self.inner.step(action_mm)
        self._state = state
        reward = combine_reward(reward_vec, self.weights)
        info["reward_vector"] = reward_vec
        info["preference_weights"] = self.weights
        return self._encode(state), reward, done, False, info
