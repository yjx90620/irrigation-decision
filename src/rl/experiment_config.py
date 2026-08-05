"""Single immutable RL experiment configuration (audit-v3, 任务 2.1).

The audit found gamma, shaping gamma, device and seed scattered across the
env, training scripts and model construction - the gamma=1.0 ablation arm
even trained PPO with gamma=1.0 while the env's potential shaping still
used a hardcoded 0.995, so the "single-factor gamma ablation" was
confounded. One config object now carries every knob an RL run needs;
training and the env read the SAME gamma/shaping/water-normalizer from it,
and every run's full config is saved as JSON with a config hash.

Strict ablations must vary exactly one field and keep everything else
(device, seed, env, scenario sequence, steps, architecture, normalizer)
identical.
"""

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

ShapingMode = Literal["none", "potential"]
WaterNormalizer = Literal["annual_quota", "max_action"]
Device = Literal["cpu", "cuda", "auto"]


@dataclass(frozen=True)
class RLExperimentConfig:
    gamma: float = 0.995
    shaping_mode: ShapingMode = "potential"
    water_normalizer: WaterNormalizer = "annual_quota"
    device: Device = "cpu"
    seed: int = 0
    deterministic_torch: bool = True
    # training budget: requested vs actual must both be recorded
    requested_timesteps: int | None = None
    actual_timesteps: int | None = None

    def validate(self) -> None:
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError(f"Invalid gamma: {self.gamma}")
        if self.shaping_mode not in ("none", "potential"):
            raise ValueError(f"Invalid shaping_mode: {self.shaping_mode}")
        if self.water_normalizer not in ("annual_quota", "max_action"):
            raise ValueError(f"Invalid water_normalizer: {self.water_normalizer}")
        if self.device not in ("cpu", "cuda", "auto"):
            raise ValueError(f"Invalid device: {self.device}")

    def __post_init__(self) -> None:
        # audit-v3 (2.1): an invalid config must not exist even momentarily -
        # validate at construction, not only when hashing/saving.
        self.validate()

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path) -> None:
        self.validate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(self.to_dict(), fh, ensure_ascii=False, indent=2, default=str)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def config_hash(self) -> str:
        self.validate()
        payload = json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def with_updates(self, **kwargs) -> "RLExperimentConfig":
        """Derive a strict-ablation variant; only the named field changes."""
        return RLExperimentConfig(**{**self.to_dict(), **kwargs})


# The paper-2 primary arm (audit-v2 P0-9 winner): potential shaping with a
# per_quota water normalizer. gamma is the SAME value PPO and the env
# shaping use - a strict single-factor ablation must vary only gamma via
# with_updates(gamma=...).
PRIMARY_CONFIG = RLExperimentConfig(
    gamma=0.995, shaping_mode="potential", water_normalizer="annual_quota",
    device="cpu", seed=0, requested_timesteps=400_000,
)
