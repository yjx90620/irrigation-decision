"""audit-v2 P0-9 ablation evaluation: gamma=1.0 arm (superseded by the
config-driven scripts/evaluate_arm.py; kept as a thin wrapper)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_arm import main as _arm_main
from experiment_config import RLExperimentConfig


def main():
    _arm_main(RLExperimentConfig(gamma=1.0, seed=0, device="cuda", requested_timesteps=400_000))


if __name__ == "__main__":
    main()
