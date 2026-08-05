"""audit-v2 P0-9 ablation evaluation: per_quota reward-rebalanced arm
(superseded by the config-driven scripts/evaluate_arm.py; kept as a thin
wrapper that evaluates the PRIMARY config)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_arm import main as _arm_main
from experiment_config import PRIMARY_CONFIG


def main():
    _arm_main(PRIMARY_CONFIG)


if __name__ == "__main__":
    main()
