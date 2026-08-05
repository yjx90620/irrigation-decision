"""audit-v3 regeneration: PRIMARY arm (per_quota potential shaping,
gamma=0.995, CPU) - 6 models. Failures raise (2.3)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _regen_launcher import run_batch


def main():
    run_batch("RLExperimentConfig(device='cpu')", label="PPO TRAINING")


if __name__ == "__main__":
    main()
