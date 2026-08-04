"""audit-v3 ablation: gamma=1.0 arm (single-factor: only gamma changes,
CUDA for exploration only). Failures raise."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _regen_launcher import run_batch


def main():
    run_batch("RLExperimentConfig(gamma=1.0, device='cuda')", label="GAMMA1 TRAINING")


if __name__ == "__main__":
    main()
