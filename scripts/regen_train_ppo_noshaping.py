"""audit-v3 ablation: no-shaping arm (shaping_mode='none' vs potential) -
isolates the PBRS shaping's contribution. Failures raise."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _regen_launcher import run_batch


def main():
    run_batch("RLExperimentConfig(shaping_mode='none', device='cuda')", label="NOSHAPING TRAINING")


if __name__ == "__main__":
    main()
