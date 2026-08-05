"""audit-v3 ablation: per_action water normalizer arm (the audit-v2
P0-9 diagnosis arm: water penalty /max_action, showing the 10:1 scale
imbalance). Failures raise."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _regen_launcher import run_batch


def main():
    run_batch("RLExperimentConfig(water_normalizer='max_action', device='cuda')",
              label="PERACTION TRAINING")


if __name__ == "__main__":
    main()
