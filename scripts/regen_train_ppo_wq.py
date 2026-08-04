"""audit-v2 P0-9 reward-rebalanced arm: train the 6 per_quota models.
audit-v3 (2.3): failures raise (see scripts/_regen_launcher.py)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _regen_launcher import run_batch


def main():
    run_batch("wq", train_kwargs=", device='cuda', water_norm='per_quota'", label="WQ TRAINING")


if __name__ == "__main__":
    main()
