"""audit-v3 regeneration: per_quota (=annual_quota) arm - 6 models.

NOTE (audit-v3 7.x): the per_quota arm IS the PRIMARY arm now
(PRIMARY_CONFIG = annual_quota water normalizer), so this script is a
kept-for-history alias of regen_train_ppo.py - running either trains the
same 6 models. Failures raise (2.3)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _regen_launcher import run_batch


def main():
    # same config expression as regen_train_ppo.py (PRIMARY_CONFIG:
    # potential shaping, gamma=0.995, annual_quota normalizer, CPU)
    run_batch("RLExperimentConfig(device='cpu')", label="WQ (=PRIMARY) TRAINING")


if __name__ == "__main__":
    main()
