"""audit-v2 regeneration prep: smoke-test the PPO training loop with the
new env (fallow fast-forward + actual-irrigation accounting) before
committing hours of full training. Uses seed=99 (main() only trains
seeds 0..n) and cleans up its artifacts."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "rl"))
sys.path.insert(0, str(ROOT / "src" / "data"))
sys.path.insert(0, str(ROOT / "src" / "sim"))

from train_rotation_compare import train

OUT_DIR = ROOT / "data" / "processed"


def main():
    print("=== smoke: train(mode='direct', total_timesteps=5000, seed=99) ===", flush=True)
    model_path, vecnorm_path = train("direct", total_timesteps=5000, seed=99)
    print("smoke OK:", model_path.name, vecnorm_path.name, flush=True)

    for p in OUT_DIR.glob("ppo_rotation_direct_seed99_*"):
        p.unlink()
    print("smoke artifacts cleaned", flush=True)


if __name__ == "__main__":
    main()
