"""audit-v3 stall probe: find which (site, year) combos drive the rotation
env into a pathological AquaCrop state (the 'stuck straggler': a worker
pegged at 100% while the rest idle at the lockstep barrier).

Each (site, year in TRAIN_YEARS) episode runs in a fresh subprocess with a
wall-clock timeout; a fixed moderate action schedule (10mm every decision
day) exercises irrigation without policy complexity. Combos that do not
finish an episode within the timeout are reported - those are the ones to
exclude from training or patch.

Usage: python scripts/_probe_env_stalls.py
"""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "rl"))
sys.path.insert(0, str(ROOT / "src" / "data"))
sys.path.insert(0, str(ROOT / "src" / "sim"))

from config import SITES
from residual_gym_env import TRAIN_YEARS

TIMEOUT_S = 90
HARD_ACTIONS = [0.0, 10.0, 20.0]  # test a few action levels


def _probe_one(site_id: str, year: int, action: float) -> bool:
    cmd = [sys.executable, str(Path(__file__).resolve()),
           "--site", site_id, "--year", str(year), "--action", str(action)]
    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_S)
        return True
    except subprocess.TimeoutExpired:
        return False


def main():
    bad = []
    t0 = time.time()
    for site_id in SITES:
        for year in TRAIN_YEARS:
            for action in HARD_ACTIONS:
                ok = _probe_one(site_id, year, action)
                status = "ok" if ok else "HANG"
                if not ok:
                    bad.append((site_id, year, action))
                print(f"{site_id:22s} {year} action={action:>4} -> {status}", flush=True)
    print(f"\nHANGING combos ({len(bad)}):")
    for site_id, year, action in bad:
        print(f"  {site_id} {year} action={action}")
    print(f"probe finished in {time.time() - t0:.0f}s")
    return 1 if bad else 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--site", default=None)
    parser.add_argument("--year", type=int, default=None)
    parser.add_argument("--action", type=float, default=None)
    args = parser.parse_args()

    if args.site is not None and args.year is not None and args.action is not None:
        # single-combo subprocess mode: run one episode, exit 0 on finish
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "rl"))
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "data"))
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "sim"))
        from experiment_config import PRIMARY_CONFIG
        from rotation_env import RotationIrrigationEnv

        env = RotationIrrigationEnv(args.site, "loam", args.year, config=PRIMARY_CONFIG)
        state = env.reset()
        n = 0
        done = False
        while not done and n < 400:
            state, reward, done, info = env.step(args.action)
            n += 1
        print(json.dumps({"steps": n, "yield": info["total_yield_t_ha"]}))
        raise SystemExit(0)

    raise SystemExit(main())
