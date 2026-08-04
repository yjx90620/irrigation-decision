"""audit-v2 P0-9: train the reward-rebalanced arm (water_norm='per_quota',
water penalty normalized by the 450mm annual quota instead of the 40mm
max action) - the audit's suggested fix for the measured ~10:1 water-vs-
yield reward imbalance. 6 models, 2 concurrent, CUDA. Artifacts land
under ppo_rotation_{mode}_wq{_seedN}_* so they never clobber the
per_action (primary) or gamma1 (ablation) arms.
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "data" / "processed" / "_regen_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

MODES = ["direct", "residual"]
SEEDS = [0, 1, 2]
CONCURRENCY = 2


def train_cmd(mode, seed):
    return [
        sys.executable, "-c",
        "import sys; sys.path.insert(0, 'src/rl'); sys.path.insert(0, 'src/data'); "
        "sys.path.insert(0, 'src/sim'); "
        "from train_rotation_compare import train; "
        f"train(mode='{mode}', seed={seed}, device='cuda', water_norm='per_quota'); "
        f"print('TRAINED wq {mode} seed={seed} OK', flush=True)",
    ]


def main():
    jobs = [(mode, seed) for mode in MODES for seed in SEEDS]
    procs = {}
    pending = list(jobs)
    while pending or procs:
        while len(procs) < CONCURRENCY and pending:
            mode, seed = pending.pop(0)
            name = f"wq_{mode}_seed{seed}"
            log = open(LOG_DIR / f"ppo_{name}.log", "w")
            proc = subprocess.Popen(train_cmd(mode, seed), cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            procs[name] = (proc, log)
            print(f"[{time.strftime('%H:%M:%S')}] launched {name} (pid {proc.pid})", flush=True)
        done = []
        for name, (proc, log) in procs.items():
            rc = proc.poll()
            if rc is not None:
                log.close()
                print(f"[{time.strftime('%H:%M:%S')}] {name} exited rc={rc}", flush=True)
                done.append(name)
        for name in done:
            del procs[name]
        if procs:
            time.sleep(30)
    print("ALL WQ TRAINING DONE", flush=True)


if __name__ == "__main__":
    main()
