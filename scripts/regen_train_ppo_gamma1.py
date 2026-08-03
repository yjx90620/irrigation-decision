"""audit-v2 P0-9 ablation: train the 6 rotation-PPO models with gamma=1.0.

The regenerated gamma=0.995 policies collapsed toward water-minimizing
behavior (learning curves: irrigation 250->60mm, yield flat ~8.5 t/ha vs
the rules' 13-16), because the single terminal yield bonus is discounted
by gamma^T ~ 0.45 over a ~122-step episode while per-step water penalties
accumulate undiscounted. The audit's P0-9 requires comparing gamma=1.0.
Artifacts land under ppo_rotation_{mode}_gamma1{_seedN}_* so the two arms
never clobber each other.
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
        f"train(mode='{mode}', seed={seed}, gamma=1.0, device='cuda'); "
        f"print('TRAINED gamma1 {mode} seed={seed} OK', flush=True)",
    ]


def main():
    jobs = [(mode, seed) for mode in MODES for seed in SEEDS]
    procs = {}
    pending = list(jobs)
    while pending or procs:
        while len(procs) < CONCURRENCY and pending:
            mode, seed = pending.pop(0)
            name = f"gamma1_{mode}_seed{seed}"
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
    print("ALL GAMMA1 TRAINING DONE", flush=True)


if __name__ == "__main__":
    main()
