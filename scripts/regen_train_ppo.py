"""audit-v2 regeneration: train all 6 rotation-PPO models (direct/residual
x seeds 0,1,2) with bounded concurrency (3 at a time - each training
spawns 10 env workers, 3 x 10 = 30 on a 36-core box). Uses the same
train() the comparison script calls, so the final artifacts land at the
exact paths train_rotation_compare.main() later looks up for evaluation.
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
CONCURRENCY = 3


def train_cmd(mode, seed):
    return [
        sys.executable, "-c",
        "import sys; sys.path.insert(0, 'src/rl'); sys.path.insert(0, 'src/data'); "
        "sys.path.insert(0, 'src/sim'); "
        "from train_rotation_compare import train; "
        f"train(mode='{mode}', seed={seed}); print('TRAINED {mode} seed={seed} OK', flush=True)",
    ]


def main():
    jobs = [(mode, seed) for mode in MODES for seed in SEEDS]
    procs = {}
    pending = list(jobs)
    while pending or procs:
        while len(procs) < CONCURRENCY and pending:
            mode, seed = pending.pop(0)
            name = f"{mode}_seed{seed}"
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
    print("ALL PPO TRAINING DONE", flush=True)


if __name__ == "__main__":
    main()
