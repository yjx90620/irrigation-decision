"""Shared bounded-concurrency training launcher (audit-v3, 2.1/2.3).

Each arm script passes a config EXPRESSION (e.g.
"RLExperimentConfig(gamma=1.0, device='cuda')"); the launcher builds a
per-(mode,seed) subprocess that constructs the config and calls
train(mode, config, seed). audit-v3 (2.3): ANY job failure raises.
"""

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "data" / "processed" / "_regen_logs"
CONCURRENCY = 2

MODES = ["direct", "residual"]
SEEDS = [0, 1, 2]

_HEAD = (
    "import sys; sys.path.insert(0, 'src/rl'); sys.path.insert(0, 'src/data'); "
    "sys.path.insert(0, 'src/sim'); "
    "from experiment_config import RLExperimentConfig; "
    "from train_rotation_compare import train; "
)


def build_train_cmd(mode, seed, config_expr: str) -> list:
    return [
        sys.executable, "-c",
        _HEAD
        + f"cfg = {config_expr}; "
        + f"train(mode='{mode}', config=cfg, seed={seed}); "
        + f"print('TRAINED {mode} seed={seed} OK', flush=True)",
    ]


def run_batch(config_expr: str, label: str, modes=None, seeds=None, concurrency=CONCURRENCY) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    jobs = [(mode, seed) for mode in (modes or MODES) for seed in (seeds or SEEDS)]
    procs = {}
    pending = list(jobs)
    failures = []
    while pending or procs:
        while len(procs) < concurrency and pending:
            mode, seed = pending.pop(0)
            name = f"{mode}_seed{seed}"
            log = open(LOG_DIR / f"ppo_{name}.log", "w")
            proc = subprocess.Popen(build_train_cmd(mode, seed, config_expr), cwd=ROOT,
                                    stdout=log, stderr=subprocess.STDOUT)
            procs[name] = (proc, log)
            print(f"[{time.strftime('%H:%M:%S')}] launched {name} (pid {proc.pid})", flush=True)
        done = []
        for name, (proc, log) in procs.items():
            rc = proc.poll()
            if rc is not None:
                log.close()
                print(f"[{time.strftime('%H:%M:%S')}] {name} exited rc={rc}", flush=True)
                if rc != 0:
                    failures.append({"job": name, "returncode": rc,
                                     "log": str(LOG_DIR / f"ppo_{name}.log")})
                done.append(name)
        for name in done:
            del procs[name]
        if procs:
            time.sleep(30)
    if failures:
        raise RuntimeError(f"{len(failures)} {label} job(s) failed: {failures}")
    print(f"ALL {label} DONE", flush=True)
