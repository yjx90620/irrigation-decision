"""Shared bounded-concurrency training launcher (audit-v3, 2.1/2.3).

Each arm script passes a config EXPRESSION (e.g.
"RLExperimentConfig(gamma=1.0, device='cuda')"); the launcher builds a
per-(mode,seed) subprocess that constructs the config and calls
train(mode, config, seed). audit-v3 (2.3): ANY job failure raises.

audit-v3 (re-run): per-job STALL DETECTION. The retrain reproduced the
known 'stuck straggler' pathology (one SubprocVecEnv worker pegged at
100% on a pathological AquaCrop site/year/state while the other workers
sit idle waiting - same signature run_training_with_watchdog.py documents).
A stalled job produces no new checkpoints, so: if a job runs longer than
CHECKPOINT_TIMEOUT_S without any new checkpoint for its run_name, kill its
whole process tree (taskkill /F /T) and relaunch it (bounded by
MAX_RESTARTS). A hard per-job wall-clock cap (HARD_CAP_S) bounds a job
that somehow keeps 'progressing' without checkpoints.
"""

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "data" / "processed" / "_regen_logs"
CKPT_DIR = ROOT / "data" / "processed" / "ppo_checkpoints"
CONCURRENCY = 2

# audit-v3: stall detection parameters (contended runs take ~3-4x longer
# than the ~15-20min checkpoint cadence, so the timeout is generous).
CHECKPOINT_TIMEOUT_S = 45 * 60  # no NEW checkpoint for 45 min -> stall
HARD_CAP_S = 4 * 60 * 60  # absolute per-job cap (normal job ~40 min)
MAX_RESTARTS = 3

MODES = ["direct", "residual"]
SEEDS = [0, 1, 2]

_HEAD = (
    "import sys; sys.path.insert(0, 'src'); sys.path.insert(0, 'src/rl'); "
    "sys.path.insert(0, 'src/data'); sys.path.insert(0, 'src/sim'); "
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


def run_name(mode, seed) -> str:
    return f"ppo_rotation_{mode}" if seed == 0 else f"ppo_rotation_{mode}_seed{seed}"


def latest_checkpoint_mtime(run: str) -> float:
    """Newest ppo_checkpoints file whose name starts with the run_name
    (excluding the final model - final.zip is not a checkpoint)."""
    best = 0.0
    for p in CKPT_DIR.glob(f"{run}_*_steps.zip"):
        best = max(best, p.stat().st_mtime)
    return best


def kill_tree(pid: int) -> None:
    """Kill the process and its whole tree (Windows taskkill /T); on
    failure the subprocess reaper handles leftovers."""
    subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                   capture_output=True, text=True)
    print(f"[{time.strftime('%H:%M:%S')}] killed tree of pid {pid}", flush=True)


def run_batch(config_expr: str, label: str, modes=None, seeds=None, concurrency=CONCURRENCY) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    jobs = [(mode, seed) for mode in (modes or MODES) for seed in (seeds or SEEDS)]
    # name -> {proc, log, run, launched_at, last_ckpt, restarts}
    procs = {}
    pending = list(jobs)
    failures = []
    while pending or procs:
        while len(procs) < concurrency and pending:
            mode, seed = pending.pop(0)
            name = f"{mode}_seed{seed}"
            run = run_name(mode, seed)
            log = open(LOG_DIR / f"ppo_{name}.log", "w")
            proc = subprocess.Popen(build_train_cmd(mode, seed, config_expr), cwd=ROOT,
                                    stdout=log, stderr=subprocess.STDOUT)
            procs[name] = {
                "proc": proc, "log": log, "run": run,
                "launched_at": time.time(), "last_ckpt": latest_checkpoint_mtime(run),
                "restarts": 0,
            }
            print(f"[{time.strftime('%H:%M:%S')}] launched {name} (pid {proc.pid})", flush=True)

        done = []
        for name, job in procs.items():
            proc, log = job["proc"], job["log"]
            rc = proc.poll()
            if rc is not None:
                log.close()
                print(f"[{time.strftime('%H:%M:%S')}] {name} exited rc={rc}", flush=True)
                if rc != 0:
                    failures.append({"job": name, "returncode": rc,
                                     "log": str(LOG_DIR / f"ppo_{name}.log")})
                done.append(name)
                continue

            # audit-v3 stall detection: no new checkpoint in too long, or a
            # hard wall-clock cap hit -> kill the tree and relaunch.
            now = time.time()
            latest = latest_checkpoint_mtime(job["run"])
            stalled = (now - max(latest, job["last_ckpt"])) > CHECKPOINT_TIMEOUT_S
            over_cap = (now - job["launched_at"]) > HARD_CAP_S
            if (stalled or over_cap) and job["restarts"] < MAX_RESTARTS:
                job["restarts"] += 1
                print(f"[{time.strftime('%H:%M:%S')}] {name} STALLED "
                      f"(no checkpoint {int((now - max(latest, job['last_ckpt'])) / 60)} min, "
                      f"restart {job['restarts']}/{MAX_RESTARTS})", flush=True)
                log.close()
                kill_tree(proc.pid)
                mode, seed = name.rsplit("_seed", 1)
                seed = int(seed)
                log = open(LOG_DIR / f"ppo_{name}.log", "w")
                new_proc = subprocess.Popen(build_train_cmd(mode, seed, config_expr), cwd=ROOT,
                                            stdout=log, stderr=subprocess.STDOUT)
                job.update({
                    "proc": new_proc, "log": log, "launched_at": now,
                    "last_ckpt": latest_checkpoint_mtime(job["run"]),
                })
                print(f"[{time.strftime('%H:%M:%S')}] relaunched {name} (pid {new_proc.pid})", flush=True)
            elif over_cap and job["restarts"] >= MAX_RESTARTS:
                failures.append({"job": name, "returncode": "stalled-3x",
                                 "log": str(LOG_DIR / f"ppo_{name}.log")})
                done.append(name)

        for name in done:
            procs.pop(name, None)
        if procs:
            time.sleep(30)
    if failures:
        raise RuntimeError(f"{len(failures)} {label} job(s) failed: {failures}")
    print(f"ALL {label} DONE", flush=True)
