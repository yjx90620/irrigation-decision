"""Run train_rotation_compare.py's training under an external watchdog that
kills and restarts on a stall, regardless of root cause.

Two rounds of fixing the *specific* trigger (year-gap soil carryover,
then a critical-depletion safety floor) each failed to fully close the
gap - the second one still left one SubprocVecEnv worker pegged at 100%
CPU while the other seven sat idle waiting on it for 20+ continuous
seconds, the classic "stuck straggler blocks the lockstep rollout"
signature. Chasing a third specific fix has diminishing odds of being
the last one needed, so this is a general safety net instead: if no new
checkpoint file appears within STALL_TIMEOUT_S, treat the run as stuck,
kill the whole process tree, and start over. Each restart discards
progress since the last checkpoint (no mid-run resume - keeping this
simple rather than half-building resume logic under time pressure), but
that bounds the maximum wasted time to one checkpoint interval instead of
an entire unattended run silently going nowhere.
"""

import subprocess
import sys
import time
from pathlib import Path

CHECKPOINT_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "ppo_checkpoints"
PYTHON = Path(__file__).resolve().parents[2] / ".venv" / "Scripts" / "python.exe"
TRAIN_SCRIPT = Path(__file__).resolve().parent / "train_rotation_compare.py"

STALL_TIMEOUT_S = 900  # 15 min - a healthy checkpoint interval is ~7-11 min
POLL_INTERVAL_S = 30
MAX_ATTEMPTS = 8


def latest_checkpoint_mtime(run_names) -> float:
    times = [
        f.stat().st_mtime
        for run_name in run_names
        for f in CHECKPOINT_DIR.glob(f"{run_name}_*_steps.zip")
    ]
    return max(times) if times else 0.0


def kill_process_tree(proc: subprocess.Popen):
    subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)


def run_with_watchdog():
    run_names = ["ppo_rotation_direct", "ppo_rotation_residual"]
    final_models = [
        Path(__file__).resolve().parents[2] / "data" / "processed" / f"{name}_final.zip" for name in run_names
    ]

    for attempt in range(1, MAX_ATTEMPTS + 1):
        if all(p.exists() for p in final_models):
            print("both models already trained, nothing to do")
            return

        print(f"=== attempt {attempt}/{MAX_ATTEMPTS} ===", flush=True)
        proc = subprocess.Popen([str(PYTHON), str(TRAIN_SCRIPT)])
        last_progress_time = time.time()
        last_seen_mtime = latest_checkpoint_mtime(run_names)

        while proc.poll() is None:
            time.sleep(POLL_INTERVAL_S)
            current_mtime = latest_checkpoint_mtime(run_names)
            if current_mtime > last_seen_mtime:
                last_seen_mtime = current_mtime
                last_progress_time = time.time()
                print(f"  progress: new checkpoint at {time.ctime(current_mtime)}", flush=True)
            elif time.time() - last_progress_time > STALL_TIMEOUT_S:
                print(f"  STALLED: no new checkpoint in {STALL_TIMEOUT_S}s, killing and restarting", flush=True)
                kill_process_tree(proc)
                break
        else:
            rc = proc.returncode
            print(f"  process exited on its own with code {rc}", flush=True)
            if rc == 0:
                continue  # loop will see final_models exist and stop, or move to next mode

    print("gave up after max attempts" if not all(p.exists() for p in final_models) else "done")


if __name__ == "__main__":
    run_with_watchdog()
