"""Task sensitivity index: how much a site's system yield depends on
irrigation policy quality. This is the second factor in paper 3's
two-factor transfer risk predictor (环境距离 x 任务敏感度) - see
docs/系统升级方案.md and docs/papers/论文三_跨区域动态迁移.md.

The prototype-stage version of this idea used NSGA-II Pareto front width
as the sensitivity proxy. That's not available uniformly here: only the
double-cropping sites have a cross-season Pareto front
(optimize_cross_season.py), and Ningxia's single-crop system doesn't fit
that decision-variable structure at all. Sensitivity needs a metric that
applies to every site regardless of cropping system, so this uses the
gap between rainfed and full-irrigation system yield instead - cheap to
compute (2 rotation runs per site, no optimization needed) and it's the
same quantity paper 1's baseline analysis already leans on (data/README.md's
"雨养/充分灌溉产量对比").

sensitivity = (full_irrigation_yield - rainfed_yield) / full_irrigation_yield

High sensitivity = policy quality matters a lot at that site (e.g.
Ningxia) = a transferred policy's mistakes are expensive there. Low
sensitivity (e.g. Shaanxi Guanzhong) = mistakes are cheap because rainfed
already gets close to the ceiling, which is the mechanism paper 3's
existing finding needs (Shaanxi Guanzhong transferred well *despite* a
large environmental distance from its sources, because errors don't cost
much there).

The original version ran all 5 sites x 2 policies inline in one process
with no incremental saving and no timeout, and it hung: 15+ hours of one
CPU core pegged at ~100%, zero output. EVAL_YEARS is consecutive
(2011-2020, unlike scan_allocation.py's original gapped list), so this
wasn't that specific bug, but it's the same broader lesson -
multi-year continuous rotation under an *extreme* policy (rainfed = zero
irrigation, ever) can apparently still drive AquaCrop's solver into a
pathological state, not just year-gapped carryover. Given that, this now
runs each (site, policy) pair as its own subprocess under a wall-clock
timeout - the same defense-in-depth pattern scan_allocation.py ended up
needing - with incremental per-site saving so a stuck combo costs at most
one timeout instead of the whole run silently vanishing.
"""

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))

import pandas as pd
from aquacrop import IrrigationManagement

from config import SITES
from rotation import run_rotation_series

EVAL_YEARS = list(range(2011, 2021))  # 10 consecutive years
COMBO_TIMEOUT_S = 90  # 10 years/policy normally takes ~10-15s; well past
# that and it's the same class of pathological state found in scan_allocation.py

OUT_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "task_sensitivity.csv"
PYTHON = Path(__file__).resolve().parents[2] / ".venv" / "Scripts" / "python.exe"


def rainfed():
    return IrrigationManagement(irrigation_method=0)


def full_irrigation():
    return IrrigationManagement(irrigation_method=1, SMT=[100, 100, 100, 100])


def _run_policy(site_id: str, soil_key: str, policy: str) -> float:
    """Mean annual system yield under `policy` ('rainfed' or 'full'). Runs
    in-process - this is the single-combo body invoked as a fresh
    subprocess by compute_sensitivity(), not called directly for a full
    site from the parent."""
    irr = rainfed if policy == "rainfed" else full_irrigation
    df = run_rotation_series(site_id, soil_key, EVAL_YEARS, irr, irr)
    return df.groupby("year")["dry_yield_t_ha"].sum().mean()


def _run_policy_subprocess(site_id: str, soil_key: str, policy: str):
    cmd = [str(PYTHON), str(Path(__file__).resolve()), "--site", site_id, "--soil", soil_key, "--policy", policy]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=COMBO_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        print(f"  {site_id} {policy}: TIMED OUT after {COMBO_TIMEOUT_S}s, skipped", flush=True)
        return None
    last_line = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
    if result.returncode == 0 and last_line.startswith("{"):
        return json.loads(last_line)["yield"]
    print(f"  {site_id} {policy}: FAILED (exit {result.returncode}): {result.stderr[-200:]}", flush=True)
    return None


def compute_sensitivity(site_id: str, soil_key: str = "loam") -> dict:
    rain_yield = _run_policy_subprocess(site_id, soil_key, "rainfed")
    full_yield = _run_policy_subprocess(site_id, soil_key, "full")

    sensitivity = (
        (full_yield - rain_yield) / full_yield if (full_yield and rain_yield and full_yield > 0) else float("nan")
    )
    print(f"  {site_id}: rainfed={rain_yield} full={full_yield} sensitivity={sensitivity}", flush=True)
    return {
        "site_id": site_id, "soil": soil_key,
        "rainfed_yield": rain_yield, "full_irrigation_yield": full_yield,
        "sensitivity": sensitivity,
    }


def main(site_id=None, soil_key="loam", policy=None):
    if policy is not None:
        # single-(site,policy) mode: print exactly one JSON line, nothing else
        print(json.dumps({"yield": _run_policy(site_id, soil_key, policy)}))
        return

    existing = pd.read_csv(OUT_PATH) if OUT_PATH.exists() else pd.DataFrame()
    done = set(existing["site_id"]) if not existing.empty else set()

    rows = [] if existing.empty else [existing]
    for sid in SITES:
        if sid in done:
            print(f"skip {sid}, already computed")
            continue
        rows.append(pd.DataFrame([compute_sensitivity(sid)]))
        pd.concat(rows, ignore_index=True).sort_values("sensitivity").to_csv(OUT_PATH, index=False)

    print(f"saved -> {OUT_PATH}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--site", default=None)
    parser.add_argument("--soil", default="loam")
    parser.add_argument("--policy", default=None, choices=["rainfed", "full"])
    args = parser.parse_args()
    main(site_id=args.site, soil_key=args.soil, policy=args.policy)
