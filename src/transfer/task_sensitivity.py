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
import statistics
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))

import pandas as pd
from aquacrop import IrrigationManagement

from config import SITES
from rotation import run_rotation_series

# P1-1b (docs/审计修复计划.md): was range(2011,2021), which overlapped
# 2018-2020 with the RL/optimization experiments' held-out TEST_YEARS
# (2018-2022) - a "target-domain-independent" sensitivity metric was
# quietly built partly from the period it's meant to help predict
# transfer risk *for*. 2000-2009 is fully inside TRAIN_YEARS (1982-2010,
# residual_gym_env.py) and disjoint from every test period used anywhere
# in this project.
EVAL_YEARS = list(range(2000, 2010))  # 10 consecutive years, all in TRAIN_YEARS
COMBO_TIMEOUT_S = 90  # 10 years/policy normally takes ~10-15s; well past
# that and it's the same class of pathological state found in scan_allocation.py

OUT_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "task_sensitivity.csv"
# P1 (docs/审计修复计划.md): sys.executable, not a hardcoded venv path -
# the previous version only worked on Windows with a venv at this exact
# location relative to the repo.
PYTHON = sys.executable


def rainfed():
    return IrrigationManagement(irrigation_method=0)


def full_irrigation():
    return IrrigationManagement(irrigation_method=1, SMT=[100, 100, 100, 100])


def _run_policy(site_id: str, soil_key: str, policy: str) -> dict:
    """Per-year system yields under `policy` ('rainfed' or 'full'), as
    {year: yield_t_ha}. Runs in-process - this is the single-combo body
    invoked as a fresh subprocess by compute_sensitivity(), not called
    directly for a full site from the parent. Returning the full per-year
    series (not just the mean) is what lets the parent quantify how
    uncertain the sensitivity point estimate is (audit-v3 6.6)."""
    irr = rainfed if policy == "rainfed" else full_irrigation
    df = run_rotation_series(site_id, soil_key, EVAL_YEARS, irr, irr)
    return df.groupby("year")["dry_yield_t_ha"].sum().to_dict()


def _run_policy_subprocess(site_id: str, soil_key: str, policy: str):
    cmd = [str(PYTHON), str(Path(__file__).resolve()), "--site", site_id, "--soil", soil_key, "--policy", policy]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=COMBO_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        print(f"  {site_id} {policy}: TIMED OUT after {COMBO_TIMEOUT_S}s, skipped", flush=True)
        return None
    last_line = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
    if result.returncode == 0 and last_line.startswith("{"):
        payload = json.loads(last_line)
        return {int(y): v for y, v in payload["yields"].items()}
    print(f"  {site_id} {policy}: FAILED (exit {result.returncode}): {result.stderr[-200:]}", flush=True)
    return None


def compute_sensitivity(site_id: str, soil_key: str = "loam") -> dict:
    rain_years = _run_policy_subprocess(site_id, soil_key, "rainfed")
    full_years = _run_policy_subprocess(site_id, soil_key, "full")

    # P1 补充 (docs/审计修复计划.md): a legitimate rainfed_yield of exactly
    # 0.0 (complete crop failure under zero irrigation, plausible at the
    # driest sites) is falsy in Python, so `rain_yield and ...` used to
    # mark a real, computable, maximally-informative result (sensitivity
    # == 1.0) as missing. Check for None (subprocess failure/timeout)
    # explicitly instead of relying on truthiness.
    def _mean(years):
        return sum(years.values()) / len(years) if years else None

    rain_mean, full_mean = _mean(rain_years), _mean(full_years)
    sensitivity = (
        (full_mean - rain_mean) / full_mean
        if (full_mean is not None and rain_mean is not None and full_mean > 0)
        else float("nan")
    )

    # audit-v3 (6.6): the point estimate carries no uncertainty - report
    # the across-year spread of the annual sensitivity
    # (full_y - rain_y) / full_y so papers can quote sensitivity as
    # mean +/- spread instead of a bare number. Years where full
    # irrigation yields 0 (pathological) are dropped from the annual
    # series.
    annual = []
    for year in rain_years:
        if year not in full_years:
            continue
        full_y = full_years[year]
        if full_y is None or full_y <= 0:
            continue
        annual.append((full_y - rain_years[year]) / full_y)
    sensitivity_std = statistics.stdev(annual) if len(annual) >= 2 else float("nan")
    sensitivity_min = min(annual) if annual else float("nan")
    sensitivity_max = max(annual) if annual else float("nan")

    print(
        f"  {site_id}: rainfed={rain_mean} full={full_mean} sensitivity={sensitivity:.3f} "
        f"across-year std={sensitivity_std:.3f}", flush=True
    )
    return {
        "site_id": site_id, "soil": soil_key,
        "rainfed_yield": rain_mean, "full_irrigation_yield": full_mean,
        "sensitivity": sensitivity,
        "sensitivity_across_years_std": sensitivity_std,
        "sensitivity_across_years_min": sensitivity_min,
        "sensitivity_across_years_max": sensitivity_max,
        "n_years": len(annual),
    }


def main(site_id=None, soil_key="loam", policy=None):
    if policy is not None:
        # single-(site,policy) mode: print exactly one JSON line, nothing else
        print(json.dumps({"yields": _run_policy(site_id, soil_key, policy)}))
        return

    existing = pd.read_csv(OUT_PATH) if OUT_PATH.exists() else pd.DataFrame()
    # audit-v2 (P0-13): a row that merely exists is not "done" - a site
    # whose result has NaN/empty values must be recomputed, not skipped.
    if not existing.empty and "sensitivity" in existing.columns:
        done = set(existing.loc[existing["sensitivity"].notna(), "site_id"])
    else:
        done = set()

    rows = [] if existing.empty else [existing[existing["site_id"].isin(done)]]
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
