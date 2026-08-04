"""1-D scan of the wheat/maize annual-quota split (alpha) per site and year.

Motivation: the first joint-vs-fixed-split comparison found joint
optimization *losing* to a control that pins alpha=0.5. A diagnostic
showed why - alpha matters a lot mechanically (total yield at Hebei swung
13.68 / 15.56 / 14.54 t/ha across alpha = 0.2 / 0.5 / 0.8 with SMT held
fixed, and the MaxIrrSeason caps do bind), but 0.5 happens to sit near the
optimum *at that site*, so the "control" was accidentally a strong
baseline while the joint optimizer had to search a 9-dim space on the same
evaluation budget.

That makes the more fundamental question this: does the optimal split vary
enough across sites and years that no single fixed split can serve? If it
does, allocation flexibility is worth having and the comparison should be
against the *best* fixed split rather than an arbitrary one. If it
doesn't, the honest conclusion is that a fixed rule suffices - which is
itself a useful, publishable result.

This scan is cheap (1-D, no evolutionary search) and answers that directly.

ROOT CAUSE, finally isolated: this scan's YEARS list is a non-consecutive
sample ([2011,2013,2015,2017,2018,2020]), but it was being fed through
rotation.run_rotation_series(), which carries the soil-water profile
continuously from one listed year to the next - so going from 2011 to
2013 handed the 2013 wheat season a soil profile from Oct 2011, as if the
entire unsimulated calendar year 2012 hadn't happened. That's not just
physically dubious, it's computationally pathological: isolated testing
at Beijing showed run_rotation_series(['2011','2013'], ...) (the gapped
pair) hangs indefinitely, while run_rotation_series(['2011','2012'], ...)
(consecutive, no gap) and a single year both run in under 2 seconds - the
gap itself is what breaks something inside AquaCrop's solver, not any
particular SMT/alpha extreme (every combo tested hung the same way once
the gap was there).

Fixed at the source: this now uses rotation.run_rotation_years_independent(),
which gives every sampled year its own fresh observed initial soil
moisture instead of chaining from a non-adjacent previous year - which is
also the more defensible design for what this scan is actually asking
("does the optimal split vary by year *type*"), not a continuous
trajectory question.

A subprocess-per-combo timeout is kept anyway as a second line of
defense against whatever the *next* unanticipated edge case turns out to
be, using subprocess.run(timeout=...) - the same mechanism
run_allocation_scan_all_sites.py already uses reliably for site-level
parallelism. Two other approaches were tried and abandoned first:
ProcessPoolExecutor's shutdown()/cancel_futures() only stops *accepting*
new work, it doesn't kill an already-running task (a stuck worker kept
consuming a full CPU core for minutes after the pool "moved on"); bare
multiprocessing.Process+terminate() did kill the stuck worker, but
Windows' spawn-based multiprocessing seems to conflict with numba (which
AquaCrop uses internally) badly enough that *every* combo timed out that
way, including ones that normally take under a second.
"""

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))

import numpy as np
import pandas as pd
from aquacrop import IrrigationManagement

from config import SITES
from cropping_systems import is_double_crop
from rotation import run_rotation_years_independent

COMBO_TIMEOUT_S = 60  # generous vs the ~1-2s/combo normal case (which now
# includes a fresh interpreter's numba/aquacrop import cost each time, not
# just the simulation itself); a combo still running past this is
# pathological, not just slow

ANNUAL_QUOTA_MM = 450.0
ALPHAS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
SMT_LEVELS = {"conservative": [40] * 4, "moderate": [60] * 4, "aggressive": [80] * 4}
# audit-v3 (5.1): the alpha SCAN runs on the validation window only -
# alpha selection must never use the final-test years it will be evaluated on.
from temporal_split import SPLIT

YEARS = list(SPLIT.validation_years)

OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
OUT_PATH = OUT_DIR / "allocation_scan.csv"  # merged output (all sites)
PYTHON = sys.executable  # P1 (docs/审计修复计划.md): not a hardcoded venv path


def _run_one_combo(site_id, soil_key, smt, wheat_cap, maize_cap) -> dict:
    df = run_rotation_years_independent(
        site_id, soil_key, YEARS,
        lambda: IrrigationManagement(irrigation_method=1, SMT=list(smt), MaxIrrSeason=wheat_cap),
        lambda: IrrigationManagement(irrigation_method=1, SMT=list(smt), MaxIrrSeason=maize_cap),
    )
    per_year = df.groupby("year").agg(total_yield=("dry_yield_t_ha", "sum"), total_irr=("irrigation_mm", "sum"))
    by_crop = df.groupby("crop")[["dry_yield_t_ha", "irrigation_mm"]].mean()
    return {
        "mean_total_yield": per_year["total_yield"].mean(),
        "std_total_yield": per_year["total_yield"].std(),
        "mean_total_irr": per_year["total_irr"].mean(),
        "wheat_yield": by_crop.loc["wheat", "dry_yield_t_ha"],
        "maize_yield": by_crop.loc["maize", "dry_yield_t_ha"],
        "wheat_irr": by_crop.loc["wheat", "irrigation_mm"],
        "maize_irr": by_crop.loc["maize", "irrigation_mm"],
    }


def _timeout_row(site_id, soil_key, smt_label, alpha, reason):
    return {
        "site_id": site_id, "soil": soil_key, "smt_level": smt_label, "alpha": alpha,
        "mean_total_yield": np.nan, "std_total_yield": np.nan, "mean_total_irr": np.nan,
        "wheat_yield": np.nan, "maize_yield": np.nan, "wheat_irr": np.nan, "maize_irr": np.nan,
        "failure_reason": reason,
    }


def scan_site(site_id, soil_key="loam"):
    rows = []
    for smt_label, smt in SMT_LEVELS.items():
        for alpha in ALPHAS:
            cmd = [
                str(PYTHON), str(Path(__file__).resolve()),
                "--site", site_id, "--soil", soil_key,
                "--smt-label", smt_label, "--alpha", str(alpha),
            ]
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=COMBO_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                # subprocess.run kills the child (and on failure the OS
                # reclaims it) before raising this, unlike the
                # multiprocessing.Process approaches tried above
                rows.append(_timeout_row(site_id, soil_key, smt_label, alpha, "timeout"))
                print(f"  {site_id} {smt_label} alpha={alpha:.1f} -> TIMED OUT after {COMBO_TIMEOUT_S}s, skipped", flush=True)
                continue

            last_line = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
            if result.returncode == 0 and last_line.startswith("{"):
                metrics = json.loads(last_line)
                rows.append({"site_id": site_id, "soil": soil_key, "smt_level": smt_label, "alpha": alpha, **metrics})
                print(f"  {site_id} {smt_label} alpha={alpha:.1f} -> yield={metrics['mean_total_yield']:.2f}", flush=True)
            else:
                rows.append(_timeout_row(site_id, soil_key, smt_label, alpha, f"exit_code={result.returncode}"))
                print(f"  {site_id} {smt_label} alpha={alpha:.1f} -> FAILED (exit {result.returncode}): {result.stderr[-200:]}", flush=True)

    return pd.DataFrame(rows)


def main(site_id=None, soil_key="loam", smt_label=None, alpha=None):
    if smt_label is not None and alpha is not None:
        # single-combo mode: print exactly one JSON line to stdout, nothing
        # else - this is what the subprocess.run() calls above parse
        metrics = _run_one_combo(
            site_id, soil_key, SMT_LEVELS[smt_label], ANNUAL_QUOTA_MM * alpha, ANNUAL_QUOTA_MM * (1 - alpha)
        )
        print(json.dumps(metrics))
        return

    if site_id:
        if not is_double_crop(site_id):
            print(f"skip {site_id}: single-crop site, no wheat/maize split to scan")
            return
        out_path = OUT_DIR / f"allocation_scan_{site_id}.csv"
        # P1-4 (docs/审计修复计划.md): a file existing isn't "done" if some
        # of its rows are timeouts/failures - that used to get skipped on
        # every future re-run, silently keeping bad rows forever instead
        # of ever retrying them.
        if out_path.exists():
            prior = pd.read_csv(out_path)
            if "failure_reason" not in prior.columns or prior["failure_reason"].isna().all():
                print(f"skip {site_id}, already scanned (no failures)")
                return
            print(f"{site_id}: prior scan has {prior['failure_reason'].notna().sum()} failed rows, re-scanning")
        scan_site(site_id).to_csv(out_path, index=False)
        print(f"saved -> {out_path}")
        return

    existing = pd.read_csv(OUT_PATH) if OUT_PATH.exists() else pd.DataFrame()
    if not existing.empty and "failure_reason" in existing.columns:
        failed = set(zip(existing.loc[existing["failure_reason"].notna(), "site_id"], existing.loc[existing["failure_reason"].notna(), "smt_level"]))
    else:
        failed = set()
    done = set(zip(existing["site_id"], existing["smt_level"])) - failed if not existing.empty else set()
    if failed and not existing.empty:
        existing = existing[~existing.set_index(["site_id", "smt_level"]).index.isin(failed)]

    frames = [existing] if not existing.empty else []
    for sid in SITES:
        if not is_double_crop(sid):
            print(f"skip {sid}: single-crop site, no wheat/maize split to scan")
            continue
        if all((sid, lvl) in done for lvl in SMT_LEVELS):
            print(f"skip {sid}, already scanned")
            continue
        frames.append(scan_site(sid))
        pd.concat(frames, ignore_index=True).to_csv(OUT_PATH, index=False)
    print(f"saved -> {OUT_PATH}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--site", default=None)
    parser.add_argument("--soil", default="loam")
    parser.add_argument("--smt-label", default=None, choices=list(SMT_LEVELS))
    parser.add_argument("--alpha", type=float, default=None)
    args = parser.parse_args()
    main(site_id=args.site, soil_key=args.soil, smt_label=args.smt_label, alpha=args.alpha)
