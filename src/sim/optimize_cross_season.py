"""Cross-season joint irrigation optimization for the wheat-maize rotation
(paper 1's core innovation - see docs/系统升级方案.md).

The premise: under an annual water quota, irrigating winter wheat draws the
profile down and leaves less for the following summer maize, so the two
seasons cannot be optimized independently. The decision vector therefore
includes not just each crop's soil-moisture targets but *how the annual
quota is split between them*:

    x = [SMT_wheat(4), SMT_maize(4), alpha]

where alpha is wheat's share of the annual quota (maize gets 1-alpha),
enforced through AquaCrop's MaxIrrSeason cap on each season.

The comparison this enables (paper 1 对比1): optimize each season
independently under a fixed 50/50 split, versus optimizing jointly with
alpha free - at the same total quota, so any difference is attributable to
the cross-season coupling rather than to using more water.

Objectives (all minimized, yields negated):
    f1 = -(wheat yield + maize yield)      system productivity
    f2 = total irrigation
    f3 = total deep percolation + runoff   non-productive losses
    f4 = yield CV across evaluation years  inter-annual instability

Evaluation is serial within a run. Thread-based parallelism was measured
and gives nothing here (12 threads: 18.5s vs 1 thread: 18.8s for the same
12 individuals) because AquaCrop's numba kernels don't release the GIL -
so parallelism is applied at the site level instead, by launching one
process per site.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))

import numpy as np
import pandas as pd
from aquacrop import IrrigationManagement
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from rotation import run_rotation_series

ANNUAL_QUOTA_MM = 450.0  # total irrigation available per rotation year
EVAL_YEARS = [2016, 2017, 2018, 2019, 2020]

OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


def evaluate_policy(site_id, soil_key, smt_wheat, smt_maize, alpha):
    wheat_cap = ANNUAL_QUOTA_MM * alpha
    maize_cap = ANNUAL_QUOTA_MM * (1 - alpha)

    def wheat_irr():
        return IrrigationManagement(irrigation_method=1, SMT=list(smt_wheat), MaxIrrSeason=wheat_cap)

    def maize_irr():
        return IrrigationManagement(irrigation_method=1, SMT=list(smt_maize), MaxIrrSeason=maize_cap)

    df = run_rotation_series(site_id, soil_key, EVAL_YEARS, wheat_irr, maize_irr)
    per_year = df.groupby("year").agg(
        total_yield=("dry_yield_t_ha", "sum"),
        total_irr=("irrigation_mm", "sum"),
        total_loss=("deep_perc_mm", "sum"),
    )
    per_year["total_loss"] += df.groupby("year")["runoff_mm"].sum()

    mean_yield = per_year["total_yield"].mean()
    yield_cv = per_year["total_yield"].std() / mean_yield if mean_yield > 0 else 10.0
    return mean_yield, per_year["total_irr"].mean(), per_year["total_loss"].mean(), yield_cv


class CrossSeasonProblem(Problem):
    """joint=True lets alpha vary; joint=False pins it at 0.5 (independent
    per-season optimization under an even split) for the control condition."""

    def __init__(self, site_id, soil_key, joint=True, **kwargs):
        n_var = 9 if joint else 8
        xl = [0] * 8 + ([0.1] if joint else [])
        xu = [100] * 8 + ([0.9] if joint else [])
        super().__init__(n_var=n_var, n_obj=4, xl=xl, xu=xu, **kwargs)
        self.site_id = site_id
        self.soil_key = soil_key
        self.joint = joint

    def _evaluate(self, X, out, *args, **kwargs):
        F = np.zeros((X.shape[0], 4))
        for i, x in enumerate(X):
            alpha = x[8] if self.joint else 0.5
            y, irr, loss, cv = evaluate_policy(self.site_id, self.soil_key, x[:4], x[4:8], alpha)
            F[i] = [-y, irr, loss, cv]
        out["F"] = F


def run(site_id, soil_key="loam", joint=True, pop_size=32, n_gen=20, seed=1):
    problem = CrossSeasonProblem(site_id, soil_key, joint=joint)
    result = minimize(problem, NSGA2(pop_size=pop_size), get_termination("n_gen", n_gen), seed=seed, verbose=True)

    cols = ["SMT_w1", "SMT_w2", "SMT_w3", "SMT_w4", "SMT_m1", "SMT_m2", "SMT_m3", "SMT_m4"]
    if joint:
        cols.append("alpha")
    df = pd.DataFrame(result.X, columns=cols)
    if not joint:
        df["alpha"] = 0.5
    df["total_yield_t_ha"] = -result.F[:, 0]
    df["total_irrigation_mm"] = result.F[:, 1]
    df["water_loss_mm"] = result.F[:, 2]
    df["yield_cv"] = result.F[:, 3]
    df["site_id"] = site_id
    df["mode"] = "joint" if joint else "independent"
    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--site", default="hebei_central")
    parser.add_argument("--soil", default="loam")
    parser.add_argument("--pop-size", type=int, default=32)
    parser.add_argument("--n-gen", type=int, default=20)
    parser.add_argument("--modes", nargs="+", default=["joint", "independent"])
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    for mode in args.modes:
        print(f"=== {args.site} / {args.soil} / {mode} ===")
        frames.append(run(args.site, args.soil, joint=(mode == "joint"), pop_size=args.pop_size, n_gen=args.n_gen))
    out_path = OUT_DIR / f"cross_season_pareto_{args.site}_{args.soil}.csv"
    pd.concat(frames, ignore_index=True).to_csv(out_path, index=False)
    print(f"saved -> {out_path}")
