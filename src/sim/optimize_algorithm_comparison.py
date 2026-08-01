"""Algorithm comparison for the cross-season optimization problem (paper 1
对比3: NSGA-II vs NSGA-III vs MOEA/D). Reuses optimize_cross_season.py's
CrossSeasonProblem unchanged - only the algorithm differs, so any
difference in the resulting front is attributable to the search strategy.

NSGA-III and MOEA/D both need reference directions for a 4-objective
problem; using pymoo's Riesz-energy-based get_reference_directions("energy",
...) rather than "das-dennis" since das-dennis's direction count is
combinatorially fixed (comb(n_partitions + n_obj - 1, n_obj - 1)) and
rarely lands near an arbitrary target population size, whereas "energy"
takes n_points directly - keeping all three algorithms on the same
population size for a fair comparison.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.indicators.hv import HV
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.util.ref_dirs import get_reference_directions

from optimize_cross_season import CrossSeasonProblem

POP_SIZE = 32
N_GEN = 20
N_OBJ = 4

OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


def run_algorithm(name, site_id, soil_key="loam", seed=1):
    problem = CrossSeasonProblem(site_id, soil_key, joint=True)
    ref_dirs = get_reference_directions("energy", N_OBJ, POP_SIZE, seed=seed)

    if name == "NSGA-II":
        algo = NSGA2(pop_size=POP_SIZE)
    elif name == "NSGA-III":
        algo = NSGA3(pop_size=POP_SIZE, ref_dirs=ref_dirs)
    elif name == "MOEA/D":
        algo = MOEAD(ref_dirs=ref_dirs, n_neighbors=15, prob_neighbor_mating=0.7)
    else:
        raise ValueError(name)

    result = minimize(problem, algo, get_termination("n_gen", N_GEN), seed=seed, verbose=True)
    F = result.F
    df = pd.DataFrame(F, columns=["neg_yield", "irrigation_mm", "water_loss_mm", "yield_cv"])
    df["total_yield_t_ha"] = -df["neg_yield"]
    df = df.drop(columns=["neg_yield"])
    df["algorithm"] = name
    df["site_id"] = site_id
    return df


def compare_site(site_id):
    frames = [run_algorithm(name, site_id) for name in ["NSGA-II", "NSGA-III", "MOEA/D"]]
    combined = pd.concat(frames, ignore_index=True)

    all_F = combined[["total_yield_t_ha", "irrigation_mm", "water_loss_mm", "yield_cv"]].copy()
    all_F["total_yield_t_ha"] *= -1
    ref_point = all_F.max().values * 1.05
    hv = HV(ref_point=ref_point)

    print(f"\n=== {site_id}: hypervolume (higher = better front) ===")
    for name in ["NSGA-II", "NSGA-III", "MOEA/D"]:
        sub = combined[combined.algorithm == name]
        F = sub[["total_yield_t_ha", "irrigation_mm", "water_loss_mm", "yield_cv"]].copy()
        F["total_yield_t_ha"] *= -1
        print(f"  {name:10s} hv={hv(F.values):.3e}  n_solutions={len(sub)}")

    return combined


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--site", default="hebei_central")
    args = parser.parse_args()

    df = compare_site(args.site)
    out_path = OUT_DIR / f"algorithm_comparison_{args.site}.csv"
    df.to_csv(out_path, index=False)
    print(f"\nsaved -> {out_path}")
