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


X_COLS = ["SMT_w1", "SMT_w2", "SMT_w3", "SMT_w4", "SMT_m1", "SMT_m2", "SMT_m3", "SMT_m4", "alpha"]


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
    # P1-3 (docs/审计修复计划.md): keep the decision variables, not just
    # objective values - otherwise there's no way to inspect *what* each
    # algorithm's solutions actually recommend, only how good they scored.
    X = result.X if result.X is not None else np.full((len(df), len(X_COLS)), np.nan)
    for i, col in enumerate(X_COLS):
        df[col] = X[:, i]
    df["algorithm"] = name
    df["site_id"] = site_id
    df["seed"] = seed
    return df


OBJ_COLS = ["total_yield_t_ha", "irrigation_mm", "water_loss_mm", "yield_cv"]


def _normalize(F: pd.DataFrame, obj_min: pd.Series, obj_range: pd.Series) -> np.ndarray:
    """Minimization-sense normalization to ~[0,1], using a scale frozen
    from the *combined* solution set (all algorithms, all seeds) so every
    algorithm's front is measured on the same yardstick."""
    F = F[OBJ_COLS].copy()
    F["total_yield_t_ha"] *= -1  # minimize -yield == maximize yield
    return ((F - obj_min) / obj_range).values


def compare_site(site_id, seeds=(1,)):
    frames = [run_algorithm(name, site_id, seed=seed) for name in ["NSGA-II", "NSGA-III", "MOEA/D"] for seed in seeds]
    combined = pd.concat(frames, ignore_index=True)

    # P1-3 (docs/审计修复计划.md): the old reference point took
    # all_F.max()*1.05 on an array that already had yield negated for
    # minimization - multiplying a negative number by 1.05 makes it *more*
    # negative, i.e. *better* in minimization space, exactly backwards
    # (a reference point must be dominated by every candidate solution).
    # Fixed by min-max normalizing all 4 objectives to the combined
    # solution set's own range, then using a fixed reference point just
    # outside [0,1] - verified dominated by assertion below, not assumed.
    all_F = combined[OBJ_COLS].copy()
    all_F["total_yield_t_ha"] *= -1
    obj_min = all_F.min()
    obj_range = (all_F.max() - obj_min).replace(0, 1.0)  # guard a degenerate zero-spread objective
    ref_point = np.full(len(OBJ_COLS), 1.1)
    hv = HV(ref_point=ref_point)

    print(f"\n=== {site_id}: hypervolume (higher = better front), {len(seeds)} seed(s) ===")
    summary_rows = []
    for name in ["NSGA-II", "NSGA-III", "MOEA/D"]:
        hvs = []
        for seed in seeds:
            sub = combined[(combined.algorithm == name) & (combined.seed == seed)]
            F_norm = _normalize(sub, obj_min, obj_range)
            assert (F_norm < ref_point).all(), (
                f"{name} seed={seed}: a normalized solution is not dominated by ref_point {ref_point}"
            )
            F_dedup = np.unique(F_norm, axis=0)  # exact-duplicate solutions don't add real hypervolume
            hvs.append(hv(F_dedup))
        mean_hv, std_hv = float(np.mean(hvs)), float(np.std(hvs))
        n_solutions = len(combined[(combined.algorithm == name) & (combined.seed == seeds[0])])
        print(f"  {name:10s} hv={mean_hv:.4f} (+/-{std_hv:.4f} over {len(seeds)} seeds)  n_solutions={n_solutions}")
        summary_rows.append({"algorithm": name, "site_id": site_id, "hv_mean": mean_hv, "hv_std": std_hv})

    return combined, pd.DataFrame(summary_rows)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--site", default="hebei_central")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1])
    args = parser.parse_args()

    df, summary = compare_site(args.site, seeds=args.seeds)
    out_path = OUT_DIR / f"algorithm_comparison_{args.site}.csv"
    df.to_csv(out_path, index=False)
    summary.to_csv(OUT_DIR / f"algorithm_comparison_{args.site}_hv_summary.csv", index=False)
    print(f"\nsaved -> {out_path}")
