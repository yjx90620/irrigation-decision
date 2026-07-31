"""NSGA-II multi-objective irrigation policy optimization (研究方案 4.5 策略六
/ 4.6 / 4.7).

AquaCrop-OSPy's built-in soil-moisture-target scheduling (irrigation_method=1)
triggers an irrigation that refills the root zone back up to a target %TAW
whenever it drops below that target - it does not accept a separately
specified application depth per stage. So the 8-dim [D_j, I_j] decision
vector from 研究方案 4.6 is adapted to what that scheduler actually exposes:
4 per-stage soil-moisture targets (SMT, one per AquaCrop growth stage) plus
a single shared cap on how much can be applied in one day (MaxIrr, matching
the 0-40mm action range planned for 研究内容二). Decision vector:
x = [SMT1, SMT2, SMT3, SMT4, MaxIrr].

Objectives (minimize all, so yield is negated):
  f1 = -dry_yield_t_ha
  f2 = irrigation_mm
  f3 = cost = c_w * irrigation_mm + c_s * irrigation_events
  f4 = deep_perc_mm + runoff_mm

Each individual is evaluated across EVAL_YEARS and averaged, so the front
reflects multi-year robustness rather than a single lucky/unlucky season.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))

import numpy as np
import pandas as pd
from aquacrop import AquaCropModel, Crop, InitialWaterContent, IrrigationManagement
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from soils import get_soil
from weather import load_site_weather

PLANTING_DATE = "05/01"
HARVEST_DATE = "09/15"
EVAL_YEARS = [2016, 2017, 2018, 2019, 2020]  # 5-year robustness sample
COST_WATER = 1.0  # cost per mm (relative units; plan allows low/mid/high scenarios later)
COST_START = 5.0  # cost per irrigation event


def evaluate_policy(weather_by_year, soil, smt, max_irr):
    yields, irrs, costs, losses = [], [], [], []
    for year, year_df in weather_by_year.items():
        crop = Crop("Maize", planting_date=PLANTING_DATE, harvest_date=HARVEST_DATE)
        init_wc = InitialWaterContent(value=["FC"])
        irr_mngt = IrrigationManagement(irrigation_method=1, SMT=list(smt), MaxIrr=max_irr)
        model = AquaCropModel(
            sim_start_time=f"{year}/01/01",
            sim_end_time=f"{year}/12/31",
            weather_df=year_df,
            soil=soil,
            crop=crop,
            initial_water_content=init_wc,
            irrigation_management=irr_mngt,
        )
        model.run_model(till_termination=True)
        res = model.get_simulation_results()
        flux = model.get_water_flux()
        irr_mm = res["Seasonal irrigation (mm)"].iloc[0]
        n_events = int((flux["IrrDay"] > 0).sum())
        yields.append(res["Dry yield (tonne/ha)"].iloc[0])
        irrs.append(irr_mm)
        costs.append(COST_WATER * irr_mm + COST_START * n_events)
        losses.append(flux["DeepPerc"].sum() + flux["Runoff"].sum())
    return np.mean(yields), np.mean(irrs), np.mean(costs), np.mean(losses)


class IrrigationPolicyProblem(Problem):
    def __init__(self, site_id: str, soil_key: str):
        super().__init__(n_var=5, n_obj=4, xl=[0, 0, 0, 0, 5], xu=[100, 100, 100, 100, 40])
        weather_df = load_site_weather(site_id)
        self.weather_by_year = {
            y: weather_df[(weather_df["Date"] >= f"{y}-01-01") & (weather_df["Date"] <= f"{y}-12-31")].reset_index(
                drop=True
            )
            for y in EVAL_YEARS
        }
        self.soil_key = soil_key

    def _evaluate(self, X, out, *args, **kwargs):
        F = np.zeros((X.shape[0], 4))
        for i, x in enumerate(X):
            soil = get_soil(self.soil_key)  # fresh soil object per eval, AquaCrop mutates it
            smt, max_irr = x[:4], x[4]
            mean_yield, mean_irr, mean_cost, mean_loss = evaluate_policy(
                self.weather_by_year, soil, smt, max_irr
            )
            F[i] = [-mean_yield, mean_irr, mean_cost, mean_loss]
        out["F"] = F


def run(site_id: str, soil_key: str, pop_size=40, n_gen=30):
    problem = IrrigationPolicyProblem(site_id, soil_key)
    algorithm = NSGA2(pop_size=pop_size)
    result = minimize(problem, algorithm, get_termination("n_gen", n_gen), seed=1, verbose=True)

    df = pd.DataFrame(result.X, columns=["SMT1", "SMT2", "SMT3", "SMT4", "MaxIrr"])
    df["neg_yield_t_ha"], df["irrigation_mm"], df["cost"], df["water_loss_mm"] = result.F.T
    df["yield_t_ha"] = -df["neg_yield_t_ha"]
    df = df.drop(columns=["neg_yield_t_ha"])
    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--site", default="hebei_central")
    parser.add_argument("--soil", default="loam")
    parser.add_argument("--pop-size", type=int, default=40)
    parser.add_argument("--n-gen", type=int, default=30)
    args = parser.parse_args()

    out_dir = Path(__file__).resolve().parents[2] / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    front = run(args.site, args.soil, args.pop_size, args.n_gen)
    out_path = out_dir / f"pareto_front_{args.site}_{args.soil}.csv"
    front.to_csv(out_path, index=False)
    print(f"saved {len(front)} Pareto-front policies -> {out_path}")
