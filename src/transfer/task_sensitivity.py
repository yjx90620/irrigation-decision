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
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))

import pandas as pd
from aquacrop import IrrigationManagement

from config import SITES
from rotation import run_rotation_series

EVAL_YEARS = list(range(2011, 2021))  # 10 years, matches scan_allocation.py's window

OUT_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "task_sensitivity.csv"


def rainfed():
    return IrrigationManagement(irrigation_method=0)


def full_irrigation():
    return IrrigationManagement(irrigation_method=1, SMT=[100, 100, 100, 100])


def compute_sensitivity(site_id: str, soil_key: str = "loam") -> dict:
    rain_df = run_rotation_series(site_id, soil_key, EVAL_YEARS, rainfed, rainfed)
    full_df = run_rotation_series(site_id, soil_key, EVAL_YEARS, full_irrigation, full_irrigation)

    rain_yield = rain_df.groupby("year")["dry_yield_t_ha"].sum().mean()
    full_yield = full_df.groupby("year")["dry_yield_t_ha"].sum().mean()
    sensitivity = (full_yield - rain_yield) / full_yield if full_yield > 0 else float("nan")

    return {
        "site_id": site_id, "soil": soil_key,
        "rainfed_yield": rain_yield, "full_irrigation_yield": full_yield,
        "sensitivity": sensitivity,
    }


def main():
    rows = [compute_sensitivity(site_id) for site_id in SITES]
    df = pd.DataFrame(rows).sort_values("sensitivity")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(df.round(3).to_string(index=False))
    print(f"\nsaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
