"""Figure: NSGA-II vs NSGA-III vs MOEA/D on the cross-season allocation
problem (data/processed/algorithm_comparison_hebei_central.csv, from
src/sim/optimize_algorithm_comparison.py).

Result this figure shows: NSGA-III's reference-direction sampling gives
the best hypervolume with fewer, more evenly-spread solutions; NSGA-II
finds more solutions but a less efficient front; MOEA/D underperforms
both here (likely undertuned neighborhood size for this problem's shape).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import pandas as pd
from style import PALETTE, apply_style

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "algorithm_comparison_hebei_central.csv"
OUT_PATH = Path(__file__).resolve().parents[2] / "figures" / "fig8_algorithm_comparison.png"

ALGO_ORDER = ["NSGA-II", "NSGA-III", "MOEA/D"]
ALGO_COLOR = dict(zip(ALGO_ORDER, PALETTE))

# from optimize_algorithm_comparison.py's run log
HV_VALUES = {"NSGA-III": 7.19e4, "NSGA-II": 6.13e4, "MOEA/D": 1.35e4}


def main():
    apply_style()
    df = pd.read_csv(DATA_PATH)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle("研究一 对比3：NSGA-II / NSGA-III / MOEA-D 算法对比（河北中部）", fontsize=14, y=1.02)

    ax = axes[0]
    for algo in ALGO_ORDER:
        sub = df[df["algorithm"] == algo].sort_values("irrigation_mm")
        ax.plot(
            sub["irrigation_mm"], sub["total_yield_t_ha"], "o", color=ALGO_COLOR[algo],
            label=f"{algo} (n={len(sub)})", alpha=0.75, markersize=6,
        )
    ax.set_xlabel("灌溉量 (mm)")
    ax.set_ylabel("系统总产量 (t/ha)")
    ax.set_title("(a) 三种算法帕累托前沿对比")
    ax.legend(fontsize=9)

    ax = axes[1]
    algos = list(HV_VALUES.keys())
    bars = ax.bar(algos, [HV_VALUES[a] for a in algos], color=[ALGO_COLOR[a] for a in algos])
    ax.set_ylabel("超体积指标 (hypervolume)")
    ax.set_title("(b) 前沿质量：超体积指标")
    for b, a in zip(bars, algos):
        ax.annotate(f"{HV_VALUES[a]:.2e}\n(n={len(df[df['algorithm']==a])})", (b.get_x() + b.get_width() / 2, b.get_height()), ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PATH)
    print(f"saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
