"""Figure: NSGA-II vs NSGA-III vs MOEA/D on the cross-season allocation
problem (data/processed/algorithm_comparison_hebei_central.csv +
_hv_summary.csv, from src/sim/optimize_algorithm_comparison.py),
regenerated after the P1-3 hypervolume-reference-point fix and with 3
independent seeds (docs/审计修复计划.md).

Result this figure shows: NSGA-III's reference-direction sampling still
gives the best hypervolume (now correctly normalized/oriented) with
fewer, more evenly-spread solutions; NSGA-II finds more solutions but a
less efficient front; MOEA/D underperforms both (likely undertuned
neighborhood size for this problem's shape) - same ranking as before the
fix, now with seed-to-seed variance reported instead of one run's number.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import pandas as pd
from style import PALETTE, apply_style

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "algorithm_comparison_hebei_central.csv"
HV_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "algorithm_comparison_hebei_central_hv_summary.csv"
OUT_PATH = Path(__file__).resolve().parents[2] / "figures" / "fig8_algorithm_comparison.png"

ALGO_ORDER = ["NSGA-II", "NSGA-III", "MOEA/D"]
ALGO_COLOR = dict(zip(ALGO_ORDER, PALETTE))


def main():
    apply_style()
    df = pd.read_csv(DATA_PATH)
    hv = pd.read_csv(HV_PATH).set_index("algorithm")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle("研究一 对比3：NSGA-II / NSGA-III / MOEA-D 算法对比（河北中部，3个独立seed）", fontsize=13, y=1.02)

    ax = axes[0]
    seed1 = df[df["seed"] == 1]
    for algo in ALGO_ORDER:
        sub = seed1[seed1["algorithm"] == algo].sort_values("irrigation_mm")
        ax.plot(
            sub["irrigation_mm"], sub["total_yield_t_ha"], "o", color=ALGO_COLOR[algo],
            label=f"{algo} (n={len(sub)})", alpha=0.75, markersize=6,
        )
    ax.set_xlabel("灌溉量 (mm)")
    ax.set_ylabel("系统总产量 (t/ha)")
    ax.set_title("(a) 三种算法帕累托前沿对比（seed=1）")
    ax.legend(fontsize=9)

    ax = axes[1]
    means = [hv.loc[a, "hv_mean"] for a in ALGO_ORDER]
    stds = [hv.loc[a, "hv_std"] for a in ALGO_ORDER]
    bars = ax.bar(ALGO_ORDER, means, yerr=stds, capsize=4, color=[ALGO_COLOR[a] for a in ALGO_ORDER])
    ax.set_ylabel("归一化超体积指标（3个seed均值±标准差）")
    ax.set_title("(b) 前沿质量：超体积指标")
    for b, algo in zip(bars, ALGO_ORDER):
        n = len(df[(df["algorithm"] == algo) & (df["seed"] == 1)])
        ax.annotate(
            f"{hv.loc[algo,'hv_mean']:.3f}±{hv.loc[algo,'hv_std']:.3f}\n(n={n}/seed)",
            (b.get_x() + b.get_width() / 2, b.get_height() + stds[ALGO_ORDER.index(algo)]),
            ha="center", va="bottom", fontsize=8,
        )

    plt.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PATH)
    print(f"saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
