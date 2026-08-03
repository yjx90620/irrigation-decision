"""Composite figure: baseline irrigation strategy results (研究方案 4.8/4.9),
from data/processed/baseline_experiment_results.csv (5400 simulation runs).
4 panels: yield distribution by strategy, water-yield tradeoff, per-site
rainfed-vs-full-irrigation gradient, water-loss breakdown by strategy.

audit-v2 (P0-12): these are SINGLE-SEASON PROTOTYPE baseline results
archived to data/processed/invalidated/legacy_v1/ (each crop started near
field capacity, no rotation). The script refuses to load them unless
--allow-legacy is passed.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from style import PALETTE, SITE_LABELS_CN, SITE_ORDER, STRATEGY_LABELS_CN, STRATEGY_ORDER, apply_style

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "baseline_experiment_results.csv"
LEGACY_DATA_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "processed" / "invalidated" / "legacy_v1"
    / "baseline_experiment_results.csv"
)
OUT_PATH = Path(__file__).resolve().parents[2] / "figures" / "fig1_baseline_overview.png"


def main(allow_legacy: bool):
    if allow_legacy:
        path = LEGACY_DATA_PATH
        print("WARNING: loading SINGLE-SEASON PROTOTYPE results (legacy_v1) - not valid for final claims")
    else:
        raise SystemExit(
            "fig1 reads single-season PROTOTYPE results (data/processed/invalidated/legacy_v1/) - "
            "invalidated by audit-v2 P0-12. Pass --allow-legacy for development record only."
        )
    apply_style()
    df = pd.read_csv(path)
    df["strategy_cn"] = df["strategy"].map(STRATEGY_LABELS_CN)
    df["site_cn"] = df["site_id"].map(SITE_LABELS_CN)
    strategy_order_cn = [STRATEGY_LABELS_CN[s] for s in STRATEGY_ORDER]
    site_order_cn = [SITE_LABELS_CN[s] for s in SITE_ORDER]

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle("研究一：基线灌溉策略结果总览（5 站点 × 3 土壤 × 45 年 × 8 策略）", fontsize=15, y=1.0)

    ax = axes[0, 0]
    sns.boxplot(
        data=df, x="strategy_cn", y="dry_yield_t_ha", hue="strategy_cn", order=strategy_order_cn,
        palette=PALETTE, legend=False, ax=ax,
    )
    ax.set_title("(a) 各策略产量分布")
    ax.set_xlabel("")
    ax.set_ylabel("产量 (t/ha)")
    ax.tick_params(axis="x", rotation=35)

    ax = axes[0, 1]
    summary = df.groupby("strategy").agg(irr=("irrigation_mm", "mean"), yld=("dry_yield_t_ha", "mean")).reset_index()
    summary["strategy_cn"] = summary["strategy"].map(STRATEGY_LABELS_CN)
    for _, row in summary.iterrows():
        color = PALETTE[STRATEGY_ORDER.index(row["strategy"]) % len(PALETTE)]
        ax.scatter(row["irr"], row["yld"], s=110, color=color, label=row["strategy_cn"], zorder=3)
    ax.plot(summary.sort_values("irr")["irr"], summary.sort_values("irr")["yld"], "--", color="gray", alpha=0.5, zorder=1)
    ax.set_title("(b) 节水—产量权衡（全部站点/土壤/年份平均）")
    ax.set_xlabel("灌溉量 (mm)")
    ax.set_ylabel("产量 (t/ha)")
    ax.legend(fontsize=8, loc="lower right", ncol=2)

    ax = axes[1, 0]
    pivot = (
        df[df.strategy.isin(["rainfed", "full_irrigation"])]
        .groupby(["site_id", "strategy"])["dry_yield_t_ha"]
        .mean()
        .unstack()
        .loc[SITE_ORDER]
    )
    x = range(len(pivot))
    width = 0.35
    ax.bar([i - width / 2 for i in x], pivot["rainfed"], width, label="雨养", color=PALETTE[0])
    ax.bar([i + width / 2 for i in x], pivot["full_irrigation"], width, label="充分灌溉", color=PALETTE[3])
    ax.set_xticks(list(x))
    ax.set_xticklabels(site_order_cn, rotation=20)
    ax.set_title("(c) 区域气候梯度：雨养 vs 充分灌溉产量")
    ax.set_ylabel("产量 (t/ha)")
    ax.legend()

    ax = axes[1, 1]
    loss = df.groupby("strategy")[["deep_perc_mm", "runoff_mm"]].mean().loc[STRATEGY_ORDER]
    loss.index = [STRATEGY_LABELS_CN[s] for s in loss.index]
    loss.plot(kind="bar", stacked=True, ax=ax, color=[PALETTE[1], PALETTE[5]])
    ax.set_title("(d) 各策略非生产性水损失（深层渗漏+径流）")
    ax.set_xlabel("")
    ax.set_ylabel("mm")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(["深层渗漏", "径流"])

    plt.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PATH)
    print(f"saved -> {OUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-legacy", action="store_true",
                        help="audit-v2 P0-12: load archived single-season prototype results (dev record only)")
    args = parser.parse_args()
    main(allow_legacy=args.allow_legacy)
