"""Composite figure: NSGA-II Pareto fronts across all 5 sites (研究方案
4.5 策略六 / 6节气候梯度), from data/processed/pareto_front_{site}_loam.csv.
3 panels: overlaid yield-irrigation fronts, per-site front width comparison,
parallel-coordinates of all 4 objectives for one representative site.

audit-v2 (P0-12): these are SINGLE-SEASON PROTOTYPE fronts (no rotation /
quota), archived to data/processed/invalidated/legacy_v1/. The script
refuses to load them unless --allow-legacy is passed.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import parallel_coordinates
from style import SITE_LABELS_CN, SITE_ORDER, apply_style, site_color

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
LEGACY_DATA_DIR = DATA_DIR / "invalidated" / "legacy_v1"
OUT_PATH = Path(__file__).resolve().parents[2] / "figures" / "fig2_pareto_fronts.png"


def load_fronts(data_dir: Path) -> dict:
    fronts = {}
    for site_id in SITE_ORDER:
        path = data_dir / f"pareto_front_{site_id}_loam.csv"
        if path.exists():
            fronts[site_id] = pd.read_csv(path)
    return fronts


def main(allow_legacy: bool):
    if allow_legacy:
        data_dir = LEGACY_DATA_DIR
        print("WARNING: loading SINGLE-SEASON PROTOTYPE results (legacy_v1) - not valid for final claims")
    else:
        raise SystemExit(
            "fig2 reads single-season PROTOTYPE results (data/processed/invalidated/legacy_v1/) - "
            "invalidated by audit-v2 P0-12. Pass --allow-legacy for development record only."
        )
    apply_style()
    fronts = load_fronts(data_dir)

    fig = plt.figure(figsize=(15, 11))
    fig.suptitle("研究一：NSGA-II 多目标灌溉策略帕累托前沿（壤土）", fontsize=15, y=1.0)
    gs = fig.add_gridspec(2, 2)

    ax = fig.add_subplot(gs[0, :])
    for site_id, df in fronts.items():
        df_sorted = df.sort_values("irrigation_mm")
        ax.plot(
            df_sorted["irrigation_mm"], df_sorted["yield_t_ha"], "-o", color=site_color(site_id),
            label=SITE_LABELS_CN[site_id], markersize=5, alpha=0.85,
        )
    ax.set_xlabel("灌溉量 (mm)")
    ax.set_ylabel("产量 (t/ha)")
    ax.set_title("(a) 5 站点帕累托前沿叠加对比")
    ax.legend(loc="lower right")

    ax = fig.add_subplot(gs[1, 0])
    widths = pd.DataFrame(
        {
            site_id: [df["yield_t_ha"].min(), df["yield_t_ha"].max(), df["irrigation_mm"].max()]
            for site_id, df in fronts.items()
        },
        index=["yield_min", "yield_max", "irr_max"],
    ).T.loc[SITE_ORDER]
    x = range(len(widths))
    ax.bar(x, widths["yield_max"] - widths["yield_min"], color=[site_color(s) for s in SITE_ORDER])
    ax.set_xticks(list(x))
    ax.set_xticklabels([SITE_LABELS_CN[s] for s in SITE_ORDER], rotation=20)
    ax.set_ylabel("前沿产量跨度 (t/ha)")
    ax.set_title("(b) 前沿宽度随干旱程度放大")
    for i, s in enumerate(SITE_ORDER):
        ax.annotate(f"上限{widths.loc[s,'irr_max']:.0f}mm", (i, widths.loc[s, "yield_max"] - widths.loc[s, "yield_min"]), ha="center", va="bottom", fontsize=8)

    ax = fig.add_subplot(gs[1, 1])
    rep_site = "hebei_central"
    df = fronts[rep_site].copy()
    df["group"] = pd.qcut(df["irrigation_mm"], 4, labels=["低灌溉", "中低", "中高", "高灌溉"], duplicates="drop")
    cols = ["yield_t_ha", "irrigation_mm", "cost", "water_loss_mm"]
    plot_df = df[cols + ["group"]].copy()
    for c in cols:
        plot_df[c] = (plot_df[c] - plot_df[c].min()) / (plot_df[c].max() - plot_df[c].min() + 1e-9)
    parallel_coordinates(plot_df, "group", cols=cols, ax=ax, colormap="viridis", alpha=0.6)
    ax.set_title(f"(c) 目标权衡关系（{SITE_LABELS_CN[rep_site]}，归一化）")
    ax.set_xticklabels(["产量", "灌溉量", "成本", "水损失"])
    ax.legend(fontsize=8, loc="upper right")

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
