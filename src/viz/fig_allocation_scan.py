"""Composite figure: alpha (wheat/maize annual-quota split) scan results,
from data/processed/allocation_scan.csv (src/sim/scan_allocation.py),
regenerated after the P0-1 calendar fix (docs/审计修复计划.md) - all 4
double-crop sites now included (beijing_plain previously failed all
combos under the pre-fix calendar/HIGC bugs).

Key finding this figure exists to show: the optimal split flips by site
aridity - Hebei/Henan/Beijing (drier/mid) favor wheat-heavy splits,
Shaanxi Guanzhong (wettest double-crop site) favors maize-heavy splits
instead. No single fixed alpha serves all sites, which is the core
argument for paper 1's joint-allocation innovation.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import pandas as pd
from style import PALETTE, apply_style

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "allocation_scan.csv"
OUT_PATH = Path(__file__).resolve().parents[2] / "figures" / "fig7_allocation_scan.png"

SITE_LABELS_CN = {
    "hebei_central": "河北中部", "henan_north": "河南北部", "shaanxi_guanzhong": "陕西关中",
    "beijing_plain": "北京平原",
}
SMT_LABELS_CN = {"conservative": "保守 (SMT40)", "moderate": "中等 (SMT60)", "aggressive": "激进 (SMT80)"}
SMT_ORDER = ["conservative", "moderate", "aggressive"]


def main():
    apply_style()
    df = pd.read_csv(DATA_PATH)
    if "failure_reason" in df.columns:
        df = df[df["failure_reason"].isna()].copy()
    sites = [s for s in ["hebei_central", "henan_north", "shaanxi_guanzhong", "beijing_plain"] if s in df["site_id"].unique()]

    fig, axes = plt.subplots(1, len(sites), figsize=(6 * len(sites), 5), sharey=False)
    fig.suptitle("研究一 对比1a：小麦/玉米年度配水比例 (α) 扫描 —— 最优配水随干旱程度反转", fontsize=14, y=1.03)

    for ax, site_id in zip(axes, sites):
        sub = df[df["site_id"] == site_id]
        for i, smt in enumerate(SMT_ORDER):
            s = sub[sub["smt_level"] == smt].sort_values("alpha")
            if s.empty:
                continue
            ax.errorbar(
                s["alpha"], s["mean_total_yield"], yerr=s["std_total_yield"],
                marker="o", color=PALETTE[i], label=SMT_LABELS_CN[smt], capsize=3, alpha=0.85,
            )
            best = s.loc[s["mean_total_yield"].idxmax()]
            ax.scatter([best["alpha"]], [best["mean_total_yield"]], color=PALETTE[i], s=140, marker="*", zorder=5)
        ax.set_xlabel("α (分配给小麦的年度配额比例)")
        ax.set_ylabel("系统总产量 (t/ha)")
        ax.set_title(SITE_LABELS_CN[site_id])
        ax.legend(fontsize=8)

    plt.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PATH)
    print(f"saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
