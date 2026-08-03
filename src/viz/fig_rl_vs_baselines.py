"""Composite figure: PPO v2 vs the threshold rule baseline (研究方案 5.11
对比一), from data/processed/rl_vs_baselines_comparison.csv (2018-2022,
balanced preference weights). 2 panels: yield by site/policy, irrigation
by site/policy - shows where RL matches/beats the rule and where it
still falls short (currently: Ningxia, the hardest/driest site).

audit-v2 (P0-12): this figure is built from SINGLE-SEASON PROTOTYPE
results that were archived to data/processed/invalidated/legacy_v1/ (the
research premise - no rotation, near-field-capacity starts - was
invalidated by the system upgrade). The script refuses to load them
unless --allow-legacy is passed explicitly; it must never be part of a
final paper figure run.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import pandas as pd
from style import PALETTE, SITE_LABELS_CN, SITE_ORDER, apply_style

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "rl_vs_baselines_comparison.csv"
LEGACY_DATA_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "processed" / "invalidated" / "legacy_v1"
    / "rl_vs_baselines_comparison.csv"
)
OUT_PATH = Path(__file__).resolve().parents[2] / "figures" / "fig5_rl_vs_baselines.png"
POLICY_LABELS = {"threshold": "阈值规则（40%亏缺触发）", "ppo_v2": "PPO（修复后）"}


def main(allow_legacy: bool):
    if allow_legacy:
        path = LEGACY_DATA_PATH
        print("WARNING: loading SINGLE-SEASON PROTOTYPE results (legacy_v1) - not valid for final claims")
    else:
        raise SystemExit(
            "fig5 reads single-season PROTOTYPE results (data/processed/invalidated/legacy_v1/) - "
            "invalidated by audit-v2 P0-12. Pass --allow-legacy to load them for development "
            "record only; they must not appear in final paper figures."
        )
    apply_style()
    df = pd.read_csv(path)
    df["site_cn"] = df["site_id"].map(SITE_LABELS_CN)
    site_order_cn = [SITE_LABELS_CN[s] for s in SITE_ORDER]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("研究二：PPO vs 阈值规则基线（2018—2022 测试年，平衡偏好权重）", fontsize=14, y=1.02)

    for ax, metric, title in zip(axes, ["dry_yield_t_ha", "irrigation_mm"], ["(a) 产量对比", "(b) 灌溉量对比"]):
        summary = df.groupby(["site_id", "policy"])[metric].mean().unstack().loc[SITE_ORDER]
        x = range(len(summary))
        width = 0.35
        ax.bar([i - width / 2 for i in x], summary["threshold"], width, label=POLICY_LABELS["threshold"], color=PALETTE[0])
        ax.bar([i + width / 2 for i in x], summary["ppo_v2"], width, label=POLICY_LABELS["ppo_v2"], color=PALETTE[1])
        ax.set_xticks(list(x))
        ax.set_xticklabels(site_order_cn, rotation=20)
        ax.set_title(title)
        ax.set_ylabel("t/ha" if "yield" in metric else "mm")

    axes[0].legend(fontsize=9)
    plt.figtext(
        0.5, -0.02,
        "陕西关中/河北中部/北京平原/河南北部：PPO 几乎不灌水（0mm）也能拿到接近阈值规则的产量，用水效率更高；\n"
        "宁夏灌区（最干旱站点）则相反——PPO 用水更少但产量也明显更低，说明极端缺水场景下策略还没学到位，仍有改进空间。",
        ha="center", fontsize=9, style="italic",
    )
    plt.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PATH, bbox_inches="tight")
    print(f"saved -> {OUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-legacy", action="store_true",
                        help="audit-v2 P0-12: load archived single-season prototype results (dev record only)")
    args = parser.parse_args()
    main(allow_legacy=args.allow_legacy)
