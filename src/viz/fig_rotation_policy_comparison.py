"""Composite figure: 论文二's核心对比 - residual RL vs direct RL vs two rule
baselines, from data/processed/rotation_policy_comparison.csv
(src/rl/train_rotation_compare.py, regenerated after the P0/P1 audit
fixes - docs/审计修复计划.md).

Updated for the regenerated data: 4 policies (added quota_reserving_rule,
a stronger baseline than threshold_rule - P0-5), error bars (std across
the 3 independent RL seeds x test years, or across years alone for the
deterministic rule baselines - P0-5's multi-seed requirement), and
discounted_return instead of the old undiscounted-only scalar_return
(P0-4c). 4 panels: yield, irrigation, discounted return, safety-filter
intervention rate.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import pandas as pd
from style import PALETTE, SITE_LABELS_CN, SITE_ORDER, apply_style

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "rotation_policy_comparison.csv"
OUT_PATH = Path(__file__).resolve().parents[2] / "figures" / "fig10_rotation_policy_comparison.png"

POLICY_ORDER = ["threshold_rule", "quota_reserving_rule", "ppo_direct", "ppo_residual"]
POLICY_LABELS = {
    "threshold_rule": "阈值规则", "quota_reserving_rule": "预留配额规则",
    "ppo_direct": "直接RL", "ppo_residual": "残差RL",
}
POLICY_COLOR = dict(zip(POLICY_ORDER, PALETTE))


def main():
    apply_style()
    df = pd.read_csv(DATA_PATH)
    # P1-4 (audit-v2): the comparison CSV now carries a `preference` axis
    # (the policies are preference-conditioned and evaluated under 7 sets) -
    # the headline comparison must not average across preference sets.
    df = df[df["preference"] == "balanced"]
    site_order_cn = [SITE_LABELS_CN[s] for s in SITE_ORDER]

    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    fig.suptitle("研究二 核心对比：残差RL vs 直接RL vs 两种规则基线（轮作环境，3个独立seed，2018—2022测试年）", fontsize=13, y=1.03)

    metrics = [
        ("total_yield_t_ha", "(a) 系统总产量", "t/ha"),
        ("total_irrigation_mm", "(b) 总灌溉量", "mm"),
        ("discounted_return", "(c) 折扣回报 (gamma=0.995)", ""),
        ("action_modified_rate", "(d) 安全层介入率", "比例"),
    ]
    for ax, (col, title, unit) in zip(axes, metrics):
        grouped = df.groupby(["site_id", "policy"])[col]
        mean = grouped.mean().unstack()[POLICY_ORDER].loc[SITE_ORDER]
        std = grouped.std().unstack()[POLICY_ORDER].loc[SITE_ORDER].fillna(0.0)
        x = range(len(mean))
        width = 0.2
        for i, policy in enumerate(POLICY_ORDER):
            offsets = [xi + (i - 1.5) * width for xi in x]
            ax.bar(
                offsets, mean[policy], width, yerr=std[policy], capsize=2,
                label=POLICY_LABELS[policy], color=POLICY_COLOR[policy],
            )
        ax.set_xticks(list(x))
        ax.set_xticklabels(site_order_cn, rotation=20)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(unit)

    axes[0].legend(fontsize=8)
    plt.figtext(
        0.5, -0.03,
        "两种RL策略在所有站点都大幅省水，加权折扣回报全场超过两种规则；但换成预留配额这个更强规则后，RL的原始产量反而更低——\n"
        "RL赢在用水效率的权衡上，不是全面碾压规则。宁夏灌区（单季、任务敏感度最高）预留配额规则退化为阈值规则本身（无小麦季可预留）。",
        ha="center", fontsize=9, style="italic",
    )
    plt.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PATH, bbox_inches="tight")
    print(f"saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
