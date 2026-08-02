"""Composite figure: 论文二's core comparison on the rotation environment -
residual RL vs direct RL vs threshold rule, from
data/processed/rotation_policy_comparison.csv (src/rl/train_rotation_compare.py).

This supersedes fig5_rl_vs_baselines.png (single-season prototype, see the
warning at the top of docs/papers/论文二_偏好条件化强化学习决策.md). 3
panels: yield by site/policy, irrigation by site/policy, safety-filter
intervention rate by site/policy - the third one is why residual beats
direct almost everywhere: it needs far fewer safety-filter corrections
because it starts from a sane base policy instead of raw exploration.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import pandas as pd
from style import PALETTE, SITE_LABELS_CN, SITE_ORDER, apply_style

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "rotation_policy_comparison.csv"
OUT_PATH = Path(__file__).resolve().parents[2] / "figures" / "fig10_rotation_policy_comparison.png"

POLICY_ORDER = ["threshold_rule", "ppo_direct", "ppo_residual"]
POLICY_LABELS = {"threshold_rule": "阈值规则基线", "ppo_direct": "直接RL", "ppo_residual": "残差RL"}
POLICY_COLOR = dict(zip(POLICY_ORDER, PALETTE))


def main():
    apply_style()
    df = pd.read_csv(DATA_PATH)
    site_order_cn = [SITE_LABELS_CN[s] for s in SITE_ORDER]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("研究二 核心对比：残差RL vs 直接RL vs 阈值规则（轮作环境，2018—2022测试年）", fontsize=14, y=1.03)

    metrics = [
        ("total_yield_t_ha", "(a) 系统总产量", "t/ha"),
        ("total_irrigation_mm", "(b) 总灌溉量", "mm"),
        ("action_modified_rate", "(c) 安全层介入率", "比例"),
    ]
    for ax, (col, title, unit) in zip(axes, metrics):
        summary = df.groupby(["site_id", "policy"])[col].mean().unstack()[POLICY_ORDER].loc[SITE_ORDER]
        x = range(len(summary))
        width = 0.26
        for i, policy in enumerate(POLICY_ORDER):
            offsets = [xi + (i - 1) * width for xi in x]
            ax.bar(offsets, summary[policy], width, label=POLICY_LABELS[policy], color=POLICY_COLOR[policy])
        ax.set_xticks(list(x))
        ax.set_xticklabels(site_order_cn, rotation=20)
        ax.set_title(title)
        ax.set_ylabel(unit)

    axes[0].legend(fontsize=9)
    plt.figtext(
        0.5, -0.03,
        "陕西关中（最湿润）：残差RL仅用38%的水拿到阈值规则99%的产量；宁夏灌区（最干旱、任务敏感度全场最高）两种RL都明显落后于规则——\n"
        "残差RL在几乎所有站点安全层介入率都远低于直接RL（起点是合理的基础策略，而不是从随机探索开始）。",
        ha="center", fontsize=9, style="italic",
    )
    plt.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PATH, bbox_inches="tight")
    print(f"saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
