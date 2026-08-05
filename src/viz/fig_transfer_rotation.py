"""Paper-3 rotation-era transfer figure (audit-v2/v3): zero-shot / finetuned /
scratch / full-target-expert / rule yields per target site + the two-factor
risk vs actual-gap scatter that validates the distance x sensitivity claim
(per_quota wq arm).

Reads the regenerated leave_one_out_rotation_summary_wq.csv and
two_factor_risk.csv (audit-v3 6.1: now includes the scratch and
full_target_expert arms; test years 2018-2025 from the shared split).
The legacy single-season figure is fig6_transfer_leave_one_out.py
(--allow-legacy gated).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rl"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sim.temporal_split import SPLIT
from style import PALETTE, SITE_LABELS_CN, apply_style

SUMMARY_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "leave_one_out_rotation_summary_wq.csv"
RISK_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "two_factor_risk.csv"
OUT_PATH = Path(__file__).resolve().parents[2] / "figures" / "fig_transfer_rotation.png"

CONDITION_LABELS = {
    "zero_shot": "零样本迁移", "finetuned": "目标域微调", "scratch": "目标域从头训练",
    "full_target_expert": "目标域专家(上界)", "threshold_rule": "规则基线",
}
CONDITION_COLOR = {
    "zero_shot": PALETTE[0], "finetuned": PALETTE[1], "scratch": PALETTE[2],
    "full_target_expert": PALETTE[3], "threshold_rule": "gray",
}
CONDITIONS = ["zero_shot", "finetuned", "scratch", "full_target_expert", "threshold_rule"]

# gap vs risk needs both files merged on site
ORDER = ["hebei_central", "beijing_plain", "henan_north", "shaanxi_guanzhong", "ningxia_irrigation"]


def main():
    apply_style()
    summary = pd.read_csv(SUMMARY_PATH)
    risk = pd.read_csv(RISK_PATH)
    risk_col = "site_id" if "site_id" in risk.columns else "target_site"
    m = summary.merge(risk, left_on="target_site", right_on=risk_col)

    test_years = list(SPLIT.final_test_years)
    year_label = f"{test_years[0]}—{test_years[-1]}"

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f"论文三 对比四：轮作版留一迁移 + 两因子风险验证（audit-v3 物理，{year_label}）", fontsize=13, y=1.03)

    # panel (a): yields by site x condition
    ax = axes[0]
    x = np.arange(len(ORDER))
    width = 0.16
    for i, cond in enumerate(CONDITIONS):
        vals = m.set_index("target_site").loc[ORDER, cond].values
        ax.bar(x + (i - (len(CONDITIONS) - 1) / 2) * width, vals, width, label=CONDITION_LABELS[cond],
               color=CONDITION_COLOR[cond])
    ax.set_xticks(x)
    ax.set_xticklabels([SITE_LABELS_CN[s] for s in ORDER], rotation=20)
    ax.set_ylabel("总产量 (t/ha)")
    ax.set_title("(a) 各目标站点五种条件产量")
    ax.legend(fontsize=8)

    # panel (b): two-factor risk vs actual gap (the validation scatter)
    ax = axes[1]
    gap = m["yield_gap_vs_rule"].values
    risk_val = m["two_factor_risk"].values
    r = np.corrcoef(gap, risk_val)[0, 1]
    ax.scatter(risk_val, gap, s=90, color=PALETTE[0])
    for site, g, rv in zip(m["target_site"], gap, risk_val):
        ax.annotate(SITE_LABELS_CN[site], (rv, g), textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("两因子风险 (距离 × 任务敏感度)")
    ax.set_ylabel("实际迁移缺口 vs 规则 (t/ha)")
    ax.set_title(f"(b) 两因子风险 vs 实测缺口  r={r:.3f}（n=5，方向性证据，CI 见 two_factor_validation_report.csv）")

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
