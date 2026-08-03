"""Composite figure: leave-one-out transfer results (研究方案 7.2/10.1),
from data/processed/leave_one_out_transfer.csv. 2 panels: yield by
site/condition (zero-shot/fine-tuned/threshold-rule), and environmental
distance vs transfer gap scatter (testing whether distance predicts
difficulty).

audit-v2 (P0-12): this figure is built from SINGLE-SEASON PROTOTYPE
results archived to data/processed/invalidated/legacy_v1/ (invalidated
by the system upgrade). The rotation-era transfer results live in
leave_one_out_rotation_transfer.csv and have their own figure
(fig_transfer_rotation if present); this script refuses to load the
legacy CSV unless --allow-legacy is passed.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import pandas as pd
from style import PALETTE, SITE_LABELS_CN, SITE_ORDER, apply_style

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "leave_one_out_transfer.csv"
LEGACY_DATA_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "processed" / "invalidated" / "legacy_v1"
    / "leave_one_out_transfer.csv"
)
OUT_PATH = Path(__file__).resolve().parents[2] / "figures" / "fig6_transfer_leave_one_out.png"
CONDITION_LABELS = {"zero_shot": "零样本迁移", "finetuned": "目标域微调", "threshold_rule": "阈值规则（参照）"}


def main(allow_legacy: bool):
    if allow_legacy:
        path = LEGACY_DATA_PATH
        print("WARNING: loading SINGLE-SEASON PROTOTYPE results (legacy_v1) - not valid for final claims")
    else:
        raise SystemExit(
            "fig6 reads single-season PROTOTYPE results (data/processed/invalidated/legacy_v1/) - "
            "invalidated by audit-v2 P0-12. Pass --allow-legacy for development record only."
        )
    apply_style()
    df = pd.read_csv(path)
    df["site_cn"] = df["target_site"].map(SITE_LABELS_CN)
    site_order_cn = [SITE_LABELS_CN[s] for s in SITE_ORDER]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("研究三：留一地区迁移验证（2018—2022 测试年）", fontsize=14, y=1.02)

    ax = axes[0]
    summary = df.groupby(["target_site", "condition"])["dry_yield_t_ha"].mean().unstack().loc[SITE_ORDER]
    x = range(len(summary))
    width = 0.25
    for i, cond in enumerate(["zero_shot", "finetuned", "threshold_rule"]):
        offset = (i - 1) * width
        ax.bar([j + offset for j in x], summary[cond], width, label=CONDITION_LABELS[cond], color=PALETTE[i])
    ax.set_xticks(list(x))
    ax.set_xticklabels(site_order_cn, rotation=20)
    ax.set_ylabel("产量 (t/ha)")
    ax.set_title("(a) 零样本 / 微调 / 规则基线 产量对比")
    ax.legend(fontsize=9)

    ax = axes[1]
    dist_gap = df.groupby("target_site").agg(
        dist=("nearest_source_distance", "first"),
    )
    yield_by_cond = df.groupby(["target_site", "condition"])["dry_yield_t_ha"].mean().unstack()
    dist_gap["gap"] = yield_by_cond["threshold_rule"] - yield_by_cond["zero_shot"]
    for site_id in SITE_ORDER:
        ax.scatter(dist_gap.loc[site_id, "dist"], dist_gap.loc[site_id, "gap"], s=160, color=PALETTE[SITE_ORDER.index(site_id) % len(PALETTE)], zorder=3)
        ax.annotate(SITE_LABELS_CN[site_id], (dist_gap.loc[site_id, "dist"], dist_gap.loc[site_id, "gap"]), textcoords="offset points", xytext=(8, 5), fontsize=10)
    corr = dist_gap["dist"].corr(dist_gap["gap"])
    ax.set_xlabel("到最近源站点的环境距离")
    ax.set_ylabel("产量差距（规则基线 − 零样本迁移，t/ha）")
    ax.set_title(f"(b) 环境距离 vs 迁移难度（相关系数 r={corr:.2f}）")

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
