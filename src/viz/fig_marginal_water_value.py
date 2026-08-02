"""Figure: stage-wise marginal water value (MWV), from
data/processed/marginal_water_value.csv (src/sim/marginal_water_value.py).

Key finding: maize's first growth stage (right after wheat harvest, when
residual soil moisture is lowest) has by far the highest MWV across all
5 sites - wheat's MWV stays near zero throughout. This is the mechanistic
explanation behind paper 1's allocation finding (marginal_water_value.py's
docstring, 论文一's discussion section): water is most valuable exactly
where the rotation handoff leaves the profile driest, which is also why a
fixed 50/50 split leaves yield on the table.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import pandas as pd
from style import SITE_LABELS_CN, SITE_ORDER, apply_style, site_color

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "marginal_water_value.csv"
OUT_PATH = Path(__file__).resolve().parents[2] / "figures" / "fig9_marginal_water_value.png"


def main():
    apply_style()
    df = pd.read_csv(DATA_PATH)
    df["label"] = df["crop"] + "_S" + df["stage"].astype(str)
    sites = [s for s in SITE_ORDER if s in df["site_id"].unique()]

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle("研究一 对比4：分生育阶段边际水价值 (MWV) —— 玉米生育前期(紧接小麦收获)水分价值最高", fontsize=13)

    labels = sorted(df["label"].unique(), key=lambda s: (s.split("_")[0], int(s.split("S")[-1])))
    x = range(len(labels))
    width = 0.15
    for i, site_id in enumerate(sites):
        sub = df[df["site_id"] == site_id].set_index("label").reindex(labels)
        offsets = [xi + (i - len(sites) / 2) * width + width / 2 for xi in x]
        ax.bar(offsets, sub["mwv_t_ha_per_mm"], width=width, color=site_color(site_id), label=SITE_LABELS_CN[site_id], alpha=0.85)

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("边际水价值 (t/ha per mm)")
    ax.set_xlabel("作物_生育阶段")
    ax.legend(fontsize=9)

    plt.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PATH)
    print(f"saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
