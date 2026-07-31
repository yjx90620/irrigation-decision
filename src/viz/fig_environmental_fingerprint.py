"""Composite figure: environmental fingerprint and cross-site similarity
(研究方案 6.4/6.5), from data/processed/environmental_fingerprints.csv,
site_distance_matrix.csv, site_pca.csv. 4 panels: climate radar chart,
distance heatmap, dendrogram, PCA scatter.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import MinMaxScaler
from style import SITE_LABELS_CN, SITE_ORDER, apply_style, site_color

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
OUT_PATH = Path(__file__).resolve().parents[2] / "figures" / "fig3_environmental_fingerprint.png"

RADAR_FEATURES = ["P_mm", "ET0_mm", "aridity_index", "CV_P", "max_dry_spell_days", "hot_days", "GDD"]
RADAR_LABELS_CN = ["生育期降水", "生育期ET0", "干旱指数", "降水变异", "最长无雨天", "高温日数", "有效积温"]


def main():
    apply_style()
    fp = pd.read_csv(DATA_DIR / "environmental_fingerprints.csv", index_col="site_id").loc[SITE_ORDER]
    dist = pd.read_csv(DATA_DIR / "site_distance_matrix.csv", index_col="site_id").loc[SITE_ORDER, SITE_ORDER]
    pca = pd.read_csv(DATA_DIR / "site_pca.csv", index_col="site_id").loc[SITE_ORDER]

    fig = plt.figure(figsize=(15, 13))
    fig.suptitle("研究三：站点环境指纹与区域相似性分析", fontsize=15, y=1.0)
    gs = fig.add_gridspec(2, 2)

    ax = fig.add_subplot(gs[0, 0], projection="polar")
    norm = MinMaxScaler().fit_transform(fp[RADAR_FEATURES])
    norm_df = pd.DataFrame(norm, index=fp.index, columns=RADAR_FEATURES)
    angles = np.linspace(0, 2 * np.pi, len(RADAR_FEATURES), endpoint=False).tolist()
    angles += angles[:1]
    for site_id in SITE_ORDER:
        values = norm_df.loc[site_id].tolist()
        values += values[:1]
        ax.plot(angles, values, color=site_color(site_id), label=SITE_LABELS_CN[site_id], linewidth=1.8)
        ax.fill(angles, values, color=site_color(site_id), alpha=0.06)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(RADAR_LABELS_CN, fontsize=9)
    ax.set_yticklabels([])
    ax.set_title("(a) 气候指纹雷达图（归一化）", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8)

    ax = fig.add_subplot(gs[0, 1])
    labels_cn = [SITE_LABELS_CN[s] for s in dist.index]
    sns.heatmap(
        dist.set_axis(labels_cn, axis=0).set_axis(labels_cn, axis=1), annot=True, fmt=".1f", cmap="YlOrRd",
        ax=ax, cbar_kws={"label": "标准化欧氏距离"},
    )
    ax.set_title("(b) 区域气候距离矩阵")

    ax = fig.add_subplot(gs[1, 0])
    Z = linkage(fp[RADAR_FEATURES].apply(lambda c: (c - c.mean()) / c.std()), method="ward")
    dendrogram(Z, labels=[SITE_LABELS_CN[s] for s in fp.index], ax=ax, color_threshold=0)
    ax.set_title("(c) 层次聚类树状图（Ward法）")
    ax.set_ylabel("距离")

    ax = fig.add_subplot(gs[1, 1])
    for site_id in SITE_ORDER:
        ax.scatter(pca.loc[site_id, "PC1"], pca.loc[site_id, "PC2"], s=160, color=site_color(site_id), zorder=3)
        ax.annotate(SITE_LABELS_CN[site_id], (pca.loc[site_id, "PC1"], pca.loc[site_id, "PC2"]), textcoords="offset points", xytext=(8, 8), fontsize=10)
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.axvline(0, color="gray", linewidth=0.8)
    ax.set_xlabel("PC1（主要对应干旱程度）")
    ax.set_ylabel("PC2")
    ax.set_title("(d) 主成分分析（PC1+PC2 解释 96.7% 方差）")

    plt.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PATH)
    print(f"saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
