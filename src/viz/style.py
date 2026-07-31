"""Shared matplotlib/seaborn style for all figures in this project, so
paper figures look like one consistent set instead of default-styled
one-offs. Import `apply_style()` once at the top of a plotting script.
"""

import matplotlib.pyplot as plt

# Windows-bundled CJK fonts confirmed available on this machine; without a
# CJK-capable font matplotlib silently renders Chinese labels as tofu boxes.
CJK_FONT_CANDIDATES = ["Microsoft YaHei", "SimHei", "SimSun"]

# Site order fixed by aridity (wet -> dry), matching the gradient the
# project's analyses keep landing on (data/README.md, src/transfer/README.md)
# - use this order in every multi-site figure so panels are comparable at a
# glance instead of alphabetical/insertion order.
SITE_ORDER = ["shaanxi_guanzhong", "hebei_central", "beijing_plain", "henan_north", "ningxia_irrigation"]
SITE_LABELS_CN = {
    "shaanxi_guanzhong": "陕西关中",
    "hebei_central": "河北中部",
    "beijing_plain": "北京平原",
    "henan_north": "河南北部",
    "ningxia_irrigation": "宁夏灌区",
}

STRATEGY_ORDER = [
    "rainfed",
    "threshold_60pct",
    "threshold_50pct",
    "threshold_40pct",
    "critical_stage",
    "threshold_30pct",
    "fixed_interval_14d",
    "full_irrigation",
]
STRATEGY_LABELS_CN = {
    "rainfed": "雨养",
    "threshold_60pct": "阈值60%",
    "threshold_50pct": "阈值50%",
    "threshold_40pct": "阈值40%",
    "critical_stage": "关键期保护",
    "threshold_30pct": "阈值30%",
    "fixed_interval_14d": "固定间隔14天",
    "full_irrigation": "充分灌溉",
}

# Colorblind-safe categorical palette (Okabe-Ito), used for anything keyed
# by site or strategy so colors stay stable across figures.
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000"]


def apply_style():
    plt.rcParams["font.sans-serif"] = CJK_FONT_CANDIDATES + plt.rcParams["font.sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 200
    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.3
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False


def site_color(site_id: str) -> str:
    return PALETTE[SITE_ORDER.index(site_id) % len(PALETTE)]


def strategy_color(strategy: str) -> str:
    return PALETTE[STRATEGY_ORDER.index(strategy) % len(PALETTE)]
