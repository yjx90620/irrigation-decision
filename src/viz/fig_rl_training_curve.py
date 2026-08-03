"""Composite figure: PPO training curves, v1 (stuck at zero - see
src/rl/README.md "踩过的第二个坑") vs v2 (VecNormalize-fixed), reconstructed
from checkpoints by src/rl/reconstruct_learning_curve.py. 2x3 grid: rows
are irrigation/yield, columns are the 3 evaluated scenarios (site fixed,
year=2015, balanced preference weights).

audit-v2 (P0-12): these are SINGLE-SEASON PROTOTYPE learning curves,
archived to data/processed/invalidated/legacy_v1/. The script refuses to
load them unless --allow-legacy is passed.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import pandas as pd
from style import PALETTE, SITE_LABELS_CN, apply_style

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "ppo_learning_curve.csv"
LEGACY_DATA_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "processed" / "invalidated" / "legacy_v1"
    / "ppo_learning_curve.csv"
)
OUT_PATH = Path(__file__).resolve().parents[2] / "figures" / "fig4_rl_training_curve.png"
RUN_LABELS = {"ppo_irrigation": "v1（未归一化，卡死）", "ppo_irrigation_v2": "v2（VecNormalize修复后）"}
SCENARIO_SITES = ["hebei_central", "ningxia_irrigation", "shaanxi_guanzhong"]


def main(allow_legacy: bool):
    if allow_legacy:
        path = LEGACY_DATA_PATH
        print("WARNING: loading SINGLE-SEASON PROTOTYPE results (legacy_v1) - not valid for final claims")
    else:
        raise SystemExit(
            "fig4 reads single-season PROTOTYPE results (data/processed/invalidated/legacy_v1/) - "
            "invalidated by audit-v2 P0-12. Pass --allow-legacy for development record only."
        )
    apply_style()
    df = pd.read_csv(path)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    fig.suptitle("研究二：PPO 训练曲线（checkpoint 逐点评估重建，2015年固定场景）", fontsize=15, y=1.02)

    for col, site_id in enumerate(SCENARIO_SITES):
        site_df = df[df.site_id == site_id]
        ax_irr, ax_yield = axes[0, col], axes[1, col]
        for i, run in enumerate(["ppo_irrigation", "ppo_irrigation_v2"]):
            run_df = site_df[site_df.run == run].sort_values("timesteps")
            color = PALETTE[i]
            ax_irr.plot(run_df.timesteps, run_df.irrigation_mm, "-o", color=color, label=RUN_LABELS[run], markersize=3)
            ax_yield.plot(run_df.timesteps, run_df.dry_yield_t_ha, "-o", color=color, markersize=3)
        ax_irr.set_title(SITE_LABELS_CN[site_id])
        ax_irr.set_ylabel("灌溉量 (mm)" if col == 0 else "")
        ax_yield.set_ylabel("产量 (t/ha)" if col == 0 else "")
        ax_yield.set_xlabel("训练步数")
        ax_yield.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))

    axes[0, 0].legend(fontsize=9, loc="upper left")
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
