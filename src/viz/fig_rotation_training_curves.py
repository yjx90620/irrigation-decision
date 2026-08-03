"""Paper-2 training-diagnosis figure (对比一, rotation era): PPO yield and
irrigation over training steps, reconstructed from the ~100k-step
checkpoints by scripts/reconstruct_rotation_learning_curve.py (3 fixed
evaluation scenarios, balanced preference). Mean +/- std across the 3
independent seeds per mode.

audit-v2 (P0-9): this figure is exactly the evidence that the gamma=0.995
policies converge to water-minimizing behavior - irrigation collapses
toward ~60-80mm while yield plateaus well below the rule baselines (the
terminal yield bonus is discounted away over the ~122-step episode). The
gamma=1.0 arm's curves land in the same CSV with _gamma1 run names once
that ablation finishes.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import pandas as pd
from style import PALETTE, apply_style

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "ppo_rotation_learning_curve.csv"
OUT_PATH = Path(__file__).resolve().parents[2] / "figures" / "fig_rl_training_curves_rotation.png"

MODE_LABELS = {"direct": "直接RL (γ=0.995)", "residual": "残差RL (γ=0.995)"}
MODE_COLOR = {"direct": PALETTE[0], "residual": PALETTE[1]}


def base_run(policy: str) -> str:
    # "ppo_rotation_direct_seed1" -> "ppo_rotation_direct" (keep _gammaN tags)
    return re.sub(r"_seed\d+$", "", policy)


def main():
    apply_style()
    df = pd.read_csv(DATA_PATH)
    df["mode"] = df["policy"].map(base_run)
    # keep only the gamma=0.995 arm (gamma1 checkpoints are evaluated later)
    df = df[~df["mode"].str.contains("_gamma1", na=False)]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    fig.suptitle("论文二 对比一：PPO 训练过程（3 固定评估情景 × 3 seed 均值±标准差，平衡偏好）", fontsize=13, y=1.03)

    for ax, metric, title in zip(axes, ["total_yield_t_ha", "total_irrigation_mm"],
                                 ["(a) 系统总产量随训练步数", "(b) 总灌溉量随训练步数"]):
        for mode in ["direct", "residual"]:
            sub = df[df["mode"] == f"ppo_rotation_{mode}"]
            g = sub.groupby("timesteps")[metric].agg(["mean", "std"])
            ax.plot(g.index, g["mean"], "o-", color=MODE_COLOR[mode], label=MODE_LABELS[mode])
            ax.fill_between(g.index, g["mean"] - g["std"], g["mean"] + g["std"],
                            color=MODE_COLOR[mode], alpha=0.15)
        # rule-baseline reference lines for yield (balanced evaluation, mean over sites in the figure)
        if metric == "total_yield_t_ha":
            ax.axhline(15.45, color="gray", linestyle="--", linewidth=1, label="预留配额规则 (15.5 t/ha 参照)")
            ax.axhline(14.21, color="darkgray", linestyle=":", linewidth=1, label="阈值规则 (14.2 t/ha 参照)")
        ax.set_xlabel("训练步数")
        ax.set_ylabel("t/ha" if "yield" in metric else "mm")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.set_xticks([100000, 200000, 300000, 400000])

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
