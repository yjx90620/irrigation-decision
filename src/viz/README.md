论文用图。每个 `fig_*.py` 读 `data/processed/` 里的结果，产出一张多面板复合图到根目录 `figures/`，重跑对应的数据脚本后重跑这里的脚本就能更新图。

- **`style.py`** — 全项目共用的配色、站点/策略的中文标签和固定顺序（按干旱程度从湿到干排：陕西关中→河北中部→北京平原→河南北部→宁夏灌区）、CJK 字体设置。所有画图脚本先 `apply_style()`。**改字体/配色/站点顺序改这一个文件就行，不要在每张图里各自定义。**
- **`fig_baseline_overview.py`** → `figures/fig1_baseline_overview.png` — 基线实验网格四联图：各策略产量分布、节水—产量权衡、区域气候梯度、非生产性水损失。
- **`fig_pareto_fronts.py`** → `figures/fig2_pareto_fronts.png` — NSGA-II 帕累托前沿：5站点叠加对比、前沿宽度随干旱程度变化、目标权衡平行坐标图。
- **`fig_environmental_fingerprint.py`** → `figures/fig3_environmental_fingerprint.png` — 环境指纹雷达图、区域距离热力图、层次聚类树状图、PCA散点图。
- **`fig_rl_training_curve.py`** → `figures/fig4_rl_training_curve.png` — PPO 训练曲线（v1卡死 vs v2修复后，逐checkpoint重建），2x3面板（灌溉量/产量 × 3个代表站点）。
- **`fig_rl_vs_baselines.py`** → `figures/fig5_rl_vs_baselines.png` — PPO v2 vs 阈值规则基线，5站点产量+灌溉量对比。

## 还没做的（等对应研究进展到那一步再补）

- 研究二：偏好模式对比（稳产/节水/经济/平衡四种，目前评估都用的是单一平衡权重）、RL vs 规则 vs NSGA-II 三方综合对比（目前只对比了 RL vs 规则，NSGA-II 前沿还没接进来）、单个决策序列的时间线图（哪天灌了多少水、土壤墒情怎么变化）
- 研究三：留一地区迁移实验的结果对比图、负迁移检测效果图

## 给后续复杂制图留的接口

论文可能需要更细粒度的图（比如某一年某个策略逐日的土壤墒情曲线），现在的 `baseline_experiment_results.csv` 只存了每次模拟的最终汇总指标，没有逐日轨迹。如果要做这类图，需要回 `src/sim/experiment.py` / `src/rl/evaluate_policy.py` 加一个"保存完整逐日 `model.get_water_flux()` / `get_crop_growth()` 结果"的选项——不建议对全部 5400 次跑都存（数据量太大且大部分用不上），跑一个针对性的小批次（比如挑几个代表站点/年份/策略）专门存逐日数据即可。
