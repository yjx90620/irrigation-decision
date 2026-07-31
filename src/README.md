这里存放项目代码，按研究阶段分三个子目录：

- [data/](data/README.md) — 站点/土壤配置，气象数据下载（Open-Meteo + AgERA5），数据一致性检验
- [sim/](sim/README.md) — AquaCrop 数字农田：基线灌溉策略、实验网格、NSGA-II 多目标优化
- [rl/](rl/README.md) — 强化学习决策模型：环境封装、安全动作层、PPO 训练与评估

运行顺序大致是 `data/` → `sim/` → `rl/`，后一个目录的脚本依赖前一个目录产出的数据/结果（各目录 README 里有说明）。
