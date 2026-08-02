# 已失效结果（invalidated）

这个目录下的所有结果都建立在 `src/sim/rotation.py` / `src/rl/rotation_env.py` 的一个已确认 bug 之上：
小麦→玉米的季节交接固定用玉米播种日 06-15 作为时间锚点，但小麦的实际模拟结束日期是
GDD 驱动的，实测存在小麦收获晚于 06-15 的年份（例：河北中部2013年收获06-22）——这种情况下
玉米季的初始土壤水分状态其实来自尚未发生的"未来"几天，跨年度的交接也有类似问题。

这不是唯一问题。同一批结果还牵涉：年度硬配额在临界亏缺时被安全层悄悄突破（不再是硬约束）、
RL 训练的站点采样按 episode 而非按 transition 均匀（双季站点被系统性过采样）、奖励被稠密的
逐步水分胁迫代理项主导而非终端产量、residual 动作离散裁剪后有重复值、评估用未折扣回报而
PPO 实际按 gamma=0.995 优化、超体积参考点方向算错。完整清单和修复计划见
[docs/审计修复计划.md](../../docs/审计修复计划.md)（对应 `~/.claude/plans/unified-inventing-lampson.md`
的仓库内副本）。

**这里的文件不是被删除，是作为开发记录保留**——展示了从"单季原型"到"轮作模型"这次升级踩过的
坑，本身有诊断价值。但在 P0 问题修复并通过验收测试之前，**这里的任何数值都不能被论文引用为
结论**。

| 文件 | 原路径 | 说明 |
| --- | --- | --- |
| `allocation_scan*.csv` | `data/processed/` | α配水扫描（论文一对比1a） |
| `cross_season_pareto_*.csv` | `data/processed/` | 跨季联合优化前沿（论文一对比1/1b） |
| `marginal_water_value.csv` | `data/processed/` | 阶段边际水价值（论文一对比4） |
| `algorithm_comparison_hebei_central.csv` | `data/processed/` | NSGA-II/III/MOEA-D对比（论文一对比3，超体积参考点单独有算法层面的bug） |
| `rotation_policy_comparison.csv`、`ppo_rotation_*`、`ppo_checkpoints/ppo_rotation_*` | `data/processed/` | 论文二核心对比（残差RL vs 直接RL vs 规则）及其训练产物 |
| `task_sensitivity.csv` | `data/processed/` | 论文三任务敏感度指标（另有零产量被误判缺失值的独立bug） |

对应图表在 [figures/invalidated/](../../figures/invalidated/)。
