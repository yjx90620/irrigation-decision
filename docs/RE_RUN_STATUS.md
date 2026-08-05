# 全量重跑状态与恢复笔记（2026-08-05，audit-v3）

> 机器/会话重启恢复用。背景任务会全部终止；cron（会话级）会丢失。按本文件重续。
> 代码均已提交（分支 audit-fix-v3，远端已同步）；`git log --oneline -5` 可确认最新提交。

## 0. 一键恢复命令（按顺序）

```bat
cd D:\jasonworkspace\IrrigationDecision
git pull origin audit-fix-v3        :: 若重启后在其他机器/目录
.venv\Scripts\python.exe -m pytest src -q   :: 应 85+ passed
```

## 1. 已完成（最终数据，无需重跑）

- 任务敏感度 `data/processed/task_sensitivity.csv`（含 6.6 跨年不确定性列）
- 作物期指纹 `environmental_fingerprints.csv` + 距离矩阵/PCA/聚类（6.5，40 列阶段特征）
- α 配水扫描 `allocation_scan*.csv` + `best_fixed_alpha.csv`（新最优 α：北京 0.5/河北 0.5/河南 0.5/陕西 0.2）
- MWV v2 `marginal_response_v2.csv`（固定剂量法；840 行，772 未定义——如实记录"边际响应基本为零"）
- 算法对比 `algorithm_comparison_hebei_central*.csv`（NSGA-II 0.826 > NSGA-III 0.792 > MOEA/D 0.289）
- 气候对比5 part B `allocation_scan_future.csv` + `best_fixed_alpha_future.csv`（仅陕西 0.2→0.1）
- 图：fig3 / fig7 / fig8 已按新数据生成
- 论文二节水声明已改（5.5—22%）、论文三清单已更新、两因子 bootstrap 报告（旧数据版）已生成

## 2. 进行中但被重启打断（需重跑/续跑）

| 任务 | 状态 | 恢复命令 |
| --- | --- | --- |
| 气候对比5 part A（robustness） | 第 2 次重跑中被打断；**磁盘上的 climate_robustness.csv 是错的（宁夏 0.00/0，两次配额 bug 已修：single_crop=True 工厂）** | `.venv\Scripts\python.exe src\sim\climate_comparison.py --parts a`（~10 分钟，覆盖写） |
| 跨季优化 1a | 跑了 ~3h 无站点产出（可能正常偏慢，**也可能同样踩 AquaCrop 病态**，见 §4） | `.venv\Scripts\python.exe src\sim\run_cross_season_all_sites.py` |
| 跨季优化 1b | 同上 | `.venv\Scripts\python.exe src\sim\run_cross_season_1b_all_sites.py` |
| RL stall 探针 | 435 组合跑到 ~40%（北京+河北全过，固定动作未复现） | `.venv\Scripts\python.exe scripts\_probe_env_stalls.py` |

## 3. 【阻塞项】RL 重训连续卡死（AquaCrop 病态）

- **现象**：修复奖励 bug（3.2 每次收获付产量奖金）后，主臂重训 2 次都在 ~4 分钟后卡死——
  2 个 worker 100% 占满（45+ 分钟），其余 18 个空转（lockstep 等待），无任何 checkpoint。
- **已排除**：不是奖励 bug（修复后 env 测试 17 passed）；不是随机（seed 确定性→重启无意义，
  相同序列再踩同一点）；固定动作探针（0/10/20mm×全部站点×年份）至今未复现。
- **线索**：P0-4 worker 按站点固定（每站 2 worker），卡死的 2 个 worker = 同一站点的两个。
  概率上 4/4 作业卡死 → 该病态对某站点某动作模式是"高概率命中"。
- **候选根因**：`aquacrop/core.py:278` 主日循环 `while model_is_finished is False`——若时钟不推进
  则无限循环（烧满 CPU）。`calculate_HIGC` 已打补丁（200k 上限），`root_development:159` EndProf
  循环有界（layeri 递增到 Soil_nLayer）——均已排除。
- **下一步诊断**（恢复后）：
  1. 给 `residual_gym_env.reset()` 加临时 ep 日志（环境变量门控，追加 `site,year` 到文件）；
  2. 前台单跑一个 `direct seed0` 作业（15 分钟超时），卡死后读日志尾部 → 精确 (site, year)；
  3. 对该 (site, year) 用不同动作深度复现 → 定位 AquaCrop 内部死循环点 → 打补丁（同 HIGC 做法）
     或从训练域排除（如实记录 caveat）。
- **绕过选项**：训练域排除病态 (site, year) 并写进论文局限；或用 watchdog 模式（seed 偏移重启）
  但会破坏"3 seed 可复现"承诺，不推荐。

## 4. 风险提示

- **跨季 1a/1b 也可能踩同一 AquaCrop 病态**（NSGA-II 评估同样跑 AquaCrop）：3h 无产出可疑，
  恢复后先确认其进程是否在跑（CPU 活跃度），卡死则同样用探针定位。
- **cron 已丢失**：重启后不再有每 17 分钟监控；用下面的恢复链手动推进。
- 后台任务 ID（已失效，仅作记录）：bg_5b53a25c(探针) bg_2b226c51(气候A) bg_4b9d20cb(1a) bg_3cbb9e5f(1b)。

## 5. 恢复后的推进链（依赖顺序）

1. 诊断并解决 RL 卡死（§3）→ 重启主臂重训 `scripts\regen_train_ppo.py`
   （启动器已带 stall 检测+重启，`scripts\_regen_launcher.py`）
2. 主臂完成后：`scripts\evaluate_arm.py` → 依次 `regen_train_ppo_gamma1.py` /
   `regen_train_ppo_peraction.py` / `regen_train_ppo_noshaping.py`，每批后评估
   （gamma1 用 `evaluate_gamma1.py`；peraction/noshaping 用
   `python -c "import sys; sys.path.insert(0,'scripts'); from evaluate_arm import main;
   from experiment_config import RLExperimentConfig;
   main(RLExperimentConfig(water_normalizer='max_action'))"`，noshaping 用
   `RLExperimentConfig(shaping_mode='none')`）
3. 跨季 1a/1b（与 RL 并行无依赖）→ 8 个 cross_season_pareto*.csv
4. 迁移：`.venv\Scripts\python.exe -m src.transfer.leave_one_out_rotation --seed 0`
   （单 seed；5 折叠 × 5 臂，~2-3h）→ `two_factor_predictor.py` → `validate_two_factor.py`
5. 图：fig3/fig7/fig8/fig10/fig_transfer_rotation/fig_rl_training_curves_rotation
6. `scripts\collect_paper_claims.py` → `scripts\build_result_validity_matrix.py --mark-valid`
7. 重写论文一/二/三终稿 → 打时间戳 release

## 6. 常用坑（本会话踩过）

- cmd.exe 下 python -c 多行字符串会静默失败 → 一律写成脚本文件再跑
- 启动器日志块缓冲（log 文件 0 字节是正常，看 shell 输出文件的 STALLED/DONE 行）
- `latest_checkpoint_mtime` 为 0 时 stall 计时器需要启动宽限期（已修）
- 数据产物 resume 判定按内容（列/行数），schema 变更后要删旧文件再重跑
  （clean_stale_audit_v3.py 可复用）
