# 结果溯源（result provenance）与复现链

audit-v3 (7.7)：本项目所有**论文引用的数字**必须能追溯到 `data/processed/` 下的一个文件、
一行计算和一条重跑命令——不允许手敲进论文。本文档是这条溯源链的总索引；权威状态以
[docs/RESULT_VALIDITY_MATRIX.csv](RESULT_VALIDITY_MATRIX.csv)（每条产物：状态+失效原因+重跑命令）
和 [data/processed/paper_claims.csv](../data/processed/paper_claims.csv)（每条论文数字：取值+来源文件+
重跑命令，`scripts/collect_paper_claims.py --check` 验证漂移）为准。

## 运行链（全量重跑顺序）

1. **气候与站点数据**：`python src/data/download_cmip6.py`（CMIP6 未来气候，10 文件 + manifest）
2. **轮作基线/任务敏感度**：`python src/transfer/task_sensitivity.py`（雨养 vs 充分灌溉，10 年窗口，
   2000-2009，跨年敏感性分布列为 audit-v3 6.6 新增）
3. **α 配水扫描**：`python src/sim/scan_allocation.py`（validation 窗口 2011-2017，audit-v3 5.1；
   同时产出 `best_fixed_alpha.csv`——对比1b 的 fixed_best 对照唯一来源，audit-v3 7.1）
4. **跨季优化 1a/1b**：`python src/sim/run_cross_season_all_sites.py`（validation 窗口）
5. **边际水价值 v2**：`python src/sim/marginal_water_value_v2.py`（固定剂量脉冲法，audit-v3 5.3 替代
   SMT 扰动版；阶段日历单一来源 `CROP_STAGES` in `src/sim/cropping_systems.py`，audit-v3 6.5）
6. **算法对比**：`python src/sim/optimize_algorithm_comparison.py`（NSGA-II/III/MOEA-D，3 seed）
7. **轮作 RL 训练**：`scripts/regen_train_ppo.py` + `scripts/_gamma1.py`/`_peraction.py`/`_noshaping.py`
   （消融臂，失败即退出传播）；配置对象 `RLExperimentConfig`（PRIMARY_CONFIG = per_quota 势函数
   shaping、gamma=0.995、CPU），每臂 3 seed；训练产出 `ppo_rotation_*_final.zip` +
   `*_vecnormalize.pkl` + `*_site_transitions.csv`
8. **RL 评估**：`scripts/evaluate_arm.py`（7 偏好集 × 平衡，产出 `ppo_rotation_wq_comparison.csv` 等）
9. **迁移/微调**：`python -m src.transfer.leave_one_out_rotation`（5 折叠 × 5 条件：zero_shot/finetuned/
   scratch/full_target_expert/threshold_rule，audit-v3 6.1；测试年从 `sim.temporal_split.SPLIT` 读取，
   final_test = 2018-2025）
10. **两因子风险**：`python src/transfer/two_factor_predictor.py`（距离 × 任务敏感度；指纹与距离矩阵
    前置重跑：`python src/transfer/fingerprint.py` + `python src/transfer/similarity_analysis.py`）
11. **两因子验证报告**：`python scripts/validate_two_factor.py`（Pearson/Spearman + 分层 bootstrap CI +
    leave-one-out 降级表，固定 seed 20260804；audit-v3 6.2/6.3）
12. **图表**：`python src/viz/fig_*.py`（输入齐后重跑；fig3 雷达特征已按新指纹 schema 修复）
13. **论文数字**：`python scripts/collect_paper_claims.py`（写 paper_claims.csv）→ 论文按该文件取值
14. **矩阵收尾**：`python scripts/build_result_validity_matrix.py --mark-valid`（把 regenerate_needed
    翻转为 valid_regenerated，与重跑日志对照）

## 约定（所有结果一致遵守）

- **时间分割**（audit-v3 5.1）：development 1982-2010 / validation 2011-2017 / final_test 2018-2025，
  单一来源 `src/sim/temporal_split.py` 的 `SPLIT`。选择类分析（α 扫描/跨季优化/MWV）只用 validation；
  所有"测试年"结论只用 final_test。
- **配置**：`RLExperimentConfig`（`src/rl/experiment_config.py`）不可变、构造时校验、hash 敏感；
  PPO 训练与评估共享同一配置对象（γ 消融由 `gamma1` 臂的独立 launcher 提供，失败传播）。
- **随机性**：`seed_everything()` 固定每个 seed 的 Python/numpy/torch；3 个独立 seed 是正式模式。
- **PBRS**：严格势函数 shaping（`Phi(terminal)=0`、无作物切换重置），恒等式成立由
  `src/rl/test_audit_v3_env.py` 验收。
- **记账**：施水按模型实际施水量（irr_cum 差值），配额为硬约束；安全层介入率只用安全规则标志。
- **单一来源**：阶段日历 `CROP_STAGES`（cropping_systems.py）、时间分割 `SPLIT`（temporal_split.py）、
  固定 α `best_fixed_alpha.csv`（scan_allocation.py 派生）、论文数字 `paper_claims.csv`
  （collect_paper_claims.py 派生）——任何一处手写重复即视为 bug。

## 失效与重跑记账

- audit-v3 修改了轮作物理（日历/休耕）、RL 环境（严格 PBRS/系统级产量/实际施水）、时间分割、
  迁移臂与指纹 schema，因此 **audit-v2 生成的全部轮作期结果在矩阵中标记 regenerate_needed**
  （见 RESULT_VALIDITY_MATRIX.csv 的 reason 列），重跑完成后用 `--mark-valid` 翻转。
- 每次重跑后执行 `python scripts/collect_paper_claims.py --check`：任何与数据不一致的注册数字都会
  以 DRIFT 形式报出，论文改写必须基于无漂移的注册表。
