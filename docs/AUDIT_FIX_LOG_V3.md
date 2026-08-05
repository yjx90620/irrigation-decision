# 第三轮审计整改日志（audit-v3 fix log）

对应外部第三轮审计报告（GPT，2026-08）。整改原则与前两轮相同：**逐条核实 →
只修确认属实的问题 → 每项带测试 → 旧结果归档/清理不删除 → 受影响结果标记失效并全量重跑**。

审计共 19 项指控，18 项核实属实，1 项过时（上一轮已修）。分 8 个阶段整改，
全部代码修改已提交到 `audit-fix-v3` 分支；测试基线 `pytest src` = **85 passed**
（第二轮末为 42）。

## 阶段 1（5.1/5.2）：时间分割与作物参数口径
- **5.1 时间泄漏**：统一 `src/sim/temporal_split.py` 的 `SPLIT`——
  development 1982-2010 / validation 2011-2017 / final_test 2018-2025。
  所有"选择类"分析（α 扫描、跨季优化、MWV）改用 validation 窗口；
  所有"测试年"结论只用 final_test（原 2018-2022 与旧切分混杂）。
  下游全部从 `SPLIT` 读取，禁止手写年份。
- **5.2 小麦率定口径**：`docs/methods/crop_calibration.md` 明确重分类为
  "日历可行性调整（calendar-feasibility adjustment）"——只保证收获窗内成熟，
  无真实物候观测，论文不得称"站点校准"。

## 阶段 2（2.1/3.x）：RL 环境重写
- `RLExperimentConfig`（`src/rl/experiment_config.py`）：不可变、构造时校验、
  hash 敏感；PPO 训练与环境共享同一 gamma/shaping/water-normalizer，
  消除 γ 消融混淆（旧 gamma1 臂训练用 γ=1.0 而环境 shaping 仍用 0.995）。
- `rotation_env.py` 全面重写：严格势函数 PBRS（Phi(terminal)=0、无作物切换重置）、
  系统级产量归一化（sum yields / sum refs，单双季可比）、日历法 days_to_harvest
  （不含休耕）、实际施水缺口、干预原因拆分（safety/quota/delivery 分列）、
  acute_stress 命名、safety_enabled 开关。
- 验收测试 `test_audit_v3_env.py`。

## 阶段 3（4.x）：数据硬化
- `download_cmip6.py`：ColumnStatus 严格列状态 + 双期对称排除；
  `--check-only` 真正只读；重试控制流修复（孤儿代码块删除）。
- `weather.py` 严格加载器契约（单文件、日期完整、物理范围）；
  `et0.py` 用 rh_max/rh_min（FAO eq.17）+ 参考案例测试；
  SSP 标签修正（对比5 为 HighResMIP delta-change，非 SSP 下推）。

## 阶段 4（5.3）：MWV v2
- `marginal_water_value_v2.py`：固定剂量（-d/0/+d 脉冲 + 实际施水 + 中心差分）
  替代 SMT 扰动；完整键集 + 显式 status（undefined_no_actual_delta）。

## 阶段 5（6.1）：迁移臂
- `leave_one_out_rotation.py`：新增 `scratch`（同 5 万步预算，公平对照）与
  `full_target_expert`（20 万步上界，明确标注不作公平比较）臂；
  training_budget_source/target 列；修复包式 `from sim.` 导入缺 src 路径的
  潜在 bug（此前从未在 audit-v3 代码上跑过训练）。

## 阶段 6（6.2-6.6）：统计诚实性
- 6.2/6.3：`scripts/validate_two_factor.py` 重写为可复现报告——Pearson/Spearman +
  分层 bootstrap 95% CI（固定 seed 20260804）+ 留一降级表；实测两因子与距离的
  **差值 CI 含 0**、去掉宁夏后两因子相关从 0.971 掉到 0.635——结论口径降级为
  "方向性/探索性证据"。
- 6.4/6.5：指纹按作物期细化——`CROP_STAGES`（cropping_systems.py 单一来源，
  与 MWV v2 共享）+ `fingerprint.py` 每站统一计算的作物期气候特征（40 列）。
- 6.6：`task_sensitivity.py` 增加跨年敏感性分布（std/min/max/n_years）。

## 阶段 7（7.1-7.7）：论文数字与文档
- 7.1：`scripts/collect_paper_claims.py` → `paper_claims.csv`（13 项论文数字的
  单一来源，`--check` 检测漂移）；`best_fixed_alpha.csv` 由扫描派生为单一来源
  （修掉代码里 henan 0.8/beijing 0.7 与扫描结果不一致的陈旧硬编码）。
- 7.2：论文二节水声明修正为实测 5.5—22%（均值约 17%），原 15—25% 有误。
- 7.3：`build_result_validity_matrix.py` 重写：audit-v3 影响的所有产物标记
  regenerate_needed（带重跑命令），`--mark-valid` 供重跑后翻转。
- 7.4：fig3 指纹雷达用了已不存在的列名（audit-v2 schema 变更后从未再生成）——
  改用真实 cool/warm 特征 + 实际 PCA 方差；fig10/fig_transfer_rotation 年份
  从 SPLIT 读取；迁移图展示 5 臂。
- 7.5：7 个失效链接指向归档图并标注"仅作升级前对照"；`scripts/check_doc_links.py`。
- 7.7：`docs/RESULT_PROVENANCE.md` 全链溯源。

## 阶段 8：legacy 隔离
- 单季原型数据脚本（train_ppo/reconstruct_learning_curve/transfer/leave_one_out/
  instance_weighted）无 `--allow-legacy` 直接拒绝运行；`test_legacy_isolation.py`
  静态保证轮作期管线零 legacy 导入；`src/rl/README.md` 顶部加弃用横幅。

## 全量重跑（2026-08-05）
清理 203 个陈旧产物（旧环境 checkpoint/结果），按 `docs/RESULT_PROVENANCE.md`
运行链重跑：任务敏感度 → 指纹/距离矩阵 → α 扫描（新最优 α：北京 0.5/河北 0.5/
河南 0.5/陕西 0.2，方向反转结论保持）→ RL 主臂 6 模型（400k×3 seed）→ 消融
（gamma1/peraction/noshaping）→ 评估 → 学习曲线 → 跨季 1a/1b → MWV v2 →
算法对比 → 对比5 气候实验 → 迁移（5 折叠×5 臂）→ 两因子报告 → 图表 → 论文终稿。

（重跑完成后：`python scripts/build_result_validity_matrix.py --mark-valid`、
`python scripts/collect_paper_claims.py --check`，状态在
[docs/RESULT_VALIDITY_MATRIX.csv](RESULT_VALIDITY_MATRIX.csv) 与
[docs/RESULT_PROVENANCE.md](RESULT_PROVENANCE.md)。）
