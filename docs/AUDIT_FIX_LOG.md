# 第二轮审计整改日志（audit-v2 fix log）

对应外部第二轮审计报告（GPT，2026-08）。整改原则与第一轮相同：**逐条核实 →
只修确认属实的问题 → 每项带测试 → 旧结果归档不删除 → 受影响结果标记失效并计划重跑**。
核实方法：对每条声称，用当前代码 + 实际数据实证（数值实验/文件检查），区分
"属实（本轮修复）"、"上一轮已修复（过时指控）"、"已披露局限（只需措辞核对）"。

测试基线：`pytest src` = **42 passed**（第一轮末为 25）。
编译检查：`compileall src` 通过。

---

## P0 级

### P0-1 轮作时间轴不连续（休耕空档未模拟）— 属实，已修复
- **原问题**：小麦实际收获→玉米固定播种之间的空档（实测 7–36 天，均值 18–21 天）
  的降水/蒸散被完全跳过，剖面被直接传递；玉米收获→下季小麦播种、宁夏 7 个月冬季
  休耕同样被跳过。
- **核实证据**：`_audit_diag_calendar`（已存 data/calibration/wheat_harvest_calendar_audit.csv）：
  4 双季站点 × 1982–2025，空档 7–36 天（均值 17.5–21.1 天）；旧代码
  `run_rotation_year` 直接把小麦季末剖面传给玉米。
- **报告中的"晚收"部分为过时指控**：当前小麦率定参数下 0/44 年晚于 06/15
  （报告引用的"北京13/44"是老参数数据，上一轮 P0-1 已修）。
- **修改**（src/sim/rotation.py、src/sim/cropping_systems.py）：
  每季仿真窗口改为 [上一季**实际**收获日+1, 本季收获]，`off_season=True`——
  休耕期以裸土水量平衡真实模拟；轮作日历常量移入 cropping_systems.py 单一来源；
  `run_rotation_series` 拒绝非连续年份（防跨空档串联）。
- **测试**：test_rotation_calendar.py 新增
  `test_offseason_fallow_water_balance_conserves_water`（休耕不凭空造水）、
  `test_nonconsecutive_years_are_not_chained`；原 4 项日历测试全过。
- **是否改变论文结论**：是——玉米初始墒情改变（休耕干燥效应实测使灌水量需求 +20mm/季量级）。
  全部轮作结果失效（见 RESULT_VALIDITY_MATRIX.csv）。

### P0-2 夏玉米 132 天品种被 112 天窗口截断 — 属实，已修复
- **原问题**：默认 Maize MaturityCD=132 天 > 06/15–10/05 窗口（112 天），充分灌溉下
  从不自然成熟，产量是未完成灌浆的截断值。
- **核实证据**：实测充分灌溉 4 站点 × 5 年 **0/20 季自然成熟**（收获日恒为 10/05）；
  105 天品种（按 105/132=0.7955 缩放 Senescence/HIstart/MaxRooting/YldForm/Emergence）
  全年份 09/28 成熟（余量 7 天），产量 10.3–11.6 vs 截断值 ~9.0 t/ha（+25%）。
  宁夏春玉米用默认 132 天品种在其真实窗口（04/25–09/30）下 09/04 成熟，无需改。
- **修改**（src/sim/cropping_systems.py）：`SUMMER_MAIZE_PARAMS`（105 天，双季站点）、
  `SPRING_MAIZE_PARAMS`（默认，宁夏）、统一作物工厂 `build_crop(site_id, crop_name)`；
  rotation.py / rotation_env.py 不再手工构造 Crop。物候验证口径：每条结果新增
  `matured_naturally`（AquaCrop crop_mature 标志）。
- **测试**：`test_summer_maize_matures_naturally`（4 站点 × 5 年）、
  `test_ningxia_spring_maize_matures_naturally`、`test_build_crop_selects_site_cultivar`。
- **是否改变论文结论**：是——夏玉米产量普遍上移约 25%，全部轮作结果失效。
- **参数来源**：data/calibration/crop_parameter_sources.csv +
  docs/methods/crop_calibration.md；105 天为**文献/情景参数**（华北夏玉米 ~100–110 天），
  非站点实测品种。

### P0-3 年度配额按"请求水量"记账，实际施水量偏低 37.5% — 属实，已修复
- **原问题（细项）**：AquaCrop 施水量 ≤ 请求深度（实测请求 40mm 恒定只施 25mm），
  而 `quota_used` 与水的奖励项用请求值——训练高估用水约 60%，配额叙事描述的是
  从未实际施用的水量。（报告主指控"配额可被安全层突破"为过时指控：上一轮已改为
  硬约束并有断言+3 项测试。）
- **核实证据**：26/26 步请求 40mm → irr_cum 差值恒为 25.0mm。
- **修改**（src/rl/rotation_env.py step()）：`actual_applied = irr_cum 差值`，
  `quota_used`、`water/cost` 奖励、`days_since_last_irr`、`last_irr_mm` 全部改用实际值；
  info 保留 `applied_mm`（请求后）与 `actual_model_irrigation_mm`（实际）区分。
- **测试**：test_rotation_env.py 新增 `test_quota_accounting_uses_actual_model_delta`、
  `test_water_reward_uses_actual_not_requested`、`test_episode_total_irrigation_matches_quota_used`。
- **是否改变论文结论**：是——PPO 训练奖励信号改变，论文二全部 RL 模型需重训。

### P0-4 迁移微调评估加载错误 VecNormalize — 过时指控（上一轮已修复）
- 当前代码：`ft_model, ft_vecnorm = train_rotation_policy(...)` 后
  `load_policy(ft_model, ft_vecnorm, "residual")`（src/transfer/leave_one_out_rotation.py），
  `load_policy` 内置 strict_pairing 断言（train_rotation_compare.py）。

### P0-5 环境指纹与任务不匹配 — 大部分过时（上一轮已修），剩余部分已核对
- 已修：指纹按真实轮作日历分冷/暖季、限 TRAIN_YEARS（src/transfer/fingerprint.py），
  非报告引用的"05-01~09-15 单季"。
- 剩余已知局限（已如实记录在文件 docstring）：土壤块为常量（SoilGrids 不可用）、
  无休耕期特征、类别变量参与距离的口径——记入论文三局限，不构成本轮代码改动。

### P0-6 超体积参考点方向错误 — 过时指控（上一轮已修复）
- 当前代码（optimize_algorithm_comparison.py）：先归一化（ideal/nadir），
  `ref_point=1.1` 并逐点断言 `F_norm < ref_point` 全部成立。

### P0-7 CMIP6 下载器把不完整响应当成功 — 属实，已修复（数据待重下）
- **核实证据**：现有 10 个文件全部存在 `shortwave_radiation_sum_CMCC_CM2_VHR4`
  整列为空（历史+未来）；未来期 EC_Earth3P_HR 各变量 95% 覆盖（缺约 365 天）；
  下载器 `if out_path.exists(): skip`，无任何验证。
- **修改**（src/data/download_cmip6.py）：`validate_cmip6_frame`（日期范围完整、
  无重复、必需变量×模式列非空，整列为空即 `DataValidationError`）、
  `common_models`（历史/未来配对）、原子写入 + manifest sidecar（含覆盖率/哈希/许可）、
  quarantine 目录、`--check-only` 只读检查；每次运行重新验证已存在文件。
  climate_scenario.py 的 `__main__` 现在报告每个变量的模式数（透明度）。
- **测试**：src/data/test_download_cmip6.py（6 项）。
- **现状**：`--check-only` 实测全部 10 个文件不满足新标准（CMCC 辐射空洞）——重下
  属重跑阶段任务；若上游 API 本身缺该列，辐射集合将明确记录为 6 模式而非静默降级。
- **是否改变论文结论**：气候对比（论文一对比5、论文三时域迁移）尚未跑，无现存结果受影响。

### P0-8 未来天气按数组位置合并 — 属实，已修复
- **修改**（src/sim/climate_scenario.py build_future_weather）：辐射/风/湿度改为按
  `Date` 与原始 Open-Meteo 文件 `one_to_one` merge，任何错位日抛
  `DataValidationError`（新增）。其余（逐模式先算 delta、辐射/风/湿度已扰动、海拔
  用站点真实值）为上一轮已修，核实属实。

### P0-9 奖励偏向较长较早季节 — 主指控过时，剩余项已处理
- 主指控（逐步累加 mean_stress）为过时：上一轮已改为势函数塑形 + 终端单次发放。
- 剩余两项本轮处理：(a) **风险指标口径**——RL 的 `risk` 项实为急性胁迫惩罚
  （tr_ratio<0.5），与优化侧的多年产量 CV 不是同一目标；论文二措辞已加说明
  （后续重训时可重命名为 `acute_stress_penalty`）；(b) **γ=0.995 的集长依赖**——
  已报告双口径回报（undiscounted/discounted），γ=1.0 对照实验列为论文二开放事项。

### P0-10 Residual"按构造从规则开始" — 过时指控（上一轮已修复）
- 当前 RESIDUAL_DELTAS=[-20,5,10,15,20]（裁剪后 5 个不重复动作，有测试），
  文档已改为"结构先验，非初始性能等于规则"表述（论文二同步修正）。

### P0-11 reset(seed=...) 不控制抽样 — 过时指控（上一轮已修复）
- 当前 `reset(seed=...)` 会重设 `self._rng`（test_seeding.py 4 项测试覆盖）。
- 报告附带的"偏好用 Dirichlet 而非归一化均匀"建议：确为更标准的单纯形采样，
  但当前文档未声称均匀；列为 P2 改进（重训时可选），不构成本轮改动。

### P0-12 旧单季原型结果未归档 — 属实，已修复
- **核实**：110 个单季原型产物（baseline 网格、pareto_front、rl_vs_baselines、
  leave_one_out_transfer、ppo_irrigation* 模型/检查点/学习曲线、transfer_* 模型、
  fig1–6）此前位于 processed/figures 根目录、未归档。
- **修改**：scripts/archive_legacy_v1.py 以 `git mv` 移入
  `data/processed/invalidated/legacy_v1/` 与 `figures/invalidated/legacy_v1/`，
  附逐文件 SHA-256 迁移清单（migration_manifest.csv）与 PROTOTYPE 标记；fig1/2/4/5/6
  与 batch_optimize_loam.py 拒绝加载/生成原型产物，除非显式 `--allow-legacy`。
- **是否改变论文结论**：否（这些结果此前已被 README 标注原型，本轮改为物理隔离）。

### P0-13 文件存在即跳过 — 属实，已修复（最严重脚本）
- **核实**：盘点 ~20 处 file-exists/csv-content skip（run_cross_season_all_sites、
  run_cross_season_1b、batch_optimize_loam、marginal_water_value、task_sensitivity、
  leave_one_out_rotation 等；scan_allocation/run_allocation_scan 上一轮已有
  failure_reason 修复）。
- **修改**：新增 src/utils/（atomic_io.py 原子写入 + run_manifest.py 内容级完成判定
  `csv_result_complete`）；上述脚本的跳过条件改为"必需列齐全 + 行数达标 + 无 NaN"，
  或（原型生成器）加 --allow-legacy 闸门。
- **剩余**：完整 RunManifest（code/config/input 哈希、状态机）为 P2 工程化阶段；
  本轮以内容判定先堵住"失败被误报完成"。

---

## P1 级

| 编号 | 问题 | 处置 |
| --- | --- | --- |
| P1-1 | "独立优化"实为固定 50/50 联合 SMT | 已修：mode 标签改为 `fixed_split_alpha_0.50` / `fixed_best`（optimize_cross_season.py）；论文对比1b 已用 best-fixed-alpha 强化基线 |
| P1-2 | 边际水价值定义（ΔSMT 不保证 Δwater>0） | 已修：明确为 SMT 局部敏感性；Δwater≤0 输出 NaN 不编造比值；新增 baseline/perturbed 配对列（marginal_water_value.py） |
| P1-3 | RL 用完美未来天气 | 已披露（论文二 §2.2、论文三 §4.4：仅实现完美预报上界）；本轮未实现 no-forecast 变体，保留为开放事项 |
| P1-4 | 偏好策略只在平衡权重评估 | 已修：train_rotation_compare.py 新增 `PREFERENCE_EVAL_SETS`（4 极端 + 2 两两权衡 + balanced），PPO 模型全偏好集评估并输出 `preference` 列 |
| P1-5 | 迁移协议归纳/传导区分 | 已核对：指纹限 TRAIN_YEARS（上一轮已修）；instance_weighted 措辞在论文三中已限定"元数据辅助"；协议标注留待重跑阶段 |
| P1-6 | 双因子 n=5 样本量 | 已披露（论文三：方向性证据，非统计显著结论）；不扩大站点（超出本项目范围） |
| P1-7 | ET0 参数透明 | 已修：文档化 RH 近似假设、截断统计（n_days_clamped_at_zero）、验证阈值（|bias|<0.3/RMSE<0.8）；风速 mean 与海拔为上一轮已修 |
| P1-8 | 初始墒情是方法假设 | 已核对措辞：论文一/二"真实观测"改为"ERA5-Land 再分析（网格尺度），非田间实测剖面"；敏感性实验列重跑阶段 |
| P1-9 | 作物参数缺来源链 | 已修：data/calibration/crop_parameter_sources.csv + docs/methods/crop_calibration.md + 率定数据存档（wheat/maize/ningxia 三份 CSV） |
| P1-10 | Open-Meteo vs AgERA5 非独立验证 | 措辞已核对（论文一：一致性检验，非独立验证）；重命名脚本留待 P2 |
| P1-11 | 随机重复与元数据 | 已部分修（算法对比 3 seed、RL 3 seed 为上一轮）；seed/Python/依赖/commit 元数据纳入 RunManifest（P2） |

---

## 过时指控清单（核实为上一轮已修复，本轮未改动）

P0-1 晚收计数、P0-3 主指控（配额非硬约束）、P0-4、P0-5 主体、P0-6、P0-8 的
ET0 扰动部分、P0-9 主指控、P0-10、P0-11；以及"转义文件名 #U7814"（不存在，
全部 docs 链接可解析）、"宁夏 CMIP6 缺失"（文件齐全）、task_sensitivity
`bool(0)` bug（已修）、`.venv` 硬编码（已换 sys.executable）。

## 结果失效与重跑

- 修改 P0-1/P0-2（物理）→ 全部轮作仿真/优化/RL/迁移结果失效；
- 修改 P0-3（奖励/记账）→ 全部 PPO 模型与迁移模型失效；
- 修改 P1-4（评估轴）→ RL 对比表重生成（不重训）。
- 全部状态见 `docs/RESULT_VALIDITY_MATRIX.csv`；重跑顺序按 README 运行链：
  轮作基线 → α 扫描 → 跨季优化(1a/1b) → 边际水价值 → 算法对比 → 轮作 PPO(3 seed)
  → 迁移/微调 → 两因子风险 → 气候情景 → 图表。
- CMIP6 数据：现有 10 文件不满足新验证标准，需重下（P0-7）。

## 本轮新增/修改测试（42 passed）

- src/data/test_download_cmip6.py（6 项新增）
- src/sim/test_rotation_calendar.py（+4 项：成熟率×2、品种工厂、休耕水量、非连续拒绝）
- src/rl/test_rotation_env.py（+3 项：实际施水记账、奖励口径、季总水量一致性）

## 全量重跑进展（2026-08-03，进行中）

| 结果 | 状态 | 关键数字（audit-v2 物理） |
| --- | --- | --- |
| 任务敏感度 task_sensitivity.csv | ✅ | 北京 0.647 / 河北 0.607 / 河南 0.561 / 陕西 0.139 / 宁夏 0.813 |
| α 配水扫描 allocation_scan*.csv | ✅ | 最优 α：北京 0.6 / 河北 0.5 / 河南 0.5 / 陕西 0.3——"无统一最优、随站点反转"结论保持 |
| PPO 重训（γ=0.995，6 模型 × 3 seed） | ✅ | 见下"γ 折扣发现" |
| RL 对比 rotation_policy_comparison.csv | ✅ | 平衡偏好：direct 11.9 t/ha/157mm、residual 12.6/183、预留配额规则 15.5/357、阈值规则 14.2/362；7 偏好集评估显示策略对偏好输入真实响应 |
| PPO 检查点 + 学习曲线 | ✅ | ppo_rotation_learning_curve.csv + fig_rl_training_curves_rotation.png |
| CMIP6 数据 | ✅ 8/10 | 4 站点 × 历史/未来，降级 manifest（辐射 6 模式）；宁夏 2 文件待 2026-08-04 日配额重置 |
| 跨季优化 1a/1b、边际水价值、算法对比 | 🔄 | L2 流水线运行中 |
| 迁移/微调 leave_one_out_rotation | 🔄 4/5 | 北京/河北/河南/陕西折叠完成，宁夏运行中 |
| 两因子风险 two_factor_risk.csv | ⏳ | 依赖迁移完成后自动运行 |
| γ=1.0 对照臂（P0-9 审计要求） | 🔄 | 训练中（bg_5b1c40ab） |

### γ 折扣发现（P0-9 的实证结论，论文二必须如实报告）

再生后的 γ=0.995 策略在学习曲线上**收敛到"最小化用水"而非"优化年产量"**：
灌溉量随训练从 250→60mm 骤降，产量停在 ~8.5 t/ha（固定 3 情景评估），远低于规则
基线 13-16 t/ha。机制：逐决策步的水惩罚（-actual/40）在 ~122 步 episode 中无折扣
累加，而单一季末产量奖励被 γ^122≈0.45 折现——训练目标与"年度总产量/用水/成本/风险"
的研究目标不等价。审计 P0-9 要求比较 γ=1.0：对照臂训练中，若 γ=1.0 关闭产量缺口，
论文须双报两个奖励配置并说明取舍（γ=1.0 与"年度目标"更一致，γ=0.995 仅 PPO 惯例）。

**γ 对照臂最终裁决（2026-08-04，6/6 模型 × 3 seed 全量评估）**：
γ=1.0 **没有**关闭产量缺口——direct 11.92→11.95 t/ha（+1%）、residual 12.62→12.28
（-12%），用水量也几乎不变。**产量缺口的根源是奖励尺度失衡，不是折扣**：实测单个
episode 内水惩罚总绝对值 -3.15 是折现后季末产量奖励 +0.32 的约 10 倍（scripts/
_reward_scale_audit.py），γ 从 0.995 改 1.0 只把产量信号放大 ~1.8 倍，远不足以
翻转 10:1 的失衡。**wq 臂（水项按 annual_quota=450 归一化，惩罚缩小 ~11 倍）是
审计 P0-9 建议的实际修复方向**，其训练/评估结果将决定论文二主奖励配置。

**wq 臂最终裁决（2026-08-04，6/6 模型 × 3 seed 全量评估）——奖励重构生效**：
per_quota 臂成为论文二正式主臂。direct 全站点均值 13.89 t/ha / 296mm、residual
13.52 / 268mm，vs 预留配额规则 15.45 / 357mm——**产量缺口收窄到 1.56—1.93 t/ha
（重构前 2.8—3.5），省水 17—25%**；偏好条件化在全 198—409mm 用水范围真正生效
（重构前偏好只驱动 154—243mm 窄区间）；宁夏从重构前 ~50% 规则产量改善到 ~70%。
论文二对比三/摘要/图已按 wq 主臂重写；per_action 与 γ=1.0 作为消融记录。论文三
迁移正在用 wq 策略重跑（leave_one_out_rotation_transfer_wq.csv，GPU 路径在
Windows spawn 下死锁已改 CPU）。
