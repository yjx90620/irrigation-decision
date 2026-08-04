# 作物参数与率定方法（crop calibration）

本文档记录本项目所有作物参数的来源与率定方法（P1-9，audit-v2），逐参数
记录见 `data/calibration/crop_parameter_sources.csv`。原则：**代码里的每个
物候参数必须能追溯到（文献/包默认值/本仓库率定脚本 + 实测范围）之一**，没有
实测资料的一律标注为"文献/情景参数"，不得写成"已验证当地品种"。

## 1. 冬小麦（WheatGDD，GDD 驱动，Tbase=0）

### 种植制度
华北双季站点：冬小麦 10/10 播种 → 夏玉米 06/15 播种（小麦收获必须早于 06/15，
否则轮作日历断言抛 `RotationCalendarError`）。河南/陕西/河北/北京为双季，
宁夏无冬小麦（10/10-06/25 窗口累计 GDD 仅 ~1833，标准品种需 2200，不适宜）。

### 品种参数（audit-v3 5.2 重分类：calendar-feasibility adjustment）
- 标准品种：AquaCrop-OSPy 自带 WheatGDD（Maturity=2200, Senescence=1600,
  HIstart=1200），仅河南直接可用（GDD 余量全年份达标）。
- 河北（Maturity=1825）、陕西（1850）、北京（1600）：由
  `src/sim/calibrate_wheat_maturity.py` 调整——向下搜索最短 Maturity，
  使**开发窗口（1982-2010，audit-v3 5.1）**全部年份在充分灌溉下（最晚成熟
  情形）收获日距 06/15 留有 ≥7 天安全边际；Senescence/HIstart 按标准品种的
  固定比例随 Maturity 缩放。率定数据见 `data/calibration/wheat_harvest_calendar_audit.csv`。
- **口径（audit-v3 5.2）**：该过程是**日历可行性调整（calendar-feasibility
  adjustment）**——只保证"能在收获窗口前成熟"，没有真实播种/抽穗/开花/成熟
  观测，不是作物生理校准，也不作为模型泛化证据。论文中不得称"站点校准"，
  只能称"情景参数化（保证日历可行）"。

## 2. 夏玉米（Maize，日历驱动，双季站点）

### 问题（audit-v2 P0-2，实测）
AquaCrop 自带 Maize 品种 MaturityCD=132 天，而夏玉米窗口 06/15-10/05 仅
112 天。旧代码直接用了 132 天品种：充分灌溉下 **0/20 季自然成熟**，收获日
恒为窗口末端 10/05（被截断），产量比成熟时低约 25%（9.0 → 11.3 t/ha）。

### 修复
采用 105 天日历品种（MaturityCD=105，Senescence/HIstart/MaxRooting/YldForm/
Emergence 按 105/132=0.7955 同比例缩放），依据：
- 华北夏玉米实际熟期 ~100-110 天（郑单 958 等中早熟品种，文献/生产常识）；
- 105 天 + 06/15 播种 = 09/28 成熟，距 10/05 余量 7 天，全年份达标
  （`data/calibration/maize_cultivar_calibration.csv`：4 站点 × 44 年
  充分灌溉，0 年晚收，产量 10.3-11.6 t/ha）。
- 状态：**文献/情景参数，非站点实测品种**——我们没有当地品种的田间试验
  资料，105 天是"与华北夏玉米生育期一致的合理情景选择"，不是"验证过的
  当地品种"。

## 3. 春玉米（宁夏，日历驱动，单季）

宁夏 04/25 播种、09/30 收获（158 天窗口），自带 132 天品种 ~09/04 成熟，
余量 26 天，无需调整（`data/calibration/ningxia_spring_maize_calibration.csv`
：44 年 0 晚收，产量 13.5-14.8 t/ha）。状态：**包默认值 + 本窗口验证**。

## 4. 物候验证口径

`rotation.run_season()` 的每条结果含 `matured_naturally`（AquaCrop
`crop_mature` 标志）：**成熟**（收获日 < 固定窗口末端）为 True；被窗口截断
或作物死亡为 False。正式结果不得含未成熟玉米季（验收测试
`test_summer_maize_matures_naturally` 等强制）。

## 5. 局限

- 小麦率定以"充分灌溉 = 最晚成熟"为最坏情形（胁迫会加速衰老，只会更早），
  以 1982-2025 观测气象为准；CMIP6 未来气候下的成熟稳定性未验证（对比5
  未跑）。
- 玉米 105 天为文献情景值，未用田间试验率定；如后续获得当地品种资料应
  替换并重跑全部轮作结果。
