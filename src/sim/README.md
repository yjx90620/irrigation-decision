AquaCrop 数字农田：基线灌溉策略仿真 + NSGA-II 多目标优化（对应研究方案 4.5—4.9）。依赖 `src/data/` 的气象和土壤配置。

- **`weather.py`** — 把 Open-Meteo 气象 CSV 转成 AquaCrop-OSPy 要求的格式。**注意：`weather_df` 内部按列位置（不是列名）取值**（MinTemp/MaxTemp/Precipitation/ReferenceET/Date 顺序），改这个文件时别把 Date 列顺序改错，AquaCrop 会用错列且不报错，只会算出离谱的结果。
- **`run_baseline.py`** — 最小的端到端验证脚本（单站点单年份，雨养/充分灌溉/阈值三种策略），用来确认气象+土壤+作物+模型能跑通。改动仿真链路后先跑这个，几秒钟出结果。
- **`strategies.py`** — 8 种基线灌溉策略定义（雨养、充分灌溉、固定间隔、4 档阈值、关键期保护）。
- **`experiment.py`** — 全量实验网格：5 站点 × 3 土壤 × 45 年 × 8 策略（5400 次模拟），产出 `data/processed/baseline_experiment_results.csv`。跑一次大约 30 分钟。
- **`optimize_nsga2.py`** — 单个 site×soil 组合的 NSGA-II 四目标（产量/灌溉量/成本/水损失）优化，产出 `data/processed/pareto_front_{site}_{soil}.csv`。决策变量是 AquaCrop 4 个生育阶段的土壤水分目标（SMT）+ 单日最大灌水量上限——不是方案原文里的“每阶段固定灌水量”，因为 AquaCrop 内置调度器是“亏了就补到目标水位”，不支持指定每次的固定灌水深度。单次跑一组大约 40—50 分钟。
- **`batch_optimize_loam.py`** — 循环跑完剩余 site（壤土），已跑完的会自动跳过。

## 已知结果

见根目录 [data/README.md](../../data/README.md) 里的"基线实验网格结果"和"NSGA-II 帕累托前沿"两节。
