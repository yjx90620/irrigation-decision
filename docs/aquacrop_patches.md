# aquacrop-ospy 补丁说明

`.venv/` 不纳入版本控制，所以对已安装第三方包的修改不会随仓库保留——每次重新
`pip install -r requirements.txt` 后都需要重新执行：

```bash
.venv/Scripts/python patches/patch_aquacrop_higc.py
```

脚本本身是幂等的（已打过补丁会直接跳过），可以放心重复运行。

## 修的是什么问题

`aquacrop.initialize.calculate_HIGC.calculate_HIGC()` 内部是一个没有迭代上限
的 `while` 循环，用于反推收获指数增长系数 (HIGC)。正常情况下几十到几百次迭代
就能收敛，但当模拟作物在严重水分胁迫下生育期日历发生塌缩——`HIstartCD` 反而
晚于 `HIendCD`——传入的 `crop_YldFormCD`（循环里的 `tHI`）会变成负数。这会让
`np.exp(-HIGC * tHI)` 随 HIGC 增大而溢出，导致循环的收敛目标 `HIest` 永远卡在
0，循环因此真正地无限循环下去（已实测验证：跑满 200 万次迭代仍未跳出）。

这条链路很可能是本项目整个开发过程中反复出现的"卡死"问题的统一根因：

- 北京站点 α 扫描在 2013 年这个特定年份卡死（不区分灌溉策略）
- `task_sensitivity.py` 连续 10 年雨养策略卡死 15+ 小时
- 排除北京站点之后，RL 训练在随机探索阶段仍反复卡死（连续 4 次尝试、每次
  900 秒超时都没能保存一个 checkpoint）

此前针对每个具体触发场景做的规避（`run_rotation_years_independent` 消除年份
断档、`CRITICAL_DEPLETION_FRAC` 安全下限、外部 watchdog 重启机制、各脚本里的
`subprocess.run(timeout=...)` 防护）都只是在从外部绕开这个机制的某一种触发路
径，不是真正的根因修复，这也是之前几轮"修了又卡"的原因。

## 修了什么

给循环加了 20 万次的迭代上限（对应 HIGC 最大到 200——公开的作物参数表里 HIGC
基本都远小于 1.0，所以这个上限只有在上述退化情形下才会真正被触及），并抑制随
之产生的 `overflow encountered in exp` 噪音警告（整个项目日志里反复出现的那条
警告，现在知道了它的真正含义：不是"数值不稳定但问题不大"，而是"这次很可能要
卡死了"）。

验证：退化输入（负 `tHI`）现在 ~0.2 秒内返回，不再无限循环；正常作物参数下的
输出与补丁前完全一致（两种情况都远在迭代上限之内就收敛）。

补丁文件：[patches/patch_aquacrop_higc.py](../patches/patch_aquacrop_higc.py)
