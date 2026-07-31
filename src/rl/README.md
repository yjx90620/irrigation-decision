偏好条件化强化学习灌溉决策（对应研究方案 5.4—5.12）。依赖 `src/sim/` 的基线实验结果（用于校准产量奖励的产量上限）。

- **`env.py`** — 核心环境 `IrrigationEnv`：把 AquaCropModel 包装成可逐 3 天交互的环境。
  - 用 `irrigation_method=5`（固定深度）实现外部逐日控制：这个方法每天都会重新读 `IrrMngt.depth`，所以在两次 `run_model(num_steps=1, initialize_model=False)` 之间改这个值，就能让外部策略决定当天灌多少。已经用手工探针验证过（第1天设15mm，`irr_cum` 立刻变15，后两天设0则保持不变）。
  - 仿真窗口从**播种日**开始，不是从1月1日开始——播种前 AquaCrop 走的是一套独立的"休耕期"子模型，不会读我们设的 `IrrMngt`，播种前的动作会被静默忽略。
  - `safety_filter()`：动作安全层（方案5.8），会拦掉这几类动作——根区已接近饱和还硬要灌、未来3天预报有大雨还要灌、离收获太近灌了也没用、超过季节剩余水量预算。
  - 奖励是4维向量（yield_proxy/water/cost/risk），产量项用当季 `tr_ratio`（水分胁迫比）做逐步代理，季末加一个用**该站点实际充分灌溉产量**（从 `baseline_experiment_results.csv` 读，土壤类型对充分灌溉产量影响很小所以只按站点分）归一化的奖励。
- **`gym_env.py`** — Gymnasium 封装 `GymIrrigationEnv`：每个 episode 随机采样 站点×土壤×年份×偏好权重，让一个策略学会响应偏好向量而不是死记一个场景。`gymnasium.utils.env_checker.check_env` 会报"非确定性"警告——这是故意的（域随机化），不是 bug，见代码注释。
- **`test_env_random_policy.py`** / **`benchmark_device.py`** — 冒烟测试和吞吐量基准，不是正式流程的一部分。
- **`evaluate_policy.py`** — 评估框架：给定任意策略函数（规则或训练好的 SB3 模型），在 站点×土壤×年份×偏好 网格上跑，产出和基线实验、NSGA-II 前沿同口径的指标（产量/灌溉量/安全层介入率等），可以直接比较。
- **`train_ppo_smoke_test.py`** — 5000 步的冒烟测试，只确认训练循环能跑通，不是真训练。
- **`train_ppo.py`** — 正式训练：100万步，8 个并行环境（`SubprocVecEnv`），域随机化覆盖 5 站点×3 土壤×训练年份（1981—2010，方案5.10的切分），验证年（2011—2017）和测试年（2018—2025）留出不参与训练。用 `VecNormalize` 做观测归一化（原因见下面的踩坑记录），每个 checkpoint 都配一份对应的归一化参数（`save_vecnormalize=True`）。改 `RUN_NAME` 常量可以并行跑多个版本互不覆盖。
- **`reconstruct_learning_curve.py`** — 训练日志本身没存成 CSV（stable-baselines3 默认只打印到控制台），这个脚本事后把 `ppo_checkpoints/` 里所有 checkpoint 拿固定场景（3个站点×2015年×平衡偏好）评估一遍，拼出学习曲线存到 `data/processed/ppo_learning_curve.csv`。可以随时重跑，已经评估过的 (run, timesteps) 组合会跳过——**注意是按 (run, timesteps) 去重，不是只按 timesteps**，因为不同训练版本（v1/v2）的 checkpoint 步数会重叠，只按步数去重会把后面版本的 checkpoint 误判成"评估过了"直接跳过（真的踩过这个坑，见下面 git 历史）。

## GPU 加速的结论

这台机器有 RTX 3090，测过 GPU vs CPU 的吞吐量（`benchmark_device.py`）：**只快 1.12 倍**（248 fps → 278 fps）。原因是环境本身的瓶颈在 AquaCrop 的 CPU 仿真（numpy 计算），策略网络只是个很小的 MLP，搬到 GPU 的收益被 CPU-GPU 传输开销抵消了大半——stable-baselines3 自己也会在这种情况下报警告（"PPO 用 MlpPolicy 时不建议上 GPU"）。所以目前训练继续用 CPU 跑，没有为了这点提升重启已经训练到一半的任务。如果以后研究内容三用更大的共享编码器网络，GPU 收益可能会明显一些，到时候再重新测。

## 踩过的坑：策略退化成"永远不灌水"

第一版 100 万步正式训练跑完后评估，发现无论站点/年份/偏好，输出永远是 0mm——包括宁夏 2020 年这种严重缺水、产量已经崩到 8.65 t/ha 的场景。查训练日志发现 `approx_kl` 和 `clip_fraction` 在 n_updates≈2400 时就已经精确等于 0，说明策略早早就停止更新了。

根因是奖励量纲没对齐：`water`/`cost` 用的是原始 mm/成本数值（单步最多 -40/-45），`yield_proxy` 却停留在 0—1 量级，两者相差 30—40 倍——不管方案里写的偏好权重是多少，水/成本惩罚在加权和里都会碾压产量项。再加上 stable-baselines3 的 `ent_coef` 默认是 0（没有熵奖励鼓励探索），策略一旦发现"不灌水"能避免这个惩罚，就再也不会跳出来看灌水的长期产量收益了。

修复（`env.py` + `train_ppo.py`）：把 `water`/`cost` 按单步最大可能值归一化到大致 [-1, 0]，和 `yield_proxy` 同量级；训练加 `ent_coef=0.01`。用 10 万步小规模验证过：熵从 -1.6 缓慢降到 -0.7（不再是直接归零），平均回报从 ~7 涨到 ~10，确认策略还在正常探索和进步，才重新跑完整的 100 万步。

## 踩过的第二个坑：熵没崩，但策略"模式"还是卡在不灌水

上面这个修复过的版本训练到 70 万步，`entropy_loss`/`approx_kl` 看着都正常（没有归零），但用 `reconstruct_learning_curve.py` 把从 5 万到 70 万步的每个 checkpoint 拿固定场景（含宁夏 2015 年这种明确缺水的年份，产量只有 7.9 t/ha）评估一遍，发现确定性输出**从头到尾都是 0mm，一次没变过**。策略的"众数"卡住了，即便分布还留着一点熵没探索完。

根因是状态特征完全没做归一化，量纲差了大约 600 倍——`tr_ratio`/`depletion_frac` 是 0—1，`biomass`/`gdd_cum`/`remaining_water_budget` 是几百到几百的量级。没有归一化的话，大量纲但信息量低的特征（`biomass`、`gdd_cum` 基本只是日历时间的代理）在梯度里会把小量纲但真正该看的信号（`depletion_frac`、`tr_ratio`）淹没掉，网络可能在还没学会"看水分亏缺"之前就已经把"别灌水"锁死成局部最优了。

修复：训练时用 `VecNormalize(norm_obs=True)` 包一层，每个 checkpoint 连同归一化统计量一起存（`CheckpointCallback(save_vecnormalize=True)`），`evaluate_policy.py` 的 `load_ppo_policy()` 会自动找配对的归一化文件并应用。5万步小规模验证：同一策略在宁夏2015/2018/2020年分别给出10mm/20mm/20mm——不再是死板的全0，训练预算只有卡住那版的1/20就已经是质的差别。

修复后的完整 100 万步训练用 `RUN_NAME="ppo_irrigation_v2"` 单独跑，第一版（卡住不灌水的那个）保留没杀掉，它跑完的学习曲线本身就是个有意思的"负结果"记录（存在 `data/processed/ppo_learning_curve.csv`，`ppo_irrigation` 前缀的 checkpoint）。

## v2 训练完成后的评估结果

`ppo_irrigation_v2` 跑完 100 万步。学习曲线（`data/processed/ppo_learning_curve.csv`，见 `figures/fig4_rl_training_curve.png`）显示策略学到了合理、随场景变化的行为：

- **陕西关中**（最湿润站点）：灌溉量很快收敛到 0mm 并保持，产量维持在 14.6 t/ha 高位——策略正确学会了"这个站点基本不需要灌溉"，和基线实验里雨养≈充分灌溉的结论一致。
- **宁夏灌区**（最干旱站点）：灌溉量随训练推进从 45mm 逐步涨到 80—120mm，产量从 9.1 t/ha 涨到 ~12 t/ha——策略确实在学习"这里需要更多水"，方向正确。
- **河北中部**：灌溉量和产量都有明显波动（不是单调收敛），训练还不算完全稳定。

和阈值规则基线的对比（`data/processed/rl_vs_baselines_comparison.csv`，2018—2022测试年，见 `figures/fig5_rl_vs_baselines.png`）：

- 陕西关中/河北中部/北京平原/河南北部：PPO 几乎不灌水（0mm）就能拿到接近阈值规则的产量（差距普遍 <0.4 t/ha）——**用水效率明显更高**。
- 宁夏灌区：PPO 用水更少（40mm vs 阈值规则 215mm）但产量也明显更低（13.2 vs 14.6 t/ha）——**这里 PPO 还没有超过简单规则**，说明极端缺水场景下的学习还不充分，可能需要：更长的训练、专门增加干旱年份/干旱站点在域随机化里的采样权重（当前是均匀采样，极端场景样本天然稀少）、或者更细的奖励塑形。

这是诚实的阶段性结果——不是"RL全面吊打规则"的故事，是"RL在容易的场景已经学得比规则更省水，在最难的场景还没追上"，这个对比本身对论文讨论"强化学习 vs 规则基线"这个问题是有价值的材料。
