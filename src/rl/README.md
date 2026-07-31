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
- **`train_ppo.py`** — 正式训练：100万步，8 个并行环境（`SubprocVecEnv`），域随机化覆盖 5 站点×3 土壤×训练年份（1981—2010，方案5.10的切分），验证年（2011—2017）和测试年（2018—2025）留出不参与训练。

## GPU 加速的结论

这台机器有 RTX 3090，测过 GPU vs CPU 的吞吐量（`benchmark_device.py`）：**只快 1.12 倍**（248 fps → 278 fps）。原因是环境本身的瓶颈在 AquaCrop 的 CPU 仿真（numpy 计算），策略网络只是个很小的 MLP，搬到 GPU 的收益被 CPU-GPU 传输开销抵消了大半——stable-baselines3 自己也会在这种情况下报警告（"PPO 用 MlpPolicy 时不建议上 GPU"）。所以目前训练继续用 CPU 跑，没有为了这点提升重启已经训练到一半的任务。如果以后研究内容三用更大的共享编码器网络，GPU 收益可能会明显一些，到时候再重新测。

## 当前状态

正式训练（`train_ppo.py`）在跑，checkpoint 存在 `data/processed/ppo_checkpoints/`，跑完后用 `evaluate_policy.py` 在验证/测试年份上和阈值规则、NSGA-II 前沿做对比。
