# Irrigation Decision

灌溉决策研究项目仓库。总体方案见 [docs/研究方案.md](docs/研究方案.md)：数字农田与多目标灌溉基准构建 → 多目标协同灌溉决策模型 → 跨区域动态迁移与适应性验证，三个研究内容依次衔接。

## 当前进度

| 研究内容 | 状态 | 说明 |
| --- | --- | --- |
| 一、数字农田与多目标灌溉基准 | 数据+基线+帕累托前沿已跑通 | 见 [data/README.md](data/README.md)、[src/sim/README.md](src/sim/README.md) |
| 二、多目标协同灌溉决策模型 | 环境+安全层+评估框架已完成，正式 PPO 训练进行中 | 见 [src/rl/README.md](src/rl/README.md) |
| 三、跨区域动态迁移与适应性验证 | 环境指纹与区域相似性分析已完成，迁移方法本身还没开始 | 见 [src/transfer/README.md](src/transfer/README.md)，后续依赖研究二训练好的策略网络 |

## 仓库结构

```
docs/            研究方案与文档
figures/         论文用图（复合图，PNG）
data/
  raw/           下载的原始数据（气象/土壤）
  processed/     实验结果、帕累托前沿、训练产物
src/
  data/          数据获取与站点/土壤配置
  sim/           AquaCrop 基线仿真、灌溉策略、NSGA-II 优化
  rl/            强化学习环境、Gym 封装、PPO 训练与评估
  transfer/      环境指纹、区域相似性分析（跨区域迁移）
  viz/           论文复合图的生成脚本
```

每个子目录都有自己的 README，说明该目录下脚本的用途和运行顺序。**`data/` 下所有原始数据和实验结果都纳入 git 版本控制**（不是只能靠脚本重新生成）——训练结果来之不易（比如 PPO 训练踩过的坑，见 `src/rl/README.md`），保留原始数据是为了论文制图时能随时回来重新组合、深挖，不用重新跑一遍。

## 环境搭建

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt
```

CDS（AgERA5）需要在 `~/.cdsapirc` 配置个人 API key（去 https://cds.climate.copernicus.eu 免费注册获取），非必需——默认数据源是免注册的 Open-Meteo。

## 数据与代码约定

- 原始数据、实验结果、训练产物（csv/zip 等）都纳入 git 版本控制，同时也都能靠 `src/` 下的脚本复现（见各目录 README 里的运行顺序）——两者不冲突，复现脚本是为了能重新生成/扩展，入库是为了不必每次都重新生成。只有真正的临时中间文件（比如 AgERA5 下载过程中的原始 NetCDF 格点文件）才会被 `.gitignore`。
- 长耗时任务（AgERA5 批量下载、基线实验网格、NSGA-II、PPO 训练）都设计成可后台运行、可断点续跑（已完成的会自动跳过）。
- 论文制图脚本在 `src/viz/`，产出图在 `figures/`，见 [src/viz/README.md](src/viz/README.md)。
