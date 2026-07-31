这里存放研究相关数据。原始数据文件（csv/xlsx 等）默认不纳入 git 版本控制（见根目录 `.gitignore`），通过 `src/data/` 下的脚本可复现下载。

## 当前状态（研究内容一：数据准备）

- **气象数据**：`data/raw/weather/{site_id}_1981_2025_openmeteo.csv`，通过 `src/data/download_weather_openmeteo.py` 从 Open-Meteo Archive API（基于 ERA5 再分析）下载，覆盖 5 个站点、1981—2025 年，日尺度 Tmax/Tmin/降水/ET0/辐射/风速/相对湿度。这是方案中的二级/冻结数据集，作为正式数据不可用时的基准。
- **AgERA5（CDS 正式数据）**：已配置 `~/.cdsapirc`（本机用户目录，不在仓库内）并验证 API key 有效、可正常下载。后续可切换到 `sis-agrometeorological-indicators` 数据集作为一级正式数据源。
- **土壤数据**：SoilGrids 当前网络不可达，暂用 `src/data/soils.py` 中 AquaCrop-OSPy 内置的标准土壤（SandyLoam/Loam/ClayLoam）代替，对应方案中的砂壤土/壤土/黏壤土。
- **站点配置**：见 `src/data/config.py`（5 个代表区域坐标、历史年份范围）。
