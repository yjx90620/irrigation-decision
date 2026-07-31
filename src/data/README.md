数据获取与配置。运行顺序：

1. **`config.py`** — 5 个代表站点坐标（北京平原/河北中部/河南北部/陕西关中/宁夏灌区）、历史年份范围（1981—2025）。其他脚本都从这里读站点/年份配置。
2. **`download_weather_openmeteo.py`** — 从 Open-Meteo Archive API（ERA5 再分析）下载全部 5 站点 × 1981—2025 年日尺度气象数据，存到 `data/raw/weather/`。免注册、免密钥，是全周期建模的**主数据源**。已跑完，可直接用；重跑会自动跳过已下载的站点。
3. **`download_weather_agera5.py`** — 从 CDS 下载官方 AgERA5 数据，作为正式数据来源和抽样校验用（不是全周期主数据源——45 年 × 5 站点逐日请求太慢，见脚本内注释）。需要 `~/.cdsapirc` 配置 API key。目前只拉了 2024 年验证样本。
4. **`validate_agera5_vs_openmeteo.py`** — 用 AgERA5 验证样本和同期 Open-Meteo 逐日比对（偏差/RMSE/相关系数），确认 Open-Meteo 可靠后再放心用它做全周期建模。结果见 `data/README.md`。
5. **`soils.py`** — AquaCrop-OSPy 内置标准土壤（SandyLoam/Loam/ClayLoam），对应方案里的砂壤土/壤土/黏壤土。SoilGrids 目前网络不可达，暂时用这个代替。

## 已知限制

- SoilGrids、NASA POWER 在当前网络环境下连不上（TLS 握手失败），CDS 反而是通的——不是账号问题，像是本地代理只放行了部分域名。如果以后想用真实土壤数据，需要先解决这个网络问题。
