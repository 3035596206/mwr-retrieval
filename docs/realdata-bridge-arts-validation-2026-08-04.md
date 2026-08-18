# 真实数据桥接与 ARTS 验证记录，2026-08-04

## 目标

把成都真实微波辐射计观测 `Obs_BT` 与 CDS ERA5 压力层数据接通，形成可训练、可审计、可追溯的真实数据桥接集；随后用 ARTS 清空前向模型对桥接廓线做全量前向验证，检查真实观测到前向模型链路是否可用。

本记录不包含 PPT 汇报文案修改，只记录项目工程与实验进展。

## 输入数据

- 观测亮温：`D:\project-504-data\raw\mwr\chengdu\chengdu_obs_bt.json`
- ERA5 压力层：`D:\project-504-data\raw\era5\cds\chengdu\pressure-levels\year=2026\month=05\part=dXX-XX\era5_pressure_202605_dXX-XX.nc`
- 仪器通道：`config/instruments.json` 中 `chengdu-21ch`，共 21 个通道
- 垂直网格：`src/mwr_retrieval/grids.py` 中 48 层物理网格，0-10 km AGL
- ARTS runner：WSL `Ubuntu-24.04`，`/home/inkp/miniconda3/envs/arts/bin/python`，`scripts/run_arts_profile.py --server`

## 新增脚本

### `scripts/build_chengdu_realdata_bridge.py`

功能：构建真实数据桥接集。

处理流程：

1. 读取成都 `Obs_BT` JSON，检查每条记录是否为 21 通道有限亮温。
2. 递归读取 CDS ERA5 pressure-level NetCDF 文件。
3. 选择最接近成都站点的 ERA5 网格点，本次为 `lat=30.75, lon=104.0`。
4. 提取 ERA5 `t/r/z`，换算 AGL 高度，并按项目 48 层网格做层平均。
5. 按 UTC 整点匹配观测和 ERA5 廓线。
6. 输出 `bridge_dataset.npz`、`bridge_stats.json`、`manifest.json`。
7. 使用 `--register-catalog` 时写入 `processing_runs`、`data_assets`、`project_assets` 和 `asset_lineage`。

关键输出数组：

- `X`：观测亮温，形状 `(n, 21)`
- `T`：ERA5 温度标签，形状 `(n, 48)`
- `RH`：ERA5 相对湿度标签，形状 `(n, 48)`
- `q/logq/P`：湿度和压力辅助标签，形状 `(n, 48)`
- `heights/layer_edges/layer_thickness`：48 层网格信息
- `channel_frequencies_ghz`：成都 21 通道频率
- `train_mask/val_mask/test_mask`：按日期的时间顺序切分

正式运行命令：

```powershell
wsl.exe -d Ubuntu-24.04 -- /home/inkp/miniconda3/envs/arts/bin/python /mnt/d/project-504/mwr-retrieval-main/scripts/build_chengdu_realdata_bridge.py --obs-json /mnt/d/project-504-data/raw/mwr/chengdu/chengdu_obs_bt.json --era5-root /mnt/d/project-504-data/raw/era5/cds/chengdu/pressure-levels --register-catalog --catalog-data-root /mnt/d/project-504-data
```

正式产物：

- `results/chengdu_realdata_bridge/run_id=20260804T075728Z-d5b0fa9c/bridge_dataset.npz`
- `results/chengdu_realdata_bridge/run_id=20260804T075728Z-d5b0fa9c/bridge_stats.json`
- `results/chengdu_realdata_bridge/run_id=20260804T075728Z-d5b0fa9c/manifest.json`

桥接结果：

- 原始观测记录：`163`
- 可用 21 通道记录：`163`
- 成功匹配 ERA5：`163`
- 未匹配：`0`
- 时间范围：`2026_05_13 16:00:00` 至 `2026_05_30 15:00:00`
- 匹配时间差：全部为 `0 h`
- 使用 ERA5 日文件：`18`
- 载入 ERA5 有效廓线：`410`
- 训练/验证/测试样本数：`84 / 24 / 55`
- 亮温范围：`19.72-302.87 K`
- 温度标签范围：`230.49-300.88 K`
- 相对湿度标签范围：`3.12-99.88 %`

数据湖登记：

- 桥接 run id：`20260804T075728Z-d5b0fa9c`
- 桥接输出 asset id：`4c163386-897c-43a6-8b98-eadbde0e6e41`
- 输入资产：`1` 个观测 JSON + `18` 个 ERA5 日文件
- lineage：`19` 条 `derived_from`

## ARTS 真实数据前向验证

### `scripts/validate_chengdu_realdata_arts.py`

功能：读取桥接集，用 ARTS 清空前向模型计算模拟亮温，并与真实观测亮温做 `O-B` 残差审计。

这里的 `O-B` 定义为：

```text
obs_minus_arts_k = observed_bt_k - arts_bt_k
```

该验证用于确认真实观测、ERA5 廓线、通道表、ARTS runner 和数据湖产物能够连成闭环。它不是最终反演精度评估，因为当前 ARTS runner 仍为清空：`cloudboxOff()`，未显式引入云液水、降水和实际通道响应函数。

正式运行命令：

```powershell
wsl.exe -d Ubuntu-24.04 -- /home/inkp/miniconda3/envs/arts/bin/python /mnt/d/project-504/mwr-retrieval-main/scripts/validate_chengdu_realdata_arts.py --bridge-dataset /mnt/d/project-504/mwr-retrieval-main/results/chengdu_realdata_bridge/run_id=20260804T075728Z-d5b0fa9c/bridge_dataset.npz --output-dir /mnt/d/project-504/mwr-retrieval-main/results/chengdu_realdata_arts_validation --all --register-catalog --catalog-data-root /mnt/d/project-504-data
```

正式产物：

- `results/chengdu_realdata_arts_validation/run_id=20260804T080459Z-22fec53d/arts_validation_predictions.npz`
- `results/chengdu_realdata_arts_validation/run_id=20260804T080459Z-22fec53d/arts_validation_stats.json`
- `results/chengdu_realdata_arts_validation/run_id=20260804T080459Z-22fec53d/manifest.json`

全量验证结果：

- 验证样本：`163 / 163`
- ARTS 成功：`163 / 163`
- 失败：`0`
- 整体 `O-B` bias：`6.86 K`
- 整体 `O-B` RMSE：`9.74 K`
- 整体 `O-B` 标准差：`6.91 K`
- 中位绝对残差：`4.39 K`
- 持久 ARTS 进程性能：首样本约 `0.95 s`，后续均值约 `0.006 s/sample`

分波段残差：

| 波段 | 通道数 | bias(K) | RMSE(K) | std(K) | 说明 |
| --- | ---: | ---: | ---: | ---: | --- |
| K 22-31 GHz | 7 | 6.15 | 7.27 | 3.88 | 水汽低频通道存在正偏差 |
| V 51-58 GHz | 7 | 8.18 | 10.41 | 6.44 | 氧气温度通道低频侧偏差较大 |
| W 89 GHz | 1 | 19.56 | 23.72 | 13.42 | 云液水、降水和窗区散射/发射最敏感 |
| G 183 GHz | 5 | 1.95 | 2.32 | 1.26 | 当前最稳定，说明水汽线附近链路较好 |
| 229 GHz | 1 | 14.48 | 16.54 | 8.00 | 高湿、云水和通道响应需重点审计 |

主要异常通道：

- `89.00 GHz`：RMSE `23.72 K`，bias `19.56 K`
- `51.26 GHz`：RMSE `17.13 K`，bias `16.22 K`
- `229.00 GHz`：RMSE `16.54 K`，bias `14.48 K`
- `52.28 GHz`：RMSE `15.65 K`，bias `14.66 K`
- `53.86 GHz`：RMSE `12.80 K`，bias `12.75 K`

表现较好的通道：

- `184.31 GHz`：RMSE `1.38 K`
- `185.11 GHz`：RMSE `2.31 K`
- `186.31 GHz`：RMSE `2.37 K`
- `187.81 GHz`：RMSE `2.42 K`
- `58.00 GHz`：RMSE `2.53 K`

数据湖登记：

- 验证 run id：`20260804T080459Z-22fec53d`
- 输入 asset id：`4c163386-897c-43a6-8b98-eadbde0e6e41`
- 输出 asset id：`4b7cd5d9-c930-4164-9326-7df629a35df4`
- lineage：`4c163386-897c-43a6-8b98-eadbde0e6e41 validated_by 4b7cd5d9-c930-4164-9326-7df629a35df4`

## 结论

1. 成都真实观测亮温、ERA5 48 层标签、成都 21 通道频率表、ARTS 前向模型和数据湖 catalog 已形成闭环。
2. 当前桥接集可作为后续成都环境训练、真实观测 OEM、ARTS 前向一致性校正的基础数据资产。
3. ARTS 清空前向可以稳定处理全量真实桥接样本，说明接口、维度、单位和运行环境已基本打通。
4. 残差结构显示：183 GHz 水汽带最稳定；89 GHz、229 GHz 和 51-53 GHz 是后续精度提升的主要突破口。

## 后续改进方向

1. 引入云筛选或云液水资料，对 89 GHz 和 229 GHz 大残差样本做分组诊断。
2. 补充地面气象、雨滴/降水标志、云底高或再分析云液水路径，避免把有云样本直接按清空处理。
3. 核查成都仪器真实通道响应、带宽、极化和定标版本，替代当前仅使用中心频率的简化通道配置。
4. 将 `O-B` 残差按时段、湿度、温度、云况和通道分组，形成样本筛选规则。
5. 用桥接集训练或微调统计模型时，优先保留 183 GHz 水汽通道作为稳定约束，并对 89/229 GHz 设置更严格 QC 或更大的观测误差。
6. 在 OEM 中把 `S_e` 从固定噪声扩展为按通道、按天气状态调整的观测误差协方差。
7. 等研究组 cloudy ARTS agenda 或云微物理输入准备好后，开展 cloudy ARTS 前向验证，再进入真实观测 OEM 主实验。
