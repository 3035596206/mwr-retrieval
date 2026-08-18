# 工作流说明

## 当前推荐入口

| 工作流 | 入口 | 输入 | 输出 |
| --- | --- | --- | --- |
| 数据目录初始化/登记 | `python -m mwr_retrieval.cli.data` | 本地文件 | SQLite catalog、manifest、审计报告 |
| ERA5 下载 | `python scripts/download_era5.py` | CDS 参数 | `raw/era5/cds/`、下载任务、manifest |
| 成都 48 层（探空） | `scripts/pipeline_sounding_48layer.py` | 亮温 JSON、探空目录 | 兼容的历史 `results/` 输出 |
| 成都 48 层（ERA5） | `scripts/pipeline_filtered_48layer.py` | 亮温、GRIB、探空 | 兼容的历史 `results/` 输出 |
| 成都真实数据桥接 | `scripts/build_chengdu_realdata_bridge.py` | 数据湖 Obs_BT、CDS ERA5 NetCDF、通道表 | `bridge_dataset.npz`、stats、manifest、catalog lineage |
| 成都真实数据 ARTS 验证 | `scripts/validate_chengdu_realdata_arts.py` | 桥接数据集、ARTS runner | ARTS 模拟亮温、O-B 残差、验证 stats、catalog lineage |
| ARTS 前向模型 | `ForwardModel(backend="arts")` 或 `ARTS_FORWARD_MODEL_COMMAND` | 廓线 JSON、通道表、ARTS agenda | 亮温 `brightness_temperature_k` |
| OEM 实验 | `scripts/run_oem_*.py --forward arts` | 先验、亮温、ARTS 前向模型 | OEM 诊断结果 |

## 前向模型策略

当前主前向模型已切换为 ARTS，对齐研究所和师兄的工作流。`config.DEFAULT_FORWARD_BACKEND` 默认为 `arts`，默认通道集为成都 21 通道。MonoRTM 保留为历史基线后端，simple RTM 只用于快速算法闭环。

ARTS 后端通过本地 runner 接入。推荐设置：

```powershell
$env:ARTS_FORWARD_MODEL_COMMAND = 'wsl -d Ubuntu-24.04 -- /home/inkp/miniconda3/envs/arts/bin/python /mnt/d/project-504/mwr-retrieval-main/scripts/run_arts_profile.py --server'
$env:ARTS_FORWARD_MODEL_PERSISTENT = '1'
python scripts/run_oem_baseline.py --forward arts --n-samples 100
```

本机默认配置已经指向上述 WSL runner，并启用持久进程模式；显式设置 `ARTS_FORWARD_MODEL_COMMAND` 只在切换到研究组 agenda、其他 conda 环境或其他 WSL 发行版时需要。runner 协议记录在 [`config/forward_model.json`](../config/forward_model.json)：脚本从 stdin 读取廓线/通道 JSON，并向 stdout 写出包含 `brightness_temperature_k` 的 JSON。

## 真实数据桥接与验证

当前成都真实数据链路已打通：`Obs_BT` 21 通道观测可与数据湖中的 CDS ERA5 pressure-level NetCDF 按 UTC 整点精确匹配，并转换到项目 48 层物理网格。正式桥接记录见 [`docs/realdata-bridge-arts-validation-2026-08-04.md`](realdata-bridge-arts-validation-2026-08-04.md)。

正式桥接产物：

- `results/chengdu_realdata_bridge/run_id=20260804T075728Z-d5b0fa9c/bridge_dataset.npz`
- 桥接输出 asset id：`4c163386-897c-43a6-8b98-eadbde0e6e41`
- 样本数：`163`，全部精确匹配 ERA5，训练/验证/测试切分为 `84 / 24 / 55`

正式 ARTS 全量验证产物：

- `results/chengdu_realdata_arts_validation/run_id=20260804T080459Z-22fec53d/arts_validation_predictions.npz`
- 验证输出 asset id：`4b7cd5d9-c930-4164-9326-7df629a35df4`
- ARTS 成功率：`163 / 163`
- 全通道 `O-B` RMSE：`9.74 K`
- 183 GHz 水汽带 RMSE：`2.32 K`

常用命令：

```powershell
wsl.exe -d Ubuntu-24.04 -- /home/inkp/miniconda3/envs/arts/bin/python /mnt/d/project-504/mwr-retrieval-main/scripts/build_chengdu_realdata_bridge.py --obs-json /mnt/d/project-504-data/raw/mwr/chengdu/chengdu_obs_bt.json --era5-root /mnt/d/project-504-data/raw/era5/cds/chengdu/pressure-levels --register-catalog --catalog-data-root /mnt/d/project-504-data

wsl.exe -d Ubuntu-24.04 -- /home/inkp/miniconda3/envs/arts/bin/python /mnt/d/project-504/mwr-retrieval-main/scripts/validate_chengdu_realdata_arts.py --bridge-dataset /mnt/d/project-504/mwr-retrieval-main/results/chengdu_realdata_bridge/run_id=20260804T075728Z-d5b0fa9c/bridge_dataset.npz --output-dir /mnt/d/project-504/mwr-retrieval-main/results/chengdu_realdata_arts_validation --all --register-catalog --catalog-data-root /mnt/d/project-504-data
```

当前解释边界：ARTS runner 仍是清空前向，适合验证接口、单位、维度和通道一致性；89 GHz、229 GHz 以及 51-53 GHz 的较大残差需要通过云筛选、通道响应和观测误差协方差继续改进。

## 新产物

新的数据、模型、预测类流程应使用 `mwr_retrieval.artifacts.create_tracked_run_directory()` 在 `projects/<project_id>/curated/<type>/run_id=<id>/` 中创建运行目录，并写入 `processing_runs`。输出文件使用 `register_run_output()` 登记，这会写 asset manifest、关联 `project_assets`，并把输入 asset ID 与输出 asset ID 写入 `asset_lineage`。manifest 至少应记录输入 asset ID、通道 schema、前向模型配置、网格、参数、随机种子、代码版本和生成文件。

常用审计命令：

```powershell
python -m mwr_retrieval.cli.data runs --project-id project-brnn --limit 20
python -m mwr_retrieval.cli.data show-run <run-id>
python -m mwr_retrieval.cli.data lineage <asset-id>
```

## 遗留脚本

根目录中的 `dl_*`、`bulk_*`、`train_mp3000a_v*` 与早期 `run.py` 用于保留历史实验。它们并未删除或移动，但不应作为新数据下载和新实验的默认入口。新下载请使用 catalog 支持的 `scripts/download_era5.py`。
