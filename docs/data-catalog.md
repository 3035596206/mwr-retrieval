# 数据目录与本地数据湖

本项目使用“文件湖 + SQLite catalog”管理科学数据。大文件不写入数据库：ERA5 NetCDF/GRIB、探空原始文件、亮温源文件、TAPE3、模型权重和数组仍保留为文件；SQLite 只保存索引、哈希、覆盖范围、来源、下载任务和谱系。

## 初始化

默认数据根目录为 `D:\project-504-data`。它是研究所级的共享数据空间；当前工程的项目 ID 是 `project-brnn`。详见 [`shared-data-lake.md`](shared-data-lake.md)。建议设置环境变量后使用：

```powershell
$env:MWR_DATA_ROOT = 'D:\project-504-data'
python -m mwr_retrieval.cli.data init
```

也可以在每条命令中显式指定：

```powershell
python -m mwr_retrieval.cli.data --data-root D:\project-504-data init
```

初始化会创建 `raw/`、`interim/`、`projects/<project_id>/curated/`、`metadata/` 和 `tmp/`，以及 `metadata/catalog.sqlite3`。

## 资产登记

手工放入 `raw/` 的文件可以登记而不复制或移动：

```powershell
python -m mwr_retrieval.cli.data register D:\project-504-data\raw\era5\cds\...\era5_pressure_202605.nc
```

登记会计算 SHA-256、保存格式/变量等可安全获得的元数据，并生成 JSON manifest。验证文件仍未改变：

```powershell
python -m mwr_retrieval.cli.data list --kind era5
python -m mwr_retrieval.cli.data verify <asset-id>
```

## 运行记录与谱系

下载、迁移、预处理、ARTS 前向模拟、训练和评估都应写入 `processing_runs`。由已有资产生成的新资产还应写入 `asset_lineage`，这样可以从模型、训练集或预测文件追溯到 ERA5、探空、亮温和前向模型配置。

查看最近运行：

```powershell
python -m mwr_retrieval.cli.data runs --project-id project-brnn --limit 20
python -m mwr_retrieval.cli.data show-run <run-id>
```

查看某个资产的上游/下游：

```powershell
python -m mwr_retrieval.cli.data lineage <asset-id>
```

新训练或 ARTS/OEM 脚本推荐使用 `mwr_retrieval.artifacts.create_tracked_run_directory()` 创建 `projects/<project_id>/curated/<type>/run_id=<run_id>/`，并使用 `register_run_output()` 登记输出文件、写 manifest、关联项目资产和谱系。

## 历史资产

以下命令只读扫描当前仓库的 `data/`、`models/`、`models_chengdu_*` 和 `results/`；不会移动、删除、重算或覆盖任何历史文件：

```powershell
python -m mwr_retrieval.cli.data register-existing
```

报告写入 `metadata/logs/`。历史仓库文件采用 `legacy://` URI，未来迁移前仍可正确验证。

## 工作区新增源数据

`D:\project-504` 根目录下的新增源数据不属于 `mwr-retrieval-main` 仓库内部历史资产，需要使用工作区源数据命令入湖：

```powershell
# 只写计划，不复制文件
python -m mwr_retrieval.cli.data workspace-ingest-plan

# 复制到 raw/interim、校验 SHA-256、登记 catalog，并关联 project-brnn
python -m mwr_retrieval.cli.data migrate-workspace-sources
```

默认扫描以下目录：

- `chengdu_era5` → `raw/era5/grib/site=chengdu/`
- `chengdu_obs_bt` → `raw/mwr/chengdu/obs_bt/`
- `wenjiang_sounding` → `raw/radiosonde/wenjiang/station=56187/year=<yyyy>/month=<mm>/`
- `obs_bt_filtered_20260726` 和 `观测亮温筛选7.26` → `interim/mwr/chengdu/obs_bt_filtered_20260726/`

## 命名约定

- `raw/`：来源原件，只追加，不在原位处理。
- `interim/`：可再生的解析、缓存和按时段分片数据。
- `projects/<project_id>/curated/`：可复现实验输入、模型、预测与报告；每次运行必须有 `manifest.json`。
- `metadata/manifests/`：每个已登记资产的不可变元数据快照。
- `tmp/`：下载和外部模型的临时空间，可在任务结束后清理。

新接入 GNSS 时请使用 `raw/gnss/station=<id>/year=<yyyy>/month=<mm>/`，先通过通用登记命令入库；具体解析格式将在确认数据源后增加。
