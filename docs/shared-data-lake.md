# Project 504 共享数据湖规范

## 身份与边界

- **研究所/共享空间**：`project-504`
- **当前项目**：`project-brnn`
- **默认数据根目录**：`D:\project-504-data`
- 可通过环境变量 `MWR_DATA_ROOT` 指定数据盘或 NAS 上的其他根目录。

`project-504` 是研究所级数据归属，不是单一科研任务。每个承接任务必须获得独立的 `project_id`，例如 `project-oem`、`project-gnss`、`project-satellite`。

## 目录

```text
D:\project-504-data\
  raw\                         # 可复用原始数据；只追加
  interim\                     # 可再生的通用中间数据
  projects\
    project-brnn\              # 当前项目的私有产物空间
      curated\{datasets,models,predictions,reports}\
      configs\
      manifests\
  metadata\                    # catalog.sqlite3、全局 manifest、日志
  tmp\                         # 可清理的临时空间
```

原始 ERA5、探空、亮温、GNSS、卫星和辅助光谱资料只能进入 `raw/`。模型、项目训练集、预测、图件、报告只能进入 `projects/<project_id>/curated/`。

## 使用流程

### 初始化与注册项目

```powershell
python -m mwr_retrieval.cli.data init
```

首次迁移 `project-brnn` 时，迁移命令自动注册 `project-504` 与 `project-brnn`。后续项目应通过 catalog API/CLI 注册项目，并记录仓库地址、描述和负责范围。

### 复用公共资产

先查询文件 SHA-256、来源和覆盖范围，再以 `project_assets` 关联将其标记为新项目的 `input`；不得再次下载或复制内容相同的 ERA5、探空或 GNSS 原始文件。

### 新项目产物

使用 `mwr_retrieval.artifacts.create_tracked_run_directory(..., project_id="<id>")` 创建运行目录，同时在 catalog 中创建 `processing_runs` 记录。每个运行须记录输入 asset ID、通道 schema、网格、配置、随机种子、代码版本和输出 asset ID；输出文件登记时使用 `register_run_output()` 写入 `asset_lineage`。

## 完整性与迁移

迁移遵循：**预生成清单 → 分块复制到临时文件 → SHA-256 校验 → 原子落位 → catalog 登记 → 源文件复验**。

- 不移动、删除、修改或自动去重项目源码目录中的文件；
- 即使检测到同一内容的报告或模型副本，也保留首次迁移的逻辑路径；
- catalog 使用全局内容哈希识别重复，并通过 `project_assets` 保留不同项目和用途的关联；
- 压缩归档包和已展开目录需另行审计，不自动解包或删除。

## 当前项目迁移命令

```powershell
# 只写计划，不复制文件
python -m mwr_retrieval.cli.data migration-plan

# 复制、校验、登记并关联 project-brnn
python -m mwr_retrieval.cli.data migrate-project-brnn

# 查看当前项目的输入、模型、预测和报告
python -m mwr_retrieval.cli.data project-assets project-brnn
```

## 工作区新增源数据入湖

`D:\project-504` 根目录下的新增观测和再分析数据不属于仓库内部 legacy 目录，使用独立命令入湖：

```powershell
# 只写计划，不复制文件
python -m mwr_retrieval.cli.data workspace-ingest-plan

# 复制、校验、登记并关联 project-brnn
python -m mwr_retrieval.cli.data migrate-workspace-sources
```

当前约定映射：

- `chengdu_era5` → `raw/era5/grib/site=chengdu/`
- `chengdu_obs_bt` → `raw/mwr/chengdu/obs_bt/`
- `wenjiang_sounding` → `raw/radiosonde/wenjiang/station=56187/year=<yyyy>/month=<mm>/`
- `obs_bt_filtered_20260726` 和 `观测亮温筛选7.26` → `interim/mwr/chengdu/obs_bt_filtered_20260726/`

## 当前已登记的真实数据桥接产物

2026-08-04 已完成成都真实 `Obs_BT`、CDS ERA5 pressure-level NetCDF、ARTS 前向验证的最小闭环登记：

- 桥接运行：`20260804T075728Z-d5b0fa9c`
- 桥接数据 asset：`4c163386-897c-43a6-8b98-eadbde0e6e41`
- 桥接路径：`legacy://results/chengdu_realdata_bridge/run_id=20260804T075728Z-d5b0fa9c/bridge_dataset.npz`
- 输入 lineage：`1` 个成都 `Obs_BT` JSON + `18` 个 ERA5 日文件，关系为 `derived_from`
- ARTS 验证运行：`20260804T080459Z-22fec53d`
- ARTS 验证 asset：`4b7cd5d9-c930-4164-9326-7df629a35df4`
- ARTS 验证路径：`legacy://results/chengdu_realdata_arts_validation/run_id=20260804T080459Z-22fec53d/arts_validation_predictions.npz`
- 验证 lineage：桥接数据 `validated_by` ARTS 验证结果

查询命令：

```powershell
python -m mwr_retrieval.cli.data show-run 20260804T075728Z-d5b0fa9c
python -m mwr_retrieval.cli.data lineage 4c163386-897c-43a6-8b98-eadbde0e6e41
python -m mwr_retrieval.cli.data show-run 20260804T080459Z-22fec53d
python -m mwr_retrieval.cli.data lineage 4b7cd5d9-c930-4164-9326-7df629a35df4
```
