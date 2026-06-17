# Phase 5: 数据补齐 — Context

**Phase**: 5 — 数据补齐
**Goal**: 完成 ERA5 气压层数据下载，准备 MonoRTM 批量模拟
**Date**: 2026-06-17
**Mode**: standard

## Domain

完成 84 个月 (2013-2019) ERA5 气压层数据（37 标准等压面 × T/q/Z）的全量下载，合并为月文件 `pl_YYYY_MM.nc`。数据就位后 Phase 6 将使用 MonoRTM 批量生成 ~300K 条 Sim_BT 训练样本。

## Canonical Refs

- `.planning/PROJECT.md` — 项目定义、技术栈、关键决策
- `.planning/REQUIREMENTS.md` — DATA-02 (气压层下载), RTM-04 (批量 BT 管线)
- `.planning/ROADMAP.md` — Phase 5 成功标准
- `dl_pl_batch.py` — 当前 CDS 下载脚本 (3 天窗口, 37 层)
- `dl_pl_cds_v3.py` — CDS 下载 v3 版 (2 天窗口备选)
- `bulk_sim_monortm.py` — MonoRTM 批量模拟管线 (Phase 6 使用)
- `src/monortm_wrapper.py` — MonoRTM Python 封装
- `.cdsapirc` — CDS API 认证 (位于 ~/.cdsapirc)
- `reports/进度记录/工作进度6.16.md` — 最近一次工作进度

## Decisions

### 下载策略：两 key 轮换

`dl_pl_batch.py` 改为支持多个 CDS API key 循环使用。

| Key | UID | 状态 |
|-----|-----|------|
| old | 8dfcb2f7-... | 之前被队列限流，可能已恢复 |
| new | a2179a74-... | 当前使用，3 天窗口通过 cost cap |

策略：新 key 为主（已验证可用），old key 为备。当某个 key 的请求连续被 reject 或长时间未完成时，自动切换到另一个。每次切换之间间隔至少 5 分钟，避免两个 key 同时堆积请求。

窗口大小保持 3 天（已验证通过 cost cap，1.7MB/窗口，~100 min/窗口含排队）。

### 数据验证：月 concat 时验证

日文件下载期间只检查文件大小 > 200KB 和基本结构（非全 NaN）。concatenate 月文件时一次性执行：

1. **时间连续性**: 每月 1–28/30/31 日全部存在，每文件 24 个时次顺序连续
2. **值域检查**: T ∈ [170, 330] K, q ∈ [0, 0.05] kg/kg, Z ∈ [-500, 50000] m²/s²
3. **NaN 率**: 每变量 < 5%
4. **异常日记入** `.planning/phases/05-data-collection/data-issues.md`

### 存储：月 concat 后删日文件

- 月文件 `data/era5/pl_YYYY_MM.nc` 是唯一存储
- 日文件 `data/era5/_daily/` 仅作为下载阶段的中间产物
- 每月 concat 成功并经上述验证后，删除对应日文件
- 月文件纳入 `.gitignore`（已在 `mwr_retrieval/` 中），不提交到 git
- 预计 84 个月 × ~1-2 MB = ~100-170 MB 总量

## Code Context

### 现有下载脚本

- `dl_pl_batch.py:137` — 当前主力，cdsapi + session.proxies off + 3 天窗口
- `dl_pl_cds_v3.py:166` — v3 版，支持 2/3 天窗口
- `download_day()` / `download_month()` 函数已实现断点续传和重试

### 关键基础设施

- `data/era5/_daily/` — 47 个日文件的存放目录
- `/tmp/cds_batch.log` — 下载日志
- `/etc/hosts` — 已添加 `136.156.139.54 cds.climate.copernicus.eu` 绕过 DNS

### 目标管线 (Phase 6)

- `bulk_sim_monortm.py:315` — 读取 `pl_YYYY_MM.nc`, q→RH, 插值到 MWR 93 层, MonoRTM 逐时次模拟, 输出 `data/monortm_bt/bt_monortm_YYYY_MM.npz`

## Deferred Ideas

- ARCO Zarr 逐周下载替代方案 — 当前 ARCO 月请求超时，但如果 CDS 持续不稳定可以再探索
- PAMTRA 替代 MonoRTM — Phase 7 的事项
- 硬盘空间监控 — 84 个月仅 ~100MB，不需要专门监控

## Success Criteria (from ROADMAP.md)

1. 84 月气压层数据全部下载 (> 1 TB 是全区域，站点提取后 ~100 MB)
2. 月文件自动 concat: pl_YYYY_MM.nc
3. bulk_sim_monortm.py 管线就绪 ✅ (已就绪)
4. CDS 下载稳定运行 (断点续传)
