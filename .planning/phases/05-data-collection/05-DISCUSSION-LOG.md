# Phase 5 Discussion Log

**Date**: 2026-06-17
**Phase**: 5 — 数据补齐

## Decisions Made

### 下载策略
**Question**: CDS 下载策略？
**Options**: 维持现状 / 探索 ARCO Zarr / 两 key 轮换 / 先不管
**Selected**: **两 key 轮换** — 新 key 为主，old key 为备，dl_pl_batch.py 支持多 key 自动切换。窗口保持 3 天。

### 数据验证
**Question**: 日文件验证级别？
**Options**: 自动基础验证 / 月 concat 时验证 / 信任 CDS
**Selected**: **月 concat 时验证** — 日文件阶段只查文件大小。合并月文件时检查时间连续性、值域、NaN 率。

### 存储与组织
**Question**: 数据文件怎么管理？
**Options**: 月concat后删日文件 / 保留日文件 / 日文件全删
**Selected**: **月 concat 后删日文件** — 月文件是唯一存储，日文件为中间产物。

## Deferred Ideas

- ARCO Zarr 替代 — 如果 CDS 持续不稳定再探索
