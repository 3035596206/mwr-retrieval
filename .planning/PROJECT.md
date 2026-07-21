# MWR 微波辐射计大气温湿廓线反演

## What This Is

基于 BRNN（贝叶斯正则化神经网络）的地基多通道微波辐射计大气温湿廓线反演系统。利用 MP-3000A 微波辐射计 22 通道实测亮温作为输入，反演 0-10 km 范围内 93 层高度网格上的温度 T(z) 和相对湿度 RH(z) 垂直廓线。当前最优模型 v4 在 2,368 个测试样本上达到 T RMSE = 1.26 K、RH RMSE = 7.76%，全部指标超过论文基准。

## Core Value

**提供快速、准确的大气温湿廓线实时反演能力**。BRNN 推理速度（~0.5 ms/廓线）远超传统 1D-Var 物理反演（分钟级），可实现逐扫描线实时处理。精度在测试集上优于行业论文基准。

## Context

- **观测站点**: 天津 (39.16°N, 117.79°E)
- **数据周期**: MP-3000A 观测 2023-11 ~ 2024-03 (20,142 次)
- **辐射计**: MP-3000A, 22ch (K-band 8ch + V-band 14ch)
- **辐射传输**: MonoRTM v5.6 (AER Inc.) + Python 简易 RTM
- **训练框架**: PyTorch, Apple MPS 加速
- **环境**: Python 3.14, macOS ARM64
- **仓库**: https://github.com/3035596206/mwr-retrieval

## Stack

| 层 | 技术 |
|-----|------|
| 数据获取 | CDS API, ARCO Zarr, xarray, netCDF4 |
| 辐射传输 | MonoRTM v5.6 (Fortran), Python 简易 RTM |
| 数据处理 | NumPy, SciPy, pandas |
| 深度学习 | PyTorch 2.12, BRNN (256×2, BN, Dropout 0.3) |
| 可视化 | matplotlib, python-docx |
| 版本控制 | Git + GitHub |
| 报告 | Word (.docx) + PDF + Markdown |
| 项目框架 | GSD (Get Shit Done) |

## Key Decisions

| 决策 | 理由 | 结果 |
|------|------|------|
| Obs_BT 直接训练 (v4) 而非 Sim_BT | v3 实验证明 Sim→Obs domain gap 退化 3.7× | v4 为当前最优 |
| 6 个独立 BRNN (非单一多输出网络) | 各高度区间物理特性差异大，独立模型更灵活 | T RMSE 1.26K |
| 廓线分组划分 (非时间划分) | 同一廓线被重复观测 ~5.6 次，必须防泄露 | 测试集完全独立 |
| BT 订正用 Winsorize 回归 (非 OLS) | K 波段 OMB 厚尾分布，OLS 被异常值扭曲 | OMB std -49% |
| ERA5 再分析作训练标签 (非探空) | 探空样本不足 (~250)，ERA5 每小时 1 时次 | 配对样本 15,838 |
| MonoRTM 而非仅用 Python RTM | Python 模型 K 波段偏差 195K | 精度保证 |

## Requirements

### Validated (已完成)

- ✓ **RET-01**: BRNN 从 22ch 亮温反演 93 层 T(z), RH(z) 廓线 — v4 验证
- ✓ **RET-02**: T RMSE < 1.5 K — v4 达到 1.26 K
- ✓ **RET-03**: RH RMSE < 13% — v4 达到 7.76%
- ✓ **RET-04**: 廓线分组划分防止信息泄露 — v2 起实现
- ✓ **RET-05**: Winsorize BT 订正 — v4 OMB std -49%
- ✓ **RET-06**: K 波段云污染统计过滤 — v4 |OMB|>2.5σ
- ✓ **RET-07**: MonoRTM v5.6 在 ARM64 macOS 上编译运行 — 2026-06 完成
- ✓ **RET-08**: ASCII → 二进制 TAPE3 转换器 — convert_tape3.py
- ✓ **RET-09**: Word/PDF 技术报告自动生成 — generate_report.py
- ✓ **RET-10**: 专业实践报告 — MWR_专业实践报告.docx
- ✓ **RET-11**: v1 → v6 共 6 轮迭代实验完成

### Active (进行中)

- [ ] **DATA-01**: 84 个月 ERA5 气压层数据下载 (47/2556 天, 3 天窗口)
- [ ] **DATA-02**: MonoRTM 批量生成 ~300K 合成训练样本 — 管线就绪待数据
- [ ] **SIM-01**: MonoRTM 预训练 + Obs_BT 微调 (v6 增强版)
- [ ] **PHY-01**: BRNN loss 加入物理约束正则化 (静力方程/RTM 一致性)
- [ ] **VAL-01**: 独立探空数据外部交叉验证

### Planned (计划中)

- [ ] **PAMTRA-01**: PAMTRA 替代 MonoRTM 降低维护成本
- [ ] **PINN-01**: RTM 嵌入 BRNN loss (Physics-Informed Neural Network)
- [ ] **CROSS-01**: 跨仪器迁移 (MP-3000A → HATPRO)
- [ ] **TEMP-01**: 时序一致性利用 (Kalman/LSTM)
- [ ] **BIAS-01**: 按季节分组 BT 订正系数

### Out of Scope (当前不做)

- 降水条件下的廓线反演 — 散射 RTM 需要，MonoRTM 不支持
- 业务化部署 — 当前为研究原型
- 夏季数据训练 — MP-3000A 数据仅覆盖 11-3 月

## Evolution

本文档随项目推进更新。每次阶段完成后：验证过的需求移到 Validated，新涌现的需求加入 Active，不再适用的移到 Out of Scope。

---
*最后更新: 2026-06-17 after GSD initialization*
