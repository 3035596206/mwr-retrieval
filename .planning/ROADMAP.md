# Roadmap — MWR 廓线反演系统

## Overview

**7 phases** | **4 completed** | **1 in progress** | **2 planned**

| # | Phase | Goal | Status | Requirements |
|---|-------|------|--------|--------------|
| 1 | 基础搭建 | 环境 + POC 管线 | ✅ Done | DATA-01, RTM-01 |
| 2 | 基线训练 | BRNN v1-v2 基线 | ✅ Done | MODEL-01, MODEL-02 |
| 3 | 迭代优化 | v3-v6 实验 | ✅ Done | MODEL-03~05, DOCS-01~04 |
| 4 | 系统集成 | MonoRTM + 报告 | ✅ Done | RTM-02~03, INFER-01, DOCS-05 |
| 5 | 数据补齐 | ERA5 气压层 | 🔄 Active | DATA-02~03, RTM-04 |
| 6 | 规模训练 | MonoRTM 预训练 | 📋 Planned | MODEL-06, SIM-01, PHY-01 |
| 7 | 验证升级 | 交叉验证 + PAMTRA | 📋 Planned | VAL-01, PAMTRA-01, PINN-01 |

---

## Phase 1: 基础搭建 ✅

**Goal**: 构建 Python 开发环境，验证 POC 管线可行性

**Success Criteria**:
1. ERA5 单层 84 月数据全部下载
2. MonoRTM 成功编译 (bin/monortm)
3. POC 管线跑通 (744 条廓线 → BT 模拟 → BRNN 训练)
4. 6/6 模型训练成功

**Requirements**: DATA-01, RTM-01

---

## Phase 2: 基线训练 ✅

**Goal**: 使用 MP-3000A 实测数据训练 BRNN v1-v2 基线

**Success Criteria**:
1. 20,142 观测 → BRNN 训练管线跑通
2. v1: T = 3.03K (基线，有 bug)
3. v2: T = 1.45K, RH = 9.0% (首次超过论文)
4. OMB 异常值过滤 + BT 线性订正 + 廓线分组实现

**Requirements**: MODEL-01, MODEL-02

---

## Phase 3: 迭代优化 ✅

**Goal**: 系统性迭代 v3-v6，定位问题，达到最优

**Success Criteria**:
1. v3: Sim_BT 训练实验 → 发现 Domain Gap (退化 3.7×)
2. v4: 三级过滤 + Winsorize → T=1.26K, RH=7.78% (最优)
3. v6: 两阶段训练 → Sim→Obs gap -68%
4. 5 个数据缺陷识别与分类
5. v4 可行性分析文档

**Requirements**: MODEL-03~05, DOCS-01~04

---

## Phase 4: 系统集成 ✅

**Goal**: 完善 MonoRTM 集成、报告体系和工具链

**Success Criteria**:
1. MonoRTM segfault 修复 (lnfl_mod.f90 + TAPE3 转换器)
2. monortm_wrapper.py 精确 FORMAT 生成
3. retrieve_and_plot.py 反演管线
4. generate_report.py Word/PDF 报告生成
5. reports/ 报告定期更新机制建立

**Requirements**: RTM-02~03, INFER-01, DOCS-05

---

## Phase 5: 数据补齐 🔄

**Goal**: 完成 ERA5 气压层数据下载，准备 MonoRTM 批量模拟

**Success Criteria**:
1. 84 月气压层数据全部下载 (> 1 TB)
2. 月文件自动 concat: pl_YYYY_MM.nc
3. bulk_sim_monortm.py 管线就绪
4. CDS 下载稳定运行 (断点续传)

**Active work**:
- CDS 3 天窗口下载: 47/2556 天 (2013-01 完整, 2013-02 进行中)
- 进程 PID 63613, 日志 /tmp/cds_batch.log
- 下一步: 完成后自动拼接月文件

**Requirements**: DATA-02~03, RTM-04

---

## Phase 6: 规模训练 📋

**Goal**: MonoRTM 批量预训练 → 精度达到新高度

**Success Criteria**:
1. ~300K MonoRTM Sim_BT 样本生成
2. Sim_BT 预训练 → Obs_BT 微调
3. T RMSE < 1.0K (目标)
4. 物理约束正则化 (二阶导数 + RTM 一致性)

**Dependencies**: Phase 5 数据补齐

**Requirements**: MODEL-06, SIM-01, PHY-01

---

## Phase 7: 验证与升级 📋

**Goal**: 独立验证 + RTM 技术栈现代化 + 发表准备

**Success Criteria**:
1. 探空数据独立交叉验证
2. PAMTRA 替代 MonoRTM
3. PINN 实验 (RTM in loss function)
4. 学术论文草稿

**Dependencies**: Phase 6 完成

**Requirements**: VAL-01, PAMTRA-01, PINN-01

---

## Gantt 时间线

```
Phase 1   [████████] 3月下旬-4月中旬  ✅
Phase 2   [████████] 4月中旬-5月上旬  ✅
Phase 3   [████████] 5月中旬-6月上旬  ✅
Phase 4   [████████] 6月上旬-6月中旬  ✅
Phase 5   [████░░] 6月中旬-??        🔄 数据下载中
Phase 6   [░░░░░░] 待Phase5完成      📋
Phase 7   [░░░░░░] 待Phase6完成      📋
```

---
*最后更新: 2026-06-17*
