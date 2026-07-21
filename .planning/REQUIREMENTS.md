# Requirements — MWR 廓线反演系统

## v1 Requirements (当前里程碑)

### 数据获取 (DATA)
- [x] **DATA-01**: 84 个月 ERA5 单层数据下载 ✅
- [ ] **DATA-02**: 84 个月 ERA5 气压层数据下载 (47/2556 天)
- [ ] **DATA-03**: MP-3000A L2 产品数据完整处理管线

### 辐射传输 (RTM)
- [x] **RTM-01**: MonoRTM v5.6 ARM64 编译与修复 ✅
- [x] **RTM-02**: TAPE3 光谱数据 ASCII → 二进制转换 ✅
- [x] **RTM-03**: Python monortm_wrapper 封装 ✅
- [ ] **RTM-04**: MonoRTM 批量 BT 模拟管线

### 模型训练 (MODEL)
- [x] **MODEL-01**: BRNN 基线模型训练 (v1) ✅
- [x] **MODEL-02**: 质控 + BT 订正迭代 (v2) ✅
- [x] **MODEL-03**: Sim_BT vs Obs_BT 对比实验 (v3) ✅
- [x] **MODEL-04**: 三级过滤 + Winsorize (v4) ✅ — 当前最优
- [x] **MODEL-05**: Sim_BT 预训练 + Obs_BT 微调 (v6) ✅
- [ ] **MODEL-06**: MonoRTM 预训练 BRNN (待数据就位)

### 反演推理 (INFER)
- [x] **INFER-01**: retrieve_and_plot.py 端到端反演 ✅
- [ ] **INFER-02**: 批量反演与指标统计

### 文档与报告 (DOCS)
- [x] **DOCS-01**: 技术报告 (Word + PDF) ✅
- [x] **DOCS-02**: 专业实践报告 ✅
- [x] **DOCS-03**: 缺陷分析文档 ✅
- [x] **DOCS-04**: v4 可行性分析 ✅
- [x] **DOCS-05**: 报告定期更新机制 (reports/) ✅

## v2 Requirements (下一里程碑)

### 训练数据扩展
- [ ] **DATA-V2-01**: 完整 84 月气压层数据
- [ ] **DATA-V2-02**: MonoRTM 合成 ~300K 训练样本
- [ ] **DATA-V2-03**: 探空独立验证数据集

### 模型改进
- [ ] **MODEL-V2-01**: MonoRTM 预训练 + Obs_BT 微调 (T RMSE < 1.0K)
- [ ] **MODEL-V2-02**: 物理约束正则化 (静力方程 + RTM 一致性)
- [ ] **MODEL-V2-03**: 分季节 BT 订正系数

### RTM 升级
- [ ] **RTM-V2-01**: PAMTRA 替代 MonoRTM (降低维护成本)
- [ ] **RTM-V2-02**: 云散射模拟 (定量分析云污染)

## 版本对照

| 版本 | 状态 | T RMSE | RH RMSE | 关键改进 |
|------|------|--------|---------|---------|
| v1 | ✅ | 3.03 K | 16.6% | 基线 |
| v2 | ✅ | 1.45 K | 9.0% | OMB 过滤 + BT 订正 |
| v3 | ✅ | 2.65 K | 12.0% | 论文方案, 发现域差异 |
| v4 | ✅ ★ | 1.26 K | 7.8% | 三级过滤 + Winsorize |
| v6 | ✅ | 1.92 K | 10.3% | 两阶段训练 |
| v7 | 📋 | — | — | MonoRTM 预训练版 (计划) |

---
*最后更新: 2026-06-17*
