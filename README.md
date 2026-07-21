# MWR 大气温湿廓线反演系统

基于 BRNN + OEM 的地基多通道微波辐射计大气温湿廓线反演系统。利用 MP-3000A 微波辐射计 22 通道实测亮温反演 0-10 km 范围内 93 层高度网格上的温度 T(z) 和相对湿度 RH(z) 垂直廓线。

**核心论文**：朱柳桦《基于地基微波辐射计的多方法反演大气温湿廓线研究》（2023，南京信息工程大学）第 3 章  
**仓库**：https://github.com/3035596206/mwr-retrieval  
**上次更新**：2026-07-21

---

## 核心指标

### BRNN v4（统计反演，当前最佳）

| 指标 | 数值 | 论文基准 |
|------|------|----------|
| T RMSE | **1.26 K** | <1.5 K |
| RH RMSE | **7.76%** | <13% |
| 推理速度 | ~0.5 ms/廓线 | — |

### MonoRTM OEM n=100（物理反演，self-consistent）

| 指标 | Prior → Posterior |
|------|-------------------|
| T RMSE | 2.64 → **2.02 K** |
| RH RMSE | 6.49 → **6.20%** |
| BT RMS | 5.03 → **0.61 K** |
| 收敛率 | **99%** |
| DOFS | **2.21** |

---

## 当前进度总览

```
BRNN 统计反演  ████████████████████ 100%  v4 当前最佳 (T=1.26K, RH=7.76%)
OEM 物理反演    ████████████████░░░░  80%  simple n=100 + MonoRTM n=100 基线完成
MonoRTM 编译    ████████████████████ 100%  macOS + Linux (WSL) 双平台
TAPE3 光谱数据  ████████████████████ 100%  已下载并转二进制
S_a 协方差      ████████████████░░░░  80%  v4-derived S_a (14x14) 已生成
EOF/PCA 降维    ████████████░░░░░░░░  60%  T/RH 各 5 EOF，基础就绪
LWC 云天 OEM    ████████░░░░░░░░░░░░  40%  21d synthetic 实验完成
ERA5 气压层     ████████░░░░░░░░░░░░  05%  47/2556 天 (CDS 3天窗口)
BRNN+OEM 桥接   █░░░░░░░░░░░░░░░░░░░  05%  待 MP-3000A Obs_BT 数据
```

---

## 技术路线

```
                    ┌─ BRNN v4 ────────────→ T(z), RH(z) (统计反演，0.5ms)
                    │
Obs_BT ──→ QC/订正 ─┤
                    │
                    └─ OEM / 1D-Var ──────→ T(z), RH(z), 后验误差, AK, DOFS (物理反演)
                         │
                         ├─ x_a: ERA5 / BRNN v4 first guess
                         ├─ H(x): MonoRTM v5.6 / simple RTM
                         ├─ S_a: v4 残差协方差 / 指数相关
                         └─ 状态: 14d 粗分层 → EOF/PCA → 21d T+RH+LWC
```

---

## 项目结构

```
mwr-retrieval/
├── src/                          核心代码
│   ├── brnn_model.py             BRNN 网络定义 (PyTorch, 6 子模型)
│   ├── train.py                  训练脚本
│   ├── evaluate.py               评估/可视化
│   ├── forward_model.py          前向模型统一接口 (simple/MonoRTM)
│   ├── brightness_temp.py        Python 简化辐射传输模型
│   ├── monortm_wrapper.py        MonoRTM v5.6 Python 封装
│   ├── oem.py                    OEM 求解器 (LM/Gauss-Newton)
│   ├── oem_state.py              状态向量打包/解包
│   ├── oem_covariance.py         S_a / S_e 协方差构造
│   ├── oem_observation.py        观测算子与 S_e
│   ├── qc_correction.py          ERA5 质控 + 偏差订正
│   ├── era5_preprocess.py        ERA5 下载/预处理
│   └── sounding_process.py       探空数据处理/LWC 估算
├── scripts/                      OEM 运行脚本
│   ├── run_oem_synthetic.py      Synthetic closure test
│   ├── run_oem_201301.py         2013-01 ERA5 OEM
│   └── evaluate_oem.py           OEM 诊断评估
├── models_mp3000a_v4/       ★   当前最佳 BRNN 模型 (6个 .pt)
├── models_mp3000a_v6/            两阶段训练模型
├── bin/
│   ├── monortm                   MonoRTM macOS 版
│   └── monortm_linux             MonoRTM Linux/WSL 版
├── data/
│   ├── TAPE3/TAPE3_bin           TAPE3 二进制光谱数据
│   └── era5/                     ERA5 单层 84 月 + 气压层参考
├── results/                      实验结果
│   ├── mp3000a_v4_results.pkl   v4 完整测试结果
│   ├── oem_201301_self_consistent_monortm_n100/  MonoRTM OEM n=100
│   ├── oem_covariance/sa_v4.pkl  v4-derived S_a
│   ├── oem_pca_state/            EOF/PCA 降维结果
│   └── oem_lwc_synthetic/        LWC 云天合成实验
├── docs/                         设计文档
│   ├── oem_retrieval_plan.md     OEM 实施计划 (2026-07-13)
│   ├── 论文方法融合下一阶段计划_2026-07-21.md   最新路线图
│   ├── oem_cloudy_extension_design.md           云天扩展设计
│   ├── brnn_oem_bridge_plan.md                  BRNN+OEM 桥接方案
│   └── lwc_extension_roadmap.md                 液态水路线图
├── reports/                      报告与进度记录
│   ├── 技术报告/MWR_Retrieval_Report_v4_CN.docx
│   └── 进度记录/
├── config.py                     全局配置
├── run.py                        主流水线
├── train_mp3000a_v4.py      ★   当前最佳训练脚本
└── requirements.txt              Python 依赖
```

---

## 已完成事项

### BRNN 统计反演（v1-v6，6 轮迭代）
- [x] **v1**：基础 BRNN，合成数据端到端验证
- [x] **v2**：首个超论文模型 (T=1.45K, RH=9.0%)
- [x] **v3**：Sim_BT 训练，确认 Sim->Obs domain gap 退化 3.7x
- [x] **v4 ★**：Obs_BT 直接训练 + Winsorize BT 订正 + 廓线分组防泄露 (T=1.26K, RH=7.76%)
- [x] **v6**：Sim_BT 两阶段训练 (T=1.92K, Sim->Obs gap 0.61K)

### OEM 物理反演框架
- [x] LM/Gauss-Newton OEM 求解器 + 有限差分 Jacobian
- [x] 状态向量打包/解包（14d 粗分层 T7+RH7）
- [x] S_a / S_e 协方差构造（指数相关 + v4-derived）
- [x] 前向模型统一接口（simple RTM + MonoRTM 双后端）
- [x] Averaging kernel、DOFS、posterior covariance 诊断
- [x] Synthetic closure test 通过
- [x] ERA5 2013-01 self-consistent / forward mismatch / MonoRTM 三类 POC
- [x] MonoRTM self-consistent **n=100** 基线
- [x] v4-derived S_a 生成 (14x14)
- [x] EOF/PCA 状态降维基础
- [x] 21 维 T+RH+LWC synthetic 实验

### MonoRTM & 环境
- [x] MonoRTM v5.6 源码编译 (macOS ARM64 + Linux x86-64)
- [x] TAPE3 下载 + ASCII->二进制转换 (convert_tape3.py)
- [x] WSL Ubuntu-24.04 环境 + Python 3.12 venv
- [x] MonoRTMForwardModel 接入 OEM

### 报告与文档
- [x] 技术报告 MWR_Retrieval_Report_v4_CN (Word + PDF)
- [x] 专业实践报告
- [x] OEM 实施计划 + 论文方法融合路线图
- [x] 云天扩展 + BRNN-OEM 桥接设计文档

---

## 下一阶段计划

> 详见 [docs/论文方法融合下一阶段计划_2026-07-21.md](docs/论文方法融合下一阶段计划_2026-07-21.md)

### P0（2 周）：闭环验证
- [ ] 扩大 MonoRTM OEM 基线：n=200/500/744
- [ ] 通道与 Obs_BT 资产审计：MP-3000A 22ch / HATPRO 14ch 映射表
- [ ] 虚拟多仰角 OEM 机制试验：A/B/C 三组对照

### P1（3-6 周）：混合反演部件
- [ ] EOF/PCA 状态向量对照：10d/14d/16d
- [ ] S_a 三层递进：指数相关 -> v4 残差 -> 场景条件
- [ ] BRNN 先验桥接：22ch 直接桥接 或 14ch 适配

### P2（2-3 月）：Surrogate + 云天
- [ ] MonoRTM residual surrogate (NN 加速 H(x))
- [ ] LWC 云天分阶段 OEM：弱液云 -> 多云层 -> cloud-dependent S_e
- [ ] 不确定度校准：PICP / MPIW / CRPS / reliability diagram

---

## 快速开始

### WSL / Linux 环境

```bash
cd /mnt/d/project-504/mwr-retrieval-main
source .venv-wsl/bin/activate
python3 -c "import torch, config; print(f'Torch {torch.__version__}, {config.N_LAYERS} layers ok')"
```

### 复现 BRNN v4 训练

```bash
python train_mp3000a_v4.py
# 预期: T_RMSE=1.26K, RH_RMSE=7.76%
```

### 运行 MonoRTM OEM (2013-01)

```bash
python scripts/run_oem_201301.py
# 使用 bin/monortm_linux + data/TAPE3/TAPE3_bin
```

### 生成报告

```bash
python generate_report.py
```

---

## 关键决策记录

| 决策 | 理由 | 结果 |
|------|------|------|
| Obs_BT 直接训练 (v4) 而非 Sim_BT | v3 实验证明 Sim->Obs domain gap 退化 3.7x | v4 为当前最优 |
| 6 个独立 BRNN (非单一多输出) | 各高度区间物理特性差异大 | T RMSE 1.26K |
| 廓线分组划分 (非时间划分) | 同一廓线被重复观测 ~5.6 次，必须防泄露 | 测试集完全独立 |
| Winsorize 回归 BT 订正 (非 OLS) | K 波段 OMB 厚尾分布 | OMB std -49% |
| MonoRTM 物理前向为主 | Python simple RTM K 波段偏差 ~195K | 精度保证 |
| 先做 self-consistent 验证 | Forward mismatch 会导致 T 严重退化 | 明确 H(x) 不一致风险 |

---

## 历史版本

| Tag | 说明 | 指标 |
|-----|------|------|
| v2.0 | 首版超论文 | T=1.45K, RH=9.0% |
| v4.0 ★ | 当前最佳 | T=1.26K, RH=7.76% |
| v6.0 | Sim_BT 两阶段 | T=1.92K |
