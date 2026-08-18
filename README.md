# MWR 大气温湿廓线反演系统

基于 BRNN + OEM 的地基多通道微波辐射计大气温湿廓线反演系统。利用 MP-3000A 微波辐射计 22 通道实测亮温反演 0-10 km 范围内 93 层高度网格上的温度 T(z) 和相对湿度 RH(z) 垂直廓线。

**核心论文**：朱柳桦《基于地基微波辐射计的多方法反演大气温湿廓线研究》（2023，南京信息工程大学）第 3 章  
**仓库**：https://github.com/3035596206/mwr-retrieval  
**上次更新**：2026-07-21

> **数据与实验状态说明**：历史 BRNN/OEM 数字来自不同数据集和部分 self-consistent 实验，不能替代严格独立探空验证。成都 21 通道 48 层流程当前仍为小样本研究验证；其 RH 产品尚未通过基线与独立性检验。详见 [`docs/data-catalog.md`](docs/data-catalog.md) 和 [`docs/workflows.md`](docs/workflows.md)。

---

## 数据目录与新工作流

新数据默认保存在独立目录 `D:\project-504-data`（可通过 `MWR_DATA_ROOT` 覆盖），原始科学文件保留在文件系统，SQLite catalog 保存哈希、覆盖范围、来源和谱系。初始化与历史资产只读登记：

```powershell
python -m mwr_retrieval.cli.data init
python -m mwr_retrieval.cli.data register-existing
```

工作区根目录 `D:\project-504` 下的成都 ERA5、成都亮温和温江探空源文件，先生成入湖计划，再复制、校验、登记并关联到 `project-brnn`：

```powershell
python -m mwr_retrieval.cli.data workspace-ingest-plan
python -m mwr_retrieval.cli.data migrate-workspace-sources
```

新 ERA5 下载请使用 `python scripts/download_era5.py ...`；根目录的 `dl_*`、`bulk_*` 和版本化 MP-3000A 脚本保留为 legacy 历史工作流，不建议用于新增任务。

---

## 核心指标

### BRNN v4（统计反演，当前最佳）

| 指标 | 数值 | 论文基准 |
|------|------|----------|
| T RMSE | **1.26 K** | <1.5 K |
| RH RMSE | **7.76%** | <13% |
| 推理速度 | ~0.5 ms/廓线 | — |

### ARTS OEM（物理反演，当前主线）

前向模型主线已切换为 ARTS，以对齐研究所和同组工作流。`ForwardModel()` 默认使用 `backend="arts"` 和成都 21 通道；本地 ARTS agenda 通过 `ARTS_FORWARD_MODEL_COMMAND` 或 `arts_runner` 接入。MonoRTM 结果保留为历史基线。详见 [`docs/arts-forward-model.md`](docs/arts-forward-model.md)。

### MonoRTM OEM n=100（历史基线，self-consistent）

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
OEM 物理反演    ████████████████░░░░  80%  ARTS 主后端已切换，待跑 ARTS baseline
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
                         ├─ H(x): ARTS / pyarts（主线）；MonoRTM/simple 为历史/调试后端
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
│   ├── forward_model.py          前向模型统一接口 (ARTS 主线，simple/MonoRTM legacy)
│   ├── arts_forward_model.py     ARTS runner/pyarts 适配层
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
- [x] 前向模型统一接口（ARTS 主后端 + simple/MonoRTM legacy 后端）
- [x] Averaging kernel、DOFS、posterior covariance 诊断
- [x] Synthetic closure test 通过
- [x] ERA5 2013-01 self-consistent / forward mismatch / MonoRTM 三类 POC
- [x] MonoRTM self-consistent **n=100** 基线
- [x] v4-derived S_a 生成 (14x14)
- [x] EOF/PCA 状态降维基础
- [x] 21 维 T+RH+LWC synthetic 实验

### ARTS / MonoRTM & 环境
- [x] ARTS 被设为默认 OEM 前向模型后端（通过 runner/command 接入本地 agenda）
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
- [ ] 接入研究组 ARTS runner，并建立成都 21 通道 ARTS OEM 基线：n=100/200/500
- [ ] 通道与 Obs_BT 资产审计：MP-3000A 22ch / HATPRO 14ch 映射表
- [ ] 虚拟多仰角 OEM 机制试验：A/B/C 三组对照

### P1（3-6 周）：混合反演部件
- [ ] EOF/PCA 状态向量对照：10d/14d/16d
- [ ] S_a 三层递进：指数相关 -> v4 残差 -> 场景条件
- [ ] BRNN 先验桥接：22ch 直接桥接 或 14ch 适配

### P2（2-3 月）：Surrogate + 云天
- [ ] ARTS residual surrogate (NN 加速 H(x))
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

### 运行 ARTS OEM (2013-01)

```bash
python scripts/run_oem_201301.py --forward arts --n-samples 100
# 本机默认通过 WSL Ubuntu-24.04 调用持久 ARTS runner:
# /home/inkp/miniconda3/envs/arts/bin/python scripts/run_arts_profile.py --server
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
| ARTS 物理前向为主 | 对齐研究所和同组 ARTS 工作流，支持成都 21 通道和后续云天扩展 | 新主线 |
| MonoRTM 保留为历史基线 | 已有 n=100 self-consistent 结果，可用于交叉验证 | legacy |
| 先做 self-consistent 验证 | Forward mismatch 会导致 T 严重退化 | 明确 H(x) 不一致风险 |

---

## 历史版本

| Tag | 说明 | 指标 |
|-----|------|------|
| v2.0 | 首版超论文 | T=1.45K, RH=9.0% |
| v4.0 ★ | 当前最佳 | T=1.26K, RH=7.76% |
| v6.0 | Sim_BT 两阶段 | T=1.92K |

## 当前断点 (2026-07-22)

### 已完成工作

**P0 闭环验证（全部完成）**：
- [x] MonoRTM OEM baseline：simple RTM n=100 + MonoRTM n=20/n=100 三组基线
- [x] 通道审计：HATPRO 14ch / MP-3000A 22ch / MonoRTM 三套体系对照
- [x] ForwardModel + MonoRTM 扩展自定义 frequencies 参数
- [x] src/oem_geometry.py 仰角/通道几何筛选模块
- [x] scripts/run_oem_baseline.py 固定配置基线脚本
- [x] scripts/scan_rh_sensitivity.py RH S_a/S_e 参数敏感性扫描
- [x] README 全面更新至 2026-07-21 状态

**关键数值**：

| 实验 | T prior→post | RH prior→post | BT prior→post | 收敛率 | DOFS |
|------|-------------|---------------|---------------|--------|------|
| BRNN v4 | — | — | — | — | — |
|  | **1.26 K** (整层) | **7.76%** (整层) | — | — | — |
| OEM simple n=100 | 2.64→**1.95K** | 6.43→6.06% | 1.59→0.54K | 97.0% | 2.10 |
| OEM MonoRTM n=100 | 2.64→**2.02K** | 6.49→6.20% | 5.03→0.61K | 99.0% | 2.21 |

**分层精度诊断**：

| 高度区间 | BRNN v4 | OEM MonoRTM (ΔT/ΔRH) |
|----------|---------|----------------------|
| 0-0.5 km | T=1.09K, RH=5.78% | +0.42K, +0.67% |
| 2-8 km | T=1.30K, RH=9.50% | +0.07K, +0.28% |
| 5-10 km | T=1.50K, RH=10.31% | **−0.01K, +0.10%** |

> **核心发现**：OEM 改善集中在 0-2 km 近地层；5-10 km T 反演完全依赖先验（不升反降）。DOFS 仅 2.2/14，高空信息量极低。

### 下一阶段优先级

1. **P1**: 接入研究组 ARTS runner，完成成都 21 通道 self-consistent baseline
2. **P1**: 改善 RH 约束 — `scripts/scan_rh_sensitivity.py --forward arts` 参数扫描
3. **P1**: S_a 对比实验 — 指数相关 vs v4-derived S_a
4. **P2**: 增强诊断输出 — 0-500m 指标 + posterior uncertainty

> **待用户提供**：D:\project-504 下新增的成都 ERA5 + 温江探空数据，用于新站点训练/测试。
