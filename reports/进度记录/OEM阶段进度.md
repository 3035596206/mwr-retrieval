# OEM 最优估计反演 — 阶段进度报告

> 更新时间：2026-07-13
> 阶段：复查 + WSL/MonoRTM 接入完成

---

## 一、总体进度

| 阶段 | 状态 | 内容 |
|------|------|------|
| Day 1-3 | ✅ | 设计 + 基础模块 |
| Day 4 | ✅ | 合成闭环测试 |
| Day 5-7 | ✅ | 2013-01 POC + 评估 |
| Day 8-9 | ⏭️ | BRNN first guess（阻塞：缺 MP-3000A Obs_BT，BRNN 11ch vs HATPRO 14ch）|
| Day 10-11 | ✅ | LWC/云天设计文档 |
| Day 12-14 | ✅ | **WSL + MonoRTM 接入（本次复查完成）** |
| Day 15 | ✅ | 阶段报告更新 |

完成度：**80%** — WSL Linux 前向模型链路已打通。

---

## 二、本次复查成果（2026-07-13）

### 2.1 环境验证

| 检查项 | 结果 |
|--------|------|
| WSL | Ubuntu-24.04, Running, Version 2 |
| Python venv | .venv-wsl, Python 3.12.3，全部依赖满足 |
| 编译工具链 | gfortran 13.3, gcc 13.3, make 4.3, libnetcdf-dev |
| 模块导入 | OEMSolver, OEMStatePacker, BRNN, MonoRTM, ForwardModel |### 2.2 OEM 实验拆分

 修改：已新增 、、，输出目录自动区分。

| 实验 | n | T RMSE (prior->post) | RH RMSE | BT RMS | 收敛 | DOFS |
|------|---|:---:|:---:|:---:|:---:|:---:|
| Self-consistent (simple RTM) | 20 | 2.38K -> 2.06K (+13%) | 7.59% -> 6.73% (+11%) | 1.50->0.54K | 100% | 2.09 |
| Non-self-consistent (simple RTM) | 20 | 2.38K -> 27.00K (-1036%) ❌ | 7.59% -> 7.21% | 22.92->20.15K | 100% | 2.21 |
| **Self-consistent (MonoRTM)** | 5 | **2.41K -> 2.00K (+17%)** | **8.97% -> 7.84% (+13%)** | **6.41->0.75K (+88%)** | **100%** | **2.25** |

关键结论：
- Self-consistent OEM 两种前向模型均通过验证
- Non-self-consistent 确认了 forward-model mismatch 导致 T 退化（之前报告的 27.16K 已独立复现）
- MonoRTM 单廓线含 Jacobian 耗时 ~7s，批量可用

### 2.3 MonoRTM 编译与链路

| 步骤 | 结果 |
|------|------|
| TAPE3_bin | 231KB，LNFL 二进制，可用 |
| 编译产物 | ，ELF 64-bit x86-64，4.9MB |
| 编译参数 |  |
| 模拟验证 | 5/5 廓线通过，K-band 180-223K，V-band 261-280K（物理合理） |
| 代码修复 |  L51:  ->  +  |

### 2.4 Simple RTM vs MonoRTM 对比

| 波段 | Simple RTM | MonoRTM | 差异 |
|------|-----------|---------|:---:|
| K-band 22-31 GHz | 15-28 K | 195-224 K | ~+190K |
| V-band 51-58 GHz | 197-282 K | 259-283 K | +/-27K |

**simple RTM 仅可用于 OEM 算法闭环，物理结论必须用 MonoRTM。**
---

## 三、关键设计决策（更新）

| 决策 | 选择 | 依据 |
|------|------|------|
| 状态向量 | 14维 T7+RH7 | Plan §4.1 |
| 前向模型 | simple RTM -> **MonoRTM ✅** | 已接入并验证 |
| S_a | 指数相关 sigma_T=2K, sigma_RH=8% | Plan §6.1 |
| S_e | 对角 K-band 1.5K, V-band 0.5K | Plan §7 |
| BRNN v4 first guess | **阻塞**：11ch MP-3000A vs 14ch HATPRO | 需重训或通道映射 |
| PAMTRA | 暂缓，不阻塞主线 | 长线方案 |

---

## 四、后续工作路线

### 立即可推进

| 优先级 | 任务 |
|--------|------|
| P0 | MonoRTM 大样本 self-consistent OEM (--n-samples 100) |
| P1 | 从 mp3000a_v4_results.pkl 构造 S_a |
| P1 | 实现 EOF/PCA 状态向量降维 |
| P2 | LWC 云天 OEM 实验（21维） |

### 阻塞项

| 阻塞 | 原因 |
|------|------|
| BRNN v4 + OEM 混合 | 缺 Obs_BT；BRNN 11ch vs 管线 14ch |
| PAMTRA | 需独立 conda + Fortran（长线） |

### 快速启动命令


---

## 五、对外汇报

### 简短版（口头/微信）

> BRNN v4 已达到 T=1.26K、RH=7.76%。本阶段完成 OEM 框架闭环验证，在 WSL Ubuntu 中编译了 MonoRTM Linux 版。Self-consistent OEM 中 T 改善 13-17%、RH 改善 11-13%、BT residual 下降 64-88%。TAPE3_bin 和 monortm_linux 已就绪，ForwardModel(backend="monortm") 可直接调用。后续重点：大样本 MonoRTM OEM、BRNN 通道适配、LWC 云天扩展。

### 正式版（书面汇报）

> 在 BRNN v4 统计反演基础上，本阶段推进了 OEM 物理约束反演的工程化落地：
>
> 1. **WSL Linux 前向模型链路**：Ubuntu-24.04 (WSL2) 中编译 MonoRTM v5.6 (gfortran 13)，TAPE3_bin 已生成。ForwardModel 通过 backend="monortm" 一键切换。
> 2. **OEM 自洽验证**：ERA5 2013-01 self-consistent 实验中 T/RH/BT 全面改善；非自洽实验确认了 forward-model mismatch 导致的 T 退化。
> 3. **MonoRTM OEM POC**：5 条廓线 T RMSE 2.41->2.00K (+17%)，RH 8.97->7.84% (+13%)，BT RMS 6.41->0.75K (+88%)，收敛 100%。
> 4. **已知限制**：BRNN v4 为 11ch MP-3000A 模型 vs 14ch HATPRO；缺 MP-3000A Obs_BT；PAMTRA 未推进。

---

## 六、文件导航




---

## 七、第二轮实施成果（2026-07-13 更新）

### 7.1 全部实验汇总

| 实验 | n | 前向 | 状态向量 | T改善 | RH改善 | BT改善 |
|------|:---:|------|------|:---:|:---:|:---:|
| Self-consistent simple | 20 | simple | 14d | +13% | +11% | +64% |
| Self-consistent MonoRTM | 100 | monortm | 14d | +10.1% | +6.1% | +87.8% |
| v4-derived S_a | 10 | monortm | 14d | +22% | -1% | — |
| EOF/PCA 10d | 10 | monortm | 10d | -15% | -2% | +49% |
| RH sweep (best) | 10 | monortm | 14d | +28.3% | +10.2% | +89% |
| LWC 21d cloud | 5 | monortm | 21d | 稳定 | 稳定 | ICLWC 92% |

### 7.2 新增产出



### 7.3 下一轮建议


