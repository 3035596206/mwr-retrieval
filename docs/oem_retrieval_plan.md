# OEM 最优估计反演实施计划

> 更新时间：2026-07-13 复查版  
> 项目：MWR 大气温湿廓线反演  
> 目标：在现有 BRNN v4 最佳模型基础上，维护已跑通的 OEM / 1D-Var 框架和 WSL/Linux MonoRTM 前向模型链路，并继续推进大样本 OEM、协方差优化、EOF/PCA 降维和云天液态水约束。

---

## 0. 当前复查结论（2026-07-13）

本次复查后，项目状态已经从“基础 OEM 框架已实现”推进到“WSL + MonoRTM 物理前向模型链路已打通”。当前重点不再是环境接入，而是扩大 MonoRTM OEM 样本、构造更真实的背景协方差、引入 EOF/PCA 降维，并为 LWC 云天 OEM 做准备。

### 0.1 已新增文件

OEM 核心模块已经存在：

```text
src/oem.py
src/oem_state.py
src/oem_covariance.py
src/oem_observation.py
src/forward_model.py
```

OEM 脚本已经存在：

```text
scripts/run_oem_synthetic.py
scripts/run_oem_201301.py
scripts/evaluate_oem.py
```

OEM 结果目录已经存在：

```text
results/oem_synthetic/
results/oem_201301/
```

云天扩展设计文档已经存在：

```text
docs/oem_cloudy_extension_design.md
```

### 0.2 2013-01 OEM POC 指标

当前实验已经按输出目录拆分，避免了之前 `poc_stats.pkl` 被覆盖的问题。以 2026-07-13 的三个独立目录为准：

```text
results/oem_201301_self_consistent/
results/oem_201301_forward_mismatch/
results/oem_201301_self_consistent_monortm/
```

| 实验 | n | T RMSE | RH RMSE | BT RMS | 收敛 | DOFS |
|------|---|--------|---------|--------|------|------|
| Self-consistent simple RTM | 20 | 2.782 -> 2.121 K | 8.890 -> 8.990 % | 6.805 -> 0.680 K | 100% | 2.306 |
| Forward mismatch simple RTM | 20 | 2.377 -> 27.001 K | 7.590 -> 7.215 % | 22.917 -> 20.153 K | 100% | 2.208 |
| Self-consistent MonoRTM | 5 | 2.410 -> 1.995 K | 8.965 -> 7.836 % | 6.410 -> 0.749 K | 100% | 2.251 |

解释：

```text
1. simple RTM 自洽实验：T 和 BT 明显改善，RH 略退，说明算法链路可用但 RH 约束仍需调参。
2. forward mismatch 实验：T 严重退化，独立复现了 H(x) 与观测 BT 不一致会破坏温度反演。
3. MonoRTM 自洽实验：T/RH/BT 全部改善，是当前最可信的物理前向模型 POC。
```

结论：OEM 数值框架已通过 self-consistent 验证；真实物理路线应以 MonoRTM 为主，simple RTM 只保留为算法闭环工具。

### 0.3 WSL / Ubuntu 状态

阶段报告显示 WSL/Linux 环境已经完成：

```text
WSL: Ubuntu-24.04, Running, Version 2
Python venv: .venv-wsl, Python 3.12.3
编译工具链: gfortran 13.3, gcc 13.3, make 4.3, libnetcdf-dev
模块导入: OEMSolver, OEMStatePacker, BRNN, MonoRTM, ForwardModel
```

项目文件也验证了 WSL venv 已存在：

```text
.venv-wsl/pyvenv.cfg
command = /usr/bin/python3 -m venv /mnt/d/project-504/mwr-retrieval-main/.venv-wsl
```

注意：当前 Codex PowerShell 会话中 `wsl -l -v` 仍未正常列出 Ubuntu，输出类似“未安装发行版”。这与阶段报告和 `.venv-wsl` 证据冲突，暂按“Codex 会话无法可靠查询 WSL，但用户普通 Ubuntu/PowerShell 环境已打通”处理。

后续如需由 Codex 直接调用 WSL，应先复核当前会话权限和 WSL 可见性。

```text
wsl --status 可执行，显示 WSL 默认版本为 2（输出编码异常但命令成功）
wsl -l -v 返回退出码 1，未正常列出 Ubuntu
```

### 0.4 当前阶段判断

当前项目不再处于“先写 OEM 框架”的阶段，而应转入：

```text
1. MonoRTM 大样本 self-consistent OEM（n=100）
2. 从 mp3000a_v4_results.pkl 构造 S_a
3. 实现 EOF/PCA 状态向量降维
4. 设计并验证 21 维 T+RH+LWC 云天 OEM
5. 解决 BRNN v4 + OEM 混合路线的 Obs_BT 与通道不一致阻塞
6. 将 PAMTRA 作为长线云天方案，不阻塞 MonoRTM 主线
```

### 0.5 MonoRTM / TAPE3 状态

当前数据状态：

```text
data/TAPE3/TAPE3      已存在，137059 bytes
data/TAPE3/TAPE3_bin  已存在，235912 bytes
bin/monortm_linux     已存在，5133592 bytes
```

这是重要进展，说明 TAPE3 二进制转换和 Linux 版 MonoRTM 编译都已经完成。

当前可执行文件状态：

```text
bin/monortm           macOS Mach-O 旧二进制，保留但不用于 WSL
bin/monortm_linux     ELF 64-bit x86-64，WSL/Linux 可用
```

结论：

```text
TAPE3_bin 已就绪；
Linux/WSL 版 MonoRTM 已就绪；
ForwardModel(backend="monortm") 已可用于 OEM 小样本 POC。
```

### 0.6 深度调研报告吸收结论

本次额外读取：

```text
C:/Users/Administrator/Downloads/deep-research-report.md
```

该报告将“神经网络 + OEM”方法分为四类：

| 类别 | 核心思想 | 对本项目的适配度 |
|------|----------|------------------|
| NN 作为先验或先验协方差 | 用 NN 输出 `x_a` 或 `S_a`，再进入 OEM | 最高，直接对应 BRNN v4 first guess 和 v4-derived `S_a` |
| NN 替代前向模型 | 用 NN surrogate 近似 `H(x)`，加速 OEM | 高，适合用 MonoRTM 批量样本训练 surrogate |
| NN 误差建模/后处理 | 用 NN 修正 OEM 残差或校准不确定度 | 中高，适合处理 forward-model mismatch 和 BT residual |
| Unrolled / DeepOE | 将 OEM 迭代展开为可训练网络 | 长线，需稳定 OEM 基线后再做 |

结合当前项目进度，新的方法优先级为：

```text
P0: 稳定 MonoRTM OEM baseline
P1: BRNN / v4-derived S_a 作为 OEM 先验协方差
P1: MonoRTM surrogate forward model，用 NN 加速 H(x)
P2: 不确定度校准（ensemble / MC dropout / QRNN）
P3: DeepOE / unrolled OEM，作为论文级长线方向
```

该报告还强调，NN+OEM 不应只比较 RMSE，还要评估不确定度校准：

```text
PICP：预测区间覆盖率
MPIW：平均预测区间宽度
CRPS：连续排名概率评分
reliability diagram：置信度可靠性图
posterior covariance calibration：后验协方差校准
```

因此，后续评估体系应从：

```text
RMSE / Bias / BT residual / DOFS
```

扩展为：

```text
RMSE / Bias / BT residual / DOFS / posterior uncertainty / PICP / calibration
```

---

## 1. 总体定位

当前项目已经完成 BRNN 统计反演路线，并在 MP-3000A 数据上得到当前最佳 v4 结果：

```text
T_RMSE  = 1.2639 K
RH_RMSE = 7.759 %
```

OEM（Optimal Estimation Method，最优估计法）不应直接替代 BRNN v4，而应作为下一阶段的物理约束反演框架。推荐定位为：

```text
BRNN v4：提供快速统计反演和 first guess
OEM / 1D-Var：提供物理一致性修正、不确定度估计和观测信息量诊断
BRNN + OEM：形成混合反演系统
```

最推荐的总体路线：

```text
Obs_BT
  ↓
BRNN v4 first guess
  ↓
OEM / 1D-Var 后验修正
  ↓
T_retrieved, RH_retrieved, 后验误差, averaging kernel, DOFS
```

---

## 2. OEM 数学框架

OEM 通过同时约束背景场和观测亮温，寻找最优大气状态向量 `x`。目标函数为：

```text
J(x) =
(x - x_a)^T S_a^-1 (x - x_a)
+
(y - H(x))^T S_e^-1 (y - H(x))
```

变量含义：

| 符号 | 含义 | 本项目建议 |
|------|------|------------|
| `x` | 待反演状态向量 | T(z), RH(z)，后续扩展 LWC(z) |
| `x_a` | 背景场 / 先验 | ERA5、NCEP-FNL 或 BRNN v4 输出 |
| `S_a` | 背景误差协方差 | 从 BRNN v4 残差或 ERA5 统计估计 |
| `y` | 观测向量 | MWR 观测亮温 Obs_BT |
| `H(x)` | 前向算子 | simple RTM / MonoRTM / PAMTRA |
| `S_e` | 观测误差协方差 | 仪器误差 + OMB 残差 + 前向模型误差 |

Levenberg-Marquardt / Gauss-Newton 更新形式：

```text
x_{i+1} = x_i + Δx

Δx =
[K^T S_e^-1 K + S_a^-1 + γI]^-1
[
K^T S_e^-1 (y - H(x_i))
-
S_a^-1 (x_i - x_a)
]
```

其中：

```text
K = ∂H(x) / ∂x
γ = LM 阻尼因子
```

---

## 3. 与当前项目的关系

当前项目已经具备 OEM 所需的大部分基础组件：

| 组件 | 当前状态 | 可复用文件 |
|------|----------|------------|
| 高度网格 | 已完成，93 层 | `config.py` |
| 通道配置 | 已完成，14 通道 HATPRO + MP-3000A 训练脚本 | `config.py`, `train_mp3000a_v4.py` |
| BRNN first guess | v4 当前最佳 | `models_mp3000a_v4/` |
| 误差统计 | v1-v6 结果齐全 | `results/*.pkl` |
| 简化前向模型 | 已有，但精度有限 | `src/brightness_temp.py` |
| MonoRTM wrapper | 已有，Linux 版链路已验证 | `src/monortm_wrapper.py`, `bin/monortm_linux` |
| QC / OMB 经验 | 已有 | `train_mp3000a_v4.py`, `src/qc_correction.py` |
| LWC / 云污染分析 | 已有 | `docs/lwc_extension_roadmap.md`, `reports/进度记录/未订正缺陷.md` |

OEM 基础能力复查状态：

| 能力 | 状态 | 对应文件 / 备注 |
|------|------|------------------|
| OEM 求解器 | 已完成基础版 | `src/oem.py` |
| 状态向量降维与打包/解包 | 已完成基础版 | `src/oem_state.py` |
| S_a / S_e 构造模块 | 已完成基础版 | `src/oem_covariance.py`, `src/oem_observation.py` |
| 前向模型统一接口 | 已完成基础版 | `src/forward_model.py` |
| 有限差分雅可比矩阵 | 已完成基础版 | `src/oem.py` |
| averaging kernel、DOFS、posterior covariance | 已完成基础诊断 | `src/oem.py`, `scripts/evaluate_oem.py` |
| synthetic closure test | 已完成并输出图表 | `scripts/run_oem_synthetic.py`, `results/oem_synthetic/` |
| 2013-01 POC | 已按 self-consistent / mismatch / MonoRTM 三类拆分 | `scripts/run_oem_201301.py`, `results/oem_201301_*` |
| MonoRTMForwardModel | 已接入并完成 5 条样本 POC | `src/forward_model.py`, `scripts/run_oem_201301.py` |
| TAPE3_bin | 已生成并可用于 MonoRTM | `data/TAPE3/TAPE3_bin` |

当前仍需新增或强化的能力：

```text
1. MonoRTM 大样本 self-consistent OEM（n=100）
2. 从 mp3000a_v4_results.pkl 构造 S_a
3. EOF/PCA 状态向量降维
4. BRNN v4 first guess + Obs_BT 的真实样本 OEM（阻塞：缺 Obs_BT + 通道不一致）
5. 21维 T+RH+LWC 云天 OEM 实验
6. PAMTRA 可行性验证（长线，不阻塞 MonoRTM 主线）
```

---

## 4. 状态向量设计

不建议第一版直接使用完整 93 层 T+RH：

```text
x = [T_0, ..., T_92, RH_0, ..., RH_92]
状态维度 = 186
```

原因：

```text
MWR 通道数只有 14 或 22 个，观测维度远小于状态维度；
完整 93 层反演高度欠定，强依赖先验，容易病态和发散。
```

### 4.1 第一版推荐：粗分层控制变量

温度控制层：

```text
0-0.5 km
0.5-1 km
1-2 km
2-3 km
3-5 km
5-8 km
8-10 km
```

湿度控制层：

```text
0-0.5 km
0.5-1 km
1-2 km
2-3 km
3-5 km
5-8 km
8-10 km
```

状态维度：

```text
T-only:      7 维
T+RH:       14 维
T+RH+LWC:   21 维左右
```

优点：

```text
1. 稳定
2. 易调试
3. 雅可比矩阵计算成本低
4. 适合第一版闭环验证
```

### 4.2 第二版推荐：EOF / PCA 降维

从 ERA5 或 MP-3000A 匹配廓线中统计扰动：

```text
δx = x - mean(x)
δx ≈ E z
```

反演低维 EOF 系数 `z`：

```text
x = x_a + E z
```

推荐保留：

```text
T:  5-8 个 EOF
RH: 5-8 个 EOF
```

最终状态维度约：

```text
10-16 维
```

实施建议：

```text
第一阶段：粗分层控制变量
第二阶段：EOF / PCA 控制变量
第三阶段：LWC 状态量扩展
```

---

## 5. 背景场 `x_a` 设计

候选背景场：

| 背景来源 | 优点 | 缺点 | 用途 |
|----------|------|------|------|
| ERA5 | 物理一致，传统 OEM 路线标准 | 可能不如 BRNN 贴近 MWR 观测 | POC 和基准实验 |
| BRNN v4 | 当前精度最高，适配 Obs_BT | 黑箱误差会进入先验 | 主推荐路线 |
| NCEP-FNL | 对应论文第4章 1D-Var 路线 | 需要额外下载和处理 | 后续论文复现 |
| ERA5 + BRNN 混合 | 可兼顾物理和统计 | 增加调参复杂度 | 后期探索 |

推荐路线：

```text
POC 阶段：
  x_a = ERA5 profile + 人工扰动

真实 MP-3000A 阶段：
  x_a = BRNN v4 retrieval

论文扩展阶段：
  比较 ERA5-first-guess、BRNN-first-guess、NCEP-FNL-first-guess
```

---

## 6. 背景误差协方差 `S_a`

### 6.1 第一版：指数相关协方差

使用垂直相关结构：

```text
S_a[i,j] = σ_i σ_j exp(-|z_i - z_j| / L)
```

推荐初值：

| 变量 | 高度 | 标准差 |
|------|------|--------|
| T | 0-2 km | 1.5 K |
| T | 2-8 km | 2.0 K |
| T | 8-10 km | 2.5 K |
| RH | 0-2 km | 8 % |
| RH | 2-8 km | 12 % |
| RH | 8-10 km | 15 % |

推荐垂直相关长度：

```text
T:  1.0-2.0 km
RH: 0.5-1.0 km
```

### 6.2 第二版：从 BRNN v4 残差估计

利用当前已有结果文件：

```text
results/mp3000a_v4_results.pkl
```

计算：

```text
T_error  = T_pred  - T_true
RH_error = RH_pred - RH_true
```

然后估计：

```text
S_a = cov([T_error, RH_error])
```

这是本项目的优势：`S_a` 不需要凭空假设，可以直接来自当前最佳模型的测试误差统计。

实现注意：

```text
1. 协方差矩阵必须正定
2. 需要加正则项 S_a = S_a + λI
3. 高维协方差可先降维到粗分层或 EOF 空间
```

---

## 7. 观测误差协方差 `S_e`

第一版使用对角矩阵：

```text
S_e = diag(σ_ch^2)
```

推荐初值：

| 通道类型 | 推荐误差 |
|----------|----------|
| V 波段 51-59 GHz | 0.5-1.0 K |
| K 波段 22-31 GHz | 1.5-3.0 K |
| 云污染风险高的 K 波段 | 3.0-5.0 K |

更贴合项目的做法是从 OMB 残差估计：

```text
OMB = Obs_BT_corrected - Sim_BT
σ_ch = std(OMB_ch after QC)
```

针对云天样本：

```text
1. 膨胀 K 波段 S_e
2. 或剔除污染通道
3. 或加入 LWC 状态量
```

---

## 8. 前向模型 `H(x)`

推荐设计统一接口：

```python
class ForwardModel:
    def simulate(self, profile):
        """Return brightness temperatures."""
```

候选实现：

| 模型 | 优点 | 缺点 | 用途 |
|------|------|------|------|
| Simple RTM | Python 原生，易调试 | 精度不足 | 算法闭环 |
| MonoRTM | 已有 wrapper，物理可信 | Windows 当前环境可能不可直接运行 | 物理验证 |
| PAMTRA | 支持云，适合 LWC | 需新环境 | 云天扩展 |

推荐顺序：

```text
1. simple RTM 跑通 OEM 算法闭环
2. MonoRTM 做物理验证
3. PAMTRA 做云天和 LWC 扩展
```

注意：

```text
simple RTM 只能用于算法验证，不应用作最终物理结论。
```

---

## 9. 雅可比矩阵 `K`

第一版使用中心有限差分：

```text
K[:, j] = [H(x + ε e_j) - H(x - ε e_j)] / (2ε)
```

推荐扰动：

```text
T:   0.2 K
RH:  1.0 %
LWC: 0.01-0.05 g/m^3
```

如果使用低维状态，雅可比矩阵规模很小：

```text
14 × 7
14 × 14
22 × 14
```

不要第一版直接对完整 186 维状态做有限差分。

---

## 10. OEM 求解器设计

核心类：

```python
class OEMSolver:
    def retrieve(self, y_obs, x_a, sa, se):
        ...
```

返回结果：

```python
{
    "x_retrieved": ...,
    "x_background": ...,
    "y_obs": ...,
    "y_sim_background": ...,
    "y_sim_retrieved": ...,
    "cost_history": ...,
    "converged": ...,
    "n_iter": ...,
    "jacobian": ...,
    "averaging_kernel": ...,
    "posterior_covariance": ...,
    "dofs": ...
}
```

收敛条件：

```text
1. J 相对变化 < 1e-3
2. Δx 范数足够小
3. BT residual 不再下降
4. 达到最大迭代次数 10-15 次
```

物理边界：

```text
T:   180-330 K
RH:  0-100 %
LWC: >= 0
```

LM 阻尼策略：

```text
如果 J 下降：
  接受更新，γ 减小

如果 J 上升：
  拒绝更新，γ 增大
```

---

## 11. OEM 诊断指标

OEM 不能只报 RMSE，必须输出反演诊断。

核心指标：

```text
1. prior profile RMSE
2. posterior profile RMSE
3. prior BT residual
4. posterior BT residual
5. J_total
6. J_background
7. J_observation
8. convergence rate
9. averaging kernel
10. DOFS = trace(A)
11. posterior uncertainty
```

Averaging kernel：

```text
A = G K
```

其中：

```text
G = [K^T S_e^-1 K + S_a^-1]^-1 K^T S_e^-1
```

后验误差协方差：

```text
S_post = [K^T S_e^-1 K + S_a^-1]^-1
```

解释价值：

```text
BRNN 只能给反演结果；
OEM 还能说明结果中有多少信息来自观测、有多少来自先验。
```

---

## 12. 实验设计

### 12.1 实验 A：Synthetic closure test

目的：验证 OEM 算法实现正确。

流程：

```text
1. 取一个真实 ERA5 廓线 x_true
2. 用 H(x_true) 生成 y_true
3. 加噪声得到 y_obs
4. 构造扰动背景 x_a = x_true + noise
5. OEM 从 x_a 出发反演
6. 检查 x_oem 是否比 x_a 更接近 x_true
```

成功标准：

```text
posterior RMSE < prior RMSE
posterior BT residual < prior BT residual
cost 单调下降
converged rate > 90%
```

### 12.2 实验 B：2013-01 ERA5 + Sim_BT POC

利用现有文件：

```text
data/era5/era5_profiles_201301.pkl
data/era5/era5_bt_sim_201301.pkl
```

流程：

```text
1. ERA5 profile 作为 truth
2. bt_sim 作为 observation
3. 给 profile 加扰动作为 background
4. OEM 反演回 truth
```

成功标准：

```text
T/RH posterior 优于 background
BT residual 明显下降
```

### 12.3 实验 C：BRNN v4 first guess + OEM

目标：验证混合反演路线。

流程：

```text
1. x_a = BRNN v4 prediction
2. truth = ERA5 profile
3. y = Obs_BT_corrected 或原始 Obs_BT
4. H = MonoRTM / simple RTM
5. OEM posterior 与 BRNN prior 比较
```

注意：

```text
当前 results/mp3000a_v4_results.pkl 保存了 T/RH 预测和 truth，
但未保存 Obs_BT。要做完整实验，需要补 MP-3000A 原始 nc 数据。
```

### 12.4 实验 D：云天 / LWC OEM

目标：处理 K 波段云液态水污染。

流程：

```text
1. 状态量扩展为 T + RH + LWC
2. K 波段参与约束
3. S_e 对云污染通道自适应膨胀
4. 比较 clear-sky 与 cloudy-sky
```

成功标准：

```text
云天 K 波段 residual 降低
RH 低层反演改善
LWC 不出现非物理振荡
```

---

## 13. 推荐文件结构

新增文件：

```text
src/oem.py
src/oem_state.py
src/oem_covariance.py
src/oem_observation.py
src/forward_model.py
scripts/run_oem_synthetic.py
scripts/run_oem_201301.py
scripts/evaluate_oem.py
docs/oem_retrieval_plan.md
reports/进度记录/OEM阶段进度.md
```

职责划分：

| 文件 | 职责 |
|------|------|
| `src/oem.py` | LM / Gauss-Newton 求解器 |
| `src/oem_state.py` | 状态向量打包、解包、降维、边界约束 |
| `src/oem_covariance.py` | S_a / S_e 构造和正则化 |
| `src/oem_observation.py` | 通道选择、BT 误差、云污染通道膨胀 |
| `src/forward_model.py` | 统一前向模型接口 |
| `scripts/run_oem_synthetic.py` | 合成闭环测试 |
| `scripts/run_oem_201301.py` | ERA5 2013-01 POC |
| `scripts/evaluate_oem.py` | RMSE、BT residual、AK、DOFS、图表 |

---

## 14. 下一阶段两周实施时间表（MonoRTM 已接入版）

当前 WSL + MonoRTM 链路已经完成，小样本 MonoRTM self-consistent OEM 已通过。下一阶段重点是扩大样本、改进先验误差协方差、降低状态维度病态性，并准备云天 LWC 实验。

### 第 1-2 天：MonoRTM 大样本 self-consistent OEM

任务：

```bash
cd /mnt/d/project-504/mwr-retrieval-main
source .venv-wsl/bin/activate
python scripts/run_oem_201301.py --self-consistent --forward monortm --n-samples 100
python scripts/evaluate_oem.py
```

输出：

```text
results/oem_201301_self_consistent_monortm_n100/
100 样本 T/RH/BT 统计
收敛率、平均迭代次数、DOFS 分布
单样本耗时统计
```

验收标准：

```text
T posterior RMSE < T prior RMSE
RH posterior RMSE 不显著退化，优先目标是改善
BT RMS 降幅 > 70%
收敛率 > 90%
平均 DOFS 合理且稳定
```

### 第 3-4 天：从 BRNN v4 结果构造 `S_a`

任务：

```text
1. 读取 results/mp3000a_v4_results.pkl
2. 计算 T/RH 逐层误差和协方差
3. 映射到当前 14 维状态向量
4. 与指数相关 S_a 做对比
```

输出：

```text
src/oem_covariance.py 新增或完善 build_sa_from_v4_results()
results/oem_covariance/
S_a 热力图
逐层 prior uncertainty
```

验收标准：

```text
S_a 正定或经正则化后可逆
T/RH 不确定度量级符合 v4 误差统计
OEM 使用 v4-derived S_a 后不发散
```

### 第 5-6 天：EOF/PCA 状态向量降维

任务：

```text
1. 从 ERA5/BRNN 结果中构造 T/RH 廓线样本矩阵
2. 实现 EOF/PCA packer
3. 对比 14维粗分层 vs EOF 控制向量
```

输出：

```text
src/oem_state.py 新增 EOFStatePacker
results/oem_pca_state/
解释方差曲线
不同状态向量下的 OEM 对比
```

验收标准：

```text
前 5-8 个 EOF 可解释主要 T/RH 变化
OEM 结果比粗分层更平滑或更稳定
Jacobian 计算成本可接受
```

### 第 7-8 天：改进 RH 约束

背景：

```text
simple RTM 自洽实验中 RH 略退；
MonoRTM 小样本中 RH 改善，但样本数少。
```

任务：

```text
1. 调整 S_a 中 RH sigma 和垂直相关长度
2. 调整 K-band / V-band 的 S_e 权重
3. 对比只用 K-band、只用 V-band、全通道的 RH 反演敏感性
```

输出：

```text
results/oem_sensitivity_rh/
RH RMSE vs S_a/S_e 参数表
通道选择敏感性图
```

验收标准：

```text
找到至少一组参数使 RH 不退化或稳定改善
明确 K-band 对 RH 的信息贡献和云污染风险
```

### 第 9-10 天：LWC 云天 OEM 预实验

任务：

```text
1. 将状态向量从 14维 T+RH 扩展为 21维 T+RH+LWC
2. 设置 LWC >= 0 的边界
3. 使用 cloud-dependent S_e inflation
4. 用合成云样本做 self-consistent test
```

输出：

```text
results/oem_lwc_synthetic/
LWC prior/posterior
K-band residual before/after
T/RH 是否受 LWC 反演影响
```

验收标准：

```text
K-band residual 下降
LWC 不出现非物理负值或剧烈振荡
T/RH 不因 LWC 扩展显著退化
```

### 第 11-12 天：BRNN v4 + OEM 阻塞拆解

当前阻塞：

```text
缺 MP-3000A Obs_BT
BRNN v4 使用 MP-3000A 11ch/训练特征体系，OEM 当前是 HATPRO 14ch 管线
```

任务：

```text
1. 明确 v4 结果文件中缺哪些 OEM 必需变量
2. 检查是否可恢复 Obs_BT_corrected、surface features、profile index
3. 设计 MP-3000A 通道到 HATPRO/MonoRTM 频率的映射或重训方案
4. 决定 BRNN first guess 是短期暂缓还是新建 14ch 版本
```

输出：

```text
docs/brnn_oem_bridge_plan.md 或报告段落
BRNN + OEM 数据需求清单
通道适配方案
```

### 第 13-14 天：阶段报告与下一轮计划

任务：

```text
1. 更新 reports/进度记录/OEM阶段进度.md
2. 更新 docs/oem_retrieval_plan.md
3. 汇总 n=100 MonoRTM OEM 指标
4. 决定下一轮主线：S_a/EOF/LWC/BRNN桥接
```

输出：

```text
一份可直接发给师兄的阶段总结
一组 MonoRTM OEM 图表
下一阶段任务优先级
```

### 第 15-28 天：NN + OEM 中期路线（由深度调研补充）

深度调研报告建议不要只做传统 OEM，而要明确神经网络在 OEM 框架中的角色。本项目中期建议并行推进三条线，但优先级不同。

#### 路线 A：BRNN / NN 作为 OEM 先验

目标：

```text
用 BRNN 或 v4 残差统计改进 x_a 和 S_a。
```

短期实现：

```text
1. 使用 results/mp3000a_v4_results.pkl 构造 v4-derived S_a
2. 在 ERA5/MonoRTM OEM 中比较：
   - 固定指数相关 S_a
   - v4-derived S_a
   - 混合 S_a
3. 若 MP-3000A Obs_BT 可用，再将 BRNN 输出作为 x_a
```

验收标准：

```text
1. posterior RMSE 不劣于固定 S_a
2. posterior uncertainty 与实际误差更匹配
3. DOFS 不出现异常塌缩
```

#### 路线 B：NN surrogate forward model

目标：

```text
训练神经网络近似 MonoRTM 的 H(x)，加速 OEM 中重复调用的前向模型和雅可比计算。
```

数据来源：

```text
输入：T/RH/P/height 或降维状态向量
标签：MonoRTM 14ch BT
```

候选模型：

```text
MLP: 2-5 层，hidden 128-512
1D-CNN: 处理垂直廓线结构
Residual MLP: 预测 SimpleRTM -> MonoRTM 的残差
```

优先推荐：

```text
Residual surrogate:
BT_mono ≈ BT_simple + NN(profile)
```

原因：

```text
simple RTM 与 MonoRTM 差异很大，尤其 K-band 可达约 190K；
直接学 MonoRTM 也可行，但学习 residual 更容易解释和调试。
```

验收标准：

```text
K-band / V-band surrogate RMSE < 观测误差量级或 < 0.5-1.0K
surrogate OEM 与 MonoRTM OEM 后验接近
推理速度显著快于 MonoRTM
```

#### 路线 C：不确定度与校准

目标：

```text
让 NN+OEM 不只输出点估计，还能输出可信的不确定度。
```

候选方法：

```text
1. Ensemble：多模型散布估计不确定度
2. MC Dropout：多次 dropout 前向估计不确定度
3. QRNN：分位数回归，直接输出预测区间
4. posterior covariance calibration：用实际误差校准 OEM S_post
```

新增评估：

```text
PICP：预测区间覆盖率
MPIW：平均区间宽度
CRPS：概率评分
reliability diagram：可靠性图
```

验收标准：

```text
名义 90% 区间覆盖率接近 90%
不确定度随云污染、K-band residual、先验误差增大而增大
不确定度不系统性低估
```

#### 路线 D：DeepOE / unrolled OEM（长线）

目标：

```text
将 Gauss-Newton / LM 迭代展开为可训练网络层。
```

暂不作为当前主线，原因：

```text
1. 需要稳定 MonoRTM OEM baseline
2. 需要足够训练样本
3. 需要明确状态向量和协方差设计
4. 工程复杂度显著高于前三条路线
```

建议启动条件：

```text
MonoRTM OEM n>=1000 样本完成
surrogate forward model 误差可控
S_a / S_e / DOFS 诊断稳定
```

---

## 15. 风险与应对

### 风险 1：simple RTM 精度不足

应对：

```text
simple RTM 只用于算法闭环；
最终物理结果必须使用 MonoRTM 或 PAMTRA。
```

### 风险 2：OEM 后验 RMSE 不如 BRNN v4

这不一定代表失败。需要区分：

```text
BRNN 是统计最优；
OEM 是物理约束最优；
如果 H(x) 有系统偏差，OEM 可能被错误前向模型拉偏。
```

诊断顺序：

```text
1. BT residual 是否下降
2. cost 是否下降
3. posterior 是否更物理平滑
4. profile RMSE 是否改善
```

### 风险 3：状态维度过高导致发散

应对：

```text
从 7 维 T-only 开始；
再到 14 维 T+RH；
不要第一版直接上 186 维。
```

### 风险 4：K 波段云污染破坏 RH 反演

应对：

```text
clear-sky 阶段先降低 K 波段权重；
cloudy-sky 阶段加入 LWC；
对 K 波段 S_e 做 cloud-dependent inflation。
```

### 风险 5：MonoRTM 批量 OEM 计算成本较高

当前小样本 POC 显示 MonoRTM 单廓线含 Jacobian 耗时约 7 秒，n=100 或更大样本时计算时间会明显增加。

应对：

```text
1. 先跑 n=100，记录总耗时和失败率
2. 缓存 H(x) 和 Jacobian 中间结果
3. 优先优化有限差分次数和状态维度
4. 必要时按样本并行，而不是在单个 OEM 内部过度并行
```

### 风险 6：Codex 会话无法直接列出 WSL，但用户环境已打通

阶段报告显示 Ubuntu-24.04 WSL2、`.venv-wsl`、gfortran/gcc/make、MonoRTM Linux 版均已完成；但当前 Codex PowerShell 会话中 `wsl -l -v` 仍未正常列出 Ubuntu。

这说明问题不是单纯“从未初始化”，而是：

```text
1. 用户普通 Ubuntu/PowerShell 环境已经可用；
2. Codex 当前会话可能无法可靠查询 WSL；
3. 需要区分“用户可运行”和“Codex 可直接运行”。
```

应对：

```text
1. 用户在 Ubuntu 终端执行 WSL 命令和长任务
2. Codex 负责读写代码、更新文档、分析结果文件
3. 若需要 Codex 直接运行 WSL，先解决当前会话的 WSL 可见性
```

### 风险 7：BRNN v4 + OEM 混合路线存在通道和数据阻塞

当前 BRNN v4 使用 MP-3000A 训练体系，而 OEM/MonoRTM 当前管线是 HATPRO 14 通道。阶段报告指出阻塞项为：

```text
1. 缺 MP-3000A Obs_BT
2. BRNN v4 通道/特征体系与 HATPRO 14ch OEM 不一致
```

应对：

```text
1. 短期不强行做 BRNN v4 + OEM
2. 先完成 ERA5/MonoRTM self-consistent OEM
3. 再设计 MP-3000A 通道映射或重训 14ch/22ch 统一模型
```

### 风险 8：NN surrogate 前向模型外推失败

NN 替代 MonoRTM 后，若训练样本覆盖不足，可能在极端温湿廓线、云液态水或逆温条件下外推失败。

应对：

```text
1. surrogate 只在训练分布内使用
2. 用独立季节/极端样本测试 surrogate BT error
3. surrogate error 必须低于或可并入 S_e
4. 对超出训练范围的样本自动回退 MonoRTM
```

### 风险 9：NN 不确定度低估

深度调研报告强调，纯 NN 往往给出过度自信的点估计。若将 NN 用作 `x_a`、`S_a` 或 surrogate，必须校准不确定度。

应对：

```text
1. 使用 ensemble / MC dropout / QRNN 做不确定度估计
2. 用 PICP、MPIW、CRPS 和 reliability diagram 评估校准
3. 用实际 posterior error 校准 S_post
4. 不把未校准 NN uncertainty 当作 OEM 协方差直接使用
```

### 风险 10：训练集泄露和场景偏差

BRNN v4 已经修复过廓线分组泄露问题，但后续 surrogate、S_a、EOF/PCA 和 NN 先验都可能重新引入泄露或场景偏差。

应对：

```text
1. 按 profile / 日期 / 天气场景分组划分训练验证测试
2. 单独保留未见月份或未见天气型作为外推测试
3. 所有 NN 训练记录随机种子、样本索引和标准化参数
4. 严禁同一廓线的多次观测跨训练/测试集合
```

---

## 16. 对外汇报表述

可以向师兄/导师这样汇报：

```text
目前 BRNN v4 已达到 T=1.26K、RH=7.76% 的统计反演精度。
本阶段完成 OEM 框架闭环验证，并在 WSL Ubuntu 中编译了 MonoRTM Linux 版。
Self-consistent OEM 中，simple RTM 和 MonoRTM 均验证通过；
MonoRTM 小样本 POC 中 T RMSE 2.41->1.995K，RH 8.97->7.84%，BT RMS 6.41->0.75K，收敛率 100%，DOFS 约 2.25。
同时，非自洽实验确认 forward-model mismatch 会导致 T 严重退化。
结合最新深度调研，后续重点是四条 NN+OEM 路线：
NN 作为先验/S_a、NN surrogate 加速 MonoRTM、NN 不确定度校准，以及长期 unrolled OEM。
短期先做大样本 MonoRTM OEM、v4-derived S_a、EOF/PCA 状态向量和 LWC 云天扩展。
```

更正式的表述：

```text
本阶段在现有 BRNN 温湿廓线反演模型基础上，引入 Optimal Estimation Method。
目前已经实现 LM/Gauss-Newton OEM 求解器、状态向量打包/解包、背景与观测误差协方差构建、
有限差分雅可比矩阵、averaging kernel、DOFS 和后验误差协方差等核心诊断量。
WSL Ubuntu-24.04 中已完成 MonoRTM v5.6 Linux 编译，TAPE3_bin 已生成，并通过 ForwardModel(backend="monortm") 接入 OEM。
ERA5 2013-01 自洽实验表明，MonoRTM OEM 能同时改善 T、RH 和 BT 残差；
非自洽实验则确认了前向模型不一致会导致温度反演退化。
后续将扩大 MonoRTM OEM 样本量，构造基于 BRNN v4 残差的背景误差协方差，
并探索 EOF/PCA 降维和 LWC 云天 OEM。
根据深度调研，项目中期将重点推进 NN 作为 OEM 先验、NN surrogate forward model、
不确定度校准和 DeepOE/unrolled OEM 四类路线，其中前两类为当前最高优先级。
```

---

## 17. 最终预期成果

工程成果：

```text
一个可运行的 OEM / 1D-Var 反演框架
```

实验成果：

```text
BRNN prior vs OEM posterior 对比
```

科学成果：

```text
MWR 反演中观测信息量、先验约束、云污染影响的定量分析
```

项目升级方向：

```text
从 BRNN 统计反演复现项目
升级为 BRNN + OEM 混合物理反演系统
```

---

## 18. 2026-07-21 论文方法融合后的计划更新

结合《论文方法与当前项目融合深度调研报告》，下一阶段的主线调整为“先验证机制，再做系统融合”：

```text
MonoRTM 物理一致性基线
-> 虚拟多仰角 + 角度/通道筛选
-> S_a/S_e 与状态向量升级
-> BRNN 先验桥接
-> NN surrogate 加速
-> 云天 T/RH/LWC 联合 OEM
-> 不确定度校准
```

论文方法中最适合当前项目直接吸收的部分是多仰角观测组织、几何通道筛选、联合云液水状态和协方差一致性控制；220--600 GHz 旋转硬件属于后续验证过算法增益后的长期方向。由于 BRNN v4 的 MP-3000A 通道体系与 OEM 当前 HATPRO 14 通道链路不一致，且 MP-3000A 原始 `Obs_BT` 仍未确认，BRNN v4 暂不能直接作为 14 通道 OEM 的 `x_a`。

详细的阶段目标、代码产物、验收指标和决策闸门见：

```text
docs/论文方法融合下一阶段计划_2026-07-21.md
```
