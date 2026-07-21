# OEM 云天液态水约束扩展设计

> 创建日期：2026-07-10
> 状态：Day 10-11 设计阶段
> 依赖：OEM 基础框架（`src/oem.py`, `src/oem_state.py`, `src/oem_covariance.py`, `src/oem_observation.py`, `src/forward_model.py`）已完成

---

## 1. 问题陈述

### 1.1 核心矛盾

K 波段（22-31 GHz）对云液态水高度敏感。当廓线含云时，液态水发射微波辐射导致亮温显著升高：

| 通道 | 频率 | 云天 OMB 可达 | 正常 σ |
|------|------|-------------|--------|
| ch0 | 22.2 GHz | +50~104 K | ~1.5 K |
| ch7 | 31.4 GHz | +13.7 K (p99) | ~2.0 K |

来源：`reports/进度记录/未订正缺陷.md` 缺陷 1。

现有 QC（`qc_correction.py`）通过 ICLWC 阈值（>1250 g/m² 删除，>750 缩放）处理云天样本，但：
- QC/Rain flag 无法标记全部云污染（缺陷 1）
- 中等 OMB（10-50 K）样本仍混入训练
- K 波段 R² 仅 0.49-0.87，远低于 V 波段的 0.92-0.98

### 1.2 OEM 框架的机会

与 BRNN 的"被动感知 + 质控剔除"不同，OEM 可以通过以下机制主动处理云天：

1. **状态向量扩展**：将 LWC 加入控制变量 `x = [T, RH, LWC]`，让 OEM 同时反演云水
2. **自适应误差膨胀**：对云污染 K 波段动态膨胀 S_e，降低其在代价函数中的权重
3. **物理约束**：LWC ≥ 0 的边界约束自然排除非物理解

---

## 2. LWC 控制变量设计

### 2.1 控制层选取

沿用 T/RH 的粗分层策略（Plan §4.1），LWC 使用相同 7 层：

```text
控制高度 [m]: 500, 1000, 2000, 3000, 5000, 8000, 10000
```

状态维度：

```text
T-only:       7 维
T+RH:        14 维
T+RH+LWC:    21 维  ← 本设计目标
```

### 2.2 LWC 先验值 `x_a(LWC)`

云天 OEM 反演需要一个 LWC 先验。候选来源优先级：

| 来源 | 可用性 | 精度 | 推荐 |
|------|--------|------|------|
| ERA5 CLWC 廓线 | ✅ 已有（`era5_profiles_201301_poc.pkl`） | 模式预报量，有系统偏差 | **POC 首选** |
| BRNN v4 + 经验 LWC 估计 | 需新增 | 基于 RH/温度的经验关系 | 过渡方案 |
| 全零先验 + 大 S_a | 无需数据 | 最弱约束，易发散 | 仅用于测试 |
| 探空 ICLWC → 垂直分配 | 代码已有（`sounding_process.py`） | 粗糙 | 参考 |

**推荐**：POC 阶段使用 ERA5 CLWC（与 T/RH 同源），真实 MP-3000A 阶段使用全零先验 + 云天误差膨胀策略（§3）。

### 2.3 LWC 物理边界

```python
# OEMStatePacker 中的 LWC 边界
LWC_MIN = 0.0       # g/m³，不可为负
LWC_MAX = 3.0       # g/m³，对流云中极少超过此值

# 分层上限（考虑典型云类型）
# 0-2 km:  2.0 g/m³  （边界层积云）
# 2-8 km:  1.0 g/m³  （中层云）
# 8-10 km: 0.2 g/m³  （高层卷云，LWC 极低）
```

### 2.4 实现改动

`oem_state.py` 已支持 `coarse_LWC` 参数。只需在构造时传入：

```python
packer_cloudy = OEMStatePacker(
    mode="coarse",
    coarse_T=[500, 1000, 2000, 3000, 5000, 8000, 10000],
    coarse_RH=[500, 1000, 2000, 3000, 5000, 8000, 10000],
    coarse_LWC=[500, 1000, 2000, 3000, 5000, 8000, 10000],
)
# n_state = 21
```

`unpack()` 中对 LWC 部分的 clip：`CLWC = np.clip(CLWC_full, 0.0, 3.0)`。

---

## 3. K 波段云天误差膨胀策略

### 3.1 策略总览

```text
                     ┌─────────────────┐
    profile ──────→  │ compute ICLWC   │
                     └────────┬────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
         ICLWC < 200     200-750         > 750 g/m²
         (clear-sky)   (thin cloud)    (thick cloud)
              │               │               │
              ▼               ▼               ▼
        S_e 不变       K-band ×4       K-band ×9
                                   + V-band ×2 (远端通道)
```

### 3.2 膨胀因子函数

```python
def cloud_inflation_factor(iclwc):
    """Return multiplicative inflation factor for K-band S_e.

    Based on the physics: liquid water emission ∝ ICLWC, but the
    forward model error grows faster due to nonlinear effects
    (MonoRTM vs simple RTM gap, vertical distribution uncertainty).

    Three regimes:
      ICLWC < 200:   factor = 1.0    (clear-sky, no inflation)
      200-750:      factor = linear ramp from 1 to 4
      > 750:         factor = 9.0    (thick cloud, heavy down-weight)
    """
    if iclwc < 200:
        return 1.0
    elif iclwc < 750:
        return 1.0 + (iclwc - 200) / (750 - 200) * 3.0  # 1 → 4
    else:
        return 9.0
```

### 3.3 通道差异化膨胀

并非所有 K 波段通道受云影响程度相同：

| 通道 | 频率 GHz | LWC 敏感性 | 膨胀乘数 |
|------|----------|-----------|---------|
| K1 (ch0) | 22.24 | 中（H₂O 线主导） | ×1.5 |
| K2-K6 | 23.04-27.84 | 中 | ×1.0（基准） |
| K7 (ch6) | **31.40** | **最高** | ×2.0 |
| V1-V3 | 51.26-53.86 | 低（远翼区） | ×1.0（clear） |
| V4-V7 | 54.94-58.00 | 极低（近线心） | ×1.0 |

已有函数 `inflate_se_for_cloud()`（`oem_covariance.py`）支持基础膨胀。需扩展为**逐通道差异化膨胀**。

### 3.4 自适应策略（迭代中更新）

在第 1 次 OEM 迭代后，根据当前反演的 LWC 廓线重新计算 ICLWC，动态调整 S_e：

```python
# 伪代码
for iteration in range(max_iter):
    if iteration == 0:
        # 使用背景场的 CLWC
        iclwc = compute_iclwc(x_a_lwc)
    else:
        # 使用当前反演的 CLWC
        iclwc = compute_iclwc(x_current_lwc)

    S_e_adapted = adapt_se_for_cloud(S_e_base, iclwc)
    # ... 继续 LM 迭代
```

这个自适应循环在 `OEMSolver.retrieve()` 内部实现，无需修改外部接口。

---

## 4. Cloud Flag / CLWC 筛选策略

### 4.1 三级筛选体系

```text
Level 0 — 不筛选（全样本 OEM）
  用于合成测试和算法验证

Level 1 — 软筛选（自适应 S_e 膨胀）★ 推荐
  所有样本参与反演，云天样本通过 S_e 膨胀降低权重
  优势：无样本损失，OEM 自动平衡观测/先验

Level 2 — 硬筛选（ICLWC 阈值剔除）
  预处理阶段：ICLWC > 1250 g/m² 的廓线不参与反演
  与现有 QC 一致（config.ICLWC_DELETE_THRESHOLD）
  优势：清除最极端污染，保护 OEM 不发散

Level 3 — 双阶段（先筛选再膨胀）
  Level 2 硬筛选 → Level 1 自适应膨胀
  最保守也最安全
```

### 4.2 前向模型一致性约束

**关键洞察**：OEM 的云天性能上限取决于前向模型 H(x) 在云天条件下的精度。

| 前向模型 | 云天 LWC 支持 | OEM 云天可行性 |
|----------|-------------|--------------|
| Simple RTM | Rayleigh 近似（`_cloud_absorption`） | 仅算法验证 |
| MonoRTM case 5 | 完整云天模式 | ✅ 物理验证 |
| PAMTRA | 全散射 + 多种水凝物 | ✅ 论文级 |

**当前阻塞**：MonoRTM 在 Windows 下不可用（Plan §15 Risk 5）。Simple RTM 的 Rayleigh 近似在 ICLWC > 500 g/m² 时偏差 > 5K。

### 4.3 云天 vs 晴空分阶段实验

```text
阶段 A（当前）：
  仅 clear-sky（ICLWC < 200 g/m²）
  S_e 不膨胀，验证基础 OEM 精度

阶段 B（本设计）：
  clear + thin cloud（ICLWC < 750 g/m²）
  K 波段 ×4 膨胀，验证自适应策略

阶段 C（MonoRTM 就绪后）：
  全样本（含 ICLWC > 750）
  LWC 状态量反演 + MonoRTM 云天前向
```

---

## 5. T-RH-LWC 联合背景误差协方差

### 5.1 结构

当前 S_a 假设 T 和 RH 不相关（分块对角）。加入 LWC 后需扩展为 3×3 块结构：

```text
        ┌─────────────────────────────────┐
        │  S_TT (7×7)  │   0    │   0    │
S_a =   │──────────────┼────────┼────────│
        │      0       │ S_RHRH │   0    │
        │──────────────┼────────┼────────│
        │      0       │   0    │ S_LWCLWC│
        └─────────────────────────────────┘
```

第一版保持块对角（跨变量协方差为零），后续从 ERA5 统计 T-RH-LWC 联合协方差。

### 5.2 LWC 先验标准差

```text
σ_LWC = 0.3 g/m³  （均匀分布假设：σ ≈ (max-min)/√12 ≈ 3/3.46 ≈ 0.87）
```

保守初值：`σ_LWC = 0.5 g/m³`，后续从 ERA5 CLWC 变率统计标定。

### 5.3 LWC 垂直相关长度

云水垂直相关长度小于温度和湿度：

```text
L_LWC = 0.3-0.5 km
```

因为液态水集中在特定高度层次（云底到云顶），不似 T/RH 的大尺度相关。

---

## 6. 代码改动清单

### 6.1 需修改的现有文件

| 文件 | 改动 | 行数 |
|------|------|------|
| `src/oem_state.py` | 无改动（已支持 `coarse_LWC`） | 0 |
| `src/oem_covariance.py` | `inflate_se_for_cloud()` → 逐通道差异化膨胀 | ~15 |
| `src/oem.py` | `retrieve()` 内增加自适应 S_e 更新循环 | ~20 |
| `src/oem_observation.py` | 新增 `compute_iclwc_from_profile()` 和 `cloud_inflation_factor()` | ~25 |

### 6.2 需新增的文件

| 文件 | 职责 | 行数 |
|------|------|------|
| `scripts/run_oem_cloudy.py` | 云天 OEM 实验脚本 | ~200 |

### 6.3 总改动量

~260 行，低风险。核心求解器逻辑不变，仅增加 S_e 自适应和 LWC 状态量支持。

---

## 7. 实验设计

### 实验 E：云天 OEM（clear + thin cloud）

```text
数据：ERA5 2013-01 POC profiles（含 CLWC）
样本筛选：ICLWC < 750 g/m²（阶段 B）
控制变量：T+RH+LWC（21 维）
背景场：ERA5 profile + 扰动
S_e 策略：cloud_inflation_factor(ICLWC)
前向模型：simple RTM（Rayleigh 近似）
对比基线：T+RH only（14 维）OEM 在相同样本上
```

成功标准：

```text
1. 云天 K 波段 BT residual < 云天 prior BT residual
2. LWC 反演值与 ERA5 CLWC 的相关系数 > 0.5
3. 加入 LWC 后 T/RH RMSE 不劣于不加 LWC 的 OEM
4. LWC 廓线无负值（边界约束生效）
```

---

## 8. 实施路线

```text
第 1 步：更新 oem_covariance.py — 逐通道差异化膨胀 + 云天因子函数
第 2 步：更新 oem.py — 迭代中自适应 S_e
第 3 步：创建 run_oem_cloudy.py — 云天 POC 脚本
第 4 步：在 ICLWC < 750 子集上运行实验 E
第 5 步：对比 14 维 vs 21 维 OEM 结果
```

预估工作量：1-2 天。

---

## 9. 风险与缓解

| 风险 | 概率 | 缓解 |
|------|------|------|
| Simple RTM 云天精度不足导致 OEM 发散 | 高 | 限制 ICLWC < 500，增大 S_e 膨胀 |
| LWC 21 维欠定（14 通道约束 21 变量） | 中 | 增大 S_a(LWC) 的对角项，依赖先验约束 |
| LWC 与 RH 的 trade-off（两者都吸 K 波段） | 中 | 检查 averaging kernel 的交叉项 |
| MonoRTM 不可用导致无法物理验证 | 高 | 先在 simple RTM 上闭环，MonoRTM 就绪后再验证 |
