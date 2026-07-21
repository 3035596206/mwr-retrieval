# BRNN v4 + OEM 混合路线阻塞拆解方案

> 日期：2026-07-13
> 依据：oem_retrieval_plan.md §14 第11-12天

---

## 1. 当前阻塞分析

### 1.1 BRNN v4 架构

| 模型 | 输入维度 | 输入特征 |
|------|:---:|------|
| T_0-2km | 11 | V-band surface BT (53-59 GHz, ~5ch) + t2m + 其他 |
| T_2-8km | 28 | 全部 22ch BT + surface features (t2m,d2m,sp,rh2m,IR,time) |
| T_8-10km | 28 | 同上 |
| RH_0-2km | 28 | 同上 |
| RH_2-8km | 28 | 同上 |
| RH_8-10km | 28 | 同上 |

BRNN v4 训练数据来源：MP-3000A 22通道 + IR + 地面气象要素

### 1.2 OEM 当前管线

| 组件 | 配置 |
|------|------|
| 前向模型 | MonoRTM v5.6 Linux |
| 通道 | 14 通道 HATPRO (K-band 7ch + V-band 7ch) |
| 先验 x_a | ERA5 profile + 扰动（当前）；目标：BRNN v4 output |
| 观测 y | self-consistent MonoRTM BT（当前）；目标：MP-3000A Obs_BT |

### 1.3 阻塞项清单

| # | 阻塞 | 严重度 | 详情 |
|---|------|:---:|------|
| B1 | BRNN v4 是 22ch MP-3000A 模型，OEM 是 14ch HATPRO | 🔴 高 | 通道数、中心频率均不同，无法直接用作 x_a |
| B2 | 缺 MP-3000A 原始 Obs_BT | 🔴 高 | mp3000a_v4_results.pkl 只有 T/RH 预测，无 BT 和 surface 特征 |
| B3 | 缺 MP-3000A 通道频率映射 | 🟡 中 | 需从原始 .nc 获取 22ch 频率表，方可做 MonoRTM 模拟 |
| B4 | MonoRTM 仅支持 HATPRO 14ch | 🟡 中 | monortm_wrapper 硬编码 14 通道，需扩展 |

---

## 2. 解阻塞方案

### 方案 A：获取原始数据（推荐，但要等人给数据）



**优点**：不改模型，直接跑通 BRNN+OEM 全链路
**缺点**：依赖外部数据，无法自主推进

### 方案 B：14ch 重训 BRNN（自给自足，但工作量大）



**优点**：完全自主，打通 HATPRO 14ch + BRNN + OEM 全链路
**缺点**：需大量 MonoRTM 模拟（744×84=62,496 廓线，每廓线 ~0.3s batch ≈ 5 小时），需要 GPU 训练

### 方案 C：通道插值桥接（折中，快速验证）



**优点**：不依赖原始数据，可快速验证 BRNN+OEM 混合路线可行性
**缺点**：本质是 self-consistent 变体（BT 和 H(x) 都是 MonoRTM），无法反映真实 OMB

---

## 3. 推荐路线



---

## 4. 下一步行动清单

| # | 行动 | 优先级 |
|---|------|:---:|
| 1 | 从 train_mp3000a_v4.py L130 获取 MP-3000A 22ch 频率表 | P0 |
| 2 | 扩展 monortm_wrapper 支持自定义频率（目前硬编码 config.ALL_CHANNELS） | P0 |
| 3 | 用 22ch MonoRTM + BRNN v4 跑 self-consistent OEM POC | P1 |
| 4 | 联系师兄获取 MP-3000A 原始 .nc（或告知路径） | P1 |
| 5 | 评估方案 B 的 MonoRTM 批量模拟时间成本 | P2 |