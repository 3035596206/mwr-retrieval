# MWR大气温湿廓线反演 — 复现项目

复现论文：朱柳桦《基于地基微波辐射计的多方法反演大气温湿廓线研究》（2023，南京信息工程大学）第3章

**上次更新：2026-05-27**

---

## 当前进度总览

```
代码框架  ████████████████████ 100%  全部模块完成并通过测试
编译依赖  ████████████████████ 100%  gfortran + MonoRTM + Python包
真实数据  █████████████████░░░░  75%  单层84/84完成，气压层仅2013-01
  ├─ ERA5  ███████████████░░░░░  75%  单层84/84月，气压层0/84(仅2013-01参考)
  ├─ TAPE3  ░░░░░░░░░░░░░░░░░░░░   0%  需GitHub/Zenodo外网
  └─ MWR   ░░░░░░░░░░░░░░░░░░░░   0%  需联系论文作者
```

## 技术路线

```
ERA5再分析资料 → 质控/偏差订正 → MonoRTM/Python模拟亮温 → BRNN神经网络训练 → 温湿廓线反演 → 探空验证
```

## 项目结构

```
mwr_retrieval/
├── bin/monortm                  MonoRTM v5.6 可执行文件 (5.1 MB, arm64)
├── setup_monortm.sh             TAPE3光谱数据一键安装脚本（待网络恢复后运行）
├── download_era5.py             ERA5 CDS备选下载脚本（带代理，按月下载，CDS已失效）
├── dl_one_month.py              ARCO单月下载（独立进程，供bash循环调用）
├── dl_sync.py                  ARCO批量下载（subprocess隔离，5min超时）
├── bulk_fast.py                ARCO批量下载（30s间隔，xarray会话级隔离）
├── bulk_v3.py                  ARCO批量下载（显式清理连接池）
├── bulk_sl.py                  ARCO批量下载（单会话版，有连接复用问题）
├── download_all.py             CDS+ARCO混合下载（CDS已失效）
├── run.py                       主流水线 (download → preprocess → qc → train → evaluate)
├── config.py                    全局配置 (14通道/93层网格/QC阈值/超参数/站点坐标)
├── requirements.txt             Python依赖
├── README.md                    项目说明（本文件）
├── CHANGES.md                   修复与变更记录
├── .venv/                       Python 3.14 虚拟环境
├── docs/
│   └── lwc_extension_roadmap.md  液态水建模拓展方案（远期路线图）
├── data/
│   └── era5/
│       ├── sl_2013_01.nc        单层数据 (CDS, 529KB)
│       ├── sl_2013_01_arco.nc   单层数据 (ARCO, 1.0MB)  
│       └── pl_2013_01_arco.nc   气压层数据 (ARCO, 12MB)
└── src/
    ├── brightness_temp.py       亮温模拟 (simple/MonoRTM双后端)
    ├── monortm_wrapper.py       MonoRTM Python封装
    ├── brnn_model.py            BRNN网络定义 (PyTorch, 6个子模型)
    ├── train.py                 训练脚本
    ├── qc_correction.py         ERA5质控 + 偏差订正
    ├── era5_preprocess.py       ERA5下载/预处理
    ├── sounding_process.py      探空数据处理/LWC估算
    └── evaluate.py              评估/可视化
```

## 已完成事项

### 开发环境
- [x] Python 3.13 + PyTorch 2.11 + 全部科学计算包 (numpy, scipy, xarray, netCDF4 等)
- [x] GNU Fortran 15.2.0 (Homebrew, Apple Silicon)
- [x] CDS API 已安装并配置 `~/.cdsapirc`（Key: `e088c69b-...`）

### 代码模块（全部完成并通过测试）
- [x] `config.py` — RPG HATPRO 14通道、93层高度网格、QC阈值、BRNN超参数、北京7站坐标
- [x] `brightness_temp.py` — Python 简化辐射传输模型，14通道亮温模拟，物理输出验证通过（V波段192-287K, K波段13-25K）
- [x] `monortm_wrapper.py` — MonoRTM v5.6 完整 Python 封装（TAPE5格式输入生成、可执行文件调用、输出解析）
- [x] `qc_correction.py` — ERA5质控4步流程（湿度缩放0.9、四季逐层订正、LWC筛选、BT线性订正）
- [x] `brnn_model.py` — BRNN网络 + 6子模型管理（3高度区间×2变量），修正了 T_0-2km 输入维度bug（6维，非7维）
- [x] `train.py` — Adam优化 + MSE损失 + Early Stopping + 自动特征选择
- [x] `evaluate.py` — 逐层RMSE/Bias/STD + 散点密度图 + BT箱线图 + CSV导出
- [x] `era5_preprocess.py` — CDS下载 + 2mRH计算（Magnus公式）+ 93层插值
- [x] `sounding_process.py` — Wyoming探空下载 + LWC估算 + 93层插值
- [x] `run.py` — 5阶段流水线主控脚本

### MonoRTM 编译
- [x] 从 GitHub AER-RC/monoRTM clone 源码 v5.6
- [x] 16个 Fortran 源文件编译通过，5.1MB 可执行文件
- [x] 编译标志：`-fallow-argument-mismatch -std=legacy -fno-range-check -w -O2`
- [x] 验证可执行文件能正确读取 MONORTM.IN 输入

### 端到端测试
- [x] 500样本合成数据全流程：BT模拟 → QC → 6模型训练 → 反演 → 评估
- [x] RH 整层 RMSE ~11.4%（论文12-13%，合成数据下合理）
- [x] 所有模块协同工作正常

## 待完成事项

### 阶段一：ERA5 数据下载 ✅ 单层完成 | ⏳ 气压层待下载

**单层 84/84 月完成（83 MB）：**
| 年份 | 状态 | 来源 |
|------|------|------|
| 2013-2017 | 60/60 | ARCO GCS（周级策略） |
| 2018-2019 | 24/24 | ARCO GCS（周级+dask单线程） |

**气压层 0/84（仅参考文件）：**
| 文件 | 大小 | 内容 | 来源 |
|------|------|------|------|
| `pl_2013_01_arco.nc` | 12 MB | 气压层 2013-01 (T, Z, q) | ARCO GCS |

**CDS API 状态：新 Key 可用 ✅**

| Key | 状态 | 备注 |
|-----|------|------|
| `e088c69b-...` (旧) | 403 | 2026-05-22 成功1次后失效 |
| `a2179a74-...` (新) | 403 | 已失效 |
| `8dfcb2f7-...` (当前) | 可用 | 2026-05-26 测试通过，已接受许可证 |

**气压层下载策略（CDS 3天窗口）：**
- 3天×24h×37层×3变量 = 7992字段 → 通过成本限制
- 7天×24h×37层×3变量 = 18648字段 → 超限
- 每次 ~1-2分钟，每月 ~10窗口，84月 ~28小时

**已废弃的 ARCO 方案：**
- 周级 xarray/zarr 对单层有效（每批次 504 chunk 不触发限流）
- 对气压层（37层）数据量 37x，dask 单线程可下载但 ~7h/月，太慢
- aiohttp 内部并发会触发 GCS 隐式限流

### 其他数据

| 数据 | 状态 | 获取方式 |
|------|------|---------|
| TAPE3 (~135MB) | 待下载 | 运行 `bash setup_monortm.sh`（需GitHub/Zenodo） |
| MWR一级亮温 | 未获取 | 联系朱柳桦/鲍艳松教授（南京信息工程大学） |
| 北京南郊探空 | 代码就绪 | `python run.py --stage download`（Wyoming大学，公开） |

## 下次继续的步骤

> **2026-06-08 断点：** 6 轮迭代完成。v4 当前最佳 (T=1.26K, RH=7.8%)，v6 将 Sim→Obs gap 压缩 68%。
> **推荐模型：** `models_mp3000a_v4/brnn_*.pt`
> **GitHub：** https://github.com/3035596206/mwr-retrieval (tags: v2.0, v4.0, v6.0)
> **详细断点文档：** `/Users/ink/test/工作进度6.8.md`

### 数据现状速查

| 文件类型 | 命名格式 | 数量 | 变量 | 来源 |
|---------|---------|------|------|------|
| 单层 | `sl_YYYY_MM.nc` | 84 月 | `t2m`, `d2m`, `sp` | CDS/ARCO |
| MP-3000A | `54623_MP_3000A_...nc` | 3,453 廓线 | T, q, 22ch Obs_BT | 本地 |
| BRNN(v4) ★ | `models_mp3000a_v4/brnn_*.pt` | 6个 | T=1.26K, RH=7.8% | train_mp3000a_v4.py |
| BRNN(v6) | `models_mp3000a_v6/brnn_*.pt` | 6个 | T=1.92K, Sim→Obs gap 0.61K | train_mp3000a_v6.py |
| TAPE3 | — | 0 | 光谱数据 | 未下载 |

### 第0步：激活环境

```bash
cd /Users/ink/test/mwr_retrieval
source .venv/bin/activate
python3 -c "import torch, config; print(f'Torch {torch.__version__}, {config.N_LAYERS} layers ✓')"
```

### 第1步：复现 v4 训练（推荐）

```bash
cd /Users/ink/test/mwr_retrieval && source .venv/bin/activate
python train_mp3000a_v4.py
# 预期: T_RMSE=1.26K, RH_RMSE=7.8%
```

### 第2步：下载 TAPE3 + MonoRTM 批量模拟

```bash
# 下载 TAPE3 (135MB)
curl -L -o data/TAPE3 \
  "https://raw.githubusercontent.com/AER-RC/monoRTM/main/run/TAPE3_spectral_lines.dat.0_55.v5.0_fast"

# MonoRTM 批量模拟 (待编写)
# python train_monortm_batch.py
```

### 第3步：MonoRTM 预训练 + Obs_BT 微调 (v7)

```python
# 路线: ~300K MonoRTM 样本预训练 → 15K Obs_BT 微调
# 目标: T_RMSE < 1.1K
```

### 训练脚本速查

| 脚本 | 用途 | 结果 |
|------|------|------|
| `train_mp3000a_v4.py` ★ | 当前最佳训练脚本 | T=1.26K, RH=7.8% |
| `train_mp3000a_v6.py` | Sim_BT 两阶段训练 | T=1.92K, Sim→Obs gap 0.61K |
| `train_mp3000a_v2.py` | 首版超论文的训练脚本 | T=1.45K, RH=9.0% |
| `poc_pipeline.py` | POC 管线（模板扩展） | 已跑通 |
| `dl_pl_cds.py` | CDS 气压层批量下载 | 3天窗口，待 CDS 恢复 |

### 已知代码修复
