# STATE — MWR 廓线反演系统

## Current

**Phase 5→6 过渡** — P0 闭环验证已完成，OEM baseline 三组实验跑通

### Active Work
- BRNN v4 与 OEM MonoRTM n=100 分层精度诊断完成
- RH S_a/S_e 参数敏感性扫描脚本就绪 (`scripts/scan_rh_sensitivity.py`)

### Last Completed (2026-07-22)
- P0-1: MonoRTM OEM baseline n=20/n=100（收敛率 99%, DOFS 2.21）
- P0-2: 通道审计报告（HATPRO 14ch / MP-3000A 22ch / MonoRTM 映射）
- P0-3: ForwardModel/MonoRTM frequencies 扩展 + oem_geometry.py
- README + 断点更新
- 新增数据：成都 ERA5 + 温江探空（待训练/测试）

### Key Findings
- OEM 改善集中在 0-2 km 近地层，5-10 km 几乎无改善（DOFS 仅 2.2/14）
- BRNN v4 在全高度均优于 OEM（T=1.26K vs 2.02K, RH=7.76% vs 6.20%）
- MonoRTM BT prior 5.03K → posterior 0.61K，BT 残差改善 87.9%

### Blocked
- BRNN+OEM 桥接受阻：缺 MP-3000A Obs_BT
- 仍在推进：ERA5/MonoRTM self-consistent OEM 路线

## Quick Resume

```bash
cd /mnt/d/project-504/mwr-retrieval-main && source .venv-wsl/bin/activate
python -c "import torch, config; print(f'Torch {torch.__version__}, {config.N_LAYERS} layers')"
```

### Run OEM baseline (simple RTM, 2 min)
```bash
python scripts/run_oem_baseline.py --n-samples 100 --forward simple --seed 42
```

### Run OEM baseline (MonoRTM, 8 min)
```bash
python scripts/run_oem_baseline.py --n-samples 100 --forward monortm --seed 42
```

### Run RH sensitivity scan
```bash
python scripts/scan_rh_sensitivity.py --n-samples 30
```

## Key Paths

| 资源 | 路径 |
|------|------|
| v4 模型 | models_mp3000a_v4/ |
| MonoRTM | bin/monortm_linux |
| TAPE3 | data/TAPE3/TAPE3_bin |
| OEM baseline (new) | results/oem_baseline_*_n100_seed42/ |
| OEM baseline (old) | results/oem_201301_self_consistent_monortm_n100/ |
| v4-derived S_a | results/oem_covariance/sa_v4.pkl |
| 通道审计 | reports/通道与观测数据审计_2026-07-21.md |
| 复现指南 | ~/Desktop/MWR复现指南_2026-07-21.md |

---
*最后更新: 2026-07-22*
