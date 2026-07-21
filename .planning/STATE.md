# STATE — MWR 廓线反演系统

## Current

**Phase 5: 数据补齐** — ERA5 气压层 CDS 下载进行中

### Active Work
- CDS 3 天窗口下载: 47/2556 天 (1.8%)
- 进程: PID 63613, 日志 /tmp/cds_batch.log
- 窗口: 3 天, 37 层, 3 变量
- Key: new key (...d0c5e7)

### Last Completed
- Phase 4: MonoRTM 修复 + 报告体系建立
- v4 可行性分析文档
- K 波段液态水污染三阶段解决历程文档

### Blocked
- Phase 6 (规模训练): 等待 Phase 5 数据补齐

## Quick Resume

```bash
cd /Users/ink/test/mwr_retrieval && source .venv/bin/activate
python3 -c "import torch, config; print(f'Torch {torch.__version__}, {config.N_LAYERS} layers')"
```

### Check CDS progress
```bash
tail -20 /tmp/cds_batch.log
ls data/era5/_daily/ | wc -l
```

### Restart CDS if needed
```bash
nohup python3 -u dl_pl_batch.py > /tmp/cds_batch.log 2>&1 &
```

### Run retrieval
```bash
python3 retrieve_and_plot.py
```

### Generate reports
```bash
python3 generate_report.py
python3 generate_practice_report.py
```

## Key Paths

| 资源 | 路径 |
|------|------|
| v4 模型 | models_mp3000a_v4/ |
| MonoRTM | bin/monortm |
| TAPE3 | data/TAPE3/TAPE3_bin |
| ERA5 单层 | data/era5/sl_*.nc (85 files) |
| ERA5 气压层 | data/era5/_daily/ (47 files) |
| 报告 | reports/ |
| 结果 | results/ |

---
*最后更新: 2026-06-17*
