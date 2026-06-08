# 模型目录说明

## 目录结构约定

每个训练版本独立目录，永不覆盖：

```
models_<数据源>_<版本>/
├── brnn_T_0-2km.pt
├── brnn_T_2-8km.pt
├── brnn_T_8-10km.pt
├── brnn_RH_0-2km.pt
├── brnn_RH_2-8km.pt
├── brnn_RH_8-10km.pt
└── README.txt           # 该版本的简要说明
```

## 当前已有版本

| 目录 | 数据 | 说明 | T_RMSE | RH_RMSE |
|------|------|------|--------|---------|
| `models_mp3000a_v2/` | MP-3000A 实测 BT | **推荐使用** | 1.45 K | 9.0% |
| `models_mp3000a_v3/` | MP-3000A Sim_BT | 实验性 | 0.72 K | 5.08% |
| `models_mp3000a/` | MP-3000A 实测 BT | v1 基线 | 3.03 K | 16.6% |
| `models/` | ERA5 POC 模拟 | 概念验证 | — | — |

## 新增模型时的步骤

```bash
# 1. 新建带注释的版本目录
mkdir -p models_mp3000a_v4

# 2. 训练脚本中设置新目录
# models_dir = os.path.join(PROJ, "models_mp3000a_v4")

# 3. 自动生成说明
cat > models_mp3000a_v4/README.txt << 'EOF'
MP-3000A data v4
日期: 2026-06-XX
改进: (与v2的差异)
T_RMSE: X.XX K, RH_RMSE: X.XX%
输入: 22ch Obs_BT + QC订正 + 时序特征
训练样本: XXXXX (QC=0,Rain=0,OMB过滤后)
分割方式: 按Profile_Index分组, 70/15/15
EOF

# 4. 提交
git add models_mp3000a_v4/
git commit -m "v4模型: (简述改进)"
git push
```

## 不要删除的

- `models_mp3000a_v2/` — 当前最佳版本
- `models_mp3000a/` — v1 基线（对比参考）
- `results/mp3000a_v2_test_results.pkl` — 测试结果（复现评估需要）
