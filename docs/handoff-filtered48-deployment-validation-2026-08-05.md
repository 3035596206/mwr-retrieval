# 交付说明：成都筛选亮温 48 层反演全量验证

本文档用于把当前稳定版代码交付给学长或另一侧 AI agent，使其能够从 Git 拉取代码与模型结果、对齐本地数据湖结构、复现实验并验证输出。

## 1. 当前稳定版本

- 仓库：`git@github.com:3035596206/mwr-retrieval.git`
- 分支：`master`
- 稳定提交：`a2c064e Add full filtered sounding retrieval run`
- 核心任务：使用筛选后的成都 21 通道微波辐射计亮温，与温江探空匹配，进行 48 层温湿廓线反演。
- 当前稳定方案：`Ridge/EOF` 反演温度，`BRNN top-2 ensemble` 反演相对湿度，RH 使用趋势约束和平滑约束，并使用验证集逐高度偏差订正。
- 已验证全量样本：61 条亮温-探空匹配样本，48 个垂直层。

注意：当前交付版本是 direct-RH 稳定版，不使用后续试验中的 `log(q)` 目标或高空加权损失。那些试验在当前小样本数据上没有超过稳定版。

## 2. 从 Git 拉取代码和模型

如果是新机器：

```powershell
cd D:\project-504
git clone git@github.com:3035596206/mwr-retrieval.git mwr-retrieval-main
cd D:\project-504\mwr-retrieval-main
git checkout master
git pull origin master
git checkout a2c064e
```

如果已经有仓库：

```powershell
cd D:\project-504\mwr-retrieval-main
git fetch origin
git checkout master
git pull origin master
git checkout a2c064e
```

模型和本次稳定验证结果已经随提交放在仓库内：

```text
results/filtered_48layer_20260805_rhtrend_calibrated/
```

主要模型文件在：

```text
results/filtered_48layer_20260805_rhtrend_calibrated/models/chengdu_filtered48_ridge_model.npz
results/filtered_48layer_20260805_rhtrend_calibrated/models/seed*/brnn_RH_*.pt
```

这些 `.npz` 和 `.pt` 文件可直接从 Git 拉取，不需要另行下载模型权重。

## 3. 数据湖目录对齐

推荐本地目录结构如下：

```text
D:\project-504\mwr-retrieval-main\              # 代码仓库
D:\project-504-data\                            # 数据湖根目录
  interim\mwr\chengdu\obs_bt_filtered_20260726\
    chengdu_obs_bt_filtered.json
  raw\radiosonde\wenjiang\station=56187\year=2026\
    *.txt
```

当前复现实验依赖两个输入：

```text
D:\project-504-data\interim\mwr\chengdu\obs_bt_filtered_20260726\chengdu_obs_bt_filtered.json
D:\project-504-data\raw\radiosonde\wenjiang\station=56187\year=2026
```

亮温 JSON 需要满足：

```json
{
  "records": [
    {
      "timestamp": "2026_05_15 22:00:00",
      "channels": [/* 21 个亮温值 */]
    }
  ]
}
```

要求：

- `timestamp` 格式为 `YYYY_MM_DD HH:MM:SS`。
- `channels` 必须是 21 个通道值。
- 探空文件名需要能从末尾解析发射时次，例如文件 stem 末尾包含 `2026051600`。
- 当前匹配窗口为 `±3 h`，运行时由 `--max-delta-hours` 控制。

如果另一台机器的数据湖不在 `D:\project-504-data`，可以不改代码，运行时把 `--obs-json` 和 `--sounding-dir` 指向实际路径即可。

## 4. 环境安装

已验证环境为 Windows + Conda Python。推荐使用 CPU 版 PyTorch 即可。

```powershell
conda create -n mwr-retrieval python=3.11 -y
conda activate mwr-retrieval
python -m pip install numpy matplotlib pillow torch --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
```

如果已经有 Python 环境，可以直接检查：

```powershell
@'
import numpy, matplotlib, PIL, torch
print('numpy', numpy.__version__)
print('torch', torch.__version__)
'@ | python -
```

本实验不需要 `eccodes`，因为当前脚本使用探空匹配数据，不读取 ERA5 GRIB。`scripts/train_chengdu_era5_ridge.py` 已做按需导入，避免无关 ERA5 依赖阻塞本流程。

## 5. 代码结构

核心脚本：

```text
scripts/pipeline_sounding_48layer.py       # 当前 61 条全量验证主流水线
scripts/train_chengdu_brnn.py              # BRNN 训练、分组拆分、预测工具
scripts/train_chengdu_era5_ridge.py        # Ridge/EOF 工具函数，当前用于温度模型
src/mwr_retrieval/grids.py                 # 48 层网格与层平均
src/mwr_retrieval/thermodynamics.py        # 湿度热力学转换工具
```

当前主流程做的事情：

1. 读取筛选亮温 JSON。
2. 在探空目录中寻找最近发射时次，默认 `±3 h` 匹配。
3. 将探空温度/RH/压力转换到物理 48 层。
4. 按探空发射组拆分训练、验证、测试，避免同一条探空廓线泄漏到不同集合。
5. 使用 Ridge/EOF 训练温度反演。
6. 使用 5 个随机种子的 RH-BRNN，按高度段选择 top-2 ensemble。
7. 使用验证集估计逐高度 RH 偏差订正。
8. 保存测试集预测、全量预测、模型权重、统计文件和 61 张全量剖面对比图。

## 6. 复现实验命令

在仓库根目录运行：

```powershell
cd D:\project-504\mwr-retrieval-main

python .\scripts\pipeline_sounding_48layer.py `
  --obs-json D:\project-504-data\interim\mwr\chengdu\obs_bt_filtered_20260726\chengdu_obs_bt_filtered.json `
  --sounding-dir D:\project-504-data\raw\radiosonde\wenjiang\station=56187\year=2026 `
  --output-root D:\project-504\mwr-retrieval-main\results\filtered_48layer_20260805_rhtrend_calibrated
```

如果使用本机已经验证过的 Conda Python，也可以明确指定：

```powershell
& 'C:\Users\Administrator\miniconda3\python.exe' .\scripts\pipeline_sounding_48layer.py `
  --obs-json D:\project-504-data\interim\mwr\chengdu\obs_bt_filtered_20260726\chengdu_obs_bt_filtered.json `
  --sounding-dir D:\project-504-data\raw\radiosonde\wenjiang\station=56187\year=2026 `
  --output-root D:\project-504\mwr-retrieval-main\results\filtered_48layer_20260805_rhtrend_calibrated
```

运行成功时，终端应看到类似结果：

```text
Matched: 61 samples / 17 sounding groups
Train: 42, Val: 8, Test: 11
T: eof, alpha=0.1, n_eof=3, val_rmse=1.404K

hybrid_sample:
  T: RMSE=1.599, Bias=+0.651, MAE=1.264
  RH: RMSE=22.249, Bias=-9.298, MAE=17.706

hybrid_all_sample:
  T: RMSE=1.075, Bias=+0.153, MAE=0.791
  RH: RMSE=20.227, Bias=+1.643, MAE=15.357
```

## 7. 输出文件说明

稳定输出目录：

```text
results/filtered_48layer_20260805_rhtrend_calibrated/
```

主要文件：

```text
chengdu_filtered48_hybrid_stats.json          # 指标汇总
brnn_ensemble_stats.json                      # 各 seed / 各高度段训练信息
predictions/chengdu_filtered48_hybrid_predictions.npz
predictions/chengdu_filtered48_hybrid_predictions_all.npz
models/chengdu_filtered48_ridge_model.npz
models/seed*/brnn_RH_*.pt
figures/profile_all_cases/case_*.png
figures/profile_all_cases/montage_cases_page01.png ... page06.png
结果_20260805_RH趋势订正.md
```

全量预测文件字段：

```python
import numpy as np

p = r"D:\project-504\mwr-retrieval-main\results\filtered_48layer_20260805_rhtrend_calibrated\predictions\chengdu_filtered48_hybrid_predictions_all.npz"
d = np.load(p)
print(d.files)

T_pred = d["T_pred"]          # (61, 48)
RH_pred = d["RH_pred"]        # (61, 48)，验证集偏差订正后结果
RH_raw = d["RH_pred_raw"]     # (61, 48)，订正前结果
T_true = d["T_true"]          # (61, 48)
RH_true = d["RH_true"]        # (61, 48)
heights = d["heights"]        # (48,)
timestamps = d["timestamps"]  # (61,)
groups = d["groups"]          # (61,)
test_mask = d["test_mask"]    # (61,)
```

验证全量预测结构：

```powershell
@'
import numpy as np
p = r'D:\project-504\mwr-retrieval-main\results\filtered_48layer_20260805_rhtrend_calibrated\predictions\chengdu_filtered48_hybrid_predictions_all.npz'
d = np.load(p)
print(d['T_pred'].shape)
print(d['RH_pred'].shape)
print(d['RH_brnn_all_seeds'].shape)
print(d['timestamps'][0], d['timestamps'][-1])
'@ | python -
```

预期：

```text
(61, 48)
(61, 48)
(5, 61, 48)
2026_05_15 22:00:00 2026_05_30 15:00:00
```

## 8. 图表查看

单样本剖面对比图：

```text
figures/profile_all_cases/case_001_2026_05_15_22_00_00.png
...
figures/profile_all_cases/case_061_2026_05_30_15_00_00.png
```

每张图包含：

- 左图：温度廓线，红色为预测，蓝色虚线为探空参考。
- 右图：相对湿度廓线，红色为预测，蓝色虚线为探空参考。
- 标题包含单样本 Bias 和 RMSE。

全量拼图：

```text
figures/profile_all_cases/montage_cases_page01.png
figures/profile_all_cases/montage_cases_page02.png
figures/profile_all_cases/montage_cases_page03.png
figures/profile_all_cases/montage_cases_page04.png
figures/profile_all_cases/montage_cases_page05.png
figures/profile_all_cases/montage_cases_page06.png
```

## 9. 当前结论边界

可以汇报：

- 温度反演稳定，测试集 T RMSE 约 `1.60 K`。
- RH 相比旧版有明显改进，测试集 RH RMSE 从约 `26.64 %` 降到 `22.25 %`。
- 61 条匹配样本均已完成全量反演，并生成逐样本图表。

不要过度表述：

- 当前样本量仍小，只有 61 条匹配样本、17 个探空组。
- 全量指标包含训练和验证样本，不能作为严格泛化精度。
- 4-8 km 中高层 RH 仍是主要短板，不能说湿度全层已经高精度可用。

## 10. 常见问题

### 10.1 缺少 torch

安装 CPU 版：

```powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 10.2 缺少 matplotlib 或 PIL

```powershell
python -m pip install matplotlib pillow
```

### 10.3 `eccodes` 报错

当前 `pipeline_sounding_48layer.py` 不需要 `eccodes`。如果仍然报错，确认运行的是 `pipeline_sounding_48layer.py`，不是 ERA5 GRIB 流程。

### 10.4 匹配样本不是 61 条

检查：

- 亮温 JSON 是否为 `obs_bt_filtered_20260726/chengdu_obs_bt_filtered.json`。
- 探空目录是否指向 `station=56187/year=2026`。
- 探空文件是否完整。
- `--max-delta-hours` 是否仍为默认 `3.0`。

### 10.5 只看到一页拼图

`montage_cases.png` 是第一页副本。完整拼图是 `montage_cases_page01.png` 到 `montage_cases_page06.png`。单样本图共有 61 张。

## 11. 给接手 AI agent 的最低验证清单

1. `git checkout a2c064e`。
2. 确认数据湖两个路径存在：亮温 JSON 和温江探空目录。
3. 安装 `numpy matplotlib pillow torch`。
4. 运行 `scripts/pipeline_sounding_48layer.py`。
5. 检查终端输出是否为 `Matched: 61 samples / 17 sounding groups`。
6. 检查 `chengdu_filtered48_hybrid_predictions_all.npz` 中 `T_pred/RH_pred` 是否为 `(61, 48)`。
7. 检查 `figures/profile_all_cases` 是否有 61 张 `case_*.png` 和 6 页 `montage_cases_page*.png`。
8. 对照 `chengdu_filtered48_hybrid_stats.json`，测试集 RH RMSE 应约为 `22.25 %`。
