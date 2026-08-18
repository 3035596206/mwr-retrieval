# 成都 ERA5 反演训练效果图

这些图由 `scripts/plot_chengdu_era5_results.py` 从已保存的训练日志和测试预测中生成。

1. `01_brnn_training_curves.png`：21通道BRNN六个分层子模型的训练/验证损失。
2. `02_height_resolved_errors.png`：混合模型与单次BRNN的分高度RMSE和Bias。
3. `03_prediction_scatter.png`：全部测试样本、全部高度层的反演值与ERA5标签对比。
4. `04_example_profiles.png`：较好、中位和较难测试样本的温湿廓线。
5. `05_time_height_error_heatmap.png`：测试时段内误差随时间和高度的演变。
6. `06_model_channel_comparison.png`：方法选择和BRNN通道消融对比。

当前最佳方案为温度 `21ch Ridge + 5 EOF`、湿度 `21ch BRNN四种子集成`。测试集只有48个样本、3天，图片用于展示当前基线，不代表正式业务精度。

重新生成：

```powershell
$env:PYTHONPATH='D:\project-504\pydeps'
python scripts/plot_chengdu_era5_results.py
```
