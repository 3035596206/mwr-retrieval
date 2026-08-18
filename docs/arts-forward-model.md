# ARTS 前向模型接入规范

## 当前策略

项目主前向模型已切换为 ARTS。`ForwardModel()` 默认等价于：

```python
ForwardModel(backend="arts", frequencies=config.DEFAULT_FORWARD_CHANNELS)
```

默认通道集为成都 21 通道。MonoRTM 保留为历史基线后端，simple RTM 只用于快速算法闭环和单元测试。

## 运行方式

推荐通过研究组已有 ARTS 脚本接入：

```powershell
$env:ARTS_FORWARD_MODEL_COMMAND = 'wsl -d Ubuntu-24.04 -- /home/inkp/miniconda3/envs/arts/bin/python /mnt/d/project-504/mwr-retrieval-main/scripts/run_arts_profile.py --server'
$env:ARTS_FORWARD_MODEL_PERSISTENT = '1'
python scripts/run_oem_baseline.py --forward arts --n-samples 100
```

本机已经验证 `Ubuntu-24.04` 的 conda 环境 `/home/inkp/miniconda3/envs/arts` 可用，`pyarts` 版本为 `2.6.18`。项目默认 `config.DEFAULT_ARTS_COMMAND` 已指向 `scripts/run_arts_profile.py`，因此在这台机器上通常可以直接运行：

```powershell
python scripts/run_oem_baseline.py --forward arts --n-samples 100
```

注意：不要在 runner 命令中使用 `conda run -n arts python ...`，它在 Windows -> WSL 管道下可能不传递 stdin。应直接调用环境中的 Python：`/home/inkp/miniconda3/envs/arts/bin/python`。OEM 有大量有限差分前向调用，推荐使用 `--server` 持久进程模式；本机默认配置已经启用。

也可以在 Python 中传入 callable：

```python
from forward_model import ForwardModel

def run_arts(payload):
    ...
    return {"brightness_temperature_k": tb}

fm = ForwardModel(backend="arts", arts_runner=run_arts)
```

## Runner 输入

外部命令从 stdin 读取 JSON：

```json
{
  "profile": {
    "temperature_k": [290.0],
    "pressure_hpa": [950.0],
    "relative_humidity_percent": [70.0],
    "cloud_liquid_water_g_m3": [0.0],
    "height_m": [0.0]
  },
  "instrument": {
    "frequencies_ghz": [22.24, 23.04],
    "elevation_angle_deg": 90.0,
    "channel_response": null
  },
  "model": {
    "backend": "arts",
    "pyarts_available": true
  }
}
```

## Runner 输出

外部命令向 stdout 写 JSON：

```json
{
  "brightness_temperature_k": [238.4, 241.2]
}
```

也兼容字段名 `tb` 或 `y`。返回数组长度必须等于输入的 `instrument.frequencies_ghz`。

## 运行产物记录

每次正式 ARTS OEM 运行的 `manifest.json` 应记录：

- 输入 asset ID；
- `config/forward_model.json` 的配置版本；
- ARTS / pyarts 版本；
- spectroscopy / line catalog 版本；
- 通道 schema；
- 几何设置；
- 输出 asset ID；
- 代码版本和随机种子。
