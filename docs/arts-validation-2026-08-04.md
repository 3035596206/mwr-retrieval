# ARTS validation record, 2026-08-04

## Scope

This record verifies that the project can run the current ARTS forward-model
chain from Windows Python through the WSL pyarts runner, and that the OEM
baseline can complete a small self-consistent retrieval with ARTS.

Validated chain:

1. WSL `pyarts` runner self-test.
2. Project `ForwardModel()` default ARTS backend.
3. Persistent JSON-lines runner acceleration.
4. Small ARTS OEM baseline retrieval.

## Environment

- WSL distro: `Ubuntu-24.04`
- ARTS Python: `/home/inkp/miniconda3/envs/arts/bin/python`
- Runner: `scripts/run_arts_profile.py --server`
- Verified `pyarts` version: `2.6.18`
- Project default backend: `config.DEFAULT_FORWARD_BACKEND = "arts"`
- Project default channel set: Chengdu 21 channels

## Commands

Self-test:

```powershell
wsl -d Ubuntu-24.04 -- /home/inkp/miniconda3/envs/arts/bin/python /mnt/d/project-504/mwr-retrieval-main/scripts/run_arts_profile.py --self-test
```

Project-side forward-model smoke test:

```powershell
$env:PYTHONPATH = "D:\project-504\mwr-retrieval-main;D:\project-504\mwr-retrieval-main\src"
@'
import numpy as np
from forward_model import ForwardModel

height = np.linspace(0.0, 10000.0, 48)
profile = {
    "T": 288.15 - 6.5 * height / 1000.0,
    "P_hPa": 1013.25 * np.exp(-height / 8000.0),
    "RH": np.full_like(height, 60.0),
    "height": height,
    "CLWC": np.zeros_like(height),
}
fm = ForwardModel()
tb = fm.simulate(profile)
print(fm.backend_name, len(tb), np.isfinite(tb).all(), tb.min(), tb.max())
'@ | python -
```

OEM baseline smoke test:

```powershell
$env:PYTHONPATH = "D:\project-504\mwr-retrieval-main;D:\project-504\mwr-retrieval-main\src"
python scripts\run_oem_baseline.py --forward arts --n-samples 5 --seed 42
```

## Results

Self-test returned finite brightness temperatures and metadata:

- `backend = arts`
- `pyarts_version = 2.6.18`
- absorption species: `H2O-PWR98`, `O2-PWR98`, `N2-SelfContStandardType`
- output length for the built-in self-test: 8 channels

Project-side default `ForwardModel()` returned 21 finite brightness
temperatures. Persistent mode was active:

- first call: about `4.04 s`
- second call: about `0.003 s`

ARTS OEM baseline, `n=5`, `seed=42`:

- converged: `5 / 5` (`100%`)
- failed: `0`
- elapsed: `12 s`
- rate: `0.42 profiles/s`
- T RMSE: `2.6942 K -> 1.8144 K`
- RH RMSE: `8.53% -> 5.78%`
- BT RMS: `4.5372 K -> 0.5242 K`
- DOFS: `2.94 +/- 0.12`
- average iterations: `7.2`

Output directories:

- `results/oem_baseline_arts_n1_seed42/`
- `results/oem_baseline_arts_n5_seed42/`

## Real-data bridge validation

After the self-consistent smoke tests, the Chengdu real-data bridge was also
validated with the same ARTS runner. This run used observed 21-channel Chengdu
`Obs_BT` records matched to CDS ERA5 pressure-level profiles on the project
48-layer grid.

Bridge dataset:

- `results/chengdu_realdata_bridge/run_id=20260804T075728Z-d5b0fa9c/bridge_dataset.npz`
- bridge asset id: `4c163386-897c-43a6-8b98-eadbde0e6e41`
- samples: `163`
- shapes: `X=(163, 21)`, `T/RH/P=(163, 48)`

ARTS validation run:

- `results/chengdu_realdata_arts_validation/run_id=20260804T080459Z-22fec53d/`
- validation asset id: `4b7cd5d9-c930-4164-9326-7df629a35df4`
- catalog run id: `20260804T080459Z-22fec53d`
- lineage: bridge dataset `validated_by` ARTS validation predictions

Full-sample clear-sky O-B audit:

- requested: `163`
- successful: `163`
- failed: `0`
- overall O-B bias: `6.86 K`
- overall O-B RMSE: `9.74 K`
- median absolute O-B: `4.39 K`
- 183 GHz band RMSE: `2.32 K`
- main high-residual channels: `89 GHz`, `229 GHz`, `51.26 GHz`, `52.28 GHz`, `53.86 GHz`

Interpretation: this verifies that real observations, ERA5 labels, the Chengdu
21-channel schema, the 48-layer grid, and ARTS forward simulation are connected
end-to-end. The residuals should be treated as a clear-sky forward-model
consistency audit rather than final retrieval accuracy. Large residuals in
window and low-V-band channels motivate cloud screening, channel-response
verification, and channel-dependent observation-error tuning.

## Notes and limitations

- Current runner is clear-sky: `scripts/run_arts_profile.py` uses
  `cloudboxOff()`. Cloudy ARTS validation still requires the research-group
  cloudy agenda or a dedicated cloud absorption/scattering setup.
- The small OEM baseline is a smoke test, not a final statistical benchmark.
  It validates that ARTS can participate in Jacobian calculation and iterative
  retrieval.
- Running WSL from the Codex sandbox requires elevated execution. Without it,
  WSL can return `Wsl/Service/E_ACCESSDENIED` or non-JSON startup noise.
- PowerShell may print a local `profile.ps1` execution-policy warning after the
  command completes. This did not affect ARTS/OEM results.
