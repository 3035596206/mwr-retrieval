import numpy as np

import config
from forward_model import ForwardModel


def _profile():
    heights = np.asarray(config.HEIGHT_GRID, dtype=float)
    return {
        "T": 288.15 - 6.5 * heights / 1000.0,
        "P_hPa": 1013.25 * np.exp(-heights / 8000.0),
        "RH": np.full_like(heights, 60.0),
        "CLWC": np.zeros_like(heights),
        "height": heights,
    }


def test_arts_forward_model_uses_runner_protocol():
    seen = {}

    def fake_arts_runner(payload):
        seen["payload"] = payload
        n = len(payload["instrument"]["frequencies_ghz"])
        return {"brightness_temperature_k": [240.0] * n}

    fm = ForwardModel(backend="arts", arts_runner=fake_arts_runner)
    tb = fm.simulate(_profile())

    assert fm.backend_name == "arts"
    assert fm.n_channels == config.DEFAULT_FORWARD_N_CHANNELS
    assert tb.shape == (config.DEFAULT_FORWARD_N_CHANNELS,)
    assert seen["payload"]["instrument"]["frequencies_ghz"] == config.DEFAULT_FORWARD_CHANNELS
    assert "temperature_k" in seen["payload"]["profile"]
