"""BRNN (Batch Normalization and Robust Neural Network) model definition.

Architecture (Fig 3.9):
    Input -> BN -> FC(256) -> ReLU -> BN -> Dropout ->
             FC(256) -> ReLU -> BN -> Dropout ->
             FC -> Sigmoid -> Output

Based on Yan et al. (2020) with adaptations for ground-based MWR.
Six separate models: 3 height ranges x 2 variables (T, RH).
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn

# Allow importing config from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config


class BRNN(nn.Module):
    """BRNN neural network for atmospheric profile retrieval.

    Args:
        n_input: number of input features
        n_output: number of output features (layers in the height interval)
        hidden_size: number of nodes in hidden layers (default: 256)
        dropout_rate: dropout probability (default: 0.3)
    """

    def __init__(self, n_input, n_output, hidden_size=256, dropout_rate=0.3):
        super(BRNN, self).__init__()

        self.input_bn = nn.BatchNorm1d(n_input)

        # Hidden layer 1
        self.fc1 = nn.Linear(n_input, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(dropout_rate)

        # Hidden layer 2
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(dropout_rate)

        # Output layer
        self.fc_out = nn.Linear(hidden_size, n_output)

        # Activation functions
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Input normalization
        x = self.input_bn(x)

        # Hidden layer 1
        x = self.fc1(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.dropout1(x)

        # Hidden layer 2
        x = self.fc2(x)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.dropout2(x)

        # Output layer
        x = self.fc_out(x)
        x = self.sigmoid(x)

        return x


def build_model_name(variable, height_range):
    """Build model name string.

    Args:
        variable: 'T' or 'RH'
        height_range: (low, high) in km
    Returns:
        model name string
    """
    return f"brnn_{variable}_{height_range[0]}-{height_range[1]}km"


def get_height_range_indices(height_low_km, height_high_km, height_grid):
    """Get indices in the height grid for a given height range.

    Args:
        height_low_km: lower bound in km
        height_high_km: upper bound in km
        height_grid: list or array of heights in meters
    Returns:
        (start_idx, end_idx): slice indices
    """
    low_m = height_low_km * 1000
    high_m = height_high_km * 1000

    indices = [i for i, h in enumerate(height_grid) if low_m <= h <= high_m]
    if not indices:
        return (0, len(height_grid))
    return (indices[0], indices[-1] + 1)


class BRNNEnsemble:
    """Ensemble of 6 BRNN models for full profile retrieval.

    Models:
        T_0-2km, T_2-8km, T_8-10km
        RH_0-2km, RH_2-8km, RH_8-10km
    """

    def __init__(self, height_grid, hidden_size=256, dropout_rate=0.3,
                 device="cpu"):
        self.height_grid = height_grid
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.device = device

        self.height_ranges = [
            (0, 2),   # 0-2km
            (2, 8),   # 2-8km
            (8, 10),  # 8-10km
        ]

        self.models = {}
        self.model_configs = {}

        for var in ["T", "RH"]:
            for low, high in self.height_ranges:
                name = build_model_name(var, (low, high))
                start, end = get_height_range_indices(
                    low, high, height_grid
                )
                n_output = end - start

                # Determine input size
                if var == "T" and low == 0 and high == 2:
                    # 0-2km temperature: 53-58GHz channels (5) + surface T
                    n_input = len(config.V_SURFACE_CHANNELS) + 1
                else:
                    # All other models: 14 channels + surface T, RH, P
                    n_input = 14 + 3

                model = BRNN(n_input, n_output, hidden_size, dropout_rate)
                model.to(device)

                self.models[name] = model
                self.model_configs[name] = {
                    "variable": var,
                    "height_range": (low, high),
                    "slice": (start, end),
                    "n_input": n_input,
                    "n_output": n_output,
                }

    def get_model(self, variable, height_range):
        name = build_model_name(variable, height_range)
        return self.models[name], self.model_configs[name]

    def predict(self, models_state_dicts, tb_obs, surface_data):
        """Full profile retrieval using all 6 models.

        Args:
            models_state_dicts: dict model_name -> state_dict
            tb_obs: brightness temperatures, shape (n_samples, 14)
            surface_data: ground T, RH, P, shape (n_samples, 3)
        Returns:
            T_profile: shape (n_samples, n_layers)
            RH_profile: shape (n_samples, n_layers)
        """
        n_samples = tb_obs.shape[0]
        n_layers = len(self.height_grid)

        T_full = np.zeros((n_samples, n_layers))
        RH_full = np.zeros((n_samples, n_layers))

        for name, config in self.model_configs.items():
            model = self.models[name]
            model.load_state_dict(models_state_dicts[name])
            model.eval()

            start, end = config["slice"]
            var = config["variable"]

            # Prepare input
            if var == "T" and config["height_range"] == (0, 2):
                # Only 53-58 GHz channels + surface T
                from config import V_SURFACE_IDX
                X = np.column_stack([
                    tb_obs[:, V_SURFACE_IDX],
                    surface_data[:, 0:1],  # surface T
                ])
            else:
                X = np.column_stack([tb_obs, surface_data])

            X_tensor = torch.FloatTensor(X).to(self.device)

            with torch.no_grad():
                output = model(X_tensor).cpu().numpy()

            if var == "T":
                # Temperature: denormalize from [0,1] sigmoid range
                # Paper uses sigmoid to constrain output to reasonable range
                T_full[:, start:end] = output * 50.0 + 250.0  # ~[250, 300] K
            else:
                # RH: denormalize from [0,1] to [0, 100] %
                RH_full[:, start:end] = output * 100.0

        return T_full, RH_full

    def save_all(self, directory):
        """Save all 6 models."""
        os.makedirs(directory, exist_ok=True)
        for name, model in self.models.items():
            torch.save(model.state_dict(), os.path.join(directory, f"{name}.pt"))

    def load_all(self, directory):
        """Load all 6 models."""
        for name, model in self.models.items():
            path = os.path.join(directory, f"{name}.pt")
            if os.path.exists(path):
                model.load_state_dict(
                    torch.load(path, map_location=self.device, weights_only=True)
                )


if __name__ == "__main__":
    from config import HEIGHT_GRID

    ensemble = BRNNEnsemble(HEIGHT_GRID)
    for name, config in ensemble.model_configs.items():
        print(f"{name}: input={config['n_input']}, output={config['n_output']}, "
              f"slice={config['slice']}")
