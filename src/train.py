"""Training script for BRNN atmospheric profile retrieval models.

Trains 6 separate BRNN models (3 height ranges x 2 variables) using
ERA5 reanalysis data with quality control as training samples.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import config
from brnn_model import BRNNEnsemble, build_model_name, get_height_range_indices


def prepare_training_data(profiles, tbs, model_config):
    """Prepare input/output tensors for a specific model.

    Args:
        profiles: dict with 'T', 'RH', 'time', 'height', 't2m', 'rh2m', 'sp'
        tbs: simulated brightness temperatures, shape (n_samples, 14)
        model_config: model configuration dict
    Returns:
        X: input tensor (n_samples, n_input)
        y: target tensor (n_samples, n_output)
    """
    n_samples = profiles["T"].shape[0]
    var = model_config["variable"]
    height_range = model_config["height_range"]
    start, end = model_config["slice"]

    # Prepare surface data
    t2m = profiles["t2m"] if profiles["t2m"] is not None else profiles["T"][:, 0]
    rh2m = profiles["rh2m"] if profiles["rh2m"] is not None else profiles["RH"][:, 0]
    sp = profiles["sp"] if profiles["sp"] is not None else np.ones(n_samples) * 101300.0

    surface_data = np.column_stack([t2m, rh2m, sp / 100.0])  # Pa -> hPa

    # Build input
    if var == "T" and height_range == (0, 2):
        X = np.column_stack([tbs[:, config.V_SURFACE_IDX], t2m.reshape(-1, 1)])
    else:
        X = np.column_stack([tbs, surface_data])

    # Build target
    if var == "T":
        # Normalize to [0, 1] using typical temperature range
        y = profiles["T"][:, start:end]
        y = (y - 250.0) / 50.0  # ~[250, 300] K -> [0, 1]
    else:
        # RH in [0, 100] -> [0, 1]
        y = profiles["RH"][:, start:end] / 100.0

    return X, y


def train_model(model, train_loader, val_loader, n_epochs, lr, patience, device):
    """Train a single BRNN model.

    Args:
        model: BRNN model instance
        train_loader: training DataLoader
        val_loader: validation DataLoader
        n_epochs: max number of epochs
        lr: learning rate
        patience: early stopping patience
        device: torch device
    Returns:
        model: trained model
        history: dict of training/validation loss
    """
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(n_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                val_loss += loss.item() * X_batch.size(0)

        val_loss /= len(val_loader.dataset)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs}: train_loss={train_loss:.6f}, "
                  f"val_loss={val_loss:.6f}")

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(best_state)
    return model, history


def train_all_models(profiles, tbs, models_dir, device="cpu"):
    """Train all 6 BRNN models.

    Args:
        profiles: dict with ERA5 profiles (after QC)
        tbs: simulated brightness temperatures
        models_dir: directory to save trained models
        device: torch device
    Returns:
        ensemble: trained BRNNEnsemble
    """
    os.makedirs(models_dir, exist_ok=True)

    ensemble = BRNNEnsemble(
        config.HEIGHT_GRID,
        hidden_size=config.HIDDEN_NODES,
        dropout_rate=config.DROPOUT_RATE,
        device=device,
    )

    n_samples = profiles["T"].shape[0]
    indices = np.random.permutation(n_samples)
    split_idx = int(n_samples * 0.8)
    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]

    print(f"Training samples: {len(train_idx)}, Validation: {len(val_idx)}")

    for name, model_config in ensemble.model_configs.items():
        print(f"\n{'='*60}")
        print(f"Training {name}")
        print(f"  Input: {model_config['n_input']}, Output: {model_config['n_output']}")

        X, y = prepare_training_data(profiles, tbs, model_config)

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        train_dataset = TensorDataset(
            torch.FloatTensor(X_train), torch.FloatTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val), torch.FloatTensor(y_val)
        )

        train_loader = DataLoader(
            train_dataset, batch_size=config.BATCH_SIZE, shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=config.BATCH_SIZE, shuffle=False
        )

        model, history = train_model(
            ensemble.models[name],
            train_loader, val_loader,
            n_epochs=config.MAX_EPOCHS,
            lr=config.LEARNING_RATE,
            patience=config.EARLY_STOPPING_PATIENCE,
            device=device,
        )

        # Save model
        torch.save(model.state_dict(), os.path.join(models_dir, f"{name}.pt"))
        print(f"  Final val_loss: {history['val_loss'][-1]:.6f}")

    print(f"\nAll models saved to {models_dir}")
    return ensemble


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to training data .npz file")
    parser.add_argument("--models_dir", type=str, default=config.MODEL_DIR,
                        help="Directory to save models")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device (cpu or cuda)")
    args = parser.parse_args()

    # Load data
    data = np.load(args.data_path, allow_pickle=True)
    profiles = data["profiles"].item()
    tbs = data["tbs"]

    train_all_models(profiles, tbs, args.models_dir, args.device)
