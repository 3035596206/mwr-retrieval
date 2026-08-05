#!/usr/bin/env python3
"""Train and test a Chengdu-adapted BRNN on observed brightness temperatures.

The Chengdu brightness-temperature files contain 21 anonymous channels and no
surface/IR features, so they cannot be passed directly to the MP-3000A v4
checkpoints. This script reuses the project's BRNN architecture and six height
segments, but trains a BT-only 21-channel adaptation. Targets are Wenjiang
soundings matched within a configurable time window. Splits are grouped by
sounding launch to prevent one profile from appearing in multiple splits.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import config
from brnn_model import BRNN, get_height_range_indices


MODEL_DEFS = [
    ("brnn_T_0-2km", "T", (0, 2)),
    ("brnn_T_2-8km", "T", (2, 8)),
    ("brnn_T_8-10km", "T", (8, 10)),
    ("brnn_RH_0-2km", "RH", (0, 2)),
    ("brnn_RH_2-8km", "RH", (2, 8)),
    ("brnn_RH_8-10km", "RH", (8, 10)),
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_observations(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload["records"]
    for record in records:
        record["datetime"] = datetime.strptime(record["timestamp"], "%Y_%m_%d %H:%M:%S")
        if len(record["channels"]) != 21:
            raise ValueError(f"Expected 21 channels, got {len(record['channels'])}")
    return records


def sounding_time_from_name(path: Path) -> datetime:
    stamp = path.stem.rsplit("-", 1)[-1]
    return datetime.strptime(stamp, "%Y%m%d%H")


def parse_sounding(
    path: Path, target_height_m: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    heights = []
    temperatures = []
    humidities = []
    pressures = []

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            fields = line.split()
            if len(fields) < 17:
                continue
            try:
                temp_c = float(fields[6])
                pressure_hpa = float(fields[7])
                rh = float(fields[8])
                height_m = float(fields[16])
            except ValueError:
                continue
            if not (-100.0 <= temp_c <= 60.0):
                continue
            if not (0.0 <= rh <= 100.0):
                continue
            if not (1.0 <= pressure_hpa <= 1100.0):
                continue
            if not (-500.0 <= height_m <= 60000.0):
                continue
            heights.append(height_m)
            temperatures.append(temp_c + 273.15)
            humidities.append(rh)
            pressures.append(pressure_hpa)

    if len(heights) < 50:
        raise ValueError(f"Too few valid levels in {path}: {len(heights)}")

    heights = np.asarray(heights, dtype=np.float64)
    temperatures = np.asarray(temperatures, dtype=np.float64)
    humidities = np.asarray(humidities, dtype=np.float64)
    pressures = np.asarray(pressures, dtype=np.float64)

    order = np.argsort(heights)
    heights = heights[order]
    temperatures = temperatures[order]
    humidities = humidities[order]
    pressures = pressures[order]

    station_height = float(np.nanpercentile(heights, 0.5))
    heights_agl = heights - station_height
    keep = np.concatenate(([True], np.diff(heights_agl) > 0.5))
    heights_agl = heights_agl[keep]
    temperatures = temperatures[keep]
    humidities = humidities[keep]
    pressures = pressures[keep]

    if heights_agl[-1] < target_height_m[-1]:
        raise ValueError(f"Sounding does not reach 10 km AGL: {path}")

    t_profile = np.interp(target_height_m, heights_agl, temperatures)
    rh_profile = np.interp(target_height_m, heights_agl, humidities)
    surface = np.asarray(
        [temperatures[0], humidities[0], pressures[0]], dtype=np.float32
    )
    return t_profile.astype(np.float32), rh_profile.astype(np.float32), surface


def build_dataset(
    records: list[dict], sounding_root: Path, target_height_m: np.ndarray, max_delta_hours: float
) -> dict:
    sounding_paths = sorted(sounding_root.rglob("*.txt"))
    sounding_entries = [(sounding_time_from_name(path), path) for path in sounding_paths]
    profile_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    brightness = []
    t_targets = []
    rh_targets = []
    groups = []
    timestamps = []
    deltas = []
    surface_features = []
    skipped_profiles: dict[str, str] = {}

    for record in records:
        obs_time = record["datetime"]
        sounding_time, sounding_path = min(
            sounding_entries, key=lambda item: abs((item[0] - obs_time).total_seconds())
        )
        delta_hours = abs((sounding_time - obs_time).total_seconds()) / 3600.0
        if delta_hours > max_delta_hours:
            continue

        group = sounding_time.strftime("%Y%m%d%H")
        if group not in profile_cache and group not in skipped_profiles:
            try:
                profile_cache[group] = parse_sounding(sounding_path, target_height_m)
            except ValueError as exc:
                skipped_profiles[group] = str(exc)
        if group in skipped_profiles:
            continue

        t_profile, rh_profile, surface = profile_cache[group]
        brightness.append(record["channels"])
        t_targets.append(t_profile)
        rh_targets.append(rh_profile)
        surface_features.append(surface)
        groups.append(group)
        timestamps.append(record["timestamp"])
        deltas.append(delta_hours)

    return {
        "X": np.asarray(brightness, dtype=np.float32),
        "T": np.asarray(t_targets, dtype=np.float32),
        "RH": np.asarray(rh_targets, dtype=np.float32),
        "surface": np.asarray(surface_features, dtype=np.float32),
        "groups": np.asarray(groups),
        "timestamps": np.asarray(timestamps),
        "delta_hours": np.asarray(deltas, dtype=np.float32),
        "skipped_profiles": skipped_profiles,
    }


def grouped_split(groups: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    unique_groups = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_groups)

    n_groups = len(unique_groups)
    n_train = max(1, int(n_groups * 0.70))
    n_val = max(1, int(n_groups * 0.15))
    if n_train + n_val >= n_groups:
        n_val = max(1, n_groups - n_train - 1)

    train_groups = set(unique_groups[:n_train])
    val_groups = set(unique_groups[n_train : n_train + n_val])
    test_groups = set(unique_groups[n_train + n_val :])

    train_mask = np.asarray([group in train_groups for group in groups])
    val_mask = np.asarray([group in val_groups for group in groups])
    test_mask = np.asarray([group in test_groups for group in groups])
    split = {
        "train_groups": sorted(train_groups),
        "val_groups": sorted(val_groups),
        "test_groups": sorted(test_groups),
    }
    return train_mask, val_mask, test_mask, split


def train_one_model(
    name: str,
    variable: str,
    height_range: tuple[float, float],
    X: np.ndarray,
    T: np.ndarray,
    RH: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    hidden_size: int,
    dropout: float,
    batch_size: int,
    learning_rate: float,
    max_epochs: int,
    patience_limit: int,
    device: str,
    height_grid: np.ndarray | list[float] | None = None,
    trend_weight: float | None = None,
    smooth_weight: float | None = None,
) -> tuple[BRNN, dict]:
    grid = config.HEIGHT_GRID if height_grid is None else height_grid
    start, end = get_height_range_indices(*height_range, grid)
    target = T[:, start:end] if variable == "T" else RH[:, start:end]
    target_norm = (target - 200.0) / 100.0 if variable == "T" else target / 100.0

    if trend_weight is None:
        trend_weight = 0.25 if variable == "RH" else 0.0
    if smooth_weight is None:
        smooth_weight = 0.003 if variable == "RH" else 0.001

    if trend_weight < 0.0:
        raise ValueError("trend_weight must be non-negative")
    if smooth_weight < 0.0:
        raise ValueError("smooth_weight must be non-negative")

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X[train_mask]), torch.from_numpy(target_norm[train_mask])),
        batch_size=min(batch_size, int(train_mask.sum())),
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X[val_mask]), torch.from_numpy(target_norm[val_mask])),
        batch_size=min(batch_size, int(val_mask.sum())),
        shuffle=False,
    )

    model = BRNN(X.shape[1], end - start, hidden_size, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = nn.MSELoss(reduction="none")
    best_state = None
    best_val = float("inf")
    best_epoch = 0
    patience = 0
    history = []

    def objective(pred: torch.Tensor, yb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        data_loss = criterion(pred, yb).mean()
        trend_loss = pred.new_tensor(0.0)
        smooth_loss = pred.new_tensor(0.0)
        if trend_weight > 0.0 and pred.shape[1] >= 2:
            trend_loss = torch.mean((torch.diff(pred, dim=1) - torch.diff(yb, dim=1)) ** 2)
        if smooth_weight > 0.0 and pred.shape[1] >= 3:
            smooth_loss = torch.mean(torch.diff(pred, n=2, dim=1) ** 2)
        total = data_loss + trend_weight * trend_loss + smooth_weight * smooth_loss
        return total, data_loss, trend_loss, smooth_loss

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_sum = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss, _, _, _ = objective(pred, yb)
            loss.backward()
            optimizer.step()
            train_sum += loss.item() * xb.shape[0]

        model.eval()
        val_sum = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss, _, _, _ = objective(pred, yb)
                val_sum += loss.item() * xb.shape[0]

        train_loss = train_sum / int(train_mask.sum())
        val_loss = val_sum / int(val_mask.sum())
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val - 1e-7:
            best_val = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
        if patience >= patience_limit:
            break

    if best_state is None:
        raise RuntimeError(f"No valid checkpoint produced for {name}")
    model.load_state_dict(best_state)
    return model, {"best_epoch": best_epoch, "best_val": best_val, "history": history}

def predict_full(
    models: dict[str, BRNN],
    X: np.ndarray,
    device: str,
    height_grid: np.ndarray | list[float] | None = None,
    model_defs: list[tuple[str, str, tuple[float, float]]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    grid = config.HEIGHT_GRID if height_grid is None else height_grid
    definitions = MODEL_DEFS if model_defs is None else model_defs
    t_pred = np.zeros((len(X), len(grid)), dtype=np.float32)
    rh_pred = np.zeros_like(t_pred)
    x_tensor = torch.from_numpy(X).to(device)

    for name, variable, height_range in definitions:
        start, end = get_height_range_indices(*height_range, grid)
        model = models[name]
        model.eval()
        with torch.no_grad():
            output = model(x_tensor).cpu().numpy()
        if variable == "T":
            t_pred[:, start:end] = output * 100.0 + 200.0
        else:
            rh_pred[:, start:end] = output * 100.0
    return t_pred, rh_pred


def rmse_bias(pred: np.ndarray, truth: np.ndarray) -> dict:
    error = pred - truth
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
    }


def grouped_metrics(
    t_pred: np.ndarray,
    rh_pred: np.ndarray,
    t_true: np.ndarray,
    rh_true: np.ndarray,
    groups: np.ndarray,
) -> dict:
    unique_groups = np.unique(groups)
    t_group_pred = []
    rh_group_pred = []
    t_group_true = []
    rh_group_true = []
    for group in unique_groups:
        mask = groups == group
        t_group_pred.append(t_pred[mask].mean(axis=0))
        rh_group_pred.append(rh_pred[mask].mean(axis=0))
        t_group_true.append(t_true[mask][0])
        rh_group_true.append(rh_true[mask][0])
    return {
        "n_groups": int(len(unique_groups)),
        "T": rmse_bias(np.asarray(t_group_pred), np.asarray(t_group_true)),
        "RH": rmse_bias(np.asarray(rh_group_pred), np.asarray(rh_group_true)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Chengdu 21-channel BRNN adaptation")
    parser.add_argument("--obs-json", type=Path, default=PROJECT_ROOT / "data" / "chengdu_obs_bt.json")
    parser.add_argument("--sounding-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "chengdu_brnn_poc")
    parser.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "models_chengdu_brnn_poc")
    parser.add_argument("--max-delta-hours", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=35)
    parser.add_argument(
        "--channel-set",
        choices=["all21", "hatpro14", "high7", "vapor183"],
        default="all21",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    device = "cpu"

    records = load_observations(args.obs_json)
    target_height = np.asarray(config.HEIGHT_GRID, dtype=np.float32)
    data = build_dataset(records, args.sounding_dir, target_height, args.max_delta_hours)
    channel_indices = {
        "all21": list(range(21)),
        "hatpro14": config.CHENGDU_HATPRO14_IDX,
        "high7": config.CHENGDU_HIGH7_IDX,
        "vapor183": config.CHENGDU_183GHZ_IDX,
    }[args.channel_set]
    data["X"] = data["X"][:, channel_indices]
    if len(data["X"]) < 50 or len(np.unique(data["groups"])) < 15:
        raise RuntimeError(
            f"Insufficient matched data: {len(data['X'])} samples, "
            f"{len(np.unique(data['groups']))} sounding groups"
        )

    train_mask, val_mask, test_mask, split = grouped_split(data["groups"], args.seed)
    print(
        f"Matched {len(data['X'])} samples / {len(np.unique(data['groups']))} soundings; "
        f"train={train_mask.sum()}, val={val_mask.sum()}, test={test_mask.sum()}"
    )

    models = {}
    training = {}
    for name, variable, height_range in MODEL_DEFS:
        print(f"Training {name}...")
        model, details = train_one_model(
            name,
            variable,
            height_range,
            data["X"],
            data["T"],
            data["RH"],
            train_mask,
            val_mask,
            args.hidden_size,
            args.dropout,
            args.batch_size,
            args.learning_rate,
            args.max_epochs,
            args.patience,
            device,
        )
        models[name] = model
        training[name] = details
        torch.save(model.state_dict(), args.model_dir / f"{name}.pt")
        print(f"  best epoch={details['best_epoch']}, val={details['best_val']:.6f}")

    t_pred, rh_pred = predict_full(models, data["X"][test_mask], device)
    t_true = data["T"][test_mask]
    rh_true = data["RH"][test_mask]
    test_groups = data["groups"][test_mask]

    climatology_t = data["T"][train_mask].mean(axis=0, keepdims=True)
    climatology_rh = data["RH"][train_mask].mean(axis=0, keepdims=True)
    baseline_t = np.repeat(climatology_t, len(t_true), axis=0)
    baseline_rh = np.repeat(climatology_rh, len(rh_true), axis=0)

    metrics = {
        "model_sample": {"T": rmse_bias(t_pred, t_true), "RH": rmse_bias(rh_pred, rh_true)},
        "model_grouped": grouped_metrics(t_pred, rh_pred, t_true, rh_true, test_groups),
        "climatology_sample": {
            "T": rmse_bias(baseline_t, t_true),
            "RH": rmse_bias(baseline_rh, rh_true),
        },
        "climatology_grouped": grouped_metrics(
            baseline_t, baseline_rh, t_true, rh_true, test_groups
        ),
    }

    height_metrics = []
    for height_km in [0, 0.5, 1, 2, 3, 5, 8, 10]:
        idx = int(np.argmin(np.abs(target_height - height_km * 1000.0)))
        height_metrics.append(
            {
                "height_km": float(target_height[idx] / 1000.0),
                "T_rmse": float(np.sqrt(np.mean((t_pred[:, idx] - t_true[:, idx]) ** 2))),
                "T_bias": float(np.mean(t_pred[:, idx] - t_true[:, idx])),
                "RH_rmse": float(np.sqrt(np.mean((rh_pred[:, idx] - rh_true[:, idx]) ** 2))),
                "RH_bias": float(np.mean(rh_pred[:, idx] - rh_true[:, idx])),
            }
        )

    summary = {
        "status": "proof_of_concept",
        "warning": (
            f"Only 21 anonymous BT channels and {len(np.unique(data['groups']))} valid matched "
            "sounding groups are available. This is an adaptation/feasibility result, not a "
            "replacement for MP-3000A v4."
        ),
        "configuration": {
            "seed": args.seed,
            "max_delta_hours": args.max_delta_hours,
            "hidden_size": args.hidden_size,
            "dropout": args.dropout,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "n_channels": int(data["X"].shape[1]),
            "channel_set": args.channel_set,
            "channel_indices": channel_indices,
            "channel_frequencies_ghz": [config.CHENGDU_ALL_CHANNELS[i] for i in channel_indices],
        },
        "dataset": {
            "n_observations_total": len(records),
            "n_matched_samples": int(len(data["X"])),
            "n_sounding_groups": int(len(np.unique(data["groups"]))),
            "train_samples": int(train_mask.sum()),
            "val_samples": int(val_mask.sum()),
            "test_samples": int(test_mask.sum()),
            "train_groups": len(split["train_groups"]),
            "val_groups": len(split["val_groups"]),
            "test_groups": len(split["test_groups"]),
            "match_delta_hours_mean": float(data["delta_hours"].mean()),
            "match_delta_hours_max": float(data["delta_hours"].max()),
        },
        "split": split,
        "metrics": metrics,
        "height_metrics": height_metrics,
        "training": {
            name: {"best_epoch": details["best_epoch"], "best_val": details["best_val"]}
            for name, details in training.items()
        },
        "skipped_profiles": data["skipped_profiles"],
    }

    (args.output_dir / "chengdu_brnn_stats.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (args.output_dir / "training_history.json").write_text(
        json.dumps(training, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        args.output_dir / "chengdu_brnn_predictions.npz",
        T_pred=t_pred,
        RH_pred=rh_pred,
        T_true=t_true,
        RH_true=rh_true,
        groups=test_groups,
        timestamps=data["timestamps"][test_mask],
        heights=target_height,
        X_test=data["X"][test_mask],
    )
    metadata = {
        "model_type": "project BRNN six-segment adaptation",
        "input_features": (
            f"Observed BT channel set {args.channel_set}: "
            f"{[config.CHENGDU_ALL_CHANNELS[i] for i in channel_indices]} GHz"
        ),
        "height_grid_m": config.HEIGHT_GRID,
        "model_definitions": MODEL_DEFS,
        "configuration": summary["configuration"],
    }
    (args.model_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(json.dumps(metrics, indent=2))
    print(f"Results: {args.output_dir}")
    print(f"Models:  {args.model_dir}")


if __name__ == "__main__":
    main()
