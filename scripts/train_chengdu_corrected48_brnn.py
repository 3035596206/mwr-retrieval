#!/usr/bin/env python3
"""Train the Chengdu BRNN on the corrected 48-layer ERA5 dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import config
from train_chengdu_brnn import predict_full, rmse_bias, set_seed, train_one_model
from train_chengdu_era5_brnn import date_mean_metrics


MODEL_DEFS_48 = [
    ("brnn_T_0.5-2km", "T", (0.5, 2.0)),
    ("brnn_T_2.25-8km", "T", (2.25, 8.0)),
    ("brnn_T_8.25-10km", "T", (8.25, 10.0)),
    ("brnn_RH_0.5-2km", "RH", (0.5, 2.0)),
    ("brnn_RH_2.25-8km", "RH", (2.25, 8.0)),
    ("brnn_RH_8.25-10km", "RH", (8.25, 10.0)),
]

MODEL_DEFS_LAYER48 = [
    ("brnn_T_0-2km", "T", (0.0, 2.0)),
    ("brnn_T_2-8km", "T", (2.0, 8.0)),
    ("brnn_T_8-10km", "T", (8.0, 10.0)),
    ("brnn_RH_0-2km", "RH", (0.0, 2.0)),
    ("brnn_RH_2-8km", "RH", (2.0, 8.0)),
    ("brnn_RH_8-10km", "RH", (8.0, 10.0)),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Corrected 48-layer BRNN training")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--channel-set", choices=["all21", "hatpro14"], default="all21")
    parser.add_argument("--target-mode", choices=["corrected", "raw"], default="corrected")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=35)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    data = np.load(args.dataset)
    heights = data["heights"].astype(np.float32)
    model_definitions = MODEL_DEFS_LAYER48 if heights[0] < 500.0 else MODEL_DEFS_48
    channel_indices = (
        list(range(21)) if args.channel_set == "all21" else config.CHENGDU_HATPRO14_IDX
    )
    X = data["X"][:, channel_indices].astype(np.float32)
    target_suffix = "" if args.target_mode == "corrected" else "_raw"
    T = data[f"T{target_suffix}"].astype(np.float32)
    RH = data[f"RH{target_suffix}"].astype(np.float32)
    train_mask = data["train_mask"].astype(bool)
    val_mask = data["val_mask"].astype(bool)
    test_mask = data["test_mask"].astype(bool)

    models = {}
    training = {}
    for name, variable, height_range in model_definitions:
        print(f"Training {name}, seed={args.seed}...")
        model, details = train_one_model(
            name,
            variable,
            height_range,
            X,
            T,
            RH,
            train_mask,
            val_mask,
            args.hidden_size,
            args.dropout,
            args.batch_size,
            args.learning_rate,
            args.max_epochs,
            args.patience,
            "cpu",
            height_grid=heights,
        )
        models[name] = model
        training[name] = details
        torch.save(model.state_dict(), args.model_dir / f"{name}.pt")
        print(f"  best epoch={details['best_epoch']}, val={details['best_val']:.6f}")

    t_pred, rh_pred = predict_full(
        models,
        X[test_mask],
        "cpu",
        height_grid=heights,
        model_defs=model_definitions,
    )
    t_pred = np.clip(t_pred, 180.0, 330.0)
    rh_pred = np.clip(rh_pred, 0.0, 100.0)
    t_true = T[test_mask]
    rh_true = RH[test_mask]
    t_raw = data["T_raw"][test_mask]
    rh_raw = data["RH_raw"][test_mask]
    test_dates = data["dates"][test_mask]
    baseline_t = np.repeat(T[train_mask].mean(axis=0, keepdims=True), len(t_true), axis=0)
    baseline_rh = np.repeat(RH[train_mask].mean(axis=0, keepdims=True), len(rh_true), axis=0)

    metrics = {
        "corrected_label": {
            "T": rmse_bias(t_pred, t_true),
            "RH": rmse_bias(rh_pred, rh_true),
            "T_daily": date_mean_metrics(t_pred, t_true, test_dates),
            "RH_daily": date_mean_metrics(rh_pred, rh_true, test_dates),
        },
        "raw_era5_reference": {
            "T": rmse_bias(t_pred, t_raw),
            "RH": rmse_bias(rh_pred, rh_raw),
        },
        "climatology_corrected_label": {
            "T": rmse_bias(baseline_t, t_true),
            "RH": rmse_bias(baseline_rh, rh_true),
        },
    }
    summary = {
        "status": "chengdu_corrected48_brnn",
        "dataset": str(args.dataset),
        "configuration": {
            "target_mode": args.target_mode,
            "seed": args.seed,
            "hidden_size": args.hidden_size,
            "dropout": args.dropout,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "channel_set": args.channel_set,
            "channel_indices": channel_indices,
            "height_layers": len(heights),
            "model_definitions": model_definitions,
        },
        "metrics": metrics,
        "training": {
            name: {"best_epoch": item["best_epoch"], "best_val": item["best_val"]}
            for name, item in training.items()
        },
    }
    (args.output_dir / "chengdu_corrected48_brnn_stats.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (args.output_dir / "training_history.json").write_text(
        json.dumps(training, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        args.output_dir / "chengdu_corrected48_brnn_predictions.npz",
        T_pred=t_pred,
        RH_pred=rh_pred,
        T_true=t_true,
        RH_true=rh_true,
        T_raw=t_raw,
        RH_raw=rh_raw,
        dates=test_dates,
        timestamps=data["timestamps"][test_mask],
        heights=heights,
        X_test=X[test_mask],
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
