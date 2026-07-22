#!/usr/bin/env python3
"""Train Ridge/EOF retrievals on the corrected Chengdu 48-layer dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import config
from train_chengdu_brnn import rmse_bias
from train_chengdu_era5_brnn import date_mean_metrics
from train_chengdu_era5_ridge import predict_selected, select_model, standardize


def main() -> None:
    parser = argparse.ArgumentParser(description="Corrected 48-layer Ridge/EOF training")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "chengdu_era5_corrected48_ridge_all21",
    )
    parser.add_argument("--channel-set", choices=["all21", "hatpro14"], default="all21")
    parser.add_argument("--target-mode", choices=["corrected", "raw"], default="corrected")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.dataset)
    channel_indices = (
        list(range(21)) if args.channel_set == "all21" else config.CHENGDU_HATPRO14_IDX
    )
    X = data["X"][:, channel_indices].astype(np.float64)
    train_mask = data["train_mask"].astype(bool)
    val_mask = data["val_mask"].astype(bool)
    test_mask = data["test_mask"].astype(bool)
    X_train, X_val, X_test, x_mean, x_std = standardize(
        X[train_mask], X[val_mask], X[test_mask]
    )

    predictions = {}
    fitted_models = {}
    selections = {}
    searches = {}
    target_suffix = "" if args.target_mode == "corrected" else "_raw"
    for variable in ["T", "RH"]:
        target = data[f"{variable}{target_suffix}"].astype(np.float64)
        best, search = select_model(X_train, target[train_mask], X_val, target[val_mask])
        prediction = predict_selected(best, X_test)
        prediction = (
            np.clip(prediction, 180.0, 330.0)
            if variable == "T"
            else np.clip(prediction, 0.0, 100.0)
        )
        predictions[variable] = prediction.astype(np.float32)
        fitted_models[variable] = best
        selections[variable] = {
            "method": best["method"],
            "alpha": best["alpha"],
            "n_eof": best["n_eof"],
            "val_rmse": best["val_rmse"],
        }
        searches[variable] = search.tolist()

    t_true = data[f"T{target_suffix}"][test_mask]
    rh_true = data[f"RH{target_suffix}"][test_mask]
    t_raw = data["T_raw"][test_mask]
    rh_raw = data["RH_raw"][test_mask]
    test_dates = data["dates"][test_mask]
    baseline_t = np.repeat(data[f"T{target_suffix}"][train_mask].mean(axis=0, keepdims=True), len(t_true), axis=0)
    baseline_rh = np.repeat(data[f"RH{target_suffix}"][train_mask].mean(axis=0, keepdims=True), len(rh_true), axis=0)

    metrics = {
        "corrected_label": {
            "T": rmse_bias(predictions["T"], t_true),
            "RH": rmse_bias(predictions["RH"], rh_true),
            "T_daily": date_mean_metrics(predictions["T"], t_true, test_dates),
            "RH_daily": date_mean_metrics(predictions["RH"], rh_true, test_dates),
        },
        "raw_era5_reference": {
            "T": rmse_bias(predictions["T"], t_raw),
            "RH": rmse_bias(predictions["RH"], rh_raw),
        },
        "climatology_corrected_label": {
            "T": rmse_bias(baseline_t, t_true),
            "RH": rmse_bias(baseline_rh, rh_true),
        },
    }
    summary = {
        "status": "chengdu_corrected48_ridge_eof",
        "dataset": str(args.dataset),
        "target_mode": args.target_mode,
        "height_layers": int(len(data["heights"])),
        "channel_set": args.channel_set,
        "channel_indices": channel_indices,
        "selection": selections,
        "metrics": metrics,
    }
    (args.output_dir / "chengdu_corrected48_ridge_stats.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (args.output_dir / "model_search.json").write_text(
        json.dumps(searches, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        args.output_dir / "chengdu_corrected48_ridge_predictions.npz",
        T_pred=predictions["T"],
        RH_pred=predictions["RH"],
        T_true=t_true,
        RH_true=rh_true,
        T_raw=t_raw,
        RH_raw=rh_raw,
        dates=test_dates,
        timestamps=data["timestamps"][test_mask],
        heights=data["heights"],
    )
    np.savez_compressed(
        args.output_dir / "chengdu_corrected48_ridge_model.npz",
        X_mean=x_mean,
        X_std=x_std,
        channel_indices=np.asarray(channel_indices, dtype=np.int64),
        heights=data["heights"],
        T_weights=fitted_models["T"]["weights"],
        T_y_mean=fitted_models["T"]["y_mean"],
        T_basis=(
            fitted_models["T"]["basis"]
            if fitted_models["T"]["basis"] is not None
            else np.empty((0, 0), dtype=np.float64)
        ),
        T_profile_mean=fitted_models["T"].get(
            "profile_mean", np.empty((0, 0), dtype=np.float64)
        ),
        RH_weights=fitted_models["RH"]["weights"],
        RH_y_mean=fitted_models["RH"]["y_mean"],
        RH_basis=(
            fitted_models["RH"]["basis"]
            if fitted_models["RH"]["basis"] is not None
            else np.empty((0, 0), dtype=np.float64)
        ),
        RH_profile_mean=fitted_models["RH"].get(
            "profile_mean", np.empty((0, 0), dtype=np.float64)
        ),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
