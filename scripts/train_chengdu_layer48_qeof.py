#!/usr/bin/env python3
"""Train T and log(q) Ridge/EOF retrievals on the physical 48-layer dataset."""

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

from build_chengdu_era5_layer48_dataset import specific_humidity_to_rh
from train_chengdu_brnn import rmse_bias
from train_chengdu_era5_brnn import date_mean_metrics
from train_chengdu_era5_ridge import predict_selected, select_model, standardize


def main() -> None:
    parser = argparse.ArgumentParser(description="Physical 48-layer T/log(q) Ridge/EOF")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.dataset)
    train_mask = data["train_mask"].astype(bool)
    val_mask = data["val_mask"].astype(bool)
    test_mask = data["test_mask"].astype(bool)
    X_train, X_val, X_test, x_mean, x_std = standardize(
        data["X"][train_mask].astype(np.float64),
        data["X"][val_mask].astype(np.float64),
        data["X"][test_mask].astype(np.float64),
    )

    targets = {"T": data["T"].astype(np.float64), "logq": data["logq"].astype(np.float64)}
    predictions = {}
    fitted_models = {}
    selections = {}
    searches = {}
    for variable, target in targets.items():
        best, search = select_model(X_train, target[train_mask], X_val, target[val_mask])
        prediction = predict_selected(best, X_test)
        if variable == "T":
            prediction = np.clip(prediction, 180.0, 330.0)
        else:
            prediction = np.clip(prediction, np.log(1e-8), np.log(0.1))
        predictions[variable] = prediction.astype(np.float32)
        fitted_models[variable] = best
        selections[variable] = {
            "method": best["method"],
            "alpha": best["alpha"],
            "n_eof": best["n_eof"],
            "val_rmse": best["val_rmse"],
        }
        searches[variable] = search.tolist()

    t_true = data["T"][test_mask]
    logq_true = data["logq"][test_mask]
    q_true = data["q"][test_mask]
    rh_true = data["RH"][test_mask]
    pressure = data["P"][test_mask]
    q_pred = np.exp(predictions["logq"]).astype(np.float32)
    rh_pred = specific_humidity_to_rh(predictions["T"], q_pred, pressure).astype(np.float32)
    rh_pred_true_t = specific_humidity_to_rh(t_true, q_pred, pressure).astype(np.float32)

    baseline_t = np.repeat(data["T"][train_mask].mean(axis=0, keepdims=True), len(t_true), axis=0)
    baseline_logq = np.repeat(
        data["logq"][train_mask].mean(axis=0, keepdims=True), len(t_true), axis=0
    )
    baseline_q = np.exp(baseline_logq)
    baseline_rh = specific_humidity_to_rh(baseline_t, baseline_q, pressure)
    dates = data["dates"][test_mask]

    metrics = {
        "model": {
            "T": rmse_bias(predictions["T"], t_true),
            "logq": rmse_bias(predictions["logq"], logq_true),
            "q_g_per_kg": rmse_bias(q_pred * 1000.0, q_true * 1000.0),
            "RH": rmse_bias(rh_pred, rh_true),
            "RH_using_true_T": rmse_bias(rh_pred_true_t, rh_true),
            "T_daily": date_mean_metrics(predictions["T"], t_true, dates),
            "RH_daily": date_mean_metrics(rh_pred, rh_true, dates),
        },
        "climatology": {
            "T": rmse_bias(baseline_t, t_true),
            "logq": rmse_bias(baseline_logq, logq_true),
            "q_g_per_kg": rmse_bias(baseline_q * 1000.0, q_true * 1000.0),
            "RH": rmse_bias(baseline_rh, rh_true),
        },
    }
    summary = {
        "status": "chengdu_physical_layer48_T_logq_ridge_eof",
        "dataset": str(args.dataset),
        "selection": selections,
        "metrics": metrics,
        "notes": {
            "RH_conversion_pressure": "ERA5 layer pressure used for evaluation",
            "deployment_requirement": "surface pressure or pressure-profile prior is required to convert predicted q to RH",
        },
    }
    (args.output_dir / "chengdu_layer48_qeof_stats.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (args.output_dir / "model_search.json").write_text(
        json.dumps(searches, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        args.output_dir / "chengdu_layer48_qeof_predictions.npz",
        T_pred=predictions["T"],
        logq_pred=predictions["logq"],
        q_pred=q_pred,
        RH_pred=rh_pred,
        RH_pred_true_T=rh_pred_true_t,
        T_true=t_true,
        logq_true=logq_true,
        q_true=q_true,
        RH_true=rh_true,
        P=pressure,
        dates=dates,
        timestamps=data["timestamps"][test_mask],
        heights=data["heights"],
        layer_edges=data["layer_edges"],
        layer_thickness=data["layer_thickness"],
    )

    def model_arrays(model: dict, prefix: str) -> dict[str, np.ndarray]:
        return {
            f"{prefix}_weights": model["weights"],
            f"{prefix}_y_mean": model["y_mean"],
            f"{prefix}_basis": model["basis"]
            if model["basis"] is not None
            else np.empty((0, 0), dtype=np.float64),
            f"{prefix}_profile_mean": model.get(
                "profile_mean", np.empty((0, 0), dtype=np.float64)
            ),
        }

    np.savez_compressed(
        args.output_dir / "chengdu_layer48_qeof_model.npz",
        X_mean=x_mean,
        X_std=x_std,
        heights=data["heights"],
        layer_edges=data["layer_edges"],
        **model_arrays(fitted_models["T"], "T"),
        **model_arrays(fitted_models["logq"], "logq"),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
