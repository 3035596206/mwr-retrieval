#!/usr/bin/env python3
"""Small-sample Ridge/EOF baseline for exact-hour Chengdu ERA5 matching."""

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
from train_chengdu_brnn import load_observations, rmse_bias
from train_chengdu_era5_brnn import (
    build_exact_dataset,
    chronological_date_split,
    date_mean_metrics,
    load_era5_profiles,
)


ALPHAS = [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0]
EOF_COUNTS = [3, 5, 8, 10]


def standardize(train: np.ndarray, *others: np.ndarray):
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return ((train - mean) / std, *[(item - mean) / std for item in others], mean, std)


def fit_ridge(X: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    y_mean = y.mean(axis=0, keepdims=True)
    y_centered = y - y_mean
    regularizer = np.eye(X.shape[1], dtype=np.float64) * alpha
    weights = np.linalg.solve(X.T @ X + regularizer, X.T @ y_centered)
    return weights, y_mean


def predict_ridge(X: np.ndarray, weights: np.ndarray, y_mean: np.ndarray) -> np.ndarray:
    return X @ weights + y_mean


def select_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> tuple[dict, np.ndarray]:
    candidates = []

    for alpha in ALPHAS:
        weights, y_mean = fit_ridge(X_train, y_train, alpha)
        pred = predict_ridge(X_val, weights, y_mean)
        candidates.append(
            {
                "method": "direct",
                "alpha": alpha,
                "n_eof": None,
                "val_rmse": float(np.sqrt(np.mean((pred - y_val) ** 2))),
                "weights": weights,
                "y_mean": y_mean,
                "basis": None,
            }
        )

    profile_mean = y_train.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(y_train - profile_mean, full_matrices=False)
    for n_eof in EOF_COUNTS:
        basis = vt[: min(n_eof, len(vt))]
        train_coeff = (y_train - profile_mean) @ basis.T
        for alpha in ALPHAS:
            weights, coeff_mean = fit_ridge(X_train, train_coeff, alpha)
            coeff_pred = predict_ridge(X_val, weights, coeff_mean)
            pred = profile_mean + coeff_pred @ basis
            candidates.append(
                {
                    "method": "eof",
                    "alpha": alpha,
                    "n_eof": int(basis.shape[0]),
                    "val_rmse": float(np.sqrt(np.mean((pred - y_val) ** 2))),
                    "weights": weights,
                    "y_mean": coeff_mean,
                    "basis": basis,
                    "profile_mean": profile_mean,
                }
            )

    best = min(candidates, key=lambda item: item["val_rmse"])
    serializable = [
        {key: value for key, value in item.items() if key not in {"weights", "y_mean", "basis", "profile_mean"}}
        for item in candidates
    ]
    return best, np.asarray(serializable, dtype=object)


def predict_selected(model: dict, X: np.ndarray) -> np.ndarray:
    latent = predict_ridge(X, model["weights"], model["y_mean"])
    if model["method"] == "direct":
        return latent
    return model["profile_mean"] + latent @ model["basis"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grib", type=Path, required=True)
    parser.add_argument("--obs-json", type=Path, default=PROJECT_ROOT / "data" / "chengdu_obs_bt.json")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "chengdu_era5_ridge")
    parser.add_argument("--site-altitude-m", type=float, default=548.0)
    parser.add_argument(
        "--channel-set", choices=["all21", "hatpro14", "high7", "vapor183"], default="all21"
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = load_observations(args.obs_json)
    heights = np.asarray(config.HEIGHT_GRID, dtype=np.float32)
    profiles, audit = load_era5_profiles(
        args.grib, {record["datetime"] for record in records}, heights, args.site_altitude_m
    )
    data = build_exact_dataset(records, profiles)
    channel_indices = {
        "all21": list(range(21)),
        "hatpro14": config.CHENGDU_HATPRO14_IDX,
        "high7": config.CHENGDU_HIGH7_IDX,
        "vapor183": config.CHENGDU_183GHZ_IDX,
    }[args.channel_set]
    data["X"] = data["X"][:, channel_indices].astype(np.float64)
    train_mask, val_mask, test_mask, split = chronological_date_split(data["dates"])

    X_train, X_val, X_test, x_mean, x_std = standardize(
        data["X"][train_mask], data["X"][val_mask], data["X"][test_mask]
    )
    predictions = {}
    selections = {}
    searches = {}
    fitted_models = {}
    for variable in ["T", "RH"]:
        y = data[variable].astype(np.float64)
        best, search = select_model(X_train, y[train_mask], X_val, y[val_mask])
        pred = predict_selected(best, X_test)
        if variable == "T":
            pred = np.clip(pred, 180.0, 330.0)
        else:
            pred = np.clip(pred, 0.0, 100.0)
        predictions[variable] = pred.astype(np.float32)
        fitted_models[variable] = best
        selections[variable] = {
            "method": best["method"],
            "alpha": best["alpha"],
            "n_eof": best["n_eof"],
            "val_rmse": best["val_rmse"],
        }
        searches[variable] = search.tolist()

    t_true = data["T"][test_mask]
    rh_true = data["RH"][test_mask]
    test_dates = data["dates"][test_mask]
    base_t = np.repeat(data["T"][train_mask].mean(axis=0, keepdims=True), len(t_true), axis=0)
    base_rh = np.repeat(data["RH"][train_mask].mean(axis=0, keepdims=True), len(rh_true), axis=0)

    metrics = {
        "ridge": {
            "T": rmse_bias(predictions["T"], t_true),
            "RH": rmse_bias(predictions["RH"], rh_true),
            "T_daily": date_mean_metrics(predictions["T"], t_true, test_dates),
            "RH_daily": date_mean_metrics(predictions["RH"], rh_true, test_dates),
        },
        "climatology": {
            "T": rmse_bias(base_t, t_true),
            "RH": rmse_bias(base_rh, rh_true),
        },
    }
    summary = {
        "status": "exact_hour_era5_ridge_eof",
        "channel_set": args.channel_set,
        "channel_indices": channel_indices,
        "frequencies_ghz": [config.CHENGDU_ALL_CHANNELS[i] for i in channel_indices],
        "dataset": {
            "n_exact_matches": int(len(data["X"])),
            "train_samples": int(train_mask.sum()),
            "val_samples": int(val_mask.sum()),
            "test_samples": int(test_mask.sum()),
        },
        "split": split,
        "grib_audit": audit,
        "selection": selections,
        "metrics": metrics,
    }
    (args.output_dir / "chengdu_era5_ridge_stats.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (args.output_dir / "model_search.json").write_text(
        json.dumps(searches, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        args.output_dir / "chengdu_era5_ridge_predictions.npz",
        T_pred=predictions["T"],
        RH_pred=predictions["RH"],
        T_true=t_true,
        RH_true=rh_true,
        dates=test_dates,
        timestamps=data["timestamps"][test_mask],
        heights=heights,
        X_mean=x_mean,
        X_std=x_std,
    )
    np.savez_compressed(
        args.output_dir / "chengdu_era5_ridge_model.npz",
        X_mean=x_mean,
        X_std=x_std,
        channel_indices=np.asarray(channel_indices, dtype=np.int64),
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
