#!/usr/bin/env python3
"""Build the best current Chengdu hybrid test product.

Temperature comes from the all-channel Ridge/EOF model. Relative humidity is
the mean prediction of four all-channel BRNN initializations.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np


def metrics(pred: np.ndarray, truth: np.ndarray) -> dict:
    error = pred - truth
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results_root = args.project_root / "results"
    ridge_path = results_root / "chengdu_era5_ridge_all21" / "chengdu_era5_ridge_predictions.npz"
    brnn_paths = [
        results_root / "chengdu_era5_all21" / "chengdu_era5_brnn_predictions.npz"
    ] + [Path(path) for path in sorted(glob.glob(str(results_root / "chengdu_era5_all21_seed*" / "chengdu_era5_brnn_predictions.npz")))]

    ridge = np.load(ridge_path)
    brnn_runs = [np.load(path) for path in brnn_paths]
    for run in brnn_runs:
        for key in ["RH_true", "dates", "timestamps", "heights"]:
            if not np.array_equal(brnn_runs[0][key], run[key]):
                raise ValueError(f"BRNN runs do not share {key}")
    for key in ["T_true", "dates", "timestamps", "heights"]:
        if not np.array_equal(ridge[key], brnn_runs[0][key]):
            raise ValueError(f"Ridge and BRNN products do not share {key}")

    t_pred = ridge["T_pred"]
    rh_pred = np.mean([run["RH_pred"] for run in brnn_runs], axis=0).astype(np.float32)
    t_true = ridge["T_true"]
    rh_true = brnn_runs[0]["RH_true"]

    summary = {
        "status": "best_current_chengdu_hybrid",
        "temperature_model": "all21 Ridge + 5 EOF",
        "humidity_model": f"all21 BRNN ensemble ({len(brnn_runs)} seeds)",
        "n_test_samples": int(len(t_true)),
        "test_dates": sorted(set(str(item) for item in ridge["dates"])),
        "metrics": {"T": metrics(t_pred, t_true), "RH": metrics(rh_pred, rh_true)},
        "ridge_prediction_source": str(ridge_path),
        "brnn_prediction_sources": [str(path) for path in brnn_paths],
        "deployment_components": {
            "T_model": str(results_root / "chengdu_era5_ridge_all21" / "chengdu_era5_ridge_model.npz"),
            "RH_model_dirs": [
                str(args.project_root / "models_chengdu_era5_all21"),
                str(args.project_root / "models_chengdu_era5_all21_seed1"),
                str(args.project_root / "models_chengdu_era5_all21_seed7"),
                str(args.project_root / "models_chengdu_era5_all21_seed21"),
            ],
        },
    }
    (args.output_dir / "chengdu_era5_hybrid_stats.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    np.savez_compressed(
        args.output_dir / "chengdu_era5_hybrid_predictions.npz",
        T_pred=t_pred,
        RH_pred=rh_pred,
        T_true=t_true,
        RH_true=rh_true,
        dates=ridge["dates"],
        timestamps=ridge["timestamps"],
        heights=ridge["heights"],
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
