#!/usr/bin/env python3
"""Combine corrected 48-layer Ridge temperature and BRNN humidity predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def metrics(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    error = pred - truth
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build corrected 48-layer hybrid product")
    parser.add_argument("--ridge-prediction", type=Path, required=True)
    parser.add_argument("--brnn-prediction", type=Path, action="append", required=True)
    parser.add_argument("--brnn-stats", type=Path, action="append")
    parser.add_argument("--allow-mixed-targets", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ridge = np.load(args.ridge_prediction)
    brnn_runs = [np.load(path) for path in args.brnn_prediction]
    for run in brnn_runs:
        for key in ["RH_true", "dates", "timestamps", "heights"]:
            if not np.array_equal(brnn_runs[0][key], run[key]):
                raise ValueError(f"BRNN runs do not share {key}")
    ridge_keys = ["dates", "timestamps", "heights"]
    if not args.allow_mixed_targets:
        ridge_keys.insert(0, "T_true")
    for key in ridge_keys:
        if not np.array_equal(ridge[key], brnn_runs[0][key]):
            raise ValueError(f"Ridge and BRNN products do not share {key}")

    t_pred = ridge["T_pred"].astype(np.float32)
    ensemble_selection = None
    if args.brnn_stats:
        if len(args.brnn_stats) != len(brnn_runs):
            raise ValueError("--brnn-stats count must match --brnn-prediction count")
        stats = [json.loads(path.read_text(encoding="utf-8")) for path in args.brnn_stats]
        rh_pred = np.zeros_like(brnn_runs[0]["RH_pred"], dtype=np.float32)
        heights = ridge["heights"]
        ensemble_selection = {}
        rh_definitions = [
            item
            for item in stats[0]["configuration"]["model_definitions"]
            if item[1] == "RH"
        ]
        for name, _, height_range in rh_definitions:
            losses = np.asarray([item["training"][name]["best_val"] for item in stats])
            selected = np.argsort(losses)[:2]
            mask = (heights >= height_range[0] * 1000.0) & (heights <= height_range[1] * 1000.0)
            rh_pred[:, mask] = np.mean(
                [brnn_runs[index]["RH_pred"][:, mask] for index in selected], axis=0
            )
            ensemble_selection[name] = [
                {
                    "prediction": str(args.brnn_prediction[index]),
                    "validation_loss": float(losses[index]),
                }
                for index in selected
            ]
    else:
        rh_pred = np.mean([run["RH_pred"] for run in brnn_runs], axis=0).astype(np.float32)
    t_true = ridge["T_true"]
    rh_true = brnn_runs[0]["RH_true"]
    t_raw = ridge["T_raw"]
    rh_raw = brnn_runs[0]["RH_raw"]
    summary = {
        "status": "best_current_chengdu_corrected48_hybrid",
        "temperature_model": "all21 Ridge/EOF",
        "humidity_model": (
            "all21 BRNN segment-wise top-2 ensemble selected by validation loss"
            if ensemble_selection is not None
            else f"all21 BRNN ensemble ({len(brnn_runs)} seeds)"
        ),
        "height_layers": int(len(ridge["heights"])),
        "mixed_targets": bool(args.allow_mixed_targets),
        "n_test_samples": int(len(t_true)),
        "test_dates": sorted(set(str(item) for item in ridge["dates"])),
        "metrics_corrected_label": {
            "T": metrics(t_pred, t_true),
            "RH": metrics(rh_pred, rh_true),
        },
        "metrics_raw_era5_reference": {
            "T": metrics(t_pred, t_raw),
            "RH": metrics(rh_pred, rh_raw),
        },
        "ridge_prediction_source": str(args.ridge_prediction),
        "brnn_prediction_sources": [str(path) for path in args.brnn_prediction],
        "brnn_ensemble_selection": ensemble_selection,
    }
    (args.output_dir / "chengdu_corrected48_hybrid_stats.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    np.savez_compressed(
        args.output_dir / "chengdu_corrected48_hybrid_predictions.npz",
        T_pred=t_pred,
        RH_pred=rh_pred,
        T_true=t_true,
        RH_true=rh_true,
        T_raw=t_raw,
        RH_raw=rh_raw,
        dates=ridge["dates"],
        timestamps=ridge["timestamps"],
        heights=ridge["heights"],
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
