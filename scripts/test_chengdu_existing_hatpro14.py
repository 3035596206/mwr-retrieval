#!/usr/bin/env python3
"""External-domain test of the existing HATPRO 14-channel BRNN on Chengdu."""

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
from brnn_model import BRNN, get_height_range_indices
from train_chengdu_brnn import (
    MODEL_DEFS,
    build_dataset,
    grouped_metrics,
    load_observations,
    rmse_bias,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sounding-dir", type=Path, required=True)
    parser.add_argument("--obs-json", type=Path, default=PROJECT_ROOT / "data" / "chengdu_obs_bt.json")
    parser.add_argument("--models-dir", type=Path, default=PROJECT_ROOT / "models")
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "results" / "chengdu_existing_hatpro14_test"
    )
    parser.add_argument("--max-delta-hours", type=float, default=3.0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = load_observations(args.obs_json)
    heights = np.asarray(config.HEIGHT_GRID, dtype=np.float32)
    data = build_dataset(records, args.sounding_dir, heights, args.max_delta_hours)
    bt14 = data["X"][:, config.CHENGDU_HATPRO14_IDX]
    surface = data["surface"]

    t_pred = np.zeros_like(data["T"])
    rh_pred = np.zeros_like(data["RH"])
    loaded = {}

    for name, variable, height_range in MODEL_DEFS:
        start, end = get_height_range_indices(*height_range, config.HEIGHT_GRID)
        if variable == "T" and height_range == (0, 2):
            model_input = np.column_stack([bt14[:, config.V_SURFACE_IDX], surface[:, 0]])
        else:
            model_input = np.column_stack([bt14, surface])

        state = torch.load(args.models_dir / f"{name}.pt", map_location="cpu", weights_only=True)
        model = BRNN(model_input.shape[1], end - start, config.HIDDEN_NODES, config.DROPOUT_RATE)
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            output = model(torch.from_numpy(model_input.astype(np.float32))).numpy()

        if variable == "T":
            t_pred[:, start:end] = output * 50.0 + 250.0
        else:
            rh_pred[:, start:end] = output * 100.0
        loaded[name] = {"n_input": int(model_input.shape[1]), "n_output": int(end - start)}

    metrics = {
        "sample": {"T": rmse_bias(t_pred, data["T"]), "RH": rmse_bias(rh_pred, data["RH"])},
        "grouped": grouped_metrics(t_pred, rh_pred, data["T"], data["RH"], data["groups"]),
    }
    summary = {
        "status": "external_domain_test",
        "warning": (
            "The existing model was trained on simulated/legacy HATPRO data and is evaluated "
            "without Chengdu-specific BT bias correction. Surface features come from matched soundings."
        ),
        "dataset": {
            "n_samples": int(len(bt14)),
            "n_sounding_groups": int(len(np.unique(data["groups"]))),
            "max_delta_hours": args.max_delta_hours,
        },
        "frequencies_ghz": config.ALL_CHANNELS,
        "models": loaded,
        "metrics": metrics,
    }
    (args.output_dir / "existing_hatpro14_stats.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    np.savez_compressed(
        args.output_dir / "existing_hatpro14_predictions.npz",
        T_pred=t_pred,
        RH_pred=rh_pred,
        T_true=data["T"],
        RH_true=data["RH"],
        groups=data["groups"],
        timestamps=data["timestamps"],
        heights=heights,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
