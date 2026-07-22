#!/usr/bin/env python3
"""Build the 48-layer Chengdu dataset with radiosonde-based ERA5 bias correction."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from train_chengdu_brnn import (
    load_observations,
    parse_sounding,
    sounding_time_from_name,
)
from train_chengdu_era5_brnn import (
    build_exact_dataset,
    chronological_date_split,
    load_era5_profiles,
)


def build_height_grid() -> np.ndarray:
    low = np.arange(500.0, 2000.0 + 1.0, 100.0)
    high = np.arange(2250.0, 10000.0 + 1.0, 250.0)
    grid = np.concatenate([low, high]).astype(np.float32)
    if len(grid) != 48:
        raise RuntimeError(f"Expected 48 layers, got {len(grid)}")
    return grid


def smooth_profile(profile: np.ndarray) -> np.ndarray:
    padded = np.pad(profile, (1, 1), mode="edge")
    return np.convolve(padded, np.ones(3, dtype=np.float64) / 3.0, mode="valid")


def profile_metrics(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    error = pred - truth
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
    }


def collect_sounding_pairs(
    sounding_dir: Path,
    grib_path: Path,
    height_grid: np.ndarray,
    start_time: datetime,
    end_time: datetime,
    site_altitude_m: float,
) -> tuple[list[dict], dict]:
    entries = []
    for path in sorted(sounding_dir.rglob("*.txt")):
        launch_time = sounding_time_from_name(path)
        if start_time - timedelta(days=1) <= launch_time <= end_time + timedelta(days=1):
            entries.append((launch_time, path))

    requested_times = {
        candidate
        for launch_time, _ in entries
        for candidate in (
            launch_time - timedelta(hours=1),
            launch_time,
            launch_time + timedelta(hours=1),
        )
    }
    era5_profiles, era5_audit = load_era5_profiles(
        grib_path, requested_times, height_grid, site_altitude_m
    )

    pairs = []
    skipped = {}
    for launch_time, path in entries:
        candidates = [
            candidate
            for candidate in (
                launch_time - timedelta(hours=1),
                launch_time,
                launch_time + timedelta(hours=1),
            )
            if candidate in era5_profiles
        ]
        if not candidates:
            skipped[path.name] = "no ERA5 profile within one hour"
            continue
        try:
            sounding_t, sounding_rh, _ = parse_sounding(path, height_grid)
        except ValueError as exc:
            skipped[path.name] = str(exc)
            continue

        nearest_seconds = min(abs((candidate - launch_time).total_seconds()) for candidate in candidates)
        used_times = [
            candidate
            for candidate in candidates
            if abs((candidate - launch_time).total_seconds()) == nearest_seconds
        ]
        era5_t = np.mean([era5_profiles[item]["T"] for item in used_times], axis=0)
        era5_rh = np.mean([era5_profiles[item]["RH"] for item in used_times], axis=0)
        pairs.append(
            {
                "launch_time": launch_time,
                "source_times": used_times,
                "sounding_T": sounding_t,
                "sounding_RH": sounding_rh,
                "era5_T": era5_t.astype(np.float32),
                "era5_RH": era5_rh.astype(np.float32),
            }
        )

    audit = {
        "candidate_sounding_files": len(entries),
        "valid_pairs": len(pairs),
        "skipped": skipped,
        "era5": era5_audit,
    }
    return pairs, audit


def evaluate_pairs(pairs: list[dict], bias_t: np.ndarray, bias_rh: np.ndarray) -> dict:
    if not pairs:
        return {"n_pairs": 0}
    sounding_t = np.stack([item["sounding_T"] for item in pairs])
    sounding_rh = np.stack([item["sounding_RH"] for item in pairs])
    era5_t = np.stack([item["era5_T"] for item in pairs])
    era5_rh = np.stack([item["era5_RH"] for item in pairs])
    corrected_t = np.clip(era5_t + bias_t, 180.0, 330.0)
    corrected_rh = np.clip(era5_rh + bias_rh, 0.0, 100.0)
    return {
        "n_pairs": len(pairs),
        "raw": {
            "T": profile_metrics(era5_t, sounding_t),
            "RH": profile_metrics(era5_rh, sounding_rh),
        },
        "corrected": {
            "T": profile_metrics(corrected_t, sounding_t),
            "RH": profile_metrics(corrected_rh, sounding_rh),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build corrected Chengdu 48-layer dataset")
    parser.add_argument("--grib", type=Path, required=True)
    parser.add_argument("--sounding-dir", type=Path, required=True)
    parser.add_argument(
        "--obs-json", type=Path, default=PROJECT_ROOT / "data" / "chengdu_obs_bt.json"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "chengdu_era5_corrected48_dataset",
    )
    parser.add_argument("--site-altitude-m", type=float, default=548.0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    heights = build_height_grid()
    records = load_observations(args.obs_json)
    requested_times = {record["datetime"] for record in records}
    era5_profiles, observation_era5_audit = load_era5_profiles(
        args.grib, requested_times, heights, args.site_altitude_m
    )
    data = build_exact_dataset(records, era5_profiles)
    train_mask, val_mask, test_mask, split = chronological_date_split(data["dates"])

    pairs, sounding_audit = collect_sounding_pairs(
        args.sounding_dir,
        args.grib,
        heights,
        min(record["datetime"] for record in records),
        max(record["datetime"] for record in records),
        args.site_altitude_m,
    )
    split_dates = {name: set(split[f"{name}_dates"]) for name in ("train", "val", "test")}
    pair_splits = {
        name: [
            item
            for item in pairs
            if item["launch_time"].strftime("%Y%m%d") in dates
        ]
        for name, dates in split_dates.items()
    }
    if len(pair_splits["train"]) < 10:
        raise RuntimeError(
            f"Only {len(pair_splits['train'])} training-date sounding pairs; bias correction is unstable"
        )

    train_delta_t = np.stack(
        [item["sounding_T"] - item["era5_T"] for item in pair_splits["train"]]
    )
    train_delta_rh = np.stack(
        [item["sounding_RH"] - item["era5_RH"] for item in pair_splits["train"]]
    )
    bias_t = smooth_profile(train_delta_t.mean(axis=0)).astype(np.float32)
    bias_rh = smooth_profile(train_delta_rh.mean(axis=0)).astype(np.float32)

    corrected_t = np.clip(data["T"] + bias_t, 180.0, 330.0).astype(np.float32)
    corrected_rh = np.clip(data["RH"] + bias_rh, 0.0, 100.0).astype(np.float32)

    sounding_evaluation = {
        name: evaluate_pairs(items, bias_t, bias_rh) for name, items in pair_splits.items()
    }
    summary = {
        "status": "chengdu_48layer_training_only_radiosonde_bias_correction",
        "height_grid": {
            "n_layers": len(heights),
            "values_m": heights.tolist(),
            "rule": "500-2000 m every 100 m; 2250-10000 m every 250 m",
        },
        "bias_correction": {
            "definition": "smoothed mean(sounding - ERA5) by height",
            "vertical_smoothing": "three-point moving average with edge replication",
            "training_sounding_pairs": len(pair_splits["train"]),
            "T_bias_profile_K": bias_t.tolist(),
            "RH_bias_profile_pct": bias_rh.tolist(),
            "T_bias_mean_K": float(bias_t.mean()),
            "RH_bias_mean_pct": float(bias_rh.mean()),
            "no_validation_or_test_soundings_used_for_estimation": True,
        },
        "dataset": {
            "n_exact_matches": len(data["X"]),
            "train_samples": int(train_mask.sum()),
            "val_samples": int(val_mask.sum()),
            "test_samples": int(test_mask.sum()),
        },
        "split": split,
        "sounding_evaluation": sounding_evaluation,
        "observation_era5_audit": observation_era5_audit,
        "sounding_audit": sounding_audit,
    }

    np.savez_compressed(
        args.output_dir / "chengdu_era5_corrected48_dataset.npz",
        X=data["X"],
        T_raw=data["T"],
        RH_raw=data["RH"],
        T=corrected_t,
        RH=corrected_rh,
        dates=data["dates"],
        timestamps=data["timestamps"],
        heights=heights,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        bias_T=bias_t,
        bias_RH=bias_rh,
    )
    (args.output_dir / "chengdu_era5_corrected48_dataset_stats.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
