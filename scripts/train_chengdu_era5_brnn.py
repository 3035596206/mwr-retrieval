#!/usr/bin/env python3
"""Train the Chengdu BRNN with exact-hour ERA5 profile targets.

This pipeline reads the user-provided GRIB file directly with ecCodes, matches
the 21-channel observations at the same UTC hour, interpolates ERA5 T/RH to the
project 93-layer AGL grid, and splits samples by calendar date to reduce leakage
from highly autocorrelated adjacent hours.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from eccodes import codes_get, codes_get_values, codes_grib_new_from_file, codes_release


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import config
from train_chengdu_brnn import (
    MODEL_DEFS,
    load_observations,
    predict_full,
    rmse_bias,
    set_seed,
    train_one_model,
)


G = 9.80665
REQUIRED_VARS = {"z", "t", "r"}


def grib_datetime(gid) -> datetime:
    date = int(codes_get(gid, "dataDate"))
    time = int(codes_get(gid, "dataTime"))
    return datetime.strptime(f"{date:08d}{time:04d}", "%Y%m%d%H%M")


def load_era5_profiles(
    grib_path: Path,
    requested_times: set[datetime],
    target_height_m: np.ndarray,
    site_altitude_m: float,
) -> tuple[dict[datetime, dict], dict]:
    raw: dict[datetime, dict[str, dict[float, float]]] = {}
    message_count = 0

    with grib_path.open("rb") as handle:
        while True:
            gid = codes_grib_new_from_file(handle)
            if gid is None:
                break
            message_count += 1
            try:
                valid_time = grib_datetime(gid)
                if valid_time not in requested_times:
                    continue
                short_name = str(codes_get(gid, "shortName"))
                if short_name not in {"z", "t", "r", "crwc"}:
                    continue
                if str(codes_get(gid, "typeOfLevel")) != "isobaricInhPa":
                    continue
                level = float(codes_get(gid, "level"))
                values = np.asarray(codes_get_values(gid), dtype=np.float64)
                if values.size != 1:
                    raise ValueError(f"Expected a 1x1 ERA5 grid, got {values.size} values")
                raw.setdefault(valid_time, {}).setdefault(short_name, {})[level] = float(values[0])
            finally:
                codes_release(gid)

    profiles = {}
    rejected = {}
    for valid_time, variables in raw.items():
        if not REQUIRED_VARS.issubset(variables):
            rejected[valid_time.isoformat()] = f"missing variables: {sorted(REQUIRED_VARS - set(variables))}"
            continue
        common_levels = sorted(
            set(variables["z"]) & set(variables["t"]) & set(variables["r"]), reverse=True
        )
        if len(common_levels) < 20:
            rejected[valid_time.isoformat()] = f"only {len(common_levels)} common pressure levels"
            continue

        geopotential = np.asarray([variables["z"][level] for level in common_levels])
        temperature = np.asarray([variables["t"][level] for level in common_levels])
        humidity = np.asarray([variables["r"][level] for level in common_levels])
        height_agl = geopotential / G - site_altitude_m

        valid = (
            np.isfinite(height_agl)
            & np.isfinite(temperature)
            & np.isfinite(humidity)
            & (temperature >= 180.0)
            & (temperature <= 330.0)
            & (humidity >= 0.0)
            & (humidity <= 100.0)
        )
        height_agl = height_agl[valid]
        temperature = temperature[valid]
        humidity = humidity[valid]
        if len(height_agl) < 15:
            rejected[valid_time.isoformat()] = "too few physically valid levels"
            continue

        order = np.argsort(height_agl)
        height_agl = height_agl[order]
        temperature = temperature[order]
        humidity = humidity[order]
        keep = np.concatenate(([True], np.diff(height_agl) > 1.0))
        height_agl = height_agl[keep]
        temperature = temperature[keep]
        humidity = humidity[keep]
        if height_agl[-1] < target_height_m[-1]:
            rejected[valid_time.isoformat()] = "profile does not reach 10 km AGL"
            continue

        t_profile = np.interp(target_height_m, height_agl, temperature).astype(np.float32)
        rh_profile = np.interp(target_height_m, height_agl, humidity).astype(np.float32)
        profiles[valid_time] = {
            "T": t_profile,
            "RH": rh_profile,
            "pressure_levels_hpa": common_levels,
            "lowest_height_agl_m": float(height_agl[0]),
            "highest_height_agl_m": float(height_agl[-1]),
        }

    audit = {
        "grib_messages_scanned": message_count,
        "requested_times": len(requested_times),
        "raw_matching_times": len(raw),
        "valid_profiles": len(profiles),
        "rejected": rejected,
        "site_altitude_m": site_altitude_m,
    }
    return profiles, audit


def build_exact_dataset(records: list[dict], profiles: dict[datetime, dict]) -> dict:
    X = []
    T = []
    RH = []
    timestamps = []
    dates = []
    for record in records:
        valid_time = record["datetime"]
        if valid_time not in profiles:
            continue
        X.append(record["channels"])
        T.append(profiles[valid_time]["T"])
        RH.append(profiles[valid_time]["RH"])
        timestamps.append(record["timestamp"])
        dates.append(valid_time.strftime("%Y%m%d"))
    return {
        "X": np.asarray(X, dtype=np.float32),
        "T": np.asarray(T, dtype=np.float32),
        "RH": np.asarray(RH, dtype=np.float32),
        "timestamps": np.asarray(timestamps),
        "dates": np.asarray(dates),
    }


def chronological_date_split(dates: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    unique_dates = np.unique(dates)
    n_dates = len(unique_dates)
    n_train = max(1, int(n_dates * 0.70))
    n_val = max(1, int(n_dates * 0.15))
    if n_train + n_val >= n_dates:
        n_val = max(1, n_dates - n_train - 1)

    train_dates = set(unique_dates[:n_train])
    val_dates = set(unique_dates[n_train : n_train + n_val])
    test_dates = set(unique_dates[n_train + n_val :])
    train_mask = np.asarray([date in train_dates for date in dates])
    val_mask = np.asarray([date in val_dates for date in dates])
    test_mask = np.asarray([date in test_dates for date in dates])
    split = {
        "train_dates": sorted(train_dates),
        "val_dates": sorted(val_dates),
        "test_dates": sorted(test_dates),
    }
    return train_mask, val_mask, test_mask, split


def date_mean_metrics(pred: np.ndarray, truth: np.ndarray, dates: np.ndarray) -> dict:
    per_date = []
    for date in np.unique(dates):
        mask = dates == date
        error = pred[mask] - truth[mask]
        per_date.append(
            {
                "date": str(date),
                "n": int(mask.sum()),
                "rmse": float(np.sqrt(np.mean(error**2))),
                "mae": float(np.mean(np.abs(error))),
                "bias": float(np.mean(error)),
            }
        )
    return {
        "n_dates": len(per_date),
        "mean_daily_rmse": float(np.mean([item["rmse"] for item in per_date])),
        "std_daily_rmse": float(np.std([item["rmse"] for item in per_date])),
        "per_date": per_date,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Chengdu exact-hour ERA5 BRNN training")
    parser.add_argument("--grib", type=Path, required=True)
    parser.add_argument("--obs-json", type=Path, default=PROJECT_ROOT / "data" / "chengdu_obs_bt.json")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "chengdu_era5_brnn")
    parser.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "models_chengdu_era5_brnn")
    parser.add_argument("--site-altitude-m", type=float, default=548.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.4)
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

    records = load_observations(args.obs_json)
    requested_times = {record["datetime"] for record in records}
    heights = np.asarray(config.HEIGHT_GRID, dtype=np.float32)
    profiles, grib_audit = load_era5_profiles(
        args.grib, requested_times, heights, args.site_altitude_m
    )
    data = build_exact_dataset(records, profiles)

    channel_indices = {
        "all21": list(range(21)),
        "hatpro14": config.CHENGDU_HATPRO14_IDX,
        "high7": config.CHENGDU_HIGH7_IDX,
        "vapor183": config.CHENGDU_183GHZ_IDX,
    }[args.channel_set]
    data["X"] = data["X"][:, channel_indices]
    if len(data["X"]) < 80 or len(np.unique(data["dates"])) < 10:
        raise RuntimeError(
            f"Insufficient exact ERA5 matches: {len(data['X'])} samples, "
            f"{len(np.unique(data['dates']))} dates"
        )

    train_mask, val_mask, test_mask, split = chronological_date_split(data["dates"])
    print(
        f"Exact ERA5 matches={len(data['X'])}, dates={len(np.unique(data['dates']))}; "
        f"train={train_mask.sum()}, val={val_mask.sum()}, test={test_mask.sum()}"
    )
    print("Split:", split)

    models = {}
    training = {}
    for name, variable, height_range in MODEL_DEFS:
        print(f"Training {name} ({args.channel_set})...")
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
            "cpu",
        )
        models[name] = model
        training[name] = details
        torch.save(model.state_dict(), args.model_dir / f"{name}.pt")
        print(f"  best epoch={details['best_epoch']}, val={details['best_val']:.6f}")

    t_pred, rh_pred = predict_full(models, data["X"][test_mask], "cpu")
    t_true = data["T"][test_mask]
    rh_true = data["RH"][test_mask]
    test_dates = data["dates"][test_mask]

    t_mean = data["T"][train_mask].mean(axis=0, keepdims=True)
    rh_mean = data["RH"][train_mask].mean(axis=0, keepdims=True)
    baseline_t = np.repeat(t_mean, len(t_true), axis=0)
    baseline_rh = np.repeat(rh_mean, len(rh_true), axis=0)

    metrics = {
        "model": {
            "T": rmse_bias(t_pred, t_true),
            "RH": rmse_bias(rh_pred, rh_true),
            "T_daily": date_mean_metrics(t_pred, t_true, test_dates),
            "RH_daily": date_mean_metrics(rh_pred, rh_true, test_dates),
        },
        "climatology": {
            "T": rmse_bias(baseline_t, t_true),
            "RH": rmse_bias(baseline_rh, rh_true),
            "T_daily": date_mean_metrics(baseline_t, t_true, test_dates),
            "RH_daily": date_mean_metrics(baseline_rh, rh_true, test_dates),
        },
    }

    height_metrics = []
    for height_km in [0, 0.5, 1, 2, 3, 5, 8, 10]:
        idx = int(np.argmin(np.abs(heights - height_km * 1000.0)))
        height_metrics.append(
            {
                "height_km": float(heights[idx] / 1000.0),
                "T_rmse": float(np.sqrt(np.mean((t_pred[:, idx] - t_true[:, idx]) ** 2))),
                "T_bias": float(np.mean(t_pred[:, idx] - t_true[:, idx])),
                "RH_rmse": float(np.sqrt(np.mean((rh_pred[:, idx] - rh_true[:, idx]) ** 2))),
                "RH_bias": float(np.mean(rh_pred[:, idx] - rh_true[:, idx])),
            }
        )

    summary = {
        "status": "exact_hour_era5_training",
        "configuration": {
            "grib": str(args.grib),
            "site_altitude_m": args.site_altitude_m,
            "seed": args.seed,
            "hidden_size": args.hidden_size,
            "dropout": args.dropout,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "channel_set": args.channel_set,
            "channel_indices": channel_indices,
            "channel_frequencies_ghz": [config.CHENGDU_ALL_CHANNELS[i] for i in channel_indices],
        },
        "grib_audit": grib_audit,
        "dataset": {
            "n_observations_total": len(records),
            "n_exact_matches": int(len(data["X"])),
            "n_dates": int(len(np.unique(data["dates"]))),
            "train_samples": int(train_mask.sum()),
            "val_samples": int(val_mask.sum()),
            "test_samples": int(test_mask.sum()),
        },
        "split": split,
        "metrics": metrics,
        "height_metrics": height_metrics,
        "training": {
            name: {"best_epoch": item["best_epoch"], "best_val": item["best_val"]}
            for name, item in training.items()
        },
    }
    (args.output_dir / "chengdu_era5_brnn_stats.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (args.output_dir / "training_history.json").write_text(
        json.dumps(training, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        args.output_dir / "chengdu_era5_brnn_predictions.npz",
        T_pred=t_pred,
        RH_pred=rh_pred,
        T_true=t_true,
        RH_true=rh_true,
        dates=test_dates,
        timestamps=data["timestamps"][test_mask],
        heights=heights,
        X_test=data["X"][test_mask],
    )
    print(json.dumps(metrics, indent=2))
    print(f"Results: {args.output_dir}")
    print(f"Models:  {args.model_dir}")


if __name__ == "__main__":
    main()
