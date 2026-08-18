#!/usr/bin/env python3
"""Build a physical 48-layer Chengdu dataset using T/log(q)/log(P) layer means."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from eccodes import codes_get, codes_get_values, codes_grib_new_from_file, codes_release


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from train_chengdu_brnn import load_observations, sounding_time_from_name
from train_chengdu_era5_brnn import chronological_date_split
from mwr_retrieval.grids import build_layer48_grid, layer_average
from mwr_retrieval.thermodynamics import (
    Q_FLOOR,
    rh_to_specific_humidity,
    saturation_vapor_pressure_hpa,
    specific_humidity_to_rh,
)


G = 9.80665


def build_layer_grid() -> tuple[np.ndarray, np.ndarray]:
    return build_layer48_grid()


def saturation_vapor_pressure_hpa(temperature_k: np.ndarray) -> np.ndarray:
    temperature_c = np.asarray(temperature_k, dtype=np.float64) - 273.15
    water = 6.112 * np.exp(17.62 * temperature_c / (243.12 + temperature_c))
    ice = 6.112 * np.exp(22.46 * temperature_c / (272.62 + temperature_c))
    blend = np.clip((temperature_c + 20.0) / 20.0, 0.0, 1.0)
    return ice * (1.0 - blend) + water * blend


def rh_to_specific_humidity(
    temperature_k: np.ndarray, relative_humidity: np.ndarray, pressure_hpa: np.ndarray
) -> np.ndarray:
    es = saturation_vapor_pressure_hpa(temperature_k)
    vapor_pressure = np.clip(relative_humidity / 100.0 * es, 0.0, pressure_hpa * 0.99)
    mixing_ratio = EPSILON * vapor_pressure / np.maximum(pressure_hpa - vapor_pressure, 1e-6)
    return np.maximum(mixing_ratio / (1.0 + mixing_ratio), Q_FLOOR)


def specific_humidity_to_rh(
    temperature_k: np.ndarray, specific_humidity: np.ndarray, pressure_hpa: np.ndarray
) -> np.ndarray:
    q = np.clip(specific_humidity, Q_FLOOR, 0.2)
    mixing_ratio = q / np.maximum(1.0 - q, 1e-8)
    vapor_pressure = mixing_ratio * pressure_hpa / (EPSILON + mixing_ratio)
    rh = 100.0 * vapor_pressure / np.maximum(saturation_vapor_pressure_hpa(temperature_k), 1e-8)
    return np.clip(rh, 0.0, 100.0)


def layer_average_from_interpolated(
    source_height: np.ndarray, source_value: np.ndarray, edges: np.ndarray
) -> np.ndarray:
    return layer_average(source_height, source_value, edges)


def raw_profile_to_layers(
    height_agl: np.ndarray,
    temperature_k: np.ndarray,
    relative_humidity: np.ndarray,
    pressure_hpa: np.ndarray,
    edges: np.ndarray,
) -> dict[str, np.ndarray]:
    valid = (
        np.isfinite(height_agl)
        & np.isfinite(temperature_k)
        & np.isfinite(relative_humidity)
        & np.isfinite(pressure_hpa)
        & (height_agl >= 0.0)
        & (temperature_k >= 180.0)
        & (temperature_k <= 330.0)
        & (relative_humidity >= 0.0)
        & (relative_humidity <= 100.0)
        & (pressure_hpa >= 1.0)
        & (pressure_hpa <= 1100.0)
    )
    z = np.asarray(height_agl[valid], dtype=np.float64)
    t = np.asarray(temperature_k[valid], dtype=np.float64)
    rh = np.asarray(relative_humidity[valid], dtype=np.float64)
    p = np.asarray(pressure_hpa[valid], dtype=np.float64)
    if len(z) < 15:
        raise ValueError("too few valid above-ground profile levels")
    order = np.argsort(z)
    z, t, rh, p = z[order], t[order], rh[order], p[order]
    keep = np.concatenate(([True], np.diff(z) > 0.5))
    z, t, rh, p = z[keep], t[keep], rh[keep], p[keep]
    if z[-1] < edges[-1]:
        raise ValueError("profile does not reach 10 km AGL")

    # Anchor the profile at ground with the lowest valid above-ground level.
    if z[0] > 0.0:
        z = np.concatenate(([0.0], z))
        t = np.concatenate(([t[0]], t))
        rh = np.concatenate(([rh[0]], rh))
        p = np.concatenate(([p[0]], p))

    q = rh_to_specific_humidity(t, rh, p)
    t_layer = layer_average_from_interpolated(z, t, edges)
    log_q_layer = layer_average_from_interpolated(z, np.log(np.maximum(q, Q_FLOOR)), edges)
    log_p_layer = layer_average_from_interpolated(z, np.log(p), edges)
    q_layer = np.exp(log_q_layer).astype(np.float32)
    p_layer = np.exp(log_p_layer).astype(np.float32)
    rh_layer = specific_humidity_to_rh(t_layer, q_layer, p_layer).astype(np.float32)
    return {"T": t_layer, "q": q_layer, "logq": log_q_layer, "P": p_layer, "RH": rh_layer}


def grib_datetime(gid) -> datetime:
    date = int(codes_get(gid, "dataDate"))
    time = int(codes_get(gid, "dataTime"))
    return datetime.strptime(f"{date:08d}{time:04d}", "%Y%m%d%H%M")


def load_era5_layer_profiles(
    grib_path: Path,
    requested_times: set[datetime],
    edges: np.ndarray,
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
                if str(codes_get(gid, "typeOfLevel")) != "isobaricInhPa":
                    continue
                variable = str(codes_get(gid, "shortName"))
                if variable not in {"z", "t", "r"}:
                    continue
                level = float(codes_get(gid, "level"))
                values = np.asarray(codes_get_values(gid), dtype=np.float64)
                if values.size != 1:
                    raise ValueError(f"Expected one ERA5 grid value, got {values.size}")
                raw.setdefault(valid_time, {}).setdefault(variable, {})[level] = float(values[0])
            finally:
                codes_release(gid)

    profiles = {}
    rejected = {}
    below_ground_counts = []
    for valid_time, variables in raw.items():
        if not {"z", "t", "r"}.issubset(variables):
            rejected[valid_time.isoformat()] = "missing z/t/r"
            continue
        levels = sorted(set(variables["z"]) & set(variables["t"]) & set(variables["r"]), reverse=True)
        height_agl = np.asarray([variables["z"][level] / G - site_altitude_m for level in levels])
        temperature = np.asarray([variables["t"][level] for level in levels])
        humidity = np.asarray([variables["r"][level] for level in levels])
        pressure = np.asarray(levels, dtype=np.float64)
        below_ground_counts.append(int(np.sum(height_agl < 0.0)))
        try:
            profiles[valid_time] = raw_profile_to_layers(
                height_agl, temperature, humidity, pressure, edges
            )
        except ValueError as exc:
            rejected[valid_time.isoformat()] = str(exc)
    audit = {
        "grib_messages_scanned": message_count,
        "requested_times": len(requested_times),
        "raw_matching_times": len(raw),
        "valid_profiles": len(profiles),
        "rejected": rejected,
        "mean_below_ground_pressure_levels_removed": float(np.mean(below_ground_counts))
        if below_ground_counts
        else 0.0,
        "surface_anchor": "lowest valid above-ground pressure level copied to 0 m AGL",
    }
    return profiles, audit


def parse_sounding_layers(path: Path, edges: np.ndarray) -> dict[str, np.ndarray]:
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
                temperature_c = float(fields[6])
                pressure_hpa = float(fields[7])
                rh = float(fields[8])
                height_m = float(fields[16])
            except ValueError:
                continue
            if (
                -100.0 <= temperature_c <= 60.0
                and 1.0 <= pressure_hpa <= 1100.0
                and 0.0 <= rh <= 100.0
                and -500.0 <= height_m <= 60000.0
            ):
                heights.append(height_m)
                temperatures.append(temperature_c + 273.15)
                humidities.append(rh)
                pressures.append(pressure_hpa)
    if len(heights) < 50:
        raise ValueError("too few valid sounding levels")
    heights = np.asarray(heights, dtype=np.float64)
    station_height = float(np.nanpercentile(heights, 0.5))
    return raw_profile_to_layers(
        heights - station_height,
        np.asarray(temperatures),
        np.asarray(humidities),
        np.asarray(pressures),
        edges,
    )


def build_observation_dataset(records: list[dict], profiles: dict[datetime, dict]) -> dict:
    matched = [(record, profiles[record["datetime"]]) for record in records if record["datetime"] in profiles]
    return {
        "X": np.asarray([item[0]["channels"] for item in matched], dtype=np.float32),
        "T_raw": np.asarray([item[1]["T"] for item in matched], dtype=np.float32),
        "RH_raw": np.asarray([item[1]["RH"] for item in matched], dtype=np.float32),
        "q_raw": np.asarray([item[1]["q"] for item in matched], dtype=np.float32),
        "logq_raw": np.asarray([item[1]["logq"] for item in matched], dtype=np.float32),
        "P": np.asarray([item[1]["P"] for item in matched], dtype=np.float32),
        "dates": np.asarray([item[0]["datetime"].strftime("%Y%m%d") for item in matched]),
        "timestamps": np.asarray([item[0]["timestamp"] for item in matched]),
    }


def collect_sounding_pairs(
    sounding_dir: Path,
    grib_path: Path,
    edges: np.ndarray,
    site_altitude_m: float,
    start_time: datetime,
    end_time: datetime,
) -> tuple[list[dict], dict]:
    entries = []
    for path in sorted(sounding_dir.rglob("*.txt")):
        launch_time = sounding_time_from_name(path)
        if start_time - timedelta(days=1) <= launch_time <= end_time + timedelta(days=1):
            entries.append((launch_time, path))
    requested = {
        candidate
        for launch_time, _ in entries
        for candidate in (launch_time - timedelta(hours=1), launch_time, launch_time + timedelta(hours=1))
    }
    era5_profiles, era5_audit = load_era5_layer_profiles(
        grib_path, requested, edges, site_altitude_m
    )
    pairs = []
    skipped = {}
    for launch_time, path in entries:
        candidates = [
            candidate
            for candidate in (launch_time - timedelta(hours=1), launch_time, launch_time + timedelta(hours=1))
            if candidate in era5_profiles
        ]
        if not candidates:
            skipped[path.name] = "no ERA5 layer profile within one hour"
            continue
        try:
            sounding = parse_sounding_layers(path, edges)
        except ValueError as exc:
            skipped[path.name] = str(exc)
            continue
        nearest = min(abs((candidate - launch_time).total_seconds()) for candidate in candidates)
        used = [
            candidate
            for candidate in candidates
            if abs((candidate - launch_time).total_seconds()) == nearest
        ]
        era5 = {
            key: np.mean([era5_profiles[item][key] for item in used], axis=0).astype(np.float32)
            for key in ("T", "RH", "q", "logq", "P")
        }
        pairs.append({"launch_time": launch_time, "sounding": sounding, "era5": era5})
    return pairs, {"candidate_files": len(entries), "valid_pairs": len(pairs), "skipped": skipped, "era5": era5_audit}


def smooth_low_freedom_bias(mean_bias: np.ndarray, centers: np.ndarray) -> np.ndarray:
    kernel = np.asarray([1.0, 2.0, 3.0, 2.0, 1.0]) / 9.0
    smoothed = np.convolve(np.pad(mean_bias, (2, 2), mode="edge"), kernel, mode="valid")
    knot_heights = np.asarray([250.0, 1000.0, 2000.0, 4000.0, 6000.0, 8000.0, 9875.0])
    knot_values = np.interp(knot_heights, centers, smoothed)
    return np.interp(centers, knot_heights, knot_values).astype(np.float32)


def profile_metrics(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    error = pred - truth
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
    }


def evaluate_correction_options(
    pairs: list[dict], bias_t: np.ndarray, bias_logq: np.ndarray
) -> dict:
    if not pairs:
        return {"n_pairs": 0}
    sounding = {key: np.stack([item["sounding"][key] for item in pairs]) for key in ("T", "RH", "q", "logq", "P")}
    era5 = {key: np.stack([item["era5"][key] for item in pairs]) for key in ("T", "RH", "q", "logq", "P")}
    options = {}
    for name, use_t, use_q in [
        ("raw", False, False),
        ("T_only", True, False),
        ("logq_only", False, True),
        ("T_logq", True, True),
    ]:
        temperature = era5["T"] + (bias_t if use_t else 0.0)
        logq = era5["logq"] + (bias_logq if use_q else 0.0)
        q = np.exp(logq)
        rh = specific_humidity_to_rh(temperature, q, era5["P"])
        options[name] = {
            "T": profile_metrics(temperature, sounding["T"]),
            "RH": profile_metrics(rh, sounding["RH"]),
            "logq": profile_metrics(logq, sounding["logq"]),
        }
    return {"n_pairs": len(pairs), "options": options}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build optimized physical 48-layer dataset")
    parser.add_argument("--grib", type=Path, required=True)
    parser.add_argument("--sounding-dir", type=Path, required=True)
    parser.add_argument("--obs-json", type=Path, default=PROJECT_ROOT / "data" / "chengdu_obs_bt.json")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "chengdu_era5_layer48_dataset",
    )
    parser.add_argument("--site-altitude-m", type=float, default=548.0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    edges, centers = build_layer_grid()
    records = load_observations(args.obs_json)
    era5_profiles, era5_audit = load_era5_layer_profiles(
        args.grib, {record["datetime"] for record in records}, edges, args.site_altitude_m
    )
    data = build_observation_dataset(records, era5_profiles)
    train_mask, val_mask, test_mask, split = chronological_date_split(data["dates"])
    pairs, sounding_audit = collect_sounding_pairs(
        args.sounding_dir,
        args.grib,
        edges,
        args.site_altitude_m,
        min(record["datetime"] for record in records),
        max(record["datetime"] for record in records),
    )
    split_dates = {name: set(split[f"{name}_dates"]) for name in ("train", "val", "test")}
    pair_splits = {
        name: [item for item in pairs if item["launch_time"].strftime("%Y%m%d") in dates]
        for name, dates in split_dates.items()
    }
    if len(pair_splits["train"]) < 10:
        raise RuntimeError("Too few training-date sounding pairs for correction")

    delta_t = np.stack(
        [item["sounding"]["T"] - item["era5"]["T"] for item in pair_splits["train"]]
    )
    delta_logq = np.stack(
        [item["sounding"]["logq"] - item["era5"]["logq"] for item in pair_splits["train"]]
    )
    bias_t = np.clip(smooth_low_freedom_bias(delta_t.mean(axis=0), centers), -5.0, 5.0)
    bias_logq = np.clip(
        smooth_low_freedom_bias(delta_logq.mean(axis=0), centers), -np.log(3.0), np.log(3.0)
    )
    correction_evaluation = {
        name: evaluate_correction_options(items, bias_t, bias_logq)
        for name, items in pair_splits.items()
    }
    validation_options = correction_evaluation["val"]["options"]
    apply_t = validation_options["T_only"]["T"]["rmse"] < validation_options["raw"]["T"]["rmse"]
    q_baseline_name = "T_only" if apply_t else "raw"
    q_corrected_name = "T_logq" if apply_t else "logq_only"
    apply_logq = (
        validation_options[q_corrected_name]["RH"]["rmse"]
        < validation_options[q_baseline_name]["RH"]["rmse"]
    )

    t_corrected = data["T_raw"] + (bias_t if apply_t else 0.0)
    logq_corrected = data["logq_raw"] + (bias_logq if apply_logq else 0.0)
    q_corrected = np.exp(logq_corrected).astype(np.float32)
    rh_corrected = specific_humidity_to_rh(t_corrected, q_corrected, data["P"]).astype(np.float32)

    summary = {
        "status": "chengdu_physical_48_layer_dataset",
        "layer_grid": {
            "n_layers": 48,
            "edges_m": edges.tolist(),
            "centers_m": centers.tolist(),
            "definition": "0-500 m one layer; 500-2000 m every 100 m; 2000-10000 m every 250 m",
        },
        "processing": {
            "below_ground_pressure_levels_removed": True,
            "temperature_interpolation": "piecewise linear in height, then layer integration",
            "humidity_interpolation": "piecewise linear log(q) in height, then layer integration",
            "pressure_interpolation": "piecewise linear log(P) in height, then layer integration",
            "RH_recomputed_from_layer_T_q_P": True,
            "surface_data_available": False,
            "surface_anchor": "lowest valid above-ground ERA5 pressure level",
        },
        "bias_correction": {
            "training_sounding_pairs": len(pair_splits["train"]),
            "low_freedom_knot_heights_m": [250, 1000, 2000, 4000, 6000, 8000, 9875],
            "T_bias_K": bias_t.tolist(),
            "logq_bias": bias_logq.tolist(),
            "apply_T_selected_on_validation": bool(apply_t),
            "apply_logq_selected_on_validation": bool(apply_logq),
        },
        "dataset": {
            "n_exact_matches": len(data["X"]),
            "train_samples": int(train_mask.sum()),
            "val_samples": int(val_mask.sum()),
            "test_samples": int(test_mask.sum()),
        },
        "split": split,
        "correction_evaluation": correction_evaluation,
        "era5_audit": era5_audit,
        "sounding_audit": sounding_audit,
    }
    np.savez_compressed(
        args.output_dir / "chengdu_era5_layer48_dataset.npz",
        X=data["X"],
        T_raw=data["T_raw"],
        RH_raw=data["RH_raw"],
        q_raw=data["q_raw"],
        logq_raw=data["logq_raw"],
        T=t_corrected.astype(np.float32),
        RH=rh_corrected,
        q=q_corrected,
        logq=logq_corrected.astype(np.float32),
        P=data["P"],
        dates=data["dates"],
        timestamps=data["timestamps"],
        heights=centers,
        layer_edges=edges,
        layer_thickness=np.diff(edges).astype(np.float32),
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        bias_T=bias_t,
        bias_logq=bias_logq,
        apply_T=np.asarray(apply_t),
        apply_logq=np.asarray(apply_logq),
    )
    (args.output_dir / "chengdu_era5_layer48_dataset_stats.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
