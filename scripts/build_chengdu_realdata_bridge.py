#!/usr/bin/env python3
"""Bridge Chengdu real Obs_BT records with ERA5 pressure-level labels.

This script produces a compact, auditable dataset for the current real-data
workflow:

* 21-channel Chengdu hourly observed brightness temperatures.
* CDS ERA5 pressure-level NetCDF files in the shared data lake.
* The project 48-layer vertical grid used by the Chengdu retrieval models.

The output directory contains a compressed dataset, a human-readable stats file,
and a manifest with source paths/checksums so the bridge can later be registered
in the data-lake catalog without guessing provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

try:
    import xarray as xr
except ModuleNotFoundError as error:  # pragma: no cover - environment guard
    raise SystemExit(
        "xarray is required to read CDS NetCDF files. On this workstation use: "
        "wsl -d Ubuntu-24.04 -- /home/inkp/miniconda3/envs/arts/bin/python "
        "/mnt/d/project-504/mwr-retrieval-main/scripts/build_chengdu_realdata_bridge.py"
    ) from error


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT.parent.parent / "project-504-data"
G = 9.80665
Q_FLOOR = 1e-6
EPSILON = 0.62198

sys.path.insert(0, str(PROJECT_ROOT / "src"))
from mwr_retrieval.grids import build_layer48_grid, layer_average  # noqa: E402
from mwr_retrieval.settings import load_instruments  # noqa: E402


def parse_obs_time(value: str) -> datetime:
    return datetime.strptime(value, "%Y_%m_%d %H:%M:%S")


def datetime64_to_datetime(value: np.datetime64) -> datetime:
    text = np.datetime_as_string(value, unit="s")
    return datetime.fromisoformat(text.replace("T", " "))


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_record(path: Path, *, include_hash: bool) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "mtime_utc": datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(),
    }
    if include_hash:
        record["sha256"] = sha256_file(path)
    return record


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
        raise ValueError("too few valid above-ground pressure levels")
    order = np.argsort(z)
    z, t, rh, p = z[order], t[order], rh[order], p[order]
    keep = np.concatenate(([True], np.diff(z) > 0.5))
    z, t, rh, p = z[keep], t[keep], rh[keep], p[keep]
    if z[-1] < edges[-1]:
        raise ValueError("profile does not reach 10 km AGL")
    if z[0] > 0.0:
        z = np.concatenate(([0.0], z))
        t = np.concatenate(([t[0]], t))
        rh = np.concatenate(([rh[0]], rh))
        p = np.concatenate(([p[0]], p))

    q = rh_to_specific_humidity(t, rh, p)
    t_layer = layer_average(z, t, edges).astype(np.float32)
    log_q_layer = layer_average(z, np.log(np.maximum(q, Q_FLOOR)), edges).astype(np.float32)
    log_p_layer = layer_average(z, np.log(p), edges).astype(np.float32)
    q_layer = np.exp(log_q_layer).astype(np.float32)
    p_layer = np.exp(log_p_layer).astype(np.float32)
    rh_layer = specific_humidity_to_rh(t_layer, q_layer, p_layer).astype(np.float32)
    return {"T": t_layer, "RH": rh_layer, "q": q_layer, "logq": log_q_layer, "P": p_layer}


def load_observations(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records", [])
    cleaned = []
    channel_lengths: dict[int, int] = {}
    skipped: list[dict[str, str]] = []
    for record in records:
        channels = np.asarray(record.get("channels", []), dtype=np.float32)
        channel_lengths[len(channels)] = channel_lengths.get(len(channels), 0) + 1
        if len(channels) != 21 or not np.all(np.isfinite(channels)):
            skipped.append({"timestamp": str(record.get("timestamp")), "reason": "invalid 21-channel BT vector"})
            continue
        item = dict(record)
        item["datetime"] = parse_obs_time(str(record["timestamp"]))
        item["channels"] = channels
        cleaned.append(item)
    if not cleaned:
        raise ValueError(f"No usable 21-channel observations in {path}")
    times = [record["datetime"] for record in cleaned]
    audit = {
        "input_records": len(records),
        "usable_records": len(cleaned),
        "skipped_records": skipped,
        "channel_length_histogram": {str(k): v for k, v in sorted(channel_lengths.items())},
        "first_timestamp": min(times).isoformat(),
        "last_timestamp": max(times).isoformat(),
        "source_file_count": len(payload.get("files", [])),
        "source_files": payload.get("files", []),
    }
    return cleaned, audit


def pressure_files(root: Path) -> list[Path]:
    patterns = ["year=*/month=*/part=*/era5_pressure_*.nc", "month=*/part=*/era5_pressure_*.nc", "part=*/era5_pressure_*.nc"]
    files: list[Path] = []
    for pattern in patterns:
        files.extend(root.glob(pattern))
    if not files:
        files.extend(root.rglob("era5_pressure_*.nc"))
    return sorted(set(files))


def first_present(dataset: xr.Dataset, names: tuple[str, ...]) -> str:
    for name in names:
        if name in dataset.variables or name in dataset.coords or name in dataset.dims:
            return name
    raise KeyError(f"None of {names} found in ERA5 dataset")


def extract_profile(
    dataset: xr.Dataset,
    *,
    time_name: str,
    level_name: str,
    lat_name: str,
    lon_name: str,
    time_index: int,
    lat_index: int,
    lon_index: int,
    edges: np.ndarray,
    site_altitude_m: float,
) -> dict[str, np.ndarray]:
    variables = {"t", "r", "z"}
    missing = sorted(variables - set(dataset.data_vars))
    if missing:
        raise ValueError(f"missing variables: {missing}")
    selector = {time_name: time_index, lat_name: lat_index, lon_name: lon_index}
    levels = np.asarray(dataset[level_name].values, dtype=np.float64)
    temperature = np.asarray(dataset["t"].isel(selector).values, dtype=np.float64).squeeze()
    humidity = np.asarray(dataset["r"].isel(selector).values, dtype=np.float64).squeeze()
    geopotential = np.asarray(dataset["z"].isel(selector).values, dtype=np.float64).squeeze()
    height_agl = geopotential / G - site_altitude_m
    return raw_profile_to_layers(height_agl, temperature, humidity, levels, edges)


def load_era5_profiles(
    root: Path,
    *,
    start: datetime,
    end: datetime,
    edges: np.ndarray,
    site_altitude_m: float,
    lat: float,
    lon: float,
) -> tuple[dict[datetime, dict[str, np.ndarray]], dict[str, Any]]:
    profiles: dict[datetime, dict[str, np.ndarray]] = {}
    files = pressure_files(root)
    if not files:
        raise FileNotFoundError(f"No ERA5 pressure NetCDF files below {root}")

    used_files: list[str] = []
    rejected: dict[str, str] = {}
    grid_points: list[dict[str, float]] = []
    below_ground_removed: list[int] = []
    for path in files:
        with xr.open_dataset(path, engine="netcdf4") as dataset:
            time_name = first_present(dataset, ("valid_time", "time"))
            level_name = first_present(dataset, ("pressure_level", "level"))
            lat_name = first_present(dataset, ("latitude", "lat"))
            lon_name = first_present(dataset, ("longitude", "lon"))
            lat_values = np.asarray(dataset[lat_name].values, dtype=np.float64)
            lon_values = np.asarray(dataset[lon_name].values, dtype=np.float64)
            lat_idx = int(np.argmin(np.abs(lat_values - lat)))
            lon_idx = int(np.argmin(np.abs(lon_values - lon)))
            file_used = False
            for time_index, time_value in enumerate(dataset[time_name].values):
                valid_time = datetime64_to_datetime(time_value)
                if valid_time < start or valid_time > end:
                    continue
                try:
                    profile = extract_profile(
                        dataset,
                        time_name=time_name,
                        level_name=level_name,
                        lat_name=lat_name,
                        lon_name=lon_name,
                        time_index=time_index,
                        lat_index=lat_idx,
                        lon_index=lon_idx,
                        edges=edges,
                        site_altitude_m=site_altitude_m,
                    )
                    profiles[valid_time] = profile
                    file_used = True
                    heights = np.asarray(dataset["z"].isel({time_name: time_index, lat_name: lat_idx, lon_name: lon_idx}).values).squeeze() / G - site_altitude_m
                    below_ground_removed.append(int(np.sum(heights < 0.0)))
                except Exception as error:  # noqa: BLE001 - audit every profile-level failure
                    rejected[valid_time.isoformat()] = f"{type(error).__name__}: {error}"
            if file_used:
                used_files.append(str(path))
                grid_points.append({"lat": float(lat_values[lat_idx]), "lon": float(lon_values[lon_idx])})

    unique_grid_points = sorted({(item["lat"], item["lon"]) for item in grid_points})
    audit = {
        "era5_root": str(root),
        "files_total": len(files),
        "files_used": len(used_files),
        "used_files": used_files,
        "loaded_profiles": len(profiles),
        "rejected_profiles": rejected,
        "requested_window_start": start.isoformat(),
        "requested_window_end": end.isoformat(),
        "target_lat": lat,
        "target_lon": lon,
        "grid_points_used": [{"lat": lat_value, "lon": lon_value} for lat_value, lon_value in unique_grid_points],
        "mean_below_ground_pressure_levels_removed": float(np.mean(below_ground_removed)) if below_ground_removed else 0.0,
        "surface_anchor": "lowest valid above-ground ERA5 pressure level copied to 0 m AGL when needed",
    }
    return profiles, audit


def chronological_masks(dates: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    unique_dates = np.unique(dates)
    n_dates = len(unique_dates)
    if n_dates < 3:
        raise ValueError("Need at least three dates to create train/val/test chronological split")
    n_train = max(1, int(n_dates * 0.70))
    n_val = max(1, int(n_dates * 0.15))
    if n_train + n_val >= n_dates:
        n_train = max(1, n_dates - 2)
        n_val = 1
    train_dates = set(unique_dates[:n_train])
    val_dates = set(unique_dates[n_train : n_train + n_val])
    test_dates = set(unique_dates[n_train + n_val :])
    train_mask = np.asarray([date in train_dates for date in dates], dtype=bool)
    val_mask = np.asarray([date in val_dates for date in dates], dtype=bool)
    test_mask = np.asarray([date in test_dates for date in dates], dtype=bool)
    split = {
        "strategy": "chronological_by_obs_date",
        "train_dates": sorted(train_dates),
        "val_dates": sorted(val_dates),
        "test_dates": sorted(test_dates),
        "train_samples": int(train_mask.sum()),
        "val_samples": int(val_mask.sum()),
        "test_samples": int(test_mask.sum()),
    }
    return train_mask, val_mask, test_mask, split


def match_records(
    records: list[dict[str, Any]], profiles: dict[datetime, dict[str, np.ndarray]], max_delta_hours: float
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    profile_times = sorted(profiles)
    if not profile_times:
        raise ValueError("No ERA5 profiles were loaded")

    matched_records = []
    unmatched = []
    for record in records:
        obs_time = record["datetime"]
        best_time = min(profile_times, key=lambda value: abs((obs_time - value).total_seconds()))
        delta_hours = abs((obs_time - best_time).total_seconds()) / 3600.0
        if delta_hours <= max_delta_hours:
            matched_records.append((record, best_time, profiles[best_time], delta_hours))
        else:
            unmatched.append({"timestamp": record["timestamp"], "nearest_era5": best_time.isoformat(), "delta_hours": delta_hours})

    if not matched_records:
        raise ValueError(f"No Obs_BT records matched ERA5 within {max_delta_hours} hours")

    dates = np.asarray([item[0]["datetime"].strftime("%Y-%m-%d") for item in matched_records])
    train_mask, val_mask, test_mask, split = chronological_masks(dates)
    dataset = {
        "X": np.asarray([item[0]["channels"] for item in matched_records], dtype=np.float32),
        "T": np.asarray([item[2]["T"] for item in matched_records], dtype=np.float32),
        "RH": np.asarray([item[2]["RH"] for item in matched_records], dtype=np.float32),
        "q": np.asarray([item[2]["q"] for item in matched_records], dtype=np.float32),
        "logq": np.asarray([item[2]["logq"] for item in matched_records], dtype=np.float32),
        "P": np.asarray([item[2]["P"] for item in matched_records], dtype=np.float32),
        "timestamps": np.asarray([item[0]["timestamp"] for item in matched_records]),
        "era5_timestamps": np.asarray([item[1].isoformat() for item in matched_records]),
        "dates": dates,
        "delta_hours": np.asarray([item[3] for item in matched_records], dtype=np.float32),
        "train_mask": train_mask,
        "val_mask": val_mask,
        "test_mask": test_mask,
    }
    exact_matches = int(np.sum(dataset["delta_hours"] == 0.0))
    audit = {
        "n_matched": len(matched_records),
        "n_unmatched": len(unmatched),
        "unmatched": unmatched[:50],
        "unmatched_truncated": len(unmatched) > 50,
        "exact_matches": exact_matches,
        "nearest_matches": len(matched_records) - exact_matches,
        "delta_hours_mean": float(np.mean(dataset["delta_hours"])),
        "delta_hours_max": float(np.max(dataset["delta_hours"])),
        "split": split,
    }
    return dataset, audit


def channel_frequencies() -> np.ndarray:
    instrument = load_instruments()["chengdu-21ch"]
    return np.asarray([channel["frequency_ghz"] for channel in instrument["channels"]], dtype=np.float32)


def stats_payload(
    *,
    run_id: str,
    obs_audit: dict[str, Any],
    era5_audit: dict[str, Any],
    match_audit: dict[str, Any],
    dataset: dict[str, np.ndarray],
    centers: np.ndarray,
    edges: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    bt = dataset["X"]
    t = dataset["T"]
    rh = dataset["RH"]
    return {
        "status": "chengdu_realdata_bridge_ready",
        "run_id": run_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "obs_json": str(args.obs_json),
            "era5_root": str(args.era5_root),
            "instrument_id": "chengdu-21ch",
            "site_id": "chengdu",
        },
        "configuration": {
            "max_delta_hours": args.max_delta_hours,
            "site_altitude_m": args.site_altitude,
            "site_latitude": args.lat,
            "site_longitude": args.lon,
            "n_channels": int(bt.shape[1]),
            "channel_frequencies_ghz": channel_frequencies().astype(float).tolist(),
            "vertical_grid": "project 48-layer physical grid, 0-10 km AGL",
            "n_layers": int(len(centers)),
        },
        "dataset": {
            "n_observations_total": obs_audit["input_records"],
            "n_observations_usable": obs_audit["usable_records"],
            "n_matched": match_audit["n_matched"],
            "n_unmatched": match_audit["n_unmatched"],
            "first_timestamp": str(dataset["timestamps"][0]),
            "last_timestamp": str(dataset["timestamps"][-1]),
            "bt_shape": list(bt.shape),
            "T_shape": list(t.shape),
            "RH_shape": list(rh.shape),
            "bt_range_k": [float(np.min(bt)), float(np.max(bt))],
            "T_range_k": [float(np.min(t)), float(np.max(t))],
            "RH_range_percent": [float(np.min(rh)), float(np.max(rh))],
            "layer_centers_m": centers.astype(float).tolist(),
            "layer_edges_m": edges.astype(float).tolist(),
        },
        "obs_audit": obs_audit,
        "era5_audit": era5_audit,
        "match_audit": match_audit,
        "catalog_registration": {
            "status": "not_attempted",
            "reason": "This bridge writes repo-local artifacts first; register after confirming data-lake catalog write access.",
        },
    }


def write_outputs(
    output_dir: Path,
    *,
    run_id: str,
    dataset: dict[str, np.ndarray],
    centers: np.ndarray,
    edges: np.ndarray,
    stats: dict[str, Any],
    sources: dict[str, Any],
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=False)
    dataset_path = output_dir / "bridge_dataset.npz"
    stats_path = output_dir / "bridge_stats.json"
    manifest_path = output_dir / "manifest.json"
    np.savez_compressed(
        dataset_path,
        **dataset,
        run_id=np.asarray(run_id),
        heights=centers,
        layer_edges=edges,
        layer_thickness=np.diff(edges).astype(np.float32),
        channel_frequencies_ghz=channel_frequencies(),
    )
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "manifest_version": 1,
        "run_id": run_id,
        "pipeline": "scripts/build_chengdu_realdata_bridge.py",
        "generated_at_utc": stats["generated_at_utc"],
        "artifact_type": "dataset",
        "outputs": {
            "dataset": str(dataset_path),
            "stats": str(stats_path),
        },
        "sources": sources,
        "summary": {
            "status": stats["status"],
            "n_matched": stats["dataset"]["n_matched"],
            "n_channels": stats["configuration"]["n_channels"],
            "n_layers": stats["configuration"]["n_layers"],
            "split": stats["match_audit"]["split"],
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"dataset": dataset_path, "stats": stats_path, "manifest": manifest_path}


def update_json_file(path: Path, updates: dict[str, Any]) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(updates)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def register_catalog_assets(
    *,
    args: argparse.Namespace,
    run_id: str,
    outputs: dict[str, Path],
    stats: dict[str, Any],
    era5_files: list[str],
) -> dict[str, Any]:
    from mwr_retrieval.artifacts import register_run_output
    from mwr_retrieval.catalog.repository import Catalog
    from mwr_retrieval.paths import DataPaths

    catalog = Catalog(DataPaths.from_value(args.catalog_data_root))
    try:
        catalog.add_project(
            args.project_id,
            "project-504",
            "BRNN temperature and humidity retrieval",
            PROJECT_ROOT.as_uri(),
            "Current MWR BRNN retrieval project",
        )
        run = catalog.create_processing_run(
            "scripts/build_chengdu_realdata_bridge.py",
            command=sys.argv,
            config={
                "obs_json": str(args.obs_json),
                "era5_root": str(args.era5_root),
                "max_delta_hours": args.max_delta_hours,
                "site_altitude_m": args.site_altitude,
                "lat": args.lat,
                "lon": args.lon,
                "n_matched": stats["dataset"]["n_matched"],
            },
            project_id=args.project_id,
            run_id=run_id,
        )
        input_assets = []
        for role, path in [("input", args.obs_json), *[("input", Path(item)) for item in era5_files]]:
            asset = catalog.register(Path(path), source_name="chengdu_realdata_bridge_input")
            catalog.link_project_asset(args.project_id, asset["asset_id"], role, catalog.paths.relative_uri(Path(path)))
            input_assets.append(asset)
        output_asset = register_run_output(
            catalog,
            outputs["dataset"],
            run_id=run["run_id"],
            input_asset_ids=[asset["asset_id"] for asset in input_assets],
            relation="derived_from",
            role="output",
            project_id=args.project_id,
            source_name="chengdu_realdata_bridge",
        )
        catalog.finish_processing_run(run["run_id"], status="completed")
        return {
            "status": "registered",
            "data_root": str(catalog.paths.root),
            "project_id": args.project_id,
            "processing_run_id": run["run_id"],
            "input_asset_count": len(input_assets),
            "input_asset_ids": [asset["asset_id"] for asset in input_assets],
            "output_asset_id": output_asset["asset_id"],
            "output_asset_uri": output_asset["uri"],
        }
    finally:
        catalog.close()


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("--obs-json", type=Path, default=DEFAULT_DATA_ROOT / "raw/mwr/chengdu/chengdu_obs_bt.json")
    root.add_argument("--era5-root", type=Path, default=DEFAULT_DATA_ROOT / "raw/era5/cds/chengdu/pressure-levels")
    root.add_argument("--output-dir", type=Path, default=None)
    root.add_argument("--site-altitude", type=float, default=548.0)
    root.add_argument("--lat", type=float, default=30.63)
    root.add_argument("--lon", type=float, default=104.04)
    root.add_argument("--max-delta-hours", type=float, default=1.0)
    root.add_argument("--skip-input-hash", action="store_true", help="Do not compute source SHA-256 checksums")
    root.add_argument("--register-catalog", action="store_true", help="Register the bridge dataset and lineage in the data-lake catalog")
    root.add_argument("--catalog-data-root", type=Path, default=DEFAULT_DATA_ROOT)
    root.add_argument("--project-id", default="project-brnn")
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
    output_dir = args.output_dir or PROJECT_ROOT / "results" / "chengdu_realdata_bridge" / f"run_id={run_id}"

    records, obs_audit = load_observations(args.obs_json)
    edges, centers = build_layer48_grid()
    obs_times = [record["datetime"] for record in records]
    delta = timedelta(hours=args.max_delta_hours)
    profiles, era5_audit = load_era5_profiles(
        args.era5_root,
        start=min(obs_times) - delta,
        end=max(obs_times) + delta,
        edges=edges,
        site_altitude_m=args.site_altitude,
        lat=args.lat,
        lon=args.lon,
    )
    dataset, match_audit = match_records(records, profiles, args.max_delta_hours)
    stats = stats_payload(
        run_id=run_id,
        obs_audit=obs_audit,
        era5_audit=era5_audit,
        match_audit=match_audit,
        dataset=dataset,
        centers=centers,
        edges=edges,
        args=args,
    )
    include_hash = not args.skip_input_hash
    sources = {
        "obs_json": source_record(args.obs_json, include_hash=include_hash),
        "era5_pressure_files": [source_record(Path(path), include_hash=include_hash) for path in era5_audit["used_files"]],
    }
    outputs = write_outputs(output_dir, run_id=run_id, dataset=dataset, centers=centers, edges=edges, stats=stats, sources=sources)
    if args.register_catalog:
        try:
            catalog_registration = register_catalog_assets(
                args=args,
                run_id=run_id,
                outputs=outputs,
                stats=stats,
                era5_files=era5_audit["used_files"],
            )
        except Exception as error:  # noqa: BLE001 - keep the bridge artifact usable even when catalog is unavailable
            catalog_registration = {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "data_root": str(args.catalog_data_root),
            }
        stats["catalog_registration"] = catalog_registration
        update_json_file(outputs["stats"], {"catalog_registration": catalog_registration})
        update_json_file(outputs["manifest"], {"catalog_registration": catalog_registration})
    print(json.dumps({"run_id": run_id, "output_dir": str(output_dir), "outputs": {k: str(v) for k, v in outputs.items()}, "summary": stats["dataset"], "split": stats["match_audit"]["split"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
