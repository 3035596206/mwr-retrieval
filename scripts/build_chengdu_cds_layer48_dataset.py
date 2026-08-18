#!/usr/bin/env python3
"""Build 48-layer dataset from CDS NetCDF daily ERA5 files (replaces GRIB reader)."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from mwr_retrieval.grids import build_layer48_grid, layer_average

# Inline functions from build_chengdu_era5_layer48_dataset (avoid eccodes dependency)
G = 9.80665
Q_FLOOR = 1e-6
EPSILON = 0.62198


def build_layer_grid():
    return build_layer48_grid()


def saturation_vapor_pressure_hpa(temperature_k):
    temperature_c = np.asarray(temperature_k, dtype=np.float64) - 273.15
    water = 6.112 * np.exp(17.62 * temperature_c / (243.12 + temperature_c))
    ice = 6.112 * np.exp(22.46 * temperature_c / (272.62 + temperature_c))
    blend = np.clip((temperature_c + 20.0) / 20.0, 0.0, 1.0)
    return ice * (1.0 - blend) + water * blend


def rh_to_specific_humidity(temperature_k, relative_humidity, pressure_hpa):
    es = saturation_vapor_pressure_hpa(temperature_k)
    vapor_pressure = np.clip(relative_humidity / 100.0 * es, 0.0, pressure_hpa * 0.99)
    mixing_ratio = EPSILON * vapor_pressure / np.maximum(pressure_hpa - vapor_pressure, 1e-6)
    return np.maximum(mixing_ratio / (1.0 + mixing_ratio), Q_FLOOR)


def specific_humidity_to_rh(temperature_k, specific_humidity, pressure_hpa):
    q = np.clip(specific_humidity, Q_FLOOR, 0.2)
    mixing_ratio = q / np.maximum(1.0 - q, 1e-8)
    vapor_pressure = mixing_ratio * pressure_hpa / (EPSILON + mixing_ratio)
    rh = 100.0 * vapor_pressure / np.maximum(saturation_vapor_pressure_hpa(temperature_k), 1e-8)
    return np.clip(rh, 0.0, 100.0)


def raw_profile_to_layers(height_agl, temperature_k, relative_humidity, pressure_hpa, edges):
    valid = (
        np.isfinite(height_agl)
        & np.isfinite(temperature_k)
        & np.isfinite(relative_humidity)
        & np.isfinite(pressure_hpa)
        & (height_agl >= 0.0)
        & (temperature_k >= 180.0) & (temperature_k <= 330.0)
        & (relative_humidity >= 0.0) & (relative_humidity <= 100.0)
        & (pressure_hpa >= 1.0) & (pressure_hpa <= 1100.0)
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
    if z[0] > 0.0:
        z = np.concatenate(([0.0], z))
        t = np.concatenate(([t[0]], t))
        rh = np.concatenate(([rh[0]], rh))
        p = np.concatenate(([p[0]], p))
    q = rh_to_specific_humidity(t, rh, p)
    t_layer = layer_average(z, t, edges)
    log_q_layer = layer_average(z, np.log(np.maximum(q, Q_FLOOR)), edges)
    log_p_layer = layer_average(z, np.log(p), edges)
    q_layer = np.exp(log_q_layer).astype(np.float32)
    p_layer = np.exp(log_p_layer).astype(np.float32)
    rh_layer = specific_humidity_to_rh(t_layer, q_layer, p_layer).astype(np.float32)
    return {"T": t_layer, "q": q_layer, "RH": rh_layer}


# ── CDS NetCDF reader ──

def load_cds_netcdf_profiles(pressure_dir, requested_times, edges, site_altitude_m,
                             lat=30.63, lon=104.04):
    pressure_files = sorted(Path(pressure_dir).glob("part=*/era5_pressure_*.nc"))
    if not pressure_files:
        raise FileNotFoundError(f"No pressure NetCDF files in {pressure_dir}")
    
    datasets = [xr.open_dataset(f, engine="netcdf4") for f in pressure_files]
    ds_all = xr.concat(datasets, dim="valid_time") if len(datasets) > 1 else datasets[0]
    
    ds_lat = ds_all["latitude"].values if "latitude" in ds_all else ds_all["lat"].values
    ds_lon = ds_all["longitude"].values if "longitude" in ds_all else ds_all["lon"].values
    lat_idx = int(np.argmin(np.abs(ds_lat - lat)))
    lon_idx = int(np.argmin(np.abs(ds_lon - lon)))
    print(f"Using grid point: lat={float(ds_lat[lat_idx]):.2f}, lon={float(ds_lon[lon_idx]):.2f}")
    
    times_np = ds_all["valid_time"].values
    levels_np = ds_all["pressure_level"].values if "pressure_level" in ds_all else ds_all["level"].values
    
    raw = {}
    n_messages = 0
    for i in range(len(times_np)):
        t_val = pd.Timestamp(times_np[i]).to_pydatetime().replace(tzinfo=None)
        if t_val not in requested_times:
            continue
        for var_name in ["z", "t", "r"]:
            if var_name not in ds_all:
                continue
            n_messages += 1
            var_data = ds_all[var_name].isel(valid_time=i, latitude=lat_idx, longitude=lon_idx)
            for j in range(len(levels_np)):
                raw.setdefault(t_val, {}).setdefault(var_name, {})[float(levels_np[j])] = float(var_data.values[j])
    
    for d in datasets:
        d.close()
    
    profiles = {}
    rejected = {}
    for valid_time, variables in raw.items():
        if not {"z", "t", "r"}.issubset(variables):
            rejected[valid_time.isoformat()] = "missing z/t/r"
            continue
        levels = sorted(set(variables["z"]) & set(variables["t"]) & set(variables["r"]), reverse=True)
        height_agl = np.asarray([variables["z"][lv] / G - site_altitude_m for lv in levels])
        temperature = np.asarray([variables["t"][lv] for lv in levels])
        humidity = np.asarray([variables["r"][lv] for lv in levels])
        pressure = np.asarray(levels, dtype=np.float64)
        try:
            profiles[valid_time] = raw_profile_to_layers(height_agl, temperature, humidity, pressure, edges)
        except ValueError as exc:
            rejected[valid_time.isoformat()] = str(exc)
    
    audit = {"netcdf_files": len(pressure_files), "messages_processed": n_messages,
             "requested_times": len(requested_times), "valid_profiles": len(profiles),
             "rejected": rejected, "site_altitude_m": site_altitude_m,
             "grid_lat": float(ds_lat[lat_idx]), "grid_lon": float(ds_lon[lon_idx])}
    return profiles, audit


# ── Main ──

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pressure-dir", required=True)
    parser.add_argument("--obs-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-stats", required=True)
    parser.add_argument("--site-altitude", type=float, default=548.0)
    parser.add_argument("--max-delta-hours", type=float, default=1.0)
    args = parser.parse_args()
    
    # Load observations
    payload = json.loads(Path(args.obs_json).read_text(encoding="utf-8"))
    obs = payload["records"]
    for rec in obs:
        rec["datetime"] = datetime.strptime(rec["timestamp"], "%Y_%m_%d %H:%M:%S")
    print(f"Loaded {len(obs)} observations")
    
    edges, centers = build_layer_grid()
    print(f"Layer grid: {len(centers)} layers")
    
    requested_times = {o["datetime"] for o in obs}
    print(f"Unique times: {len(requested_times)}")
    
    profiles, audit = load_cds_netcdf_profiles(args.pressure_dir, requested_times, edges, args.site_altitude)
    print(f"Valid profiles: {len(profiles)}, rejected: {len(audit['rejected'])}")
    
    # Match observations to ERA5
    matched = {"BT": [], "T": [], "RH": [], "times": [], "deltas": []}
    for o in obs:
        dt = o["datetime"]
        best_dt = min(profiles, key=lambda pdt: abs((dt - pdt).total_seconds() / 3600.0))
        delta = abs((dt - best_dt).total_seconds() / 3600.0)
        if delta <= args.max_delta_hours:
            p = profiles[best_dt]
            matched["BT"].append(o["channels"])
            matched["T"].append(p["T"])
            matched["RH"].append(p["RH"])
            matched["times"].append(str(dt))
            matched["deltas"].append(delta)
    
    n = len(matched["BT"])
    print(f"Matched: {n} obs within {args.max_delta_hours}h")
    if n == 0:
        print("ERROR: no matches"); sys.exit(1)
    
    BT = np.array(matched["BT"], dtype=np.float32)
    T = np.array(matched["T"], dtype=np.float32)
    RH = np.array(matched["RH"], dtype=np.float32)
    
    # Chronological split
    dates_arr = np.array([t[:10] for t in matched["times"]])
    unique_dates = np.unique(dates_arr)
    nd = len(unique_dates)
    n_train = max(1, int(nd * 0.70))
    n_val = max(1, int(nd * 0.15))
    if n_train + n_val >= nd:
        n_val = max(1, nd - n_train - 1)
    
    train_dates = set(unique_dates[:n_train])
    val_dates = set(unique_dates[n_train:n_train + n_val])
    test_dates = set(unique_dates[n_train + n_val:])
    
    train_idx = [i for i, t in enumerate(matched["times"]) if t[:10] in train_dates]
    val_idx = [i for i, t in enumerate(matched["times"]) if t[:10] in val_dates]
    test_idx = [i for i, t in enumerate(matched["times"]) if t[:10] in test_dates]
    
    np.savez_compressed(
        args.output,
        BT_train=BT[train_idx], T_train=T[train_idx], RH_train=RH[train_idx],
        BT_val=BT[val_idx], T_val=T[val_idx], RH_val=RH[val_idx],
        BT_test=BT[test_idx], T_test=T[test_idx], RH_test=RH[test_idx],
        centers=centers, edges=edges,
    )
    
    stats = {
        "status": "cds_netcdf_48layer_dataset",
        "n_observations": len(obs), "n_matched": n,
        "train": len(train_idx), "val": len(val_idx), "test": len(test_idx),
        "n_layers": len(centers),
        "deltas_mean": float(np.mean(matched["deltas"])),
        "deltas_max": float(np.max(matched["deltas"])),
        "audit": audit,
    }
    with open(args.output_stats, "w") as f:
        json.dump(stats, f, indent=2, default=str)
    
    print(f"Done. Train={len(train_idx)} Val={len(val_idx)} Test={len(test_idx)}")


if __name__ == "__main__":
    main()
