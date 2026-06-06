#!/usr/bin/env python3
"""Adapter: load CDS single-level + CDS pressure-level data for 2013-01 POC.

CDS variable names:
  single-level: t2m, d2m, sp           (dim: valid_time)
  pressure-level: t (temperature), z (geopotential), q (specific humidity)
                  (dim: valid_time, pressure_level)

The pressure-level file provides specific humidity (q), which is converted
to relative humidity (RH) before passing to the existing pipeline.

Produces the same dict format as extract_era5_profiles() so the rest of
the pipeline (interpolate_to_mwr_grid → BT simulate → QC → train) works unchanged.
"""

import os, sys
import numpy as np
import xarray as xr

sys.path.insert(0, os.path.dirname(__file__))
import config
from src.era5_preprocess import calc_2m_rh, interpolate_to_mwr_grid


def specific_humidity_to_rh(q, T, p):
    """Convert specific humidity [kg/kg] to relative humidity [%].

    q = ε * e / (p - (1-ε) * e)  →  e = q * p / (ε + (1-ε) * q)
    RH = e / e_s * 100%

    Args:
        q: specific humidity [kg/kg]
        T: temperature [K]
        p: pressure [Pa]
    Returns:
        RH [%], clipped to [0, 100]
    """
    epsilon = 0.622  # R_dry / R_vapor
    e = q * p / (epsilon + (1.0 - epsilon) * q)
    T_c = T - 273.16
    e_s = 6.1078 * np.exp(17.2693882 * T_c / (T_c + 237.3))
    rh = (e / e_s) * 100.0
    return np.clip(rh, 0.0, 100.0)


def load_cds_data(sl_path, pl_path, lat, lon):
    """Load CDS single-level + pressure-level data and merge.

    Works with both CDS native and ARCO files by auto-detecting variable names.

    Returns a dict in the same format as extract_era5_profiles().
    """
    # --- Load single-level file ---
    ds_sl = xr.open_dataset(sl_path)
    ds_sl = ds_sl.sel(latitude=lat, longitude=lon, method="nearest")

    time_var = "valid_time" if "valid_time" in ds_sl.dims else "time"
    sl_times = ds_sl[time_var].values

    # Auto-detect variable names
    t2m_key = "t2m" if "t2m" in ds_sl else "2m_temperature"
    d2m_key = "d2m" if "d2m" in ds_sl else "2m_dewpoint_temperature"
    sp_key = "sp" if "sp" in ds_sl else "surface_pressure"

    t2m = ds_sl[t2m_key].values
    d2m = ds_sl[d2m_key].values
    sp = ds_sl[sp_key].values
    ds_sl.close()

    # --- Load pressure-level file ---
    ds_pl = xr.open_dataset(pl_path)
    ds_pl = ds_pl.sel(latitude=lat, longitude=lon, method="nearest")

    pl_time_var = "valid_time" if "valid_time" in ds_pl.dims else "time"
    pl_times = ds_pl[pl_time_var].values

    # Auto-detect pressure level coordinate name
    level_key = "pressure_level" if "pressure_level" in ds_pl.dims else "level"
    levels = ds_pl.coords[level_key].values

    # Auto-detect variable names (CDS: t,z,q  ARCO: temperature,geopotential,specific_humidity)
    T_key = "t" if "t" in ds_pl else "temperature"
    Z_key = "z" if "z" in ds_pl else "geopotential"
    q_key = "q" if "q" in ds_pl else "specific_humidity"
    r_key = "r" if "r" in ds_pl else "relative_humidity"

    T = ds_pl[T_key].values        # [time, level]
    Z = ds_pl[Z_key].values        # [time, level], m²/s²

    # Get humidity — may already be RH, or may need conversion from specific humidity
    if r_key in ds_pl:
        RH = ds_pl[r_key].values   # [time, level], already RH [%]
    else:
        q = ds_pl[q_key].values    # [time, level], kg/kg
        P_bc = (np.array(levels) * 100.0)[np.newaxis, :]  # hPa→Pa, broadcast
        RH = specific_humidity_to_rh(q, T, P_bc)

    ds_pl.close()

    # --- Align time axes ---
    if len(sl_times) != len(pl_times):
        print(f"  Warning: time mismatch — SL {len(sl_times)}, PL {len(pl_times)}")
        min_len = min(len(sl_times), len(pl_times))
        sl_times = sl_times[:min_len]
        t2m = t2m[:min_len]; d2m = d2m[:min_len]; sp = sp[:min_len]
        T = T[:min_len, :]; Z = Z[:min_len, :]; RH = RH[:min_len, :]

    # --- Calculate 2m RH ---
    rh2m = calc_2m_rh(t2m, d2m, sp)

    # --- Convert times to datetime objects (QC needs .month attribute) ---
    times_dt = sl_times.astype('datetime64[s]').tolist()
    # xarray sometimes returns cftime objects — convert if needed
    if times_dt and hasattr(times_dt[0], 'strftime'):
        pass  # python datetime, all good
    elif hasattr(sl_times[0], 'strftime'):
        times_dt = [t for t in sl_times]

    # --- Build output dict ---
    result = {
        "time": np.array(times_dt),
        "P_hPa": np.array(levels),
        "Z_m": Z / 9.80665,         # geopotential height → meters
        "T": T,                     # [time, level]
        "RH": RH,                   # [time, level]
        "CLWC": np.zeros_like(T),   # placeholder
        "t2m": t2m,                 # [time]
        "rh2m": rh2m,               # [time]
        "sp": sp,                   # [time]
    }

    print(f"  Loaded {len(sl_times)} time steps, {len(levels)} pressure levels")
    print(f"  T range: {T.min():.1f} – {T.max():.1f} K")
    print(f"  RH range: {RH.min():.1f} – {RH.max():.1f} %")
    print(f"  Z (sfc) range: {result['Z_m'][:, -1].min():.0f} – {result['Z_m'][:, -1].max():.0f} m")
    print(f"  t2m range: {t2m.min():.1f} – {t2m.max():.1f} K")
    print(f"  rh2m range: {rh2m.min():.1f} – {rh2m.max():.1f} %")

    return result


if __name__ == "__main__":
    import pickle
    from src.brightness_temp import simulate_batch

    print("=" * 60)
    print("2013-01 POC Pipeline")
    print("=" * 60)

    # --- Step 1: Load & merge ---
    print("\n[1/5] Loading 2013-01 data...")
    profiles = load_cds_data(
        sl_path=os.path.join(config.ERA5_DIR, "sl_2013_01.nc"),
        pl_path=os.path.join(config.ERA5_DIR, "pl_2013_01.nc"),
        lat=config.NANJIAO_LAT,
        lon=config.NANJIAO_LON,
    )

    # --- Step 2: Interpolate to MWR 93-layer grid ---
    print("\n[2/5] Interpolating to 93-layer MWR grid...")
    era5_profiles = interpolate_to_mwr_grid(profiles)
    print(f"  Output: {era5_profiles['T'].shape[1]} layers × {era5_profiles['T'].shape[0]} samples")
    print(f"  Height range: {era5_profiles['height'][0]:.1f} – {era5_profiles['height'][-1]:.1f} m")

    cache_path = os.path.join(config.ERA5_DIR, "era5_profiles_201301.pkl")
    with open(cache_path, "wb") as f:
        pickle.dump(era5_profiles, f)
    print(f"  Saved: {cache_path}")

    # --- Step 3: Simulate brightness temperatures ---
    print("\n[3/5] Simulating brightness temperatures...")
    era5_profiles["P"] = np.ones((len(era5_profiles["time"]), config.N_LAYERS)) * 1013.0
    era5_tbs = simulate_batch(era5_profiles, add_noise=False)
    print(f"  BT shape: {era5_tbs.shape}")

    bt_path = os.path.join(config.ERA5_DIR, "era5_bt_sim_201301.pkl")
    with open(bt_path, "wb") as f:
        pickle.dump(era5_tbs, f)
    print(f"  Saved: {bt_path}")

    # --- Step 4: QC ---
    print("\n[4/5] Running QC pipeline...")
    from src.qc_correction import apply_full_qc
    qc_profiles, qc_tbs, keep_mask, qc_info = apply_full_qc(
        era5_profiles, era5_tbs, sounding_profiles=None
    )
    print(f"  Samples: {len(era5_profiles['time'])} → {keep_mask.sum()} kept")

    qc_data = {
        "profiles": qc_profiles,
        "tbs": qc_tbs,
        "keep_mask": keep_mask,
        "qc_info": qc_info,
    }
    qc_path = os.path.join(config.ERA5_DIR, "era5_qc_data_201301.npz")
    np.savez_compressed(qc_path, **qc_data)
    print(f"  Saved: {qc_path}")

    # --- Step 5: Train BRNN ---
    print("\n[5/5] Training BRNN models...")
    from src.train import train_all_models

    for key in ["T", "RH", "CLWC", "t2m", "rh2m", "sp"]:
        if key in qc_profiles and qc_profiles[key] is not None:
            qc_profiles[key] = qc_profiles[key][keep_mask]
    qc_tbs = qc_tbs[keep_mask]

    print(f"  Training on all {keep_mask.sum()} samples from 2013-01")
    os.makedirs(config.MODEL_DIR, exist_ok=True)

    device = "cpu"
    try:
        import torch
        if torch.backends.mps.is_available():
            device = "mps"
    except Exception:
        pass
    print(f"  Device: {device}")

    train_all_models(qc_profiles, qc_tbs, config.MODEL_DIR, device=device)

    print("\n" + "=" * 60)
    print("2013-01 POC pipeline complete!")
    print(f"  Profiles: {cache_path}")
    print(f"  BT:       {bt_path}")
    print(f"  QC:       {qc_path}")
    print(f"  Models:   {config.MODEL_DIR}/")
    print("=" * 60)
