#!/usr/bin/env python3
"""POC pipeline for 2013-01: uses valid CDS pressure-level profiles + surface data.

Strategy:
  - 1 day of valid CDS pressure-level data exists (_pl_201301_d01.nc, 24h × 37 levels)
  - For each hour in the month, find the closest hour-of-day template from the valid day
  - Adjust temperature profile by delta between template surface T and target surface T
  - Adjust RH profile proportionally
  - Interpolate to 93-layer MWR grid → BT simulate → QC → train

This proves the code pipeline works end-to-end with real data for Jan 2013.
Only temperature scaling is applied; absolute profile accuracy is not expected.
"""

import os, sys
import numpy as np
import xarray as xr
import pickle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
sys.path.insert(0, os.path.dirname(__file__))
import config
from src.era5_preprocess import calc_2m_rh, interpolate_to_mwr_grid


def load_template_and_surface_data():
    """Load the 1 valid CDS pressure-level day (template) and 31 days of single-level data."""
    # --- Load CDS pressure-level template (day 1, 24 hours) ---
    pl_path = os.path.join(config.ERA5_DIR, "_pl_201301_d01.nc")
    if not os.path.exists(pl_path):
        # fallback: use the test file
        pl_path = os.path.join(config.ERA5_DIR, "test_cds_pl.nc")

    ds_pl = xr.open_dataset(pl_path)
    ds_pl = ds_pl.sel(latitude=config.NANJIAO_LAT, longitude=config.NANJIAO_LON, method="nearest")

    time_var = "valid_time" if "valid_time" in ds_pl.dims else "time"
    pl_times = ds_pl[time_var].values
    level_var = "pressure_level" if "pressure_level" in ds_pl.dims else "level"
    levels = ds_pl[level_var].values

    T_key = "t" if "t" in ds_pl else "temperature"
    Z_key = "z" if "z" in ds_pl else "geopotential"
    q_key = "q" if "q" in ds_pl else "specific_humidity"
    r_key = "r" if "r" in ds_pl else "relative_humidity"

    T_template = ds_pl[T_key].values    # [time, level]
    Z_template = ds_pl[Z_key].values    # [time, level]

    if r_key in ds_pl:
        RH_template = ds_pl[r_key].values
    else:
        # Convert q → RH
        q = ds_pl[q_key].values
        P_bc = (levels * 100.0)[np.newaxis, :]
        RH_template = _specific_humidity_to_rh(q, T_template, P_bc)

    ds_pl.close()

    # --- Load single-level data for all 31 days ---
    sl_path = os.path.join(config.ERA5_DIR, "sl_2013_01.nc")
    ds_sl = xr.open_dataset(sl_path)
    ds_sl = ds_sl.sel(latitude=config.NANJIAO_LAT, longitude=config.NANJIAO_LON, method="nearest")

    sl_time_var = "valid_time" if "valid_time" in ds_sl.dims else "time"
    sl_times_all = ds_sl[sl_time_var].values  # [744]

    t2m_all = ds_sl["t2m"].values
    d2m_all = ds_sl["d2m"].values
    sp_all = ds_sl["sp"].values
    ds_sl.close()

    print(f"  Template: {len(pl_times)} hours x {len(levels)} levels "
          f"({T_template.min():.0f}-{T_template.max():.0f}K)")
    print(f"  Surface data: {len(sl_times_all)} hours "
          f"({t2m_all.min():.0f}-{t2m_all.max():.0f}K)")

    return (pl_times, levels, T_template, Z_template, RH_template,
            sl_times_all, t2m_all, d2m_all, sp_all)


def _specific_humidity_to_rh(q, T, p):
    epsilon = 0.622
    e = q * p / (epsilon + (1.0 - epsilon) * q)
    T_c = T - 273.16
    e_s = 6.1078 * np.exp(17.2693882 * T_c / (T_c + 237.3))
    return np.clip((e / e_s) * 100.0, 0.0, 100.0)


def expand_profiles(pl_times, levels, T_template, Z_template, RH_template,
                    sl_times_all, t2m_all, d2m_all, sp_all):
    """For each hour in the month, produce a pressure-level profile.

    Uses the closest hour-of-day from the template day, adjusted by
    surface temperature difference.
    """
    n_hours = len(sl_times_all)
    n_levels = len(levels)
    n_template = len(pl_times)

    # Get hour-of-day for each time
    template_hours = np.array([_get_hour(t) for t in pl_times])
    target_hours = np.array([_get_hour(t) for t in sl_times_all])

    T_out = np.zeros((n_hours, n_levels))
    Z_out = np.zeros((n_hours, n_levels))
    RH_out = np.zeros((n_hours, n_levels))
    t2m_out = np.zeros(n_hours)
    d2m_out = np.zeros(n_hours)
    sp_out = np.zeros(n_hours)

    # Surface temperature from template (for adjustment)
    template_sfc_T = T_template[:, -1].copy()  # surface is highest pressure level

    for i in range(n_hours):
        t2m_out[i] = t2m_all[i]
        d2m_out[i] = d2m_all[i]
        sp_out[i] = sp_all[i]

        # Find closest hour in template
        hour = target_hours[i]
        best_j = np.argmin(np.abs(template_hours - hour))

        # Adjust temperature profile by surface delta
        sfc_delta = t2m_all[i] - template_sfc_T[best_j]
        T_out[i, :] = T_template[best_j, :] + sfc_delta

        # Scale RH: keep same profile shape, adjust to preserve RH
        # (simplified - just use template RH, clip to valid range)
        RH_out[i, :] = np.clip(RH_template[best_j, :], 0, 100)

        # Geopotential: use template (reasonable for Beijing station)
        Z_out[i, :] = Z_template[best_j, :]

    return {
        "time": sl_times_all,
        "P_hPa": levels,
        "Z_m": Z_out / 9.80665,
        "T": T_out,
        "RH": RH_out,
        "CLWC": np.zeros_like(T_out),
        "t2m": t2m_out,
        "d2m": d2m_out,
        "sp": sp_out,
    }


def _get_hour(t):
    """Extract hour-of-day from numpy datetime64 or cftime."""
    if hasattr(t, 'hour'):
        return t.hour
    return t.astype('datetime64[h]').astype(int) % 24


if __name__ == "__main__":
    from src.brightness_temp import simulate_batch
    from src.qc_correction import apply_full_qc
    from src.train import train_all_models

    print("=" * 60)
    print("POC Pipeline: 2013-01 (1 template day → expand to 31 days)")
    print("=" * 60)

    # --- Load ---
    print("\n[1/5] Loading template and surface data...")
    (pl_times, levels, T_tpl, Z_tpl, RH_tpl,
     sl_times, t2m, d2m, sp) = load_template_and_surface_data()

    # --- Expand ---
    print("\n[2/5] Expanding profiles for full month...")
    profiles_raw = expand_profiles(
        pl_times, levels, T_tpl, Z_tpl, RH_tpl,
        sl_times, t2m, d2m, sp
    )

    # Calculate 2m RH
    rh2m = calc_2m_rh(t2m, d2m, sp)
    profiles_raw["rh2m"] = rh2m
    del profiles_raw["d2m"]

    print(f"  Created {len(profiles_raw['time'])} profiles")
    print(f"  T range: {profiles_raw['T'].min():.1f} – {profiles_raw['T'].max():.1f} K")
    print(f"  RH range: {profiles_raw['RH'].min():.1f} – {profiles_raw['RH'].max():.1f} %")

    # --- Interpolate to 93-layer grid ---
    print("\n[3/5] Interpolating to 93-layer MWR grid...")
    era5_profiles = interpolate_to_mwr_grid(profiles_raw)
    print(f"  Shape: {era5_profiles['T'].shape}")

    # Save profiles
    cache_path = os.path.join(config.ERA5_DIR, "era5_profiles_201301_poc.pkl")
    with open(cache_path, "wb") as f:
        pickle.dump(era5_profiles, f)
    print(f"  Saved: {cache_path}")

    # --- Simulate BT ---
    print("\n[4/5] Simulating brightness temperatures...")
    era5_profiles["P"] = np.ones((len(era5_profiles["time"]), config.N_LAYERS)) * 1013.0
    era5_tbs = simulate_batch(era5_profiles, add_noise=False)
    print(f"  BT shape: {era5_tbs.shape}")

    bt_path = os.path.join(config.ERA5_DIR, "era5_bt_sim_201301_poc.pkl")
    with open(bt_path, "wb") as f:
        pickle.dump(era5_tbs, f)

    # Convert times to Python datetime objects (QC needs .month attribute)
    times_py = era5_profiles["time"].astype('datetime64[s]').tolist()
    if not isinstance(times_py[0], (int, float)):
        era5_profiles["time"] = np.array(times_py)

    # --- QC ---
    print("\n[5/5] QC + Train...")
    qc_profiles, qc_tbs, keep_mask, qc_info = apply_full_qc(
        era5_profiles, era5_tbs, sounding_profiles=None
    )
    print(f"  QC: {keep_mask.sum()} / {len(era5_profiles['time'])} kept")

    # Filter
    for key in ["T", "RH", "CLWC", "t2m", "rh2m", "sp"]:
        if key in qc_profiles and qc_profiles[key] is not None:
            qc_profiles[key] = qc_profiles[key][keep_mask]
    qc_tbs = qc_tbs[keep_mask]

    device = "cpu"
    try:
        import torch
        if torch.backends.mps.is_available():
            device = "mps"
    except Exception:
        pass
    print(f"  Device: {device}, Samples: {keep_mask.sum()}")

    os.makedirs(config.MODEL_DIR, exist_ok=True)
    train_all_models(qc_profiles, qc_tbs, config.MODEL_DIR, device=device)

    print("\n" + "=" * 60)
    print("POC pipeline complete!")
    print(f"  Profiles: {cache_path}")
    print(f"  BT:       {bt_path}")
    print(f"  Models:   {config.MODEL_DIR}/")
    print("=" * 60)
