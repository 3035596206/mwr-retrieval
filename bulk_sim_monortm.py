#!/usr/bin/env python3
"""Batch MonoRTM BT simulation from monthly ERA5 pressure-level data.

For each month of ERA5 pressure-level profiles (pl_YYYY_MM.nc):
  1. Load T(z), q(z), Z(z) from netCDF
  2. Convert q → RH
  3. Interpolate to MWR 93-layer grid
  4. Generate MONORTM_PROF.IN per time step
  5. Run MonoRTM → 14-channel BT
  6. Save as parquet or combined netCDF

Resume-safe: skips months with existing BT files.

Requires: monoRTM_fixed at bin/monortm (or environment var MONORTM_PATH).
         TAPE3_bin at data/TAPE3/TAPE3_bin.

Usage:
  python3 bulk_sim_monortm.py              # all 84 months sequentially
  python3 bulk_sim_monortm.py 2013 01      # single month
  python3 bulk_sim_monortm.py --parallel 4 # 4 months in parallel
"""

import os, sys, time, calendar, pickle, subprocess, shutil, glob, tempfile
import numpy as np
import xarray as xr
from concurrent.futures import ProcessPoolExecutor, as_completed

# ---- paths ----
BASE = os.path.dirname(os.path.abspath(__file__))
ERA5_DIR = os.path.join(BASE, "data", "era5")
BT_DIR = os.path.join(BASE, "data", "monortm_bt")
TAPE3_BIN = os.path.join(BASE, "data", "TAPE3", "TAPE3_bin")
MONORTM_BIN = os.environ.get(
    "MONORTM_PATH",
    os.path.join(BASE, "bin", "monortm")
)

sys.path.insert(0, BASE)
import config
from src.monortm_wrapper import MonoRTM
from src.era5_preprocess import calc_2m_rh


os.makedirs(BT_DIR, exist_ok=True)

# ============================================================
# Profile preparation
# ============================================================

def specific_humidity_to_rh(q, T, p_hpa):
    """q [kg/kg] + T [K] + p [hPa] → RH [%]."""
    epsilon = 0.622
    p_pa = p_hpa * 100.0
    e = q * p_pa / (epsilon + (1.0 - epsilon) * q)
    T_c = T - 273.16
    e_s = 6.1078 * np.exp(17.2693882 * T_c / (T_c + 237.3))
    rh = (e / e_s) * 100.0
    return np.clip(rh, 0.0, 100.0)


def height_from_geopotential(z_m2s2):
    """Geopotential [m²/s²] → geopotential height [m]."""
    return z_m2s2 / 9.80665


def interpolate_to_mwr_grid(T_37, RH_37, Z_37, levels_hpa):
    """Interpolate 37-level ERA5 to 93-level MWR grid using log-pressure.

    Args:
        T_37: (n_time, 37) temperature [K]
        RH_37: (n_time, 37) relative humidity [%]
        Z_37: (n_time, 37) geopotential height [m]
        levels_hpa: (37,) pressure levels [hPa]

    Returns:
        dict with T, RH, P_hPa, height on 93-layer grid
    """
    n_time = T_37.shape[0]
    n_mwr = config.N_LAYERS
    mwr_heights = np.array(config.HEIGHT_GRID)

    T_mwr = np.zeros((n_time, n_mwr), dtype=np.float64)
    RH_mwr = np.zeros((n_time, n_mwr), dtype=np.float64)
    P_mwr = np.zeros((n_time, n_mwr), dtype=np.float64)

    log_p_37 = np.log(levels_hpa)

    for t in range(n_time):
        z_37 = np.maximum(Z_37[t], 0.0)
        log_p_sfc = log_p_37[-1]  # 1000 hPa level

        # Log-P model: log(P) ≈ log(P_sfc) - (z / H_scale)
        H_scale = max((z_37[-1] - z_37[0]) / (log_p_37[0] - log_p_37[-1]), 1000.0)

        for j, h_mwr in enumerate(mwr_heights):
            # Estimate log-P at this height
            log_p_est = np.interp(h_mwr, z_37, log_p_37) if h_mwr > z_37[0] else log_p_37[-1]
            P_mwr[t, j] = np.exp(log_p_est)

            # Interpolate T and RH
            T_mwr[t, j] = np.interp(h_mwr, z_37, T_37[t], left=T_37[t, 0], right=T_37[t, -1])
            RH_mwr[t, j] = np.interp(h_mwr, z_37, RH_37[t], left=RH_37[t, 0], right=RH_37[t, -1])

    return {
        "T": T_mwr,
        "RH": np.clip(RH_mwr, 0.0, 100.0),
        "P_hPa": P_mwr,
        "height": mwr_heights,
    }


def load_and_prepare_month(pl_path):
    """Load one month of ERA5 pressure-level data and prepare 93-layer profiles.

    Returns:
        profiles_dict: {T: (n,93), RH: (n,93), P_hPa: (n,93), height: (93,),
                        CLWC: (n,93), time: (n,)}
        or None if data is corrupted.
    """
    try:
        ds = xr.open_dataset(pl_path)

        # Auto-detect variable names
        T_key = "t" if "t" in ds else "temperature"
        Z_key = "z" if "z" in ds else "geopotential"
        q_key = "q" if "q" in ds else "specific_humidity"
        r_key = "r" if "r" in ds else "relative_humidity"

        # Detect time dimension
        time_key = "valid_time" if "valid_time" in ds.dims else "time"
        level_key = "pressure_level" if "pressure_level" in ds.dims else "level"

        levels = ds.coords[level_key].values  # hPa
        T_37 = ds[T_key].values  # (time, level)
        Z_37_full = ds[Z_key].values

        if r_key in ds:
            RH_37 = ds[r_key].values
        else:
            q = ds[q_key].values
            P_hpa_bc = levels[np.newaxis, :]
            RH_37 = specific_humidity_to_rh(q, T_37, P_hpa_bc)

        times = ds[time_key].values
        ds.close()

        # Convert geopotential to height
        Z_37 = height_from_geopotential(Z_37_full)

        # Interpolate to MWR 93-layer grid
        profiles = interpolate_to_mwr_grid(T_37, RH_37, Z_37, levels)

        profiles["CLWC"] = np.zeros_like(T_37[:, :config.N_LAYERS])
        profiles["time"] = times
        profiles["P_hPa"] = profiles["P_hPa"].astype(np.float64)
        profiles["T"] = profiles["T"].astype(np.float64)
        profiles["RH"] = profiles["RH"].astype(np.float64)

        return profiles

    except Exception as e:
        print(f"  ERROR loading {pl_path}: {e}")
        return None


# ============================================================
# MonoRTM batch simulation
# ============================================================

def simulate_batch_monortm(profiles):
    """Run MonoRTM for all profiles in a month.

    Args:
        profiles: dict with T (n,93), RH (n,93), P_hPa (n,93), height (93,), CLWC (n,93)
    Returns:
        tb: (n, 14) brightness temperatures [K]
    """
    rtm = MonoRTM(monortm_path=MONORTM_BIN, tape3_path=TAPE3_BIN)
    n_time = profiles["T"].shape[0]

    tb = np.zeros((n_time, len(config.ALL_CHANNELS)), dtype=np.float32)

    for t in range(n_time):
        prof = {
            "T": profiles["T"][t],
            "RH": profiles["RH"][t],
            "CLWC": profiles["CLWC"][t],
            "P_hPa": profiles["P_hPa"][t],
            "height": profiles["height"],
        }

        try:
            tb[t] = rtm.simulate(prof)
        except Exception as e:
            print(f"    t={t} FAIL: {e}")
            tb[t] = np.nan

        if (t + 1) % 100 == 0:
            print(f"    {t+1}/{n_time}...", flush=True)

    return tb


def process_month(pl_path, bt_output):
    """Load ERA5 month → MonoRTM simulate → save BT file. Returns True on success."""
    print(f"\n{'─'*50}")
    print(f"Processing: {os.path.basename(pl_path)}")

    # Load and prepare
    print("  Loading & interpolating...", flush=True)
    profiles = load_and_prepare_month(pl_path)
    if profiles is None:
        return False

    n_time = profiles["T"].shape[0]
    print(f"  {n_time} profiles, {config.N_LAYERS} layers")
    print(f"  T: {np.nanmean(profiles['T']):.1f}±{np.nanstd(profiles['T']):.1f} K")
    print(f"  RH: {np.nanmean(profiles['RH']):.1f}±{np.nanstd(profiles['RH']):.1f} %")

    # Run MonoRTM
    print("  Running MonoRTM...", flush=True)
    t0 = time.time()
    tb = simulate_batch_monortm(profiles)
    elapsed = time.time() - t0

    n_valid = np.sum(~np.isnan(tb[:, 0]))
    print(f"  {n_valid}/{n_time} profiles OK, {elapsed:.0f}s ({elapsed/n_valid:.1f}s per profile)")

    # Save
    np.savez_compressed(
        bt_output,
        tb=tb,
        time=profiles["time"],
        height=profiles["height"],
        valid_mask=~np.isnan(tb[:, 0]),
    )
    print(f"  → Saved: {bt_output} ({os.path.getsize(bt_output)//1024}KB)")
    return True


def month_output(year, month):
    return os.path.join(BT_DIR, f"bt_monortm_{year}_{month:02d}.npz")


# ============================================================
# Main
# ============================================================

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Batch MonoRTM BT simulation from ERA5 pressure-level data")
    ap.add_argument("year", nargs="?", type=int, help="Specific year")
    ap.add_argument("month", nargs="?", type=int, help="Specific month")
    ap.add_argument("--parallel", type=int, default=1, help="Number of months to process in parallel")
    args = ap.parse_args()

    if args.year and args.month:
        years_months = [(args.year, args.month)]
    else:
        years_months = [(y, m) for y in range(2013, 2020) for m in range(1, 13)]

    # Check which months are already done
    pending = []
    for y, m in years_months:
        out = month_output(y, m)
        pl = os.path.join(ERA5_DIR, f"pl_{y}_{m:02d}.nc")
        if os.path.exists(out) and os.path.getsize(out) > 50000:
            print(f"SKIP {y}-{m:02d} (BT file exists)")
        elif os.path.exists(pl) and os.path.getsize(pl) > 200000:
            pending.append((y, m, pl))
        else:
            print(f"SKIP {y}-{m:02d} (no pressure-level data)")

    if not pending:
        print("\nNo pending months to process.")
        print("Run dl_pl_cds_v2.py first to download ERA5 pressure-level data.")
        return

    print(f"\nPending: {len(pending)} months")
    print(f"MonoRTM: {MONORTM_BIN}")
    print(f"TAPE3:   {TAPE3_BIN}")
    print(f"Output:  {BT_DIR}/")
    print(f"Mode:    {'parallel × ' + str(args.parallel) if args.parallel > 1 else 'sequential'}")
    print()

    if args.parallel > 1:
        with ProcessPoolExecutor(max_workers=args.parallel) as pool:
            futures = {
                pool.submit(process_month, pl, month_output(y, m)): (y, m)
                for y, m, pl in pending
            }
            for f in as_completed(futures):
                y, m = futures[f]
                try:
                    ok = f.result()
                    print(f"  {y}-{m:02d}: {'✓' if ok else '✗'}")
                except Exception as e:
                    print(f"  {y}-{m:02d}: EXCEPTION {e}")
    else:
        ok, fail = 0, 0
        for y, m, pl in pending:
            result = process_month(pl, month_output(y, m))
            if result:
                ok += 1
            else:
                fail += 1
            print(f"  Progress: {ok} ok, {fail} fail, {len(pending)-ok-fail} remaining")

    print(f"\n{'='*50}")
    print(f"DONE. BT files in {BT_DIR}/")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
