#!/usr/bin/env python3
"""Main orchestration script for BRNN atmospheric profile retrieval.

Pipeline:
  1. Download ERA5 data (or load from cache)
  2. Download radiosonde data (or load from cache)
  3. Preprocess: interpolate to MWR height grid
  4. Simulate brightness temperatures
  5. Apply ERA5 QC and linear correction
  6. Train BRNN models
  7. Evaluate against radiosonde truth

Usage:
    python run.py --stage all          # Run full pipeline
    python run.py --stage download     # Download data only
    python run.py --stage preprocess   # Preprocess only
    python run.py --stage train        # Train only
    python run.py --stage evaluate     # Evaluate only

Before running, set up CDS API:
    Create ~/.cdsapirc with your Copernicus CDS credentials.
"""

import os
import sys
import argparse
import pickle
import numpy as np
from datetime import datetime

import config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def stage_download():
    """Download ERA5 and radiosonde data."""
    print("=" * 60)
    print("Stage 1: Data Download")
    print("=" * 60)

    from src.era5_preprocess import (
        download_era5_pressure_levels, download_era5_single_levels
    )

    os.makedirs(config.ERA5_DIR, exist_ok=True)
    os.makedirs(config.RADIOSONDE_DIR, exist_ok=True)

    area = [40.5, 115.0, 39.0, 118.0]  # [N, W, S, E]

    # Download ERA5 by year
    for year in range(config.TRAIN_YEARS[0], config.VALIDATION_YEARS[1] + 1):
        print(f"\nDownloading ERA5 for {year}...")
        pressure_file = os.path.join(config.ERA5_DIR, f"era5_pressure_{year}.nc")
        single_file = os.path.join(config.ERA5_DIR, f"era5_single_{year}.nc")

        if not os.path.exists(pressure_file):
            download_era5_pressure_levels(
                [year], list(range(1, 13)), area, pressure_file
            )
        else:
            print(f"  {pressure_file} already exists, skipping")

        if not os.path.exists(single_file):
            download_era5_single_levels(
                [year], list(range(1, 13)), area, single_file
            )
        else:
            print(f"  {single_file} already exists, skipping")

    print("\nERA5 download complete")

    # Download radiosonde data
    from src.sounding_process import download_sounding
    print("\nDownloading radiosonde data...")
    for year in range(config.TRAIN_YEARS[0], config.VALIDATION_YEARS[1] + 1):
        for month in range(1, 13):
            filepath = os.path.join(
                config.RADIOSONDE_DIR, f"sounding_54511_{year}{month:02d}.txt"
            )
            if not os.path.exists(filepath):
                download_sounding(54511, year, month, config.RADIOSONDE_DIR)
        print(f"  Year {year} done")

    print("Radiosonde download complete")


def stage_preprocess():
    """Preprocess ERA5 and radiosonde data to MWR height grid."""
    print("=" * 60)
    print("Stage 2: Data Preprocessing")
    print("=" * 60)

    from src.era5_preprocess import extract_era5_profiles, interpolate_to_mwr_grid
    from src.sounding_process import process_all_soundings

    # Process ERA5
    era5_cache_path = os.path.join(config.ERA5_DIR, "era5_profiles.pkl")
    if os.path.exists(era5_cache_path):
        print(f"Loading cached ERA5 profiles from {era5_cache_path}")
        with open(era5_cache_path, "rb") as f:
            era5_profiles = pickle.load(f)
    else:
        era5_profiles_list = []
        for year in range(config.TRAIN_YEARS[0], config.VALIDATION_YEARS[1] + 1):
            pressure_file = os.path.join(config.ERA5_DIR, f"era5_pressure_{year}.nc")
            single_file = os.path.join(config.ERA5_DIR, f"era5_single_{year}.nc")

            if not os.path.exists(pressure_file) or not os.path.exists(single_file):
                print(f"  Skipping {year}: files not found")
                continue

            print(f"  Processing {year}...")
            profiles_raw = extract_era5_profiles(
                pressure_file, single_file,
                config.NANJIAO_LAT, config.NANJIAO_LON
            )
            profiles_mwr = interpolate_to_mwr_grid(profiles_raw)
            era5_profiles_list.append(profiles_mwr)

        # Concatenate all years
        era5_profiles = {
            "time": np.concatenate([p["time"] for p in era5_profiles_list]),
            "height": era5_profiles_list[0]["height"],
            "T": np.concatenate([p["T"] for p in era5_profiles_list]),
            "RH": np.concatenate([p["RH"] for p in era5_profiles_list]),
            "CLWC": np.concatenate([p["CLWC"] for p in era5_profiles_list]),
            "t2m": np.concatenate([p["t2m"] for p in era5_profiles_list]),
            "rh2m": np.concatenate([p["rh2m"] for p in era5_profiles_list]),
            "sp": np.concatenate([p["sp"] for p in era5_profiles_list]),
        }
        print(f"  Total ERA5 samples: {len(era5_profiles['time'])}")

        with open(era5_cache_path, "wb") as f:
            pickle.dump(era5_profiles, f)
        print(f"  Saved cache to {era5_cache_path}")

    # Process radiosondes
    sounding_cache_path = os.path.join(config.RADIOSONDE_DIR, "sounding_profiles.pkl")
    if os.path.exists(sounding_cache_path):
        print(f"Loading cached sounding profiles from {sounding_cache_path}")
        with open(sounding_cache_path, "rb") as f:
            sounding_profiles = pickle.load(f)
    else:
        all_sounding_profiles = []
        for year in range(config.TRAIN_YEARS[0], config.VALIDATION_YEARS[1] + 1):
            for month in range(1, 13):
                filepath = os.path.join(
                    config.RADIOSONDE_DIR, f"sounding_54511_{year}{month:02d}.txt"
                )
                if not os.path.exists(filepath):
                    continue
                profiles = process_all_soundings(filepath)
                all_sounding_profiles.extend(profiles)

        # Organize into arrays
        if all_sounding_profiles:
            n = len(all_sounding_profiles)
            n_layers = len(all_sounding_profiles[0]["height"])
            sounding_profiles = {
                "time": [p["time"] for p in all_sounding_profiles],
                "height": all_sounding_profiles[0]["height"],
                "T": np.array([p["T"] for p in all_sounding_profiles]),
                "RH": np.array([p["RH"] for p in all_sounding_profiles]),
                "P": np.array([p["P"] for p in all_sounding_profiles]),
                "CLWC": np.array([p["CLWC"] for p in all_sounding_profiles]),
                "t2m": np.array([p["t2m"] for p in all_sounding_profiles]),
                "rh2m": np.array([p["rh2m"] for p in all_sounding_profiles]),
            }
        else:
            sounding_profiles = None

        print(f"  Total sounding profiles: "
              f"{len(all_sounding_profiles) if all_sounding_profiles else 0}")

        with open(sounding_cache_path, "wb") as f:
            pickle.dump(sounding_profiles, f)

    # Simulate brightness temperatures from ERA5 profiles
    bt_cache_path = os.path.join(config.ERA5_DIR, "era5_bt_sim.pkl")
    if os.path.exists(bt_cache_path):
        print(f"Loading cached BT from {bt_cache_path}")
        with open(bt_cache_path, "rb") as f:
            era5_tbs = pickle.load(f)
    else:
        from src.brightness_temp import simulate_batch

        # Add pressure info if missing
        if "P" not in era5_profiles or era5_profiles.get("P") is None:
            era5_profiles["P"] = np.ones((len(era5_profiles["time"]),
                                           config.N_LAYERS)) * 1013.0

        print("Simulating brightness temperatures from ERA5...")
        era5_tbs = simulate_batch(era5_profiles, add_noise=False)
        print(f"  BT shape: {era5_tbs.shape}")

        with open(bt_cache_path, "wb") as f:
            pickle.dump(era5_tbs, f)

    # Also simulate BT from sounding profiles (for bias comparison)
    if sounding_profiles is not None:
        bt_sounding_path = os.path.join(config.RADIOSONDE_DIR, "sounding_bt_sim.pkl")
        if not os.path.exists(bt_sounding_path):
            from src.brightness_temp import simulate_batch
            print("Simulating BT from sounding profiles...")
            sounding_tbs = simulate_batch(sounding_profiles, add_noise=False)
            with open(bt_sounding_path, "wb") as f:
                pickle.dump(sounding_tbs, f)
            print(f"  BT shape: {sounding_tbs.shape}")

    print("\nPreprocessing complete.")


def stage_qc():
    """Apply ERA5 quality control and linear corrections."""
    print("=" * 60)
    print("Stage 3: ERA5 Quality Control")
    print("=" * 60)

    from src.qc_correction import apply_full_qc

    # Load data
    with open(os.path.join(config.ERA5_DIR, "era5_profiles.pkl"), "rb") as f:
        era5_profiles = pickle.load(f)
    with open(os.path.join(config.ERA5_DIR, "era5_bt_sim.pkl"), "rb") as f:
        era5_tbs = pickle.load(f)

    # Load sounding if available
    sounding_cache = os.path.join(config.RADIOSONDE_DIR, "sounding_profiles.pkl")
    if os.path.exists(sounding_cache):
        with open(sounding_cache, "rb") as f:
            sounding_profiles = pickle.load(f)
    else:
        sounding_profiles = None

    qc_profiles, qc_tbs, keep_mask, qc_info = apply_full_qc(
        era5_profiles, era5_tbs, sounding_profiles
    )

    # Save QC'd data
    qc_data = {
        "profiles": qc_profiles,
        "tbs": qc_tbs,
        "keep_mask": keep_mask,
        "qc_info": qc_info,
    }
    output_path = os.path.join(config.ERA5_DIR, "era5_qc_data.npz")
    np.savez_compressed(output_path, **qc_data)
    print(f"\nQC data saved to {output_path}")
    print(f"  Original samples: {len(era5_profiles['time'])}")
    print(f"  After QC: {keep_mask.sum()}")
    print(f"  Deleted: {(~keep_mask).sum()}")


def stage_train():
    """Train BRNN models."""
    print("=" * 60)
    print("Stage 4: Training BRNN Models")
    print("=" * 60)

    from src.train import train_all_models

    # Load QC'd data
    qc_path = os.path.join(config.ERA5_DIR, "era5_qc_data.npz")
    if os.path.exists(qc_path):
        data = np.load(qc_path, allow_pickle=True)
        profiles = data["profiles"].item()
        tbs = data["tbs"]
        keep_mask = data["keep_mask"]
        # Filter to kept samples
        if "T" in profiles:
            n_orig = profiles["T"].shape[0]
            for key in ["T", "RH", "CLWC", "t2m", "rh2m", "sp"]:
                if key in profiles and profiles[key] is not None:
                    profiles[key] = profiles[key][keep_mask]
            tbs = tbs[keep_mask]
            print(f"  Filtered: {n_orig} -> {keep_mask.sum()} samples")
    else:
        # Use raw data if QC hasn't been run
        print("  No QC data found, using raw ERA5 data")
        with open(os.path.join(config.ERA5_DIR, "era5_profiles.pkl"), "rb") as f:
            profiles = pickle.load(f)
        with open(os.path.join(config.ERA5_DIR, "era5_bt_sim.pkl"), "rb") as f:
            tbs = pickle.load(f)

    # Filter to training years (2013-2016)
    train_start = np.datetime64(f"{config.TRAIN_YEARS[0]}-01-01")
    train_end = np.datetime64(f"{config.TRAIN_YEARS[1]}-12-31")
    time_mask = (profiles["time"] >= train_start) & (profiles["time"] <= train_end)
    print(f"  Training samples (2013-2016): {time_mask.sum()}")

    for key in ["T", "RH", "CLWC", "t2m", "rh2m", "sp"]:
        if key in profiles and profiles[key] is not None:
            profiles[key] = profiles[key][time_mask]
    tbs = tbs[time_mask]

    os.makedirs(config.MODEL_DIR, exist_ok=True)
    device = "cpu"
    try:
        import torch
        if torch.backends.mps.is_available():
            device = "mps"
    except Exception:
        pass

    print(f"  Device: {device}")
    train_all_models(profiles, tbs, config.MODEL_DIR, device=device)


def stage_evaluate():
    """Evaluate trained BRNN models against radiosonde truth."""
    print("=" * 60)
    print("Stage 5: Evaluation")
    print("=" * 60)

    from src.brnn_model import BRNNEnsemble
    from src.evaluate import evaluate_full_profile, plot_bt_error_boxplot
    import config as cfg

    # Load sounding profiles (truth)
    sounding_path = os.path.join(config.RADIOSONDE_DIR, "sounding_profiles.pkl")
    if not os.path.exists(sounding_path):
        print("Sounding data not found, cannot evaluate")
        return
    with open(sounding_path, "rb") as f:
        sounding = pickle.load(f)

    # Load ERA5 data for validation period (2017-2019)
    with open(os.path.join(config.ERA5_DIR, "era5_profiles.pkl"), "rb") as f:
        era5 = pickle.load(f)

    val_start = np.datetime64(f"{config.VALIDATION_YEARS[0]}-01-01")
    val_end = np.datetime64(f"{config.VALIDATION_YEARS[1]}-12-31")
    val_mask = (era5["time"] >= val_start) & (era5["time"] <= val_end)

    with open(os.path.join(config.ERA5_DIR, "era5_bt_sim.pkl"), "rb") as f:
        era5_tbs_all = pickle.load(f)
    era5_tbs = era5_tbs_all[val_mask]

    # Build surface data
    t2m = era5["t2m"][val_mask] if era5["t2m"] is not None else era5["T"][val_mask, 0]
    rh2m = era5["rh2m"][val_mask] if era5["rh2m"] is not None else era5["RH"][val_mask, 0]
    sp = era5["sp"][val_mask] if era5["sp"] is not None else np.ones(val_mask.sum()) * 101300.0
    surface_data = np.column_stack([t2m, rh2m, sp / 100.0])

    # Load trained models
    device = "cpu"
    ensemble = BRNNEnsemble(config.HEIGHT_GRID, device=device)
    ensemble.load_all(config.MODEL_DIR)

    # Retrieve profiles
    print("Running retrieval on validation period...")
    T_retrieved, RH_retrieved = ensemble.predict(
        {name: model.state_dict() for name, model in ensemble.models.items()},
        era5_tbs, surface_data
    )

    # Match times with sounding
    era5_times = era5["time"][val_mask]
    sounding_times = np.array(sounding["time"])
    n_layers = len(config.HEIGHT_GRID)

    # Find matching pairs (within 1 hour)
    T_ret_matched = []
    RH_ret_matched = []
    T_truth_matched = []
    RH_truth_matched = []

    for i, et in enumerate(era5_times):
        diff = np.abs(sounding_times - et)
        min_diff = diff.min()
        if min_diff < np.timedelta64(1, "h"):
            j = diff.argmin()
            T_ret_matched.append(T_retrieved[i])
            RH_ret_matched.append(RH_retrieved[i])
            T_truth_matched.append(sounding["T"][j])
            RH_truth_matched.append(sounding["RH"][j])

    if T_ret_matched:
        T_ret_matched = np.array(T_ret_matched)
        RH_ret_matched = np.array(RH_ret_matched)
        T_truth_matched = np.array(T_truth_matched)
        RH_truth_matched = np.array(RH_truth_matched)

        os.makedirs(config.RESULT_DIR, exist_ok=True)
        print(f"\nMatched pairs: {len(T_ret_matched)}")

        evaluate_full_profile(
            T_ret_matched, RH_ret_matched,
            T_truth_matched, RH_truth_matched,
            config.HEIGHT_GRID, config.RESULT_DIR,
            method_name="ERA5-BRNN"
        )
    else:
        print("No matching sounding pairs found for validation period.")

    # BT error boxplot (simulated vs "observed" - using sounding-simulated BT)
    bt_sounding_path = os.path.join(config.RADIOSONDE_DIR, "sounding_bt_sim.pkl")
    if os.path.exists(bt_sounding_path):
        with open(bt_sounding_path, "rb") as f:
            sounding_tbs = pickle.load(f)
        plot_bt_error_boxplot(
            era5_tbs[:len(sounding_tbs)], sounding_tbs[:len(era5_tbs)],
            config.ALL_CHANNELS,
            os.path.join(config.RESULT_DIR, "bt_error_boxplot.png"),
            title="ERA5-simulated BT vs Sounding-simulated BT"
        )

    print("\nEvaluation complete.")


def main():
    parser = argparse.ArgumentParser(
        description="MWR Atmospheric Profile Retrieval Pipeline"
    )
    parser.add_argument(
        "--stage", type=str, required=True,
        choices=["all", "download", "preprocess", "qc", "train", "evaluate"],
        help="Pipeline stage to run"
    )
    args = parser.parse_args()

    stages = {
        "download": stage_download,
        "preprocess": stage_preprocess,
        "qc": stage_qc,
        "train": stage_train,
        "evaluate": stage_evaluate,
    }

    if args.stage == "all":
        for stage_name in ["download", "preprocess", "qc", "train", "evaluate"]:
            stages[stage_name]()
    else:
        stages[args.stage]()


if __name__ == "__main__":
    main()
