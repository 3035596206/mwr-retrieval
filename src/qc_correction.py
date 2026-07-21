"""ERA5 quality control and bias correction for training sample preparation.

Implements the QC scheme from Section 3.2.2 (Fig 3.7):
1. Humidity scaling by 0.9 factor
2. Seasonal layer-wise RH bias correction
3. LWC screening (delete if ICLWC > 1250, scale if > 750)
4. Brightness temperature linear correction per channel
"""

import numpy as np
import config


def compute_iwv(rh_profile, t_profile, p_profile, height_profile):
    """Compute integrated water vapor [kg/m^2] from 0 to 10km.

    IWV = integral(rho_v * dz), rho_v from RH, T, P.
    """
    es = 6.1078 * np.exp(17.2693882 * (t_profile - 273.16) /
                          (t_profile - 35.86))  # saturation vapor pressure [hPa]
    e = rh_profile / 100.0 * es
    rho_v = e * 216.7 / t_profile  # water vapor density [g/m^3]

    # Integrate
    dz = np.diff(height_profile, prepend=0)
    iwv = np.sum(rho_v * dz) / 1000.0  # g/m^2 -> kg/m^2

    return iwv


def compute_iclwc(clwc_profile, height_profile):
    """Compute integrated cloud liquid water content [g/m^2].

    ICLWC = integral(CLWC * dz)
    """
    dz = np.diff(height_profile, prepend=0)
    iclwc = np.sum(clwc_profile * dz)
    return iclwc


def scale_humidity(rh_profiles):
    """Scale humidity profiles by 0.9 (from ERA5 vs sounding IWV regression).

    Args:
        rh_profiles: shape (n_time, n_layers) or (n_layers,)
    Returns:
        scaled RH profiles
    """
    return rh_profiles * config.RH_SCALING_FACTOR


def compute_seasonal_rh_bias(era5_rh, sounding_rh, months):
    """Compute seasonal RH bias between ERA5 and sounding for each height layer.

    Args:
        era5_rh: ERA5 RH profiles, shape (n_samples, n_layers)
        sounding_rh: sounding RH profiles, shape (n_samples, n_layers)
        months: month numbers for each sample, shape (n_samples,)
    Returns:
        bias_lookup: dict season -> array of bias per layer (n_layers,)
    """
    seasons = {
        "spring": [3, 4, 5],
        "summer": [6, 7, 8],
        "autumn": [9, 10, 11],
        "winter": [12, 1, 2],
    }

    n_layers = era5_rh.shape[1]
    bias_lookup = {}

    for season, season_months in seasons.items():
        mask = np.isin(months, season_months)
        if mask.sum() < 10:
            bias_lookup[season] = np.zeros(n_layers)
            continue
        bias = np.mean(era5_rh[mask] - sounding_rh[mask], axis=0)
        bias_lookup[season] = bias

    return bias_lookup


def correct_seasonal_rh_bias(rh_profiles, months, bias_lookup):
    """Apply seasonal layer-wise RH bias correction.

    Args:
        rh_profiles: shape (n_samples, n_layers)
        months: shape (n_samples,)
        bias_lookup: dict season -> bias array (n_layers,)
    Returns:
        corrected RH profiles
    """
    seasons = {
        "spring": [3, 4, 5],
        "summer": [6, 7, 8],
        "autumn": [9, 10, 11],
        "winter": [12, 1, 2],
    }

    corrected = rh_profiles.copy()
    season_labels = np.full(len(months), "", dtype=object)

    for season, season_months in seasons.items():
        mask = np.isin(months, season_months)
        season_labels[mask] = season
        if season in bias_lookup and mask.sum() > 0:
            corrected[mask] -= bias_lookup[season]

    return np.clip(corrected, 0, 100)


def screen_and_scale_lwc(clwc_profiles, height_profile):
    """Screen and scale liquid water content according to ICLWC thresholds.

    - ICLWC > 1250 g/m^2: flag for deletion
    - 750 < ICLWC <= 1250: scale by 750 / ICLWC
    - ICLWC <= 750: keep as is

    Args:
        clwc_profiles: shape (n_samples, n_layers)
        height_profile: shape (n_layers,)
    Returns:
        clwc_corrected: shape (n_samples, n_layers)
        delete_mask: shape (n_samples,) bool array, True = delete
    """
    n_samples = clwc_profiles.shape[0]
    delete_mask = np.zeros(n_samples, dtype=bool)
    clwc_corrected = clwc_profiles.copy()

    for i in range(n_samples):
        iclwc = compute_iclwc(clwc_profiles[i], height_profile)

        if iclwc > config.ICLWC_DELETE_THRESHOLD:
            delete_mask[i] = True
        elif iclwc > config.ICLWC_SCALE_THRESHOLD:
            clwc_corrected[i] = clwc_profiles[i] * (config.ICLWC_SCALE_FACTOR / iclwc)

    return clwc_corrected, delete_mask


def build_bt_linear_correction(tbs_sim, tbs_obs):
    """Build per-channel linear BT correction model.

    tbs_obs = a * tbs_sim + b
    After correction, tbs_sim_corrected = a * tbs_sim + b

    Args:
        tbs_sim: simulated BT, shape (n_samples, n_channels)
        tbs_obs: observed BT, shape (n_samples, n_channels)
    Returns:
        slope: shape (n_channels,)
        intercept: shape (n_channels,)
    """
    n_channels = tbs_sim.shape[1]
    slope = np.zeros(n_channels)
    intercept = np.zeros(n_channels)

    for ch in range(n_channels):
        X = tbs_sim[:, ch]
        y = tbs_obs[:, ch]
        # Linear regression y = a*X + b
        valid = np.isfinite(X) & np.isfinite(y)
        if valid.sum() < 10:
            slope[ch] = 1.0
            intercept[ch] = 0.0
            continue

        A = np.vstack([X[valid], np.ones_like(X[valid])]).T
        coeff = np.linalg.lstsq(A, y[valid], rcond=None)[0]
        slope[ch] = coeff[0]
        intercept[ch] = coeff[1]

    return slope, intercept


def apply_bt_correction(tbs_sim, slope, intercept):
    """Apply linear BT correction: tbs_corrected = slope * tbs_sim + intercept.

    Args:
        tbs_sim: simulated BT, shape (n_samples, n_channels) or (n_channels,)
        slope, intercept: shape (n_channels,)
    """
    return slope * tbs_sim + intercept


def apply_full_qc(era5_profiles, era5_tbs, sounding_profiles=None,
                  obs_tbs=None):
    """Apply the complete ERA5 QC pipeline (Fig 3.7).

    Args:
        era5_profiles: dict with 'T', 'RH', 'CLWC', 'height', 'time'
        era5_tbs: simulated BT from ERA5, shape (n_samples, 14)
        sounding_profiles: optional sounding data for bias computation
        obs_tbs: observed BT for linear correction

    Returns:
        qc_profiles: QC-corrected profiles
        qc_tbs: QC-corrected brightness temperatures
        keep_mask: boolean mask of retained samples
        qc_info: dict with QC statistics
    """
    n_samples = era5_profiles["T"].shape[0]
    height = era5_profiles["height"]

    # Step 1: Scale humidity by 0.9
    print("QC Step 1: Scaling humidity by 0.9...")
    rh_scaled = scale_humidity(era5_profiles["RH"])

    # Step 2: Seasonal layer-wise RH bias correction
    print("QC Step 2: Seasonal RH bias correction...")
    months = np.array([t.month for t in era5_profiles["time"]])

    if sounding_profiles is not None:
        # Compute bias from comparison with soundings
        bias_lookup = compute_seasonal_rh_bias(
            rh_scaled, sounding_profiles["RH"], months
        )
    else:
        # Use typical biases (paper reports ~10-13% at upper levels)
        bias_lookup = {
            "spring": np.zeros(config.N_LAYERS),
            "summer": np.zeros(config.N_LAYERS),
            "autumn": np.zeros(config.N_LAYERS),
            "winter": np.zeros(config.N_LAYERS),
        }
        # Approximate: upper levels tend to have positive bias
        for season in bias_lookup:
            for h in range(config.N_LAYERS):
                if config.HEIGHT_GRID[h] > 7000:
                    bias_lookup[season][h] = 10.0
                elif config.HEIGHT_GRID[h] > 4000:
                    bias_lookup[season][h] = 5.0

    rh_corrected = correct_seasonal_rh_bias(rh_scaled, months, bias_lookup)

    # Step 3: LWC screening and scaling
    print("QC Step 3: LWC screening and scaling...")
    clwc_corrected, delete_mask = screen_and_scale_lwc(
        era5_profiles["CLWC"], height
    )

    keep_mask = ~delete_mask
    print(f"  Deleted {delete_mask.sum()} / {n_samples} samples (LWC too high)")

    # Step 4: BT linear correction (if obs_tbs available)
    print("QC Step 4: BT linear correction...")
    if obs_tbs is not None:
        # Build correction from matching simulated and observed BT
        slope, intercept = build_bt_linear_correction(
            era5_tbs[keep_mask], obs_tbs[keep_mask]
        )
        tbs_corrected = apply_bt_correction(era5_tbs, slope, intercept)
    else:
        slope = np.ones(config.N_CHANNELS)
        intercept = np.zeros(config.N_CHANNELS)
        tbs_corrected = era5_tbs.copy()

    # Compile results
    qc_profiles = {
        "time": era5_profiles["time"],
        "height": height,
        "T": era5_profiles["T"],
        "RH": rh_corrected,
        "P": era5_profiles.get("P", np.ones((n_samples, config.N_LAYERS)) * 1013.0),
        "CLWC": clwc_corrected,
        "t2m": era5_profiles.get("t2m"),
        "rh2m": era5_profiles.get("rh2m"),
        "sp": era5_profiles.get("sp"),
    }

    qc_info = {
        "humidity_scale_factor": config.RH_SCALING_FACTOR,
        "n_deleted": int(delete_mask.sum()),
        "n_retained": int(keep_mask.sum()),
        "bt_slope": slope,
        "bt_intercept": intercept,
        "seasonal_rh_bias": bias_lookup,
    }

    return qc_profiles, tbs_corrected, keep_mask, qc_info


if __name__ == "__main__":
    # Quick test
    n_test = 100
    height = np.array(config.HEIGHT_GRID)

    dummy_profiles = {
        "time": np.array([np.datetime64("2015-06-15")] * n_test),
        "height": height,
        "T": np.random.randn(n_test, config.N_LAYERS) * 10 + 280.0,
        "RH": np.random.rand(n_test, config.N_LAYERS) * 50 + 30.0,
        "CLWC": np.random.rand(n_test, config.N_LAYERS) * 0.1,
        "t2m": np.random.randn(n_test) * 3 + 290.0,
        "rh2m": np.random.rand(n_test) * 30 + 40.0,
        "sp": np.ones(n_test) * 101300.0,
    }

    dummy_tbs = np.random.randn(n_test, 14) * 20 + 100.0

    qc_profiles, qc_tbs, keep_mask, qc_info = apply_full_qc(
        dummy_profiles, dummy_tbs
    )

    print(f"\nQC Results:")
    print(f"  Samples retained: {keep_mask.sum()} / {n_test}")
    print(f"  RH scaling: {qc_info['humidity_scale_factor']}")
