"""OEM error covariance construction: S_a (background) and S_e (observation).

Provides builders for:
  - S_a: background error covariance — from exponential correlation model
    (Plan §6.1) or from BRNN v4 residuals (Plan §6.2)
  - S_e: observation error covariance — diagonal from per-channel σ (Plan §7)

Usage:
    from oem_covariance import build_sa_exponential, build_se_diagonal

    S_a = build_sa_exponential(state_packer, sigma_T=1.5, sigma_RH=8.0)
    S_e = build_se_diagonal(forward_model, sigma_K=1.5, sigma_V=0.5)
"""

import numpy as np
import config


# ================================================================
# S_a: Background error covariance
# ================================================================

def build_sa_exponential(state_packer, sigma_T=1.5, sigma_RH=8.0,
                         length_T=1.5, length_RH=0.75,
                         reg=1e-4):
    """Build S_a using the exponential vertical correlation model (Plan §6.1).

    S_a[i,j] = σ_i · σ_j · exp(-|z_i - z_j| / L)

    Args:
        state_packer: OEMStatePacker instance (provides control heights & n_state)
        sigma_T: temperature standard deviation [K]
        sigma_RH: humidity standard deviation [%]
        length_T: vertical correlation length for T [km]
        length_RH: vertical correlation length for RH [km]
        reg: Tikhonov regularisation added to diagonal

    Returns:
        S_a: (n_state, n_state) background error covariance matrix
    """
    n = state_packer.n_state
    S_a = np.zeros((n, n))

    # Build height list for each state element
    heights_m = _state_element_heights(state_packer)

    for i in range(n):
        for j in range(n):
            kind_i = state_packer.element_kind(i)
            kind_j = state_packer.element_kind(j)

            # Cross-variable correlations are zero in this simple model
            if kind_i != kind_j:
                S_a[i, j] = 0.0
                continue

            sigma = sigma_T if kind_i == "T" else sigma_RH
            length = length_T if kind_i == "T" else length_RH

            dz_km = abs(heights_m[i] - heights_m[j]) / 1000.0
            S_a[i, j] = sigma * sigma * np.exp(-dz_km / length)

    # Regularise to ensure positive definiteness
    S_a += np.eye(n) * reg * np.mean(np.diag(S_a))

    return S_a


def build_sa_from_v4_results(results_path="results/mp3000a_v4_results.pkl",
                              state_packer=None, reg=1e-4):
    """Estimate S_a from BRNN v4 prediction residuals (Plan §6.2).

    Computes T_error = T_pred - T_true, RH_error = RH_pred - RH_true,
    then projects the error covariance into the OEM control space.

    Args:
        results_path: path to the v4 results pickle
        state_packer: OEMStatePacker; if None, uses make_default_packer()
        reg: Tikhonov regularisation

    Returns:
        S_a: (n_state, n_state) covariance matrix in OEM control space
    """
    import pickle
    from oem_state import make_default_packer

    if state_packer is None:
        state_packer = make_default_packer()

    with open(results_path, "rb") as f:
        data = pickle.load(f)

    T_err = data["T_pred"] - data["T_true"]       # (n, 93)
    RH_err = data["RH_pred"] - data["RH_true"]     # (n, 93)

    # Pack errors into OEM control space
    n_samples = T_err.shape[0]
    n_state = state_packer.n_state
    err_ctrl = np.zeros((n_samples, n_state))

    for i in range(n_samples):
        profile = {
            "T": T_err[i],
            "RH": RH_err[i],
            "CLWC": np.zeros(config.N_LAYERS),
        }
        err_ctrl[i] = state_packer.pack(profile)

    # Covariance in control space (remove mean first)
    err_ctrl -= err_ctrl.mean(axis=0, keepdims=True)
    S_a = (err_ctrl.T @ err_ctrl) / (n_samples - 1)

    # Regularise
    S_a += np.eye(n_state) * reg * np.mean(np.diag(S_a))

    return S_a


# ================================================================
# S_e: Observation error covariance
# ================================================================

def build_se_diagonal(forward_model=None, sigma_K=1.5, sigma_V=0.5,
                      frequencies=None):
    """Build a diagonal observation error covariance matrix (Plan §7).

    By default, uses per-band default σ values.  Can be overridden with
    a full per-channel sigma list.

    Args:
        forward_model: ForwardModel instance (provides frequencies & n_channels)
        sigma_K: K-band (22-32 GHz) std [K]
        sigma_V: V-band (51-58 GHz) std [K]
        frequencies: explicit frequency list (overrides forward_model)

    Returns:
        S_e: (n_channels, n_channels) diagonal matrix
    """
    if frequencies is None and forward_model is not None:
        frequencies = forward_model.frequencies
    if frequencies is None:
        frequencies = config.ALL_CHANNELS

    n = len(frequencies)
    S_e = np.zeros((n, n))

    for i, f in enumerate(frequencies):
        S_e[i, i] = sigma_K ** 2 if f < 40 else sigma_V ** 2

    return S_e


def build_se_from_channels(sigmas, frequencies=None, forward_model=None):
    """Build a diagonal S_e from a per-channel sigma list.

    Args:
        sigmas: list of per-channel standard deviations [K]
        frequencies: channel frequencies (defaults to forward_model or config)
        forward_model: ForwardModel instance

    Returns:
        S_e: (n_channels, n_channels) diagonal matrix
    """
    if frequencies is None and forward_model is not None:
        frequencies = forward_model.frequencies
    if frequencies is None:
        frequencies = config.ALL_CHANNELS

    n = len(frequencies)
    if len(sigmas) != n:
        raise ValueError(
            f"sigmas length ({len(sigmas)}) != n_channels ({n})"
        )

    S_e = np.diag(np.array(sigmas, dtype=float) ** 2)
    return S_e


def inflate_se_for_cloud(S_e, frequencies=None, forward_model=None,
                          cloud_factor=4.0, k_band_only=True,
                          per_channel_factors=None):
    """Inflate observation errors for cloud-contaminated channels.

    Supports two modes:
      1. Uniform: single cloud_factor applied to K-band (or all channels)
      2. Per-channel: per_channel_factors dict mapping channel index → factor

    Per design doc §3.3, 31.4 GHz (K7/ch6) is most LWC-sensitive and should
    get a larger inflation factor than other K-band channels.

    Args:
        S_e: existing (n_channels, n_channels) observation error covariance
        frequencies: channel frequencies
        forward_model: ForwardModel (alternative to frequencies)
        cloud_factor: default multiplicative factor for variance
        k_band_only: if True, only inflate K-band (f < 40 GHz)
        per_channel_factors: optional dict {ch_idx: factor} for per-channel tuning

    Returns:
        S_e_inflated: copy of S_e with inflated variances
    """
    if frequencies is None and forward_model is not None:
        frequencies = forward_model.frequencies
    if frequencies is None:
        frequencies = config.ALL_CHANNELS

    S_e_new = S_e.copy()
    for i, f in enumerate(frequencies):
        if not k_band_only or f < 40:
            factor = cloud_factor
            if per_channel_factors and i in per_channel_factors:
                factor = per_channel_factors[i]
            S_e_new[i, i] *= factor

    return S_e_new


def cloud_inflation_factor(iclwc):
    """Compute K-band S_e inflation factor from column-integrated LWC.

    Design doc §3.2:
      ICLWC < 200 g/m^2:   factor = 1.0    (clear-sky)
      200-750 g/m^2:       linear ramp 1.0 → 4.0
      > 750 g/m^2:          factor = 9.0    (thick cloud)

    Args:
        iclwc: float or array, integrated cloud liquid water [g/m^2]

    Returns:
        float or array, multiplicative factor for K-band S_e variance
    """
    iclwc = np.asarray(iclwc, dtype=float)
    factor = np.ones_like(iclwc)

    # Thin cloud: linear ramp
    ramp_mask = (iclwc >= 200) & (iclwc < 750)
    factor[ramp_mask] = 1.0 + (iclwc[ramp_mask] - 200.0) / (750.0 - 200.0) * 3.0

    # Thick cloud: saturate
    thick_mask = iclwc >= 750
    factor[thick_mask] = 9.0

    return float(factor) if factor.ndim == 0 else factor


def adapt_se_for_cloud(S_e_base, iclwc, frequencies=None, forward_model=None):
    """Build a cloud-adapted S_e with per-channel differentiated inflation.

    Combines cloud_inflation_factor (ICLWC-dependent) with per-channel
    sensitivity weights (design doc §3.3):
      - 31.4 GHz (ch6): 2× base factor (most LWC-sensitive)
      - 22.24 GHz (ch0): 1.5× (H2O line overlap)
      - Other K-band: 1× base factor
      - V-band: no inflation (O2-dominated, insensitive to LWC)

    Args:
        S_e_base: baseline (n_channels, n_channels) diagonal S_e
        iclwc: integrated cloud liquid water [g/m^2]
        frequencies: channel frequencies (defaults to config.ALL_CHANNELS)
        forward_model: ForwardModel instance

    Returns:
        S_e_adapted: (n_channels, n_channels) cloud-adapted S_e
    """
    if frequencies is None and forward_model is not None:
        frequencies = forward_model.frequencies
    if frequencies is None:
        frequencies = config.ALL_CHANNELS

    base_factor = cloud_inflation_factor(iclwc)

    # Per-channel sensitivity weights (relative to base K-band factor)
    # ch0 (22.24 GHz): 1.5× — H2O line + LWC overlap
    # ch1-5 (23-28 GHz): 1.0× — baseline
    # ch6 (31.40 GHz): 2.0× — strongest LWC sensitivity
    per_channel = {
        0: base_factor * 1.5,   # 22.24 GHz
        6: base_factor * 2.0,   # 31.40 GHz
    }

    return inflate_se_for_cloud(
        S_e_base,
        frequencies=frequencies,
        cloud_factor=base_factor,
        k_band_only=True,
        per_channel_factors=per_channel,
    )


def compute_iclwc_from_profile(profile):
    """Compute ICLWC [g/m^2] from an atmospheric profile dict.

    Integrates CLWC * dz over the height grid.

    Args:
        profile: dict with 'CLWC' (n_layers,) and 'height' (n_layers,)

    Returns:
        iclwc: integrated cloud liquid water [g/m^2]
    """
    clwc = np.asarray(profile["CLWC"], dtype=float)
    heights = np.asarray(profile["height"], dtype=float)
    dz = np.diff(heights, prepend=0)
    return float(np.sum(clwc * dz))


# ================================================================
# Internal helpers
# ================================================================

def _state_element_heights(state_packer):
    """Return the control height [m] for each state element.

    For coarse mode this is the control-point height; for EOF mode
    we use a representative mid-layer height.
    """
    if state_packer.mode == "coarse":
        heights = []
        for h in state_packer.coarse_T:
            heights.append(h)
        for h in state_packer.coarse_RH:
            heights.append(h)
        for h in state_packer.coarse_LWC:
            heights.append(h)
        return heights
    else:
        # EOF mode: use mid-layer heights as a proxy
        n = state_packer.n_state
        return [state_packer.heights[len(state_packer.heights) // 2]] * n
