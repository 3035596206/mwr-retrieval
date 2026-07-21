"""OEM observation handling: channel selection, BT error, cloud flagging.

Provides utilities for:
  - Selecting / masking channels based on quality or cloud flags
  - Estimating per-channel observation errors from OMB statistics
  - Cloud-dependent channel weight inflation

Usage:
    from oem_observation import ChannelSelector, estimate_se_from_omb
"""

import numpy as np
import config


class ChannelSelector:
    """Channel selection and masking for OEM retrievals.

    Allows the OEM solver to operate on a subset of channels, e.g.
    excluding K-band channels during clear-sky retrievals or inflating
    errors for cloud-contaminated channels.
    """

    def __init__(self, frequencies=None, forward_model=None):
        """Initialise with channel information.

        Args:
            frequencies: list of channel frequencies [GHz]
            forward_model: ForwardModel instance (alternative source)
        """
        if frequencies is None and forward_model is not None:
            frequencies = forward_model.frequencies
        if frequencies is None:
            frequencies = config.ALL_CHANNELS

        self.frequencies = np.array(frequencies, dtype=float)
        self.n_all = len(self.frequencies)

    # ---- Band masks ----

    @property
    def k_band_mask(self):
        """Boolean mask: True for K-band (22-32 GHz) channels."""
        return self.frequencies < 40

    @property
    def v_band_mask(self):
        """Boolean mask: True for V-band (51-58 GHz) channels."""
        return self.frequencies >= 40

    @property
    def k_band_idx(self):
        """Integer indices of K-band channels."""
        return np.where(self.k_band_mask)[0]

    @property
    def v_band_idx(self):
        """Integer indices of V-band channels."""
        return np.where(self.v_band_mask)[0]

    # ---- Channel subsetting ----

    def select(self, mask_or_indices):
        """Return a sub-selector for a subset of channels.

        Args:
            mask_or_indices: boolean mask or integer indices

        Returns:
            ChannelSelector for the subset
        """
        if hasattr(mask_or_indices, "dtype") and mask_or_indices.dtype == bool:
            indices = np.where(mask_or_indices)[0]
        else:
            indices = np.asarray(mask_or_indices, dtype=int)

        sub_freqs = self.frequencies[indices]
        cs = ChannelSelector(frequencies=sub_freqs.tolist())
        cs._parent_indices = indices
        return cs

    def apply_mask(self, array, fill=0.0):
        """Apply the channel mask to an array, filling masked entries.

        Args:
            array: (n_all,) or (n_all, n_all) array
            fill: value for masked positions

        Returns:
            masked array with fill in masked positions
        """
        raise NotImplementedError("apply_mask: use parent index mapping instead")

    # ---- Cloud-flag based selection ----

    def cloud_free_mask(self, clwc_column, threshold=50.0):
        """Identify channels least affected by cloud.

        Simplified heuristic: if column-integrated CLWC > threshold g/m²,
        flag K-band channels as potentially contaminated.

        Args:
            clwc_column: column-integrated CLWC [g/m²] or cloud flag
            threshold: CLWC threshold [g/m²]

        Returns:
            mask: boolean array, True = usable (not cloud-contaminated)
        """
        mask = np.ones(self.n_all, dtype=bool)
        if np.any(clwc_column > threshold):
            mask[self.k_band_mask] = False
        return mask


# ================================================================
# Observation error estimation
# ================================================================

def estimate_se_from_omb(omb, mad_scaling=True):
    """Estimate per-channel observation error std from OMB statistics.

    OMB = Obs_BT - Sim_BT (after QC).

    Args:
        omb: (n_samples, n_channels) OMB array
        mad_scaling: if True, use MAD (robust) instead of std

    Returns:
        sigmas: (n_channels,) per-channel standard deviation [K]
    """
    if mad_scaling:
        # MAD → σ: 1.4826 factor for Gaussian
        from scipy.stats import median_abs_deviation
        sigmas = np.array([
            median_abs_deviation(omb[:, i], scale="normal")
            for i in range(omb.shape[1])
        ])
    else:
        sigmas = np.std(omb, axis=0)

    # Ensure minimum error (0.1 K floor)
    sigmas = np.maximum(sigmas, 0.1)

    return sigmas


def compute_omb(y_obs, y_sim):
    """Compute Observation-Minus-Background (or OMB).

    Args:
        y_obs: (n_channels,) or (n_samples, n_channels) observed BT
        y_sim: same shape, simulated BT

    Returns:
        omb: same shape
    """
    return np.asarray(y_obs) - np.asarray(y_sim)


# ================================================================
# Channel error recommendations (from Plan §7)
# ================================================================

def default_channel_sigmas(frequencies=None):
    """Return recommended per-channel observation σ [K] per Plan §7.

    | Channel type       | σ [K]     |
    |--------------------|-----------|
    | V-band 51-59 GHz   | 0.5 - 1.0 |
    | K-band 22-31 GHz   | 1.5 - 3.0 |
    | Cloud-risk K-band  | 3.0 - 5.0 |
    """
    if frequencies is None:
        frequencies = config.ALL_CHANNELS

    sigmas = np.zeros(len(frequencies))
    for i, f in enumerate(frequencies):
        if f >= 40:
            sigmas[i] = 0.75       # V-band: 0.5-1.0 K
        elif f <= 24:
            sigmas[i] = 2.0        # K-band water vapour line: higher error
        else:
            sigmas[i] = 1.5        # K-band window
    return sigmas
