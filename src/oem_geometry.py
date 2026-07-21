"""OEM geometry utilities: elevation angles, horizontal path length,
channel screening, and multi-angle observation organisation.
"""

import numpy as np


def airmass_factor(elevation_deg):
    """Plane-parallel airmass factor = 1/sin(elevation)."""
    elev_rad = np.deg2rad(np.clip(elevation_deg, 0.1, 90.0))
    return 1.0 / np.sin(elev_rad)


def horizontal_path_length(elevation_deg, height_km):
    """Horizontal propagation distance at given height and elevation."""
    elev_rad = np.deg2rad(np.clip(elevation_deg, 0.5, 89.5))
    return height_km / np.tan(elev_rad)


def screen_channels_by_geometry(frequencies_ghz, elevations_deg,
                                 min_elevation=10.0,
                                 max_horizontal_path=50.0,
                                 height_km=2.0,
                                 low_opacity_channels=None):
    """Screen (angle, channel) pairs by geometric criteria.

    Rule 1: elevation too low -> remove all channels at this angle
    Rule 2: horizontal path too long for low-opacity channels -> remove

    Returns:
        mask: (n_channels, n_angles) bool, True = keep
        report: dict with screening statistics
    """
    n_ch = len(frequencies_ghz)
    n_ang = len(elevations_deg)
    mask = np.ones((n_ch, n_ang), dtype=bool)

    if low_opacity_channels is None:
        low_opacity_channels = [i for i, f in enumerate(frequencies_ghz) if f < 40.0]

    n_removed_angle = 0
    n_removed_path = 0

    for j, elev in enumerate(elevations_deg):
        if elev < min_elevation:
            mask[:, j] = False
            n_removed_angle += n_ch
            continue

        h_path = horizontal_path_length(elev, height_km)
        if h_path > max_horizontal_path:
            for ch in low_opacity_channels:
                mask[ch, j] = False
                n_removed_path += 1

    report = {
        "total_pairs": n_ch * n_ang,
        "retained": int(mask.sum()),
        "removed_by_angle": n_removed_angle,
        "removed_by_path": n_removed_path,
        "retention_rate": float(mask.sum() / (n_ch * n_ang)),
    }
    return mask, report


def build_multi_angle_observation(bt_single_angle, elevations_deg, channel_mask=None):
    """Build multi-angle observation vector from single-angle BT.

    Returns:
        y: concatenated observation vector
        y_index: list of (channel_idx, angle_idx) tuples
    """
    y_parts = []
    y_index = []
    for j, (bt, elev) in enumerate(zip(bt_single_angle, elevations_deg)):
        for i in range(len(bt)):
            if channel_mask is None or channel_mask[i, j]:
                y_parts.append(bt[i])
                y_index.append((i, j))
    return np.array(y_parts), y_index


def multi_angle_observation_size(n_channels, elevations_deg, channel_mask=None):
    """Return size of multi-angle observation vector."""
    if channel_mask is None:
        return n_channels * len(elevations_deg)
    return int(channel_mask.sum())


def build_multi_angle_se(se_single_angle, elevations_deg, channel_mask=None,
                          angle_correlation=0.0):
    """Build S_e for multi-angle observations with optional angle correlation."""
    if channel_mask is None:
        n_ch = se_single_angle.shape[0]
        n_ang = len(elevations_deg)
        n_total = n_ch * n_ang
        Se_multi = np.zeros((n_total, n_total))
        for j in range(n_ang):
            r0, r1 = j * n_ch, (j + 1) * n_ch
            Se_multi[r0:r1, r0:r1] = se_single_angle
            for k in range(j + 1, n_ang):
                c0, c1 = k * n_ch, (k + 1) * n_ch
                block = angle_correlation * se_single_angle
                Se_multi[r0:r1, c0:c1] = block
                Se_multi[c0:c1, r0:r1] = block
        return Se_multi
    else:
        n_total = int(channel_mask.sum())
        Se_multi = np.zeros((n_total, n_total))
        idx_map = {}
        count = 0
        for j in range(len(elevations_deg)):
            for i in range(se_single_angle.shape[0]):
                if channel_mask[i, j]:
                    idx_map[(i, j)] = count
                    count += 1
        for (i1, j1), idx1 in idx_map.items():
            Se_multi[idx1, idx1] = se_single_angle[i1, i1]
            for (i2, j2), idx2 in idx_map.items():
                if j1 == j2:
                    Se_multi[idx1, idx2] = se_single_angle[i1, i2]
                elif angle_correlation > 0:
                    Se_multi[idx1, idx2] = angle_correlation * se_single_angle[i1, i2]
        return Se_multi


def design_multiview_experiments(default_elevation=90.0,
                                  extra_elevations=(30.0, 45.0, 60.0, 75.0),
                                  frequencies_ghz=None):
    """Define A/B/C experiment groups for multi-angle OEM comparison.

    A: single zenith (baseline)
    B: multiple angles, no screening
    C: multiple angles + geometric/channel screening
    """
    if frequencies_ghz is None:
        from config import ALL_CHANNELS
        frequencies_ghz = ALL_CHANNELS

    experiments = {
        "A": {"label": "Single zenith (baseline)",
              "elevations": [default_elevation], "channel_mask": None},
        "B": {"label": "Multi-angle, no screening",
              "elevations": [default_elevation] + list(extra_elevations),
              "channel_mask": None},
        "C": {"label": "Multi-angle + geometric screening",
              "elevations": [default_elevation] + list(extra_elevations),
              "channel_mask": None},
    }

    mask_c, report_c = screen_channels_by_geometry(
        frequencies_ghz, experiments["C"]["elevations"])
    experiments["C"]["channel_mask"] = mask_c
    experiments["C"]["screening_report"] = report_c

    return experiments
