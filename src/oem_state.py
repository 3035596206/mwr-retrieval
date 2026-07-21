"""OEM state vector: pack / unpack / dimensionality reduction.

Provides the OEMStatePacker class that maps between:
  - The full 93-layer profile space (T, RH, optionally LWC)
  - The reduced OEM control variable space (coarse layers or EOF coefficients)

Usage:
    from oem_state import OEMStatePacker

    packer = OEMStatePacker(mode="coarse", coarse_T=[...], coarse_RH=[...])
    x = packer.pack(profile)           # profile dict → state vector
    profile = packer.unpack(x)         # state vector → profile dict
"""

import numpy as np
import config


class OEMStatePacker:
    """Map between full 93-layer profiles and reduced OEM state vectors.

    Supports two reduction modes:
      - "coarse":  Piecewise-constant coarse layers → linear interpolation
      - "eof":     EOF/PCA coefficients (to be implemented in Phase 2)

    The coarse-layer mode uses control points at specified heights and
    linearly interpolates to the full 93-layer grid.  Each control variable
    represents a layer-mean value for the height range below its node.
    """

    def __init__(self, mode="coarse", coarse_T=None, coarse_RH=None,
                 coarse_LWC=None, eof_T=None, eof_RH=None,
                 T_mean=None, RH_mean=None):
        """Initialise the state packer.

        Args:
            mode: "coarse" or "eof"
            coarse_T: list of control heights [m] for temperature.
                      Default: plan §4.1 — [500, 1000, 2000, 3000, 5000, 8000, 10000]
            coarse_RH: list of control heights [m] for humidity (same default)
            coarse_LWC: list of control heights [m] for LWC (optional)
            eof_T: (n_layers, n_eof_T) EOF basis for T (mode="eof" only)
            eof_RH: (n_layers, n_eof_RH) EOF basis for RH
            T_mean: (n_layers,) mean T profile for EOF mode
            RH_mean: (n_layers,) mean RH profile for EOF mode
        """
        self.mode = mode

        if mode == "coarse":
            self.coarse_T = np.array(coarse_T or [500, 1000, 2000, 3000,
                                                   5000, 8000, 10000],
                                     dtype=float)
            self.coarse_RH = np.array(coarse_RH or [500, 1000, 2000, 3000,
                                                     5000, 8000, 10000],
                                       dtype=float)
            self.coarse_LWC = np.array(coarse_LWC or [], dtype=float)
        elif mode == "eof":
            self.eof_T = np.asarray(eof_T) if eof_T is not None else None
            self.eof_RH = np.asarray(eof_RH) if eof_RH is not None else None
            self.T_mean = np.asarray(T_mean) if T_mean is not None else None
            self.RH_mean = np.asarray(RH_mean) if RH_mean is not None else None
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'coarse' or 'eof'.")

        self.heights = np.array(config.HEIGHT_GRID, dtype=float)
        self.n_layers = config.N_LAYERS

    # ================================================================
    # State vector dimension
    # ================================================================

    @property
    def n_state(self):
        """Total number of control variables."""
        if self.mode == "coarse":
            n = len(self.coarse_T) + len(self.coarse_RH)
            if len(self.coarse_LWC) > 0:
                n += len(self.coarse_LWC)
            return n
        elif self.mode == "eof":
            n = 0
            if self.eof_T is not None:
                n += self.eof_T.shape[1]
            if self.eof_RH is not None:
                n += self.eof_RH.shape[1]
            return n
        return 0

    @property
    def n_T(self):
        """Number of temperature control variables."""
        if self.mode == "coarse":
            return len(self.coarse_T)
        elif self.mode == "eof" and self.eof_T is not None:
            return self.eof_T.shape[1]
        return 0

    @property
    def n_RH(self):
        """Number of humidity control variables."""
        if self.mode == "coarse":
            return len(self.coarse_RH)
        elif self.mode == "eof" and self.eof_RH is not None:
            return self.eof_RH.shape[1]
        return 0

    @property
    def n_LWC(self):
        """Number of LWC control variables."""
        if self.mode == "coarse":
            return len(self.coarse_LWC)
        return 0

    # ================================================================
    # Pack: full profile → reduced state vector
    # ================================================================

    def pack(self, profile):
        """Convert a full-resolution profile to the OEM state vector.

        Args:
            profile: dict with 'T' (93,), 'RH' (93,), optionally 'CLWC' (93,)

        Returns:
            x: (n_state,) state vector
        """
        if self.mode == "coarse":
            return self._pack_coarse(profile)
        elif self.mode == "eof":
            return self._pack_eof(profile)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _pack_coarse(self, profile):
        """Extract coarse-layer control values from a full profile."""
        pieces = []
        for heights in [self.coarse_T, self.coarse_RH, self.coarse_LWC]:
            if len(heights) == 0:
                continue
            for h in heights:
                idx = self._nearest_height_idx(h)
                if heights is self.coarse_T:
                    pieces.append(profile["T"][idx])
                elif heights is self.coarse_RH:
                    pieces.append(profile["RH"][idx])
                else:
                    pieces.append(profile.get("CLWC", np.zeros(self.n_layers))[idx])
        return np.array(pieces, dtype=float)

    def _pack_eof(self, profile):
        """Project profile perturbations onto EOF basis."""
        pieces = []
        if self.eof_T is not None:
            dT = profile["T"] - self.T_mean
            pieces.append(dT @ self.eof_T)
        if self.eof_RH is not None:
            dRH = profile["RH"] - self.RH_mean
            pieces.append(dRH @ self.eof_RH)
        return np.concatenate(pieces) if pieces else np.array([])

    # ================================================================
    # Unpack: reduced state vector → full profile
    # ================================================================

    def unpack(self, x, clip=True):
        """Expand the OEM state vector to a full 93-layer profile.

        Args:
            x: (n_state,) state vector
            clip: if True, enforce physical bounds (T: 180-330, RH: 0-100)

        Returns:
            profile: dict with 'T' (93,), 'RH' (93,), 'CLWC' (93,),
                     'height' (93,), 'P_hPa' (93, computed from standard atmosphere)
        """
        if self.mode == "coarse":
            return self._unpack_coarse(x, clip=clip)
        elif self.mode == "eof":
            return self._unpack_eof(x, clip=clip)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _unpack_coarse(self, x, clip=True):
        """Map coarse control variables to full 93-layer grid via linear interpolation."""
        idx = 0

        # --- Temperature ---
        T_ctrl = x[idx:idx + self.n_T]
        idx += self.n_T
        T_full = self._interpolate_coarse(T_ctrl, self.coarse_T)

        # --- Relative humidity ---
        RH_ctrl = x[idx:idx + self.n_RH]
        idx += self.n_RH
        RH_full = self._interpolate_coarse(RH_ctrl, self.coarse_RH)

        # --- Cloud liquid water ---
        if self.n_LWC > 0:
            LWC_ctrl = x[idx:idx + self.n_LWC]
            idx += self.n_LWC
            CLWC_full = self._interpolate_coarse(LWC_ctrl, self.coarse_LWC)
        else:
            CLWC_full = np.zeros(self.n_layers)

        # Physical bounds
        if clip:
            T_full = np.clip(T_full, 180.0, 330.0)
            RH_full = np.clip(RH_full, 0.0, 100.0)
            CLWC_full = np.clip(CLWC_full, 0.0, None)

        # Standard-atmosphere pressure (simple exponential)
        P_full = 1013.25 * np.exp(-self.heights / 8000.0)

        return {
            "T": T_full,
            "P_hPa": P_full,
            "RH": RH_full,
            "CLWC": CLWC_full,
            "height": self.heights,
        }

    def _unpack_eof(self, x, clip=True):
        """Reconstruct full profile from EOF coefficients."""
        idx = 0

        if self.eof_T is not None:
            z_T = x[idx:idx + self.eof_T.shape[1]]
            idx += self.eof_T.shape[1]
            T_full = self.T_mean + self.eof_T @ z_T
        else:
            T_full = np.full(self.n_layers, 250.0)

        if self.eof_RH is not None:
            z_RH = x[idx:idx + self.eof_RH.shape[1]]
            idx += self.eof_RH.shape[1]
            RH_full = self.RH_mean + self.eof_RH @ z_RH
        else:
            RH_full = np.full(self.n_layers, 50.0)

        CLWC_full = np.zeros(self.n_layers)

        if clip:
            T_full = np.clip(T_full, 180.0, 330.0)
            RH_full = np.clip(RH_full, 0.0, 100.0)
            CLWC_full = np.clip(CLWC_full, 0.0, None)

        P_full = 1013.25 * np.exp(-self.heights / 8000.0)

        return {
            "T": T_full,
            "P_hPa": P_full,
            "RH": RH_full,
            "CLWC": CLWC_full,
            "height": self.heights,
        }

    # ================================================================
    # State vector introspection
    # ================================================================

    def element_kind(self, idx):
        """Return the kind ('T', 'RH', or 'LWC') of state element `idx`."""
        if idx < self.n_T:
            return "T"
        elif idx < self.n_T + self.n_RH:
            return "RH"
        else:
            return "LWC"

    def state_labels(self):
        """Return human-readable labels for each state element."""
        labels = []
        if self.mode == "coarse":
            for h in self.coarse_T:
                labels.append(f"T_{int(h)}m")
            for h in self.coarse_RH:
                labels.append(f"RH_{int(h)}m")
            for h in self.coarse_LWC:
                labels.append(f"LWC_{int(h)}m")
        elif self.mode == "eof":
            for i in range(self.n_T):
                labels.append(f"T_EOF{i+1}")
            for i in range(self.n_RH):
                labels.append(f"RH_EOF{i+1}")
        return labels

    # ================================================================
    # Internal helpers
    # ================================================================

    def _nearest_height_idx(self, target_m):
        """Index of the height grid nearest to `target_m` metres."""
        return int(np.argmin(np.abs(self.heights - target_m)))

    def _interpolate_coarse(self, ctrl_values, ctrl_heights):
        """Linearly interpolate coarse control values onto the 93-layer grid.

        The first control point applies from the surface up to its height;
        the last applies from its height to the top.  Interior regions are
        linear between adjacent control heights.

        Args:
            ctrl_values: (n_ctrl,) values at control heights
            ctrl_heights: (n_ctrl,) heights [m]

        Returns:
            full: (93,) interpolated profile
        """
        n_ctrl = len(ctrl_values)
        full = np.zeros(self.n_layers)

        # Convert control heights to nearest grid indices
        ctrl_idx = np.array([self._nearest_height_idx(h) for h in ctrl_heights])

        for i in range(self.n_layers):
            h_i = self.heights[i]

            if h_i <= ctrl_heights[0]:
                # Below first control point: constant = first value
                full[i] = ctrl_values[0]
            elif h_i >= ctrl_heights[-1]:
                # Above last control point: constant = last value
                full[i] = ctrl_values[-1]
            else:
                # Between two control points: linear interpolation
                for j in range(n_ctrl - 1):
                    if ctrl_heights[j] <= h_i <= ctrl_heights[j + 1]:
                        frac = ((h_i - ctrl_heights[j]) /
                                (ctrl_heights[j + 1] - ctrl_heights[j]))
                        full[i] = (ctrl_values[j] * (1 - frac) +
                                   ctrl_values[j + 1] * frac)
                        break

        return full


# ================================================================
# Factory: create the recommended first-version 14-dim packer
# ================================================================

def make_default_packer():
    """Create the default coarse-layer state packer from plan §4.1.

    T control at:  500, 1000, 2000, 3000, 5000, 8000, 10000 m   (7-dim)
    RH control at: 500, 1000, 2000, 3000, 5000, 8000, 10000 m   (7-dim)
    Total: 14 dimensions.
    """
    return OEMStatePacker(
        mode="coarse",
        coarse_T=[500, 1000, 2000, 3000, 5000, 8000, 10000],
        coarse_RH=[500, 1000, 2000, 3000, 5000, 8000, 10000],
    )


def make_t_only_packer():
    """Create a T-only coarse-layer state packer (7-dim)."""
    return OEMStatePacker(
        mode="coarse",
        coarse_T=[500, 1000, 2000, 3000, 5000, 8000, 10000],
        coarse_RH=[],
    )
