"""Unified forward model interface for OEM / 1D-Var retrieval.

Wraps multiple radiative transfer backends behind a single interface
suitable for iterative OEM solvers that need to call H(x) repeatedly
with state vectors of varying dimensionality.

Usage:
    from forward_model import ForwardModel

    fm = ForwardModel(backend="simple")
    tb = fm.simulate(profile)               # single profile → (14,)
    tb_batch = fm.simulate_batch(profiles)   # (n,93) → (n,14)
"""

import numpy as np
import config


class ForwardModel:
    """Unified forward model H(x) for OEM retrieval.

    Wraps the backend registry from brightness_temp.py and presents a
    clean interface: profile dict in → brightness temperatures out.

    The profile dict must contain:
        - T: temperature [K], (n_layers,) or (n_time, n_layers)
        - P or P_hPa: pressure [hPa]
        - RH: relative humidity [%]
        - CLWC: cloud liquid water content [g/m³]
        - height: heights [m]
    """

    def __init__(self, backend="simple", frequencies=None, **backend_kwargs):
        """Initialise the forward model.

        Args:
            backend: "simple", "monortm", or "pamtra"
            frequencies: list of channel frequencies [GHz].
                         Defaults to config.ALL_CHANNELS (14-ch HATPRO).
            **backend_kwargs: passed to get_backend (e.g. monortm_path=...)
        """
        from brightness_temp import get_backend

        self._backend = get_backend(backend, frequencies=self.frequencies, **backend_kwargs)
        self._backend_name = backend
        self.frequencies = frequencies or config.ALL_CHANNELS
        self.n_channels = len(self.frequencies)

    # ---------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------

    def simulate(self, profile):
        """Compute H(x) for a single profile.

        Args:
            profile: dict with T, P (or P_hPa), RH, CLWC, height

        Returns:
            tb: (n_channels,) ndarray of brightness temperatures [K]
        """
        return self._backend.simulate(profile)

    def simulate_batch(self, profiles_dict):
        """Compute H(x) for a batch of profiles.

        Args:
            profiles_dict: dict with arrays shaped (n_time, n_layers)

        Returns:
            tb_batch: (n_time, n_channels) ndarray [K]
        """
        return self._backend.simulate_batch(profiles_dict)

    # ---------------------------------------------------------------
    # Jacobian via finite differences (used by OEM solver)
    # ---------------------------------------------------------------

    def jacobian(self, x, state_packer, perturbation=None):
        """Compute the Jacobian matrix K = ∂H/∂x via central finite differences.

        Args:
            x: state vector (n_state,) in OEM control space
            state_packer: OEMStatePacker instance that converts x → profile
            perturbation: dict mapping state kind → perturbation size.
                          Defaults: {"T": 0.2, "RH": 1.0, "LWC": 0.02}

        Returns:
            K: (n_channels, n_state) Jacobian matrix
        """
        if perturbation is None:
            perturbation = {"T": 0.2, "RH": 1.0, "LWC": 0.02}

        n_state = len(x)
        n_channels = self.n_channels

        # Unperturbed forward
        profile0 = state_packer.unpack(x)
        tb_ref = self.simulate(profile0)

        K = np.zeros((n_channels, n_state))

        for j in range(n_state):
            eps = self._perturbation_for(j, x, state_packer, perturbation)

            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[j] += eps
            x_minus[j] -= eps

            profile_plus = state_packer.unpack(x_plus)
            profile_minus = state_packer.unpack(x_minus)

            tb_plus = self.simulate(profile_plus)
            tb_minus = self.simulate(profile_minus)

            K[:, j] = (tb_plus - tb_minus) / (2.0 * eps)

        return K

    # ---------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------

    def _perturbation_for(self, idx, x, state_packer, perturbation):
        """Return the perturbation size for a given state vector element."""
        kind = state_packer.element_kind(idx)
        return perturbation.get(kind, 0.2)

    @property
    def backend_name(self):
        return self._backend_name


# ---------------------------------------------------------------
# Convenience: build a forward model from an existing backend
# ---------------------------------------------------------------

def forward_model_from_backend(backend_instance, frequencies=None):
    """Wrap an already-constructed backend (e.g. SimpleRadiativeTransfer)
    into a ForwardModel.

    This is useful when the calling code already has a backend object
    and wants to avoid going through get_backend again.
    """
    fm = ForwardModel.__new__(ForwardModel)
    fm._backend = backend_instance
    fm._backend_name = getattr(backend_instance, "__class__", type(backend_instance)).__name__
    fm.frequencies = frequencies or config.ALL_CHANNELS
    fm.n_channels = len(fm.frequencies)
    return fm
