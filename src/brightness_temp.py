"""Brightness temperature simulation for RPG HATPRO 14-channel MWR.

Supports multiple radiative transfer backends:
  - 'simple': Python-based simplified model (no external dependencies)
  - 'monortm': MonoRTM Fortran model (requires compiled executable)
  - 'pamtra': PAMTRA Python/Fortran package (requires conda/pip install)

Usage:
    from brightness_temp import get_backend, simulate_mwr_observation

    # Use Python simplified model (default, always available)
    tb = simulate_mwr_observation(profile)

    # Use MonoRTM
    backend = get_backend('monortm', monortm_path='/usr/local/bin/monortm')
    tb = backend.simulate(profile)
"""

import numpy as np
import config


# ============================================================
# Backend registry
# ============================================================
_backends = {}


def register_backend(name, backend_instance):
    """Register a named backend."""
    _backends[name] = backend_instance


def get_backend(name="simple", **kwargs):
    """Get or create a radiative transfer backend.

    Args:
        name: 'simple', 'monortm', or 'pamtra'
        **kwargs: backend-specific arguments
            monortm_path: path to MonoRTM executable
    Returns:
        backend object with .simulate(profile) -> tb and .simulate_batch(profiles) -> tb
    """
    if name in _backends:
        return _backends[name]

    if name == "simple":
        backend = SimpleRadiativeTransfer()
    elif name == "monortm":
        from monortm_wrapper import MonoRTM
        backend = MonoRTM(monortm_path=kwargs.get("monortm_path"), tape3_path=kwargs.get("tape3_path"))
    elif name == "pamtra":
        raise NotImplementedError(
            "PAMTRA backend requires conda: conda install -c conda-forge pamtra"
        )
    else:
        raise ValueError(f"Unknown backend: {name}. Options: simple, monortm, pamtra")

    register_backend(name, backend)
    return backend


# ============================================================
# Simple Python Radiative Transfer Model
# ============================================================

class SimpleRadiativeTransfer:
    """Python-based simplified radiative transfer model.

    Implements atmospheric absorption and radiative transfer at microwave
    frequencies. This is a self-contained approximation that does not require
    external Fortran models.
    """

    def __init__(self):
        self.frequencies = config.ALL_CHANNELS

    def simulate(self, profile, add_noise=False, noise_std=0.5):
        """Simulate 14-channel brightness temperatures.

        Args:
            profile: dict with 'T', 'P', 'RH', 'CLWC', 'height'
            add_noise: add instrument noise
            noise_std: noise standard deviation [K]
        Returns:
            tb: brightness temperatures [K], shape (14,)
        """
        Tb = _compute_brightness_temperature_downgoing(
            T_profile=profile["T"],
            P_profile=profile[("P_hPa" if "P_hPa" in profile else "P")],
            RH_profile=profile["RH"],
            CLWC_profile=profile["CLWC"],
            height_profile=profile["height"],
            frequencies=self.frequencies,
        )
        if add_noise:
            Tb += np.random.randn(len(Tb)) * noise_std
        return Tb

    def simulate_batch(self, profiles_dict, add_noise=False):
        """Simulate BT for a batch of profiles.

        Args:
            profiles_dict: dict with arrays of shape (n_time, n_layers)
        Returns:
            tb_batch: shape (n_time, 14)
        """
        n_time = profiles_dict["T"].shape[0]
        tb_batch = np.zeros((n_time, config.N_CHANNELS))

        for t in range(n_time):
            profile = {
                "T": profiles_dict["T"][t],
                "P": profiles_dict["P"][t] if "P" in profiles_dict else
                    np.ones(config.N_LAYERS) * 1013.0,
                "RH": profiles_dict["RH"][t],
                "CLWC": profiles_dict["CLWC"][t],
                "height": profiles_dict["height"],
            }
            tb_batch[t] = self.simulate(profile, add_noise=add_noise)
        return tb_batch


def _oxygen_absorption(freq_ghz, T, P_hpa, e_hpa):
    """O2 volume absorption coefficient [Np/km].

    Simplified 60 GHz equivalent line approximation (eq 2-11) + 118.75 GHz wing.
    """
    theta = 300.0 / T
    P_dry = P_hpa - e_hpa
    alpha = 0.59 * P_dry / 1013.0 * theta**0.8

    f0 = 60.0
    term1 = 1.0 / ((freq_ghz - f0)**2 + alpha**2)
    term2 = 1.0 / ((freq_ghz + f0)**2 + alpha**2)
    k_o2 = 1.1e-2 * P_dry / 1013.0 * theta**2 * freq_ghz**2 * alpha * (term1 + term2)

    f1 = 118.75
    alpha1 = 1.5 * alpha
    term3 = 1.0 / ((freq_ghz - f1)**2 + alpha1**2)
    k_o2 += 0.2e-2 * P_dry / 1013.0 * theta**2 * freq_ghz**2 * alpha1 * term3

    return k_o2


def _water_vapor_absorption(freq_ghz, T, P_hpa, e_hpa):
    """H2O volume absorption coefficient [Np/km].

    22.235 GHz line + continuum absorption (eq 2-10).
    """
    theta = 300.0 / T
    rho_v = e_hpa * 216.7 / T

    alpha = 2.85 * (P_hpa / 1013.0) * theta**0.625 * (1 + 0.018 * rho_v * T / P_hpa)

    f0 = 22.235
    S = 2.0 * rho_v * theta**1.5
    k_line = S * freq_ghz**2 * alpha / ((freq_ghz - f0)**2 + alpha**2) * 1e-6
    k_cont = 2.4e-6 * rho_v * theta**2 * freq_ghz**2

    return k_line + k_cont


def _cloud_absorption(freq_ghz, T, clwc):
    """Cloud liquid water absorption [Np/km]. Rayleigh approximation."""
    if clwc <= 0:
        return 0.0

    theta = 300.0 / T
    eps0 = 77.66 - 103.3 * (1 - 1 / theta)
    eps1 = 0.0671 * theta
    eps2 = 3.52
    f = freq_ghz

    eps_p = eps2 + (eps0 - eps2) / (1 + (f / eps1)**2)
    eps_dp = (eps0 - eps2) * (f / eps1) / (1 + (f / eps1)**2) + eps1 * f

    K = (eps_p - 1j * eps_dp - 1) / (eps_p - 1j * eps_dp + 2)
    wavelength_m = 0.3 / freq_ghz
    k_m = (6 * np.pi / wavelength_m) * np.imag(-K) * 1e-6

    return k_m * clwc


def _compute_brightness_temperature_downgoing(T_profile, P_profile, RH_profile,
                                               CLWC_profile, height_profile,
                                               frequencies):
    """Solve the non-scattering radiative transfer equation.

    Tb = T_cosmic*tau(0,inf) + integral[T(z) * d_tau/dz * dz]
    """
    n_layers = len(T_profile)
    n_channels = len(frequencies)
    T_cosmic = 2.73
    Tb = np.zeros(n_channels)

    for ch, freq in enumerate(frequencies):
        d_tau = np.zeros(n_layers)
        tau_cum = np.zeros(n_layers + 1)

        for i in range(n_layers):
            es = 6.1078 * np.exp(17.2693882 * (T_profile[i] - 273.16) /
                                  (T_profile[i] - 35.86))
            e_hpa = RH_profile[i] / 100.0 * es

            k_total = (_oxygen_absorption(freq, T_profile[i], P_profile[i], e_hpa) +
                       _water_vapor_absorption(freq, T_profile[i], P_profile[i], e_hpa) +
                       _cloud_absorption(freq, T_profile[i], CLWC_profile[i]))

            dz = (height_profile[0] / 1000.0 if i == 0 else
                  (height_profile[i] - height_profile[i - 1]) / 1000.0)
            d_tau[i] = k_total * dz

        tau_cum[0] = 0.0
        for i in range(n_layers):
            tau_cum[i + 1] = tau_cum[i] + d_tau[i]

        Tb_cosmic = T_cosmic * np.exp(-tau_cum[-1])

        Tb_atm = 0.0
        for i in range(n_layers):
            Tb_atm += T_profile[i] * (1 - np.exp(-d_tau[i])) * np.exp(-tau_cum[i])

        Tb[ch] = Tb_cosmic + Tb_atm

    return Tb


# ============================================================
# Convenience functions (use default 'simple' backend)
# ============================================================

def simulate_mwr_observation(profile, add_noise=False, noise_std=0.5,
                              backend="simple", **backend_kwargs):
    """Simulate RPG HATPRO 14-channel observation from atmospheric profile.

    Args:
        profile: dict with 'T', 'P'/'P_hPa', 'RH', 'CLWC', 'height'
        add_noise: add Gaussian instrument noise
        noise_std: instrument noise std [K] (default 0.5K for HATPRO)
        backend: 'simple', 'monortm', or 'pamtra'
    Returns:
        Tb: 14-channel brightness temperatures [K]
    """
    rt = get_backend(backend, **backend_kwargs)
    return rt.simulate(profile, add_noise=add_noise, noise_std=noise_std)


def simulate_batch(profiles_dict, add_noise=False, backend="simple",
                   **backend_kwargs):
    """Simulate BT for a batch of profiles.

    Args:
        profiles_dict: dict with arrays of shape (n_time, n_layers)
        add_noise: add instrument noise
        backend: 'simple', 'monortm', or 'pamtra'
    Returns:
        Tb_batch: shape (n_time, 14)
    """
    rt = get_backend(backend, **backend_kwargs)
    return rt.simulate_batch(profiles_dict, add_noise=add_noise)


if __name__ == "__main__":
    heights = np.array(config.HEIGHT_GRID)
    T_std = 288.15 - 6.5 * heights / 1000.0
    P_std = 1013.25 * np.exp(-heights / 8000.0)

    test = {
        "T": T_std,
        "P": P_std,
        "RH": np.full_like(heights, 50.0),
        "CLWC": np.zeros_like(heights),
        "height": heights,
    }

    # Test simple backend
    print("Simple Python Backend:")
    Tb = simulate_mwr_observation(test)
    for f, tb in zip(config.ALL_CHANNELS, Tb):
        band = "K" if f < 40 else "V"
        print(f"  {band}-band {f:.2f} GHz: {tb:.2f} K")

    # Test MonoRTM backend if available
    try:
        print("\nMonoRTM Backend:")
        Tb_m = simulate_mwr_observation(test, backend="monortm")
        print("  Differences from simple model:")
        for f, tb, tbm in zip(config.ALL_CHANNELS, Tb, Tb_m):
            print(f"  {f:.2f} GHz: delta = {tbm - tb:+.2f} K")
    except (FileNotFoundError, ImportError) as e:
        print(f"  MonoRTM not available: {e}")
