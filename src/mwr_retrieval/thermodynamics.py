"""Thermodynamic conversions shared by 48-layer workflows."""

from __future__ import annotations

import numpy as np

EPSILON = 0.622
Q_FLOOR = 1e-8


def saturation_vapor_pressure_hpa(temperature_k: np.ndarray) -> np.ndarray:
    temperature_c = np.asarray(temperature_k, dtype=np.float64) - 273.15
    water = 6.112 * np.exp(17.62 * temperature_c / (243.12 + temperature_c))
    ice = 6.112 * np.exp(22.46 * temperature_c / (272.62 + temperature_c))
    blend = np.clip((temperature_c + 20.0) / 20.0, 0.0, 1.0)
    return ice * (1.0 - blend) + water * blend


def rh_to_specific_humidity(temperature_k: np.ndarray, relative_humidity: np.ndarray, pressure_hpa: np.ndarray) -> np.ndarray:
    vapor_pressure = np.clip(np.asarray(relative_humidity) / 100.0 * saturation_vapor_pressure_hpa(temperature_k), 0.0, np.asarray(pressure_hpa) * 0.99)
    mixing_ratio = EPSILON * vapor_pressure / np.maximum(np.asarray(pressure_hpa) - vapor_pressure, 1e-6)
    return np.maximum(mixing_ratio / (1.0 + mixing_ratio), Q_FLOOR)


def specific_humidity_to_rh(temperature_k: np.ndarray, specific_humidity: np.ndarray, pressure_hpa: np.ndarray) -> np.ndarray:
    q = np.clip(np.asarray(specific_humidity), Q_FLOOR, 0.2)
    mixing_ratio = q / np.maximum(1.0 - q, 1e-8)
    vapor_pressure = mixing_ratio * np.asarray(pressure_hpa) / (EPSILON + mixing_ratio)
    return np.clip(100.0 * vapor_pressure / np.maximum(saturation_vapor_pressure_hpa(temperature_k), 1e-8), 0.0, 100.0)
