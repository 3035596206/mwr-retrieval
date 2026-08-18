"""Run one ground-based MWR profile with ARTS/pyarts.

This script implements the JSON runner protocol used by
``src.arts_forward_model.ARTSForwardModel``.  It is intentionally standalone so
it can be executed from a WSL/conda ARTS environment while the rest of the
project runs from Windows Python.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from typing import Any

import numpy as np


def _saturation_vapor_pressure_pa(temperature_k: np.ndarray) -> np.ndarray:
    """Buck-style liquid-water saturation vapor pressure approximation."""
    temp_c = temperature_k - 273.15
    return 611.2 * np.exp((17.67 * temp_c) / (temp_c + 243.5))


def _profile_to_arts_fields(profile: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    temperature = np.asarray(profile["temperature_k"], dtype=float)
    pressure_hpa = np.asarray(profile["pressure_hpa"], dtype=float)
    rh_percent = np.asarray(profile["relative_humidity_percent"], dtype=float)
    height_m = np.asarray(profile["height_m"], dtype=float)

    if not (temperature.shape == pressure_hpa.shape == rh_percent.shape == height_m.shape):
        raise ValueError("temperature, pressure, RH, and height profiles must have identical shapes")
    if temperature.ndim != 1 or temperature.size < 2:
        raise ValueError("profile fields must be 1-D arrays with at least two levels")
    if np.any(~np.isfinite(temperature)) or np.any(~np.isfinite(pressure_hpa)) or np.any(~np.isfinite(height_m)):
        raise ValueError("temperature, pressure, and height profiles must be finite")
    if np.any(pressure_hpa <= 0):
        raise ValueError("pressure_hpa must be positive")

    pressure_pa = pressure_hpa * 100.0
    order = np.argsort(pressure_pa)[::-1]
    pressure_pa = pressure_pa[order]
    temperature = temperature[order]
    rh_percent = np.clip(rh_percent[order], 0.0, 100.0)
    height_m = height_m[order]

    unique = np.concatenate(([True], np.diff(pressure_pa) < 0.0))
    pressure_pa = pressure_pa[unique]
    temperature = temperature[unique]
    rh_percent = rh_percent[unique]
    height_m = height_m[unique]
    if pressure_pa.size < 2:
        raise ValueError("pressure grid must contain at least two unique levels")

    e_pa = rh_percent / 100.0 * _saturation_vapor_pressure_pa(temperature)
    h2o_vmr = np.clip(e_pa / pressure_pa, 0.0, 0.2)
    dry_air = np.clip(1.0 - h2o_vmr, 0.0, 1.0)
    o2_vmr = 0.2095 * dry_air
    n2_vmr = 0.7808 * dry_air
    vmr = np.stack([h2o_vmr, o2_vmr, n2_vmr], axis=0)
    return pressure_pa, temperature, height_m, vmr


def run_arts(payload: dict[str, Any]) -> dict[str, Any]:
    import pyarts

    profile = payload["profile"]
    instrument = payload["instrument"]
    frequencies_hz = np.asarray(instrument["frequencies_ghz"], dtype=float) * 1e9
    if frequencies_hz.ndim != 1 or frequencies_hz.size == 0:
        raise ValueError("instrument.frequencies_ghz must be a non-empty 1-D array")

    elevation_angle = float(instrument.get("elevation_angle_deg", 90.0))
    if not (0.0 < elevation_angle <= 90.0):
        raise ValueError("elevation_angle_deg must be in (0, 90]")
    zenith_angle = 90.0 - elevation_angle

    pressure_pa, temperature, height_m, vmr = _profile_to_arts_fields(profile)

    ws = pyarts.Workspace()
    ws.iy_main_agendaSet(option="Emission")
    ws.iy_surface_agendaSet(option="UseSurfaceRtprop")
    ws.surface_rtprop_agendaSet(option="Blackbody_SurfTFromt_field")
    ws.iy_space_agendaSet(option="CosmicBackground")
    ws.ppath_agendaSet(option="FollowSensorLosPath")
    ws.ppath_step_agendaSet(option="GeometricPath")
    ws.water_p_eq_agendaSet(option="MK05")
    ws.PlanetSet(option="Earth")

    ws.iy_unit = "PlanckBT"
    ws.stokes_dim = 1
    ws.atmosphere_dim = 1
    ws.p_grid = pressure_pa
    ws.lat_grid = []
    ws.lon_grid = []
    ws.z_surface = [[float(height_m[0])]]
    ws.sensor_pos = [[float(height_m[0])]]
    ws.sensor_los = [[zenith_angle]]
    ws.f_grid = frequencies_hz

    ws.abs_speciesSet(species=["H2O-PWR98", "O2-PWR98", "N2-SelfContStandardType"])
    ws.abs_lines_per_speciesSetEmpty()
    ws.propmat_clearsky_agendaAuto()

    ws.t_field = temperature.reshape((-1, 1, 1))
    ws.z_field = height_m.reshape((-1, 1, 1))
    ws.vmr_field = vmr.reshape((3, -1, 1, 1))

    ws.jacobianOff()
    ws.cloudboxOff()
    ws.sensorOff()
    ws.atmgeom_checkedCalc()
    ws.atmfields_checkedCalc()
    ws.cloudbox_checkedCalc()
    ws.sensor_checkedCalc()
    ws.yCalc()

    tb = np.asarray(ws.y.value, dtype=float).reshape(-1)
    return {
        "brightness_temperature_k": tb.tolist(),
        "metadata": {
            "backend": "arts",
            "pyarts_version": getattr(pyarts, "__version__", "unknown"),
            "absorption_species": ["H2O-PWR98", "O2-PWR98", "N2-SelfContStandardType"],
            "elevation_angle_deg": elevation_angle,
        },
    }


def _self_test_payload() -> dict[str, Any]:
    height = np.linspace(0.0, 10_000.0, 48)
    return {
        "profile": {
            "temperature_k": (288.15 - 6.5 * height / 1000.0).tolist(),
            "pressure_hpa": (1013.25 * np.exp(-height / 8000.0)).tolist(),
            "relative_humidity_percent": np.full_like(height, 60.0).tolist(),
            "cloud_liquid_water_g_m3": np.zeros_like(height).tolist(),
            "height_m": height.tolist(),
        },
        "instrument": {
            "frequencies_ghz": [22.235, 23.035, 23.835, 26.235, 30.0, 51.25, 52.28, 53.85],
            "elevation_angle_deg": 90.0,
            "channel_response": None,
        },
        "model": {"backend": "arts"},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="run a built-in profile instead of reading stdin")
    parser.add_argument("--server", action="store_true", help="read one JSON payload per line and write one JSON result per line")
    args = parser.parse_args(argv)

    if args.server:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                result = run_arts(json.loads(line))
            except Exception as error:  # pragma: no cover - defensive runner boundary
                result = {
                    "error": str(error),
                    "traceback": traceback.format_exc(limit=8),
                }
            print(json.dumps(result, ensure_ascii=False), flush=True)
        return 0

    payload = _self_test_payload() if args.self_test else json.load(sys.stdin)
    result = run_arts(payload)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
