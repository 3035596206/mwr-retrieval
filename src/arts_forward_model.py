"""ARTS forward-model backend for microwave radiometer simulations.

This adapter makes ARTS the primary forward model while keeping the project
independent from one specific local ARTS agenda. Research groups usually keep
their ARTS workspace setup, spectroscopy choices, channel response handling,
and cloud settings in site-local scripts. This class supports those scripts via
either:

1. a Python callable passed as ``arts_runner``; or
2. an external command in ``arts_command`` / ``ARTS_FORWARD_MODEL_COMMAND``.

The callable/command receives a JSON-compatible payload with atmospheric
profile arrays and must return/print JSON containing one of:
``brightness_temperature_k``, ``tb``, or ``y``.
"""

from __future__ import annotations

import importlib
import json
import os
import subprocess
from typing import Any, Callable

import numpy as np

import config


class ARTSForwardModel:
    """ARTS-backed H(x) adapter.

    Args:
        frequencies: channel center frequencies [GHz].
        arts_runner: optional Python callable that accepts a payload dict and
            returns either a sequence of BTs or a dict with
            ``brightness_temperature_k``.
        arts_command: optional external command. It is called once per profile,
            receives the payload JSON on stdin, and must print result JSON.
        elevation_angle_deg: ground-based elevation angle; 90 means zenith.
        channel_response: optional serializable channel-response metadata.
        timeout: external command timeout [s].
        require_pyarts: if True, fail unless ``pyarts`` can be imported.
    """

    def __init__(
        self,
        frequencies=None,
        arts_runner: Callable[[dict[str, Any]], Any] | None = None,
        arts_command: str | None = None,
        elevation_angle_deg: float = 90.0,
        channel_response: Any | None = None,
        timeout: float = 120.0,
        require_pyarts: bool = False,
        arts_persistent: bool | None = None,
    ):
        self.frequencies = list(frequencies if frequencies is not None else config.DEFAULT_FORWARD_CHANNELS)
        self.elevation_angle_deg = float(elevation_angle_deg)
        self.channel_response = channel_response
        self.timeout = float(timeout)
        self._pyarts_available = self._has_pyarts()
        if require_pyarts and not self._pyarts_available:
            raise ImportError(self._install_message())

        self.arts_runner = arts_runner or self._load_runner_from_env()
        self.arts_command = (
            arts_command
            or os.environ.get("ARTS_FORWARD_MODEL_COMMAND")
            or getattr(config, "DEFAULT_ARTS_COMMAND", None)
        )
        self.arts_command_persistent = self._resolve_persistent_mode(arts_persistent)
        self._process: subprocess.Popen | None = None

        if self.arts_runner is None and self.arts_command is None:
            raise RuntimeError(
                "ARTS is now the primary forward model, but no ARTS runner is configured.\n"
                "Provide one of:\n"
                "  - ForwardModel(backend='arts', arts_runner=callable)\n"
                "  - ForwardModel(backend='arts', arts_command='python path/to/arts_runner.py')\n"
                "  - environment variable ARTS_FORWARD_MODEL_COMMAND\n"
                "The runner receives JSON profile payload on stdin and returns JSON with "
                "'brightness_temperature_k'.\n"
                + self._install_message()
            )

    @staticmethod
    def _has_pyarts() -> bool:
        try:
            import pyarts  # noqa: F401
        except Exception:
            return False
        return True

    @staticmethod
    def _install_message() -> str:
        return (
            "Install ARTS/pyarts in the execution environment used for OEM, e.g. "
            "`mamba install -c rttools pyarts`, then point this adapter to the "
            "research-group ARTS agenda/runner."
        )

    @staticmethod
    def _load_runner_from_env() -> Callable[[dict[str, Any]], Any] | None:
        dotted = os.environ.get("ARTS_FORWARD_MODEL_RUNNER")
        if not dotted:
            return None
        module_name, _, attr = dotted.partition(":")
        if not module_name or not attr:
            raise ValueError("ARTS_FORWARD_MODEL_RUNNER must look like 'module.path:function_name'")
        module = importlib.import_module(module_name)
        runner = getattr(module, attr)
        if not callable(runner):
            raise TypeError(f"ARTS runner is not callable: {dotted}")
        return runner

    def _resolve_persistent_mode(self, requested: bool | None) -> bool:
        if requested is not None:
            return bool(requested)
        env_value = os.environ.get("ARTS_FORWARD_MODEL_PERSISTENT")
        if env_value is not None:
            return env_value.strip().lower() in {"1", "true", "yes", "on"}
        default_command = getattr(config, "DEFAULT_ARTS_COMMAND", None)
        if self.arts_command and default_command and self.arts_command == default_command:
            return bool(getattr(config, "DEFAULT_ARTS_COMMAND_PERSISTENT", False))
        return False

    def _payload(self, profile: dict[str, Any]) -> dict[str, Any]:
        pressure_key = "P_hPa" if "P_hPa" in profile else "P"
        clwc = profile.get("CLWC", np.zeros_like(profile["T"], dtype=float))
        return {
            "profile": {
                "temperature_k": np.asarray(profile["T"], dtype=float).tolist(),
                "pressure_hpa": np.asarray(profile[pressure_key], dtype=float).tolist(),
                "relative_humidity_percent": np.asarray(profile["RH"], dtype=float).tolist(),
                "cloud_liquid_water_g_m3": np.asarray(clwc, dtype=float).tolist(),
                "height_m": np.asarray(profile["height"], dtype=float).tolist(),
            },
            "instrument": {
                "frequencies_ghz": list(map(float, self.frequencies)),
                "elevation_angle_deg": self.elevation_angle_deg,
                "channel_response": self.channel_response,
            },
            "model": {
                "backend": "arts",
                "pyarts_available": self._pyarts_available,
            },
        }

    def _run_command(self, payload: dict[str, Any]) -> Any:
        if self.arts_command_persistent:
            return self._run_persistent_command(payload)

        result = subprocess.run(
            self.arts_command,
            input=json.dumps(payload, ensure_ascii=False),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            shell=True,
            timeout=self.timeout,
        )
        if result.returncode != 0:
            stdout = result.stdout or ""
            stderr = result.stderr or ""
            raise RuntimeError(
                f"ARTS forward-model command failed with code {result.returncode}\n"
                f"STDERR: {stderr[-1000:]}\nSTDOUT: {stdout[-1000:]}"
            )
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                "ARTS command must print JSON containing brightness_temperature_k; "
                f"failed to parse stdout: {result.stdout[-1000:]}"
            ) from error

    def _start_persistent_command(self) -> subprocess.Popen:
        if self._process is not None and self._process.poll() is None:
            return self._process

        self._process = subprocess.Popen(
            self.arts_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            shell=True,
            bufsize=1,
        )
        return self._process

    def _run_persistent_command(self, payload: dict[str, Any]) -> Any:
        process = self._start_persistent_command()
        if process.stdin is None or process.stdout is None:
            raise RuntimeError("Persistent ARTS process was not started with stdin/stdout pipes")

        try:
            process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
            process.stdin.flush()
        except (BrokenPipeError, OSError) as error:
            raise RuntimeError("Persistent ARTS process closed before receiving a profile") from error

        line = process.stdout.readline()
        if not line:
            returncode = process.poll()
            stderr = ""
            if returncode is not None and process.stderr is not None:
                stderr = process.stderr.read() or ""
            raise RuntimeError(
                "Persistent ARTS process ended without returning JSON\n"
                f"returncode={returncode}\nSTDERR: {stderr[-1000:]}"
            )

        try:
            result = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                "Persistent ARTS command must print one JSON object per line; "
                f"failed to parse line: {line[-1000:]}"
            ) from error
        if isinstance(result, dict) and "error" in result:
            raise RuntimeError(
                "Persistent ARTS runner reported an error\n"
                f"ERROR: {result.get('error')}\nTRACEBACK: {result.get('traceback', '')[-1000:]}"
            )
        return result

    def _extract_tb(self, result: Any) -> np.ndarray:
        if isinstance(result, dict):
            for key in ("brightness_temperature_k", "tb", "y"):
                if key in result:
                    result = result[key]
                    break
        tb = np.asarray(result, dtype=float)
        if tb.shape != (len(self.frequencies),):
            raise ValueError(
                f"ARTS returned BT shape {tb.shape}, expected ({len(self.frequencies)},)"
            )
        if not np.all(np.isfinite(tb)):
            raise ValueError("ARTS returned non-finite brightness temperatures")
        return tb

    def simulate(self, profile: dict[str, Any], **_: Any) -> np.ndarray:
        payload = self._payload(profile)
        if self.arts_runner is not None:
            result = self.arts_runner(payload)
        else:
            result = self._run_command(payload)
        return self._extract_tb(result)

    def simulate_batch(self, profiles_dict: dict[str, Any], **_: Any) -> np.ndarray:
        n_time = profiles_dict["T"].shape[0]
        tb_batch = np.zeros((n_time, len(self.frequencies)))
        clwc_profiles = profiles_dict.get("CLWC")
        for t in range(n_time):
            profile = {
                "T": profiles_dict["T"][t],
                "RH": profiles_dict["RH"][t],
                "CLWC": clwc_profiles[t] if clwc_profiles is not None else np.zeros_like(profiles_dict["T"][t]),
                "P_hPa": profiles_dict["P"][t] if "P" in profiles_dict else np.ones(len(profiles_dict["height"])) * 1013.0,
                "height": profiles_dict["height"],
            }
            tb_batch[t] = self.simulate(profile)
        return tb_batch

    def close(self) -> None:
        if self._process is None:
            return
        if self._process.poll() is None:
            try:
                if self._process.stdin is not None:
                    self._process.stdin.close()
                self._process.terminate()
            except OSError:
                pass
        self._process = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
