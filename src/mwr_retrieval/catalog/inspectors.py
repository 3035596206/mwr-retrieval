"""Conservative file inspectors. Inspection failures never alter source files."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def infer_format(path: Path) -> str:
    if path.name in {"TAPE3", "TAPE3_bin"}:
        return "tape3"
    suffix = path.suffix.lower()
    return {
        ".nc": "netcdf", ".nc4": "netcdf", ".grib": "grib", ".grb": "grib",
        ".json": "json", ".txt": "text", ".csv": "csv", ".xlsx": "xlsx",
        ".npz": "npz", ".npy": "npy", ".pkl": "pickle", ".pt": "torch",
        ".png": "png", ".pdf": "pdf", ".docx": "docx", ".pptx": "pptx",
        ".rar": "rar", ".ndjson": "ndjson",
    }.get(suffix, "binary")


def infer_kind(path: Path) -> str:
    parts = {item.lower() for item in path.parts}
    name = path.name.lower()
    if "tape3" in name or "tape3" in parts:
        return "ancillary_tape3"
    if path.suffix.lower() in {".pdf", ".docx", ".pptx", ".rar", ".md"} or "reports" in parts:
        return "report"
    if "era5" in name or "era5" in parts or name.startswith(("pl_", "sl_")):
        return "era5"
    if "radiosonde" in parts or "sounding" in name or "wenjiang" in parts:
        return "radiosonde"
    if "models" in parts or name.endswith(".pt"):
        return "model"
    if "results" in parts or "prediction" in name:
        return "result"
    if path.suffix.lower() in {".xlsx", ".xls"} and (
        "obs_bt" in name or "亮温" in path.name or "mwr" in parts or "bt" in name
    ):
        return "mwr"
    if "chengdu_obs_bt" in name or "mwr" in parts or "bt" in name:
        return "mwr"
    return "unknown"


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    try:
        text = str(value)
        if text == "NaT":
            return None
        return text.replace(" ", "T")
    except Exception:
        return None


def inspect(path: Path) -> dict[str, Any]:
    """Return only metadata that can be read without changing the asset."""
    result: dict[str, Any] = {"format": infer_format(path), "kind": infer_kind(path), "metadata": {}}
    fmt = result["format"]
    try:
        if fmt == "json":
            with path.open(encoding="utf-8") as handle:
                payload = json.load(handle)
            result["metadata"] = {
                "top_level_keys": sorted(payload) if isinstance(payload, dict) else None,
                "record_count": payload.get("n_records") if isinstance(payload, dict) else None,
            }
            if isinstance(payload, dict):
                result["start_time_utc"] = payload.get("start") or payload.get("time_start")
                result["end_time_utc"] = payload.get("end") or payload.get("time_end")
                channels = payload.get("channel_frequencies_ghz")
                if channels:
                    result["variables"] = [f"tb_ch{i:02d}_k" for i in range(len(channels))]
        elif fmt == "npz":
            import numpy as np
            with np.load(path, allow_pickle=False) as dataset:
                result["metadata"] = {"arrays": {key: list(dataset[key].shape) for key in dataset.files}}
                result["variables"] = list(dataset.files)
        elif fmt == "netcdf":
            import xarray as xr
            with xr.open_dataset(path) as dataset:
                result["metadata"] = {"dimensions": dict(dataset.sizes), "variables": list(dataset.data_vars)}
                result["variables"] = list(dataset.data_vars)
                for time_name in ("valid_time", "time"):
                    if time_name in dataset.coords or time_name in dataset.variables:
                        times = dataset[time_name].values
                        if len(times):
                            result["start_time_utc"] = _iso(times.min())
                            result["end_time_utc"] = _iso(times.max())
                        break
        elif fmt == "grib":
            result["metadata"] = {"inspection": "GRIB metadata is registered without decoding; use a GRIB-specific workflow for detailed coverage."}
    except Exception as error:
        result["metadata"] = {"inspection_error": f"{type(error).__name__}: {error}"}
    result["observed_at"] = datetime.now(timezone.utc).isoformat()
    return result
