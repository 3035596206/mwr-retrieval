"""Human-readable immutable asset manifests."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .paths import DataPaths


def write_asset_manifest(paths: DataPaths, asset: dict[str, Any]) -> Path:
    derived_kinds = {"model", "result", "dataset", "prediction"}
    directory = paths.manifest_root / ("derived" if asset["kind"] in derived_kinds else "raw")
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{asset['asset_id']}.json"
    payload = {"manifest_version": 1, "generated_at": datetime.now(timezone.utc).isoformat(), "asset": asset}
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return destination
