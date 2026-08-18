"""Versioned station and instrument configuration readers."""

from __future__ import annotations

import json
from pathlib import Path

from .paths import project_root


def config_path(name: str) -> Path:
    return project_root() / "config" / name


def load_sites() -> dict[str, dict]:
    with config_path("sites.json").open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return {site["site_id"]: site for site in payload["sites"]}


def load_instruments() -> dict[str, dict]:
    with config_path("instruments.json").open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return {instrument["instrument_id"]: instrument for instrument in payload["instruments"]}
