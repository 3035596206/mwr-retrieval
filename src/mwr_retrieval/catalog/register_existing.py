"""Read-only registration of assets that predate the data lake."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from ..manifest import write_asset_manifest
from ..paths import project_root
from .repository import Catalog

SKIP_DIRECTORIES = {".git", ".venv-wsl", "__pycache__"}


def historical_roots(root: Path | None = None) -> list[Path]:
    root = root or project_root()
    candidates = [root / "data", root / "models", root / "results"]
    candidates.extend(sorted(item for item in root.glob("models_chengdu_*") if item.is_dir()))
    return [item for item in candidates if item.exists()]


def register_existing(catalog: Catalog, roots: list[Path] | None = None) -> dict:
    report = {"started_at": datetime.now(timezone.utc).isoformat(), "registered": [], "skipped": [], "errors": []}
    for root in roots or historical_roots():
        for path in root.rglob("*"):
            if any(part in SKIP_DIRECTORIES for part in path.parts) or not path.is_file():
                continue
            try:
                before = catalog.connection.execute("SELECT asset_id FROM data_assets WHERE uri = ?", (catalog.paths.relative_uri(path),)).fetchone()
                asset = catalog.register(path, source_name="legacy-project")
                write_asset_manifest(catalog.paths, asset)
                (report["skipped"] if before else report["registered"]).append({"path": str(path), "asset_id": asset["asset_id"]})
            except Exception as error:
                report["errors"].append({"path": str(path), "error": f"{type(error).__name__}: {error}"})
    return report


def write_report(paths: DataPaths, report: dict) -> tuple[Path, Path]:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = paths.root / "metadata" / "logs" / f"legacy-registration-{stamp}.json"
    csv_path = paths.root / "metadata" / "logs" / f"legacy-registration-{stamp}.csv"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["status", "source", "target", "path", "asset_id", "error"])
        writer.writeheader()
        for status, rows in report.items():
            if not isinstance(rows, list):
                continue
            for row in rows:
                if isinstance(row, str):
                    row = {"path": row}
                writer.writerow({"status": status, **row})
    return json_path, csv_path
