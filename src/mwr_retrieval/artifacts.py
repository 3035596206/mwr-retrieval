"""Run-oriented artifact directories and manifests."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .catalog.repository import Catalog
from .manifest import write_asset_manifest
from .paths import DataPaths, project_root


def create_run_directory(paths: DataPaths, artifact_type: str, run_id: str | None = None,
                         project_id: str = "project-brnn") -> tuple[str, Path]:
    run_id = run_id or f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
    destination = paths.root / "projects" / project_id / "curated" / artifact_type / f"run_id={run_id}"
    destination.mkdir(parents=True, exist_ok=False)
    return run_id, destination


def write_run_manifest(directory: Path, payload: dict[str, Any]) -> Path:
    destination = directory / "manifest.json"
    destination.write_text(json.dumps({"manifest_version": 1, **payload}, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return destination


def create_tracked_run_directory(catalog: Catalog, artifact_type: str, *, pipeline: str,
                                 command: str | list[str] | None = None,
                                 config: dict[str, Any] | None = None,
                                 project_id: str = "project-brnn") -> tuple[str, Path]:
    """Create a curated run directory and matching catalog processing run."""
    if project_id == "project-brnn":
        catalog.add_project(
            "project-brnn",
            "project-504",
            "BRNN temperature and humidity retrieval",
            project_root().as_uri(),
            "Current MWR BRNN retrieval project",
        )
    run = catalog.create_processing_run(
        pipeline,
        command=command,
        config={"artifact_type": artifact_type, **(config or {})},
        project_id=project_id,
    )
    _, destination = create_run_directory(catalog.paths, artifact_type, run_id=run["run_id"], project_id=project_id)
    return run["run_id"], destination


def register_run_output(catalog: Catalog, path: Path, *, run_id: str,
                        input_asset_ids: list[str] | tuple[str, ...] = (),
                        relation: str = "derived_from", role: str = "output",
                        project_id: str = "project-brnn",
                        source_name: str | None = None,
                        source_request: dict[str, Any] | None = None) -> dict[str, Any]:
    """Register an output file and link it to all input assets for a run."""
    if project_id == "project-brnn":
        catalog.add_project(
            "project-brnn",
            "project-504",
            "BRNN temperature and humidity retrieval",
            project_root().as_uri(),
            "Current MWR BRNN retrieval project",
        )
    asset = catalog.register(path, source_name=source_name, source_request=source_request)
    logical_path = catalog.paths.relative_uri(path)
    catalog.link_project_asset(project_id, asset["asset_id"], role, logical_path)
    for parent_asset_id in input_asset_ids:
        catalog.add_asset_lineage(parent_asset_id, asset["asset_id"], relation=relation, run_id=run_id)
    write_asset_manifest(catalog.paths, asset)
    return asset
