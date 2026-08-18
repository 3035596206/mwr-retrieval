"""Ingest external workspace source files into the shared data lake."""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..manifest import write_asset_manifest
from ..paths import DataPaths, project_root, workspace_root
from .hashing import sha256_file
from .inspectors import infer_kind
from .migration import ORGANIZATION_ID, PROJECT_ID
from .repository import Catalog

SOURCE_DIRECTORIES = (
    "chengdu_era5",
    "chengdu_obs_bt",
    "wenjiang_sounding",
    "obs_bt_filtered_20260726",
    "观测亮温筛选7.26",
)
SKIP_DIRECTORIES = {".git", "__pycache__"}
WENJIANG_STATION_ID = "56187"


def source_roots(root: Path | None = None) -> list[Path]:
    root = (root or workspace_root()).resolve()
    return [root / name for name in SOURCE_DIRECTORIES if (root / name).exists()]


def _sounding_partition(source: Path) -> Path:
    match = re.search(r"SEC-56187-(\d{10})", source.stem)
    if not match:
        return Path(f"station={WENJIANG_STATION_ID}") / "unpartitioned"
    stamp = match.group(1)
    return Path(f"station={WENJIANG_STATION_ID}") / f"year={stamp[:4]}" / f"month={stamp[4:6]}"


def target_relative(source: Path, root: Path) -> Path:
    relative = source.relative_to(root)
    source_group = relative.parts[0]
    if source_group == "chengdu_era5":
        return Path("raw/era5/grib/site=chengdu") / source.name
    if source_group == "chengdu_obs_bt":
        return Path("raw/mwr/chengdu/obs_bt") / source.name
    if source_group == "wenjiang_sounding":
        return Path("raw/radiosonde/wenjiang") / _sounding_partition(source) / source.name
    if source_group in {"obs_bt_filtered_20260726", "观测亮温筛选7.26"}:
        return Path("interim/mwr/chengdu/obs_bt_filtered_20260726") / source.name
    raise ValueError(f"Unsupported workspace source group: {source_group}")


def role_for_target(target: Path) -> str:
    return "input" if target.parts[0] == "raw" else "intermediate"


def build_plan(root: Path | None = None) -> list[dict[str, Any]]:
    root = (root or workspace_root()).resolve()
    records, hashes = [], defaultdict(list)
    for source_root in source_roots(root):
        for source in source_root.rglob("*"):
            if any(part in SKIP_DIRECTORIES for part in source.parts) or not source.is_file():
                continue
            target = target_relative(source, root)
            digest = sha256_file(source)
            record = {
                "source": str(source),
                "relative_source": source.relative_to(root).as_posix(),
                "target_relative": target.as_posix(),
                "size_bytes": source.stat().st_size,
                "sha256": digest,
                "kind": infer_kind(target),
                "role": role_for_target(target),
                "project_id": PROJECT_ID,
            }
            records.append(record)
            hashes[digest].append(record["relative_source"])
    for record in records:
        record["duplicate_sources"] = hashes[record["sha256"]]
        record["is_duplicate_content"] = len(record["duplicate_sources"]) > 1
    return sorted(records, key=lambda item: item["relative_source"])


def write_plan(paths: DataPaths, plan: list[dict[str, Any]]) -> tuple[Path, Path]:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_dir = paths.root / "metadata" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    json_path = log_dir / f"workspace-ingest-plan-{PROJECT_ID}-{stamp}.json"
    csv_path = log_dir / f"workspace-ingest-plan-{PROJECT_ID}-{stamp}.csv"
    json_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    fieldnames = [
        "source",
        "relative_source",
        "target_relative",
        "size_bytes",
        "sha256",
        "kind",
        "role",
        "project_id",
        "is_duplicate_content",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{key: row.get(key) for key in fieldnames} for row in plan])
    return json_path, csv_path


def write_report(paths: DataPaths, report: dict[str, Any]) -> tuple[Path, Path]:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_dir = paths.root / "metadata" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    json_path = log_dir / f"workspace-ingest-report-{PROJECT_ID}-{stamp}.json"
    csv_path = log_dir / f"workspace-ingest-report-{PROJECT_ID}-{stamp}.csv"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["status", "source", "target", "asset_id", "error"])
        writer.writeheader()
        for status, rows in report.items():
            if not isinstance(rows, list):
                continue
            for row in rows:
                if isinstance(row, str):
                    row = {"source": row}
                writer.writerow({"status": status, **row})
    return json_path, csv_path


def atomic_copy(source: Path, target: Path, expected_hash: str) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    final_target = target
    if final_target.exists():
        if sha256_file(final_target) == expected_hash:
            return final_target
        final_target = final_target.with_name(f"{final_target.stem}--{expected_hash[:12]}{final_target.suffix}")
    staged = final_target.parent / f".{final_target.name}.copying"
    shutil.copyfile(source, staged)
    if sha256_file(staged) != expected_hash:
        staged.unlink(missing_ok=True)
        raise ValueError(f"Hash mismatch while copying {source}")
    os.replace(staged, final_target)
    return final_target


def execute_plan(catalog: Catalog, plan: list[dict[str, Any]]) -> dict[str, Any]:
    catalog.add_project(
        PROJECT_ID,
        ORGANIZATION_ID,
        "BRNN temperature and humidity retrieval",
        project_root().as_uri(),
        "Current MWR BRNN retrieval project",
    )
    run = catalog.create_processing_run(
        "workspace_source_ingest",
        command="python -m mwr_retrieval.cli.data migrate-workspace-sources",
        config={"project_id": PROJECT_ID, "planned_assets": len(plan)},
        project_id=PROJECT_ID,
    )
    report: dict[str, Any] = {
        "project_id": PROJECT_ID,
        "run_id": run["run_id"],
        "copied": [],
        "already_present": [],
        "failed": [],
        "source_verification_failed": [],
    }
    try:
        for item in plan:
            source = Path(item["source"])
            before = sha256_file(source)
            target = catalog.paths.root / item["target_relative"]
            try:
                existed = target.exists() and sha256_file(target) == item["sha256"]
                final_target = atomic_copy(source, target, item["sha256"])
                if sha256_file(source) != before:
                    report["source_verification_failed"].append(item["relative_source"])
                    continue
                asset = catalog.register(final_target, source_name="project-504 workspace ingest")
                logical_path = final_target.relative_to(catalog.paths.root).as_posix()
                catalog.link_project_asset(PROJECT_ID, asset["asset_id"], item["role"], logical_path)
                write_asset_manifest(catalog.paths, asset)
                report["already_present" if existed else "copied"].append(
                    {"source": item["relative_source"], "target": logical_path, "asset_id": asset["asset_id"]}
                )
            except Exception as error:
                report["failed"].append({"source": item["relative_source"], "error": f"{type(error).__name__}: {error}"})
    finally:
        status = "completed" if not report["failed"] and not report["source_verification_failed"] else "completed_with_errors"
        catalog.finish_processing_run(run["run_id"], status=status)
    return report
