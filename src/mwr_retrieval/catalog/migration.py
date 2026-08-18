"""Safe copy-and-verify migration from a legacy project into the shared data lake."""

from __future__ import annotations

import csv
import json
import os
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..manifest import write_asset_manifest
from ..paths import DataPaths, project_root
from .hashing import sha256_file
from .inspectors import infer_kind
from .repository import Catalog

PROJECT_ID = "project-brnn"
ORGANIZATION_ID = "project-504"
SKIP_DIRECTORIES = {".git", ".venv-wsl", "__pycache__"}


def source_roots(root: Path | None = None) -> list[Path]:
    root = root or project_root()
    roots = [root / "data", root / "models", root / "results", root / "reports"]
    roots.extend(sorted(item for item in root.glob("models_*") if item.is_dir()))
    return [item for item in roots if item.exists()]


def role_for(kind: str, source: Path) -> str:
    if kind in {"era5", "radiosonde", "mwr", "ancillary_tape3"}:
        return "input" if "data" in source.parts else "intermediate"
    if kind == "model": return "model"
    if kind == "report": return "report"
    if kind == "result": return "prediction"
    return "intermediate"


def target_relative(source: Path, root: Path) -> Path:
    relative = source.relative_to(root)
    if relative.parts[0] == "data":
        if source.name == "chengdu_obs_bt.json": return Path("raw/mwr/chengdu") / source.name
        if "TAPE3" in source.parts: return Path("raw/ancillary/monortm") / source.name
        if "era5" in source.parts:
            branch = "interim/monortm" if "bt_sim" in source.name else "interim/era5/extracted"
            return Path(branch) / source.name
        return Path("raw") / relative
    if relative.parts[0].startswith("models"):
        return Path("projects") / PROJECT_ID / "curated/models" / relative
    if relative.parts[0] == "reports":
        return Path("projects") / PROJECT_ID / "curated/reports" / relative.relative_to("reports")
    if relative.parts[0] == "results":
        if source.suffix.lower() in {".docx", ".pdf", ".pptx", ".rar"}:
            return Path("projects") / PROJECT_ID / "curated/reports/from-results" / source.name
        category = "datasets" if source.suffix.lower() in {".npz", ".csv"} and "dataset" in str(source.parent).lower() else "predictions"
        return Path("projects") / PROJECT_ID / f"curated/{category}" / relative.relative_to("results")
    return Path("projects") / PROJECT_ID / "curated/predictions" / relative


def build_plan(root: Path | None = None) -> list[dict[str, Any]]:
    root = (root or project_root()).resolve()
    records, hashes = [], defaultdict(list)
    for source_root in source_roots(root):
        for source in source_root.rglob("*"):
            if not source.is_file() or any(part in SKIP_DIRECTORIES for part in source.parts):
                continue
            digest = sha256_file(source)
            record = {
                "source": str(source), "relative_source": source.relative_to(root).as_posix(),
                "target_relative": target_relative(source, root).as_posix(), "size_bytes": source.stat().st_size,
                "sha256": digest, "kind": infer_kind(source), "role": role_for(infer_kind(source), source),
                "project_id": PROJECT_ID,
            }
            records.append(record); hashes[digest].append(record["relative_source"])
    for record in records:
        record["duplicate_sources"] = hashes[record["sha256"]]
        record["is_duplicate_content"] = len(record["duplicate_sources"]) > 1
    return sorted(records, key=lambda item: item["relative_source"])


def write_plan(paths: DataPaths, plan: list[dict[str, Any]]) -> tuple[Path, Path]:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_dir = paths.root / "metadata" / "logs"; log_dir.mkdir(parents=True, exist_ok=True)
    json_path, csv_path = log_dir / f"migration-plan-{PROJECT_ID}-{stamp}.json", log_dir / f"migration-plan-{PROJECT_ID}-{stamp}.csv"
    json_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[key for key in plan[0] if key != "duplicate_sources"])
        writer.writeheader(); writer.writerows([{key: value for key, value in row.items() if key != "duplicate_sources"} for row in plan])
    return json_path, csv_path


def atomic_copy(source: Path, target: Path, expected_hash: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha256_file(target) == expected_hash: return
        target = target.with_name(f"{target.stem}--{expected_hash[:12]}{target.suffix}")
    staged = target.parent / f".{target.name}.copying"
    shutil.copyfile(source, staged)
    if sha256_file(staged) != expected_hash:
        staged.unlink(missing_ok=True); raise ValueError(f"Hash mismatch while copying {source}")
    os.replace(staged, target)


def execute_plan(catalog: Catalog, plan: list[dict[str, Any]]) -> dict[str, Any]:
    catalog.add_project(PROJECT_ID, ORGANIZATION_ID, "BRNN temperature and humidity retrieval", project_root().as_uri(), "Current MWR BRNN retrieval project")
    run = catalog.create_processing_run(
        "legacy_project_migration",
        command="python -m mwr_retrieval.cli.data migrate-project-brnn",
        config={"project_id": PROJECT_ID, "planned_assets": len(plan)},
        project_id=PROJECT_ID,
    )
    report: dict[str, Any] = {"project_id": PROJECT_ID, "run_id": run["run_id"], "copied": [], "already_present": [], "failed": [], "source_verification_failed": []}
    try:
        for item in plan:
            source = Path(item["source"])
            before = sha256_file(source)
            target = catalog.paths.root / item["target_relative"]
            try:
                existed = target.exists() and sha256_file(target) == item["sha256"]
                atomic_copy(source, target, item["sha256"])
                if sha256_file(source) != before:
                    report["source_verification_failed"].append(item["relative_source"]); continue
                asset = catalog.register(target, source_name="project-brnn legacy migration")
                catalog.link_project_asset(PROJECT_ID, asset["asset_id"], item["role"], item["target_relative"])
                write_asset_manifest(catalog.paths, asset)
                report["already_present" if existed else "copied"].append({"source": item["relative_source"], "target": item["target_relative"], "asset_id": asset["asset_id"]})
            except Exception as error:
                report["failed"].append({"source": item["relative_source"], "error": f"{type(error).__name__}: {error}"})
    finally:
        status = "completed" if not report["failed"] and not report["source_verification_failed"] else "completed_with_errors"
        catalog.finish_processing_run(run["run_id"], status=status)
    return report
