"""Catalog persistence API."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

from ..paths import DataPaths
from .hashing import sha256_file
from .inspectors import inspect
from .schema import migrate


class Catalog:
    def __init__(self, paths: DataPaths):
        self.paths = paths
        self.paths.init()
        self.connection = sqlite3.connect(self.paths.catalog_path)
        self.connection.row_factory = sqlite3.Row
        migrate(self.connection)

    def close(self) -> None:
        self.connection.close()

    def register(self, path: Path, *, source_name: str | None = None, source_url: str | None = None,
                 source_request: dict[str, Any] | None = None) -> dict[str, Any]:
        path = path.resolve()
        if not path.is_file():
            raise ValueError(f"Asset must be an existing file: {path}")
        uri, digest, details = self.paths.relative_uri(path), sha256_file(path), inspect(path)
        existing = self.connection.execute(
            "SELECT * FROM data_assets WHERE uri = ? OR content_hash = ?", (uri, digest)
        ).fetchone()
        if existing:
            return dict(existing)
        asset_id = str(uuid.uuid4())
        metadata = details.get("metadata", {})
        with self.connection:
            self.connection.execute(
                """INSERT INTO data_assets(asset_id,kind,format,uri,content_hash,size_bytes,source_name,source_url,source_request_json,metadata_json)
                   VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (asset_id, details["kind"], details["format"], uri, digest, path.stat().st_size,
                 source_name, source_url, json.dumps(source_request or {}, ensure_ascii=False),
                 json.dumps(metadata, ensure_ascii=False)),
            )
            self.connection.execute(
                """INSERT INTO asset_coverage(asset_id,start_time_utc,end_time_utc,variables_json)
                   VALUES(?,?,?,?)""",
                (asset_id, details.get("start_time_utc"), details.get("end_time_utc"),
                 json.dumps(details.get("variables", []), ensure_ascii=False)),
            )
        return dict(self.connection.execute("SELECT * FROM data_assets WHERE asset_id = ?", (asset_id,)).fetchone())

    def create_processing_run(self, pipeline: str, *, command: str | list[str] | None = None,
                              config: dict[str, Any] | None = None, config_hash: str | None = None,
                              git_revision: str | None = None, software_environment: str | None = None,
                              status: str = "running", project_id: str | None = None,
                              run_id: str | None = None) -> dict[str, Any]:
        """Record a reproducible data-lake operation.

        A processing run is used for downloads, migrations, preprocessing,
        training, forward-model simulations, and evaluations. Raw source
        downloads may have no parent assets; derived outputs should additionally
        call :meth:`add_asset_lineage`.
        """
        run_id = run_id or str(uuid.uuid4())
        command_text = " ".join(command) if isinstance(command, list) else command
        config_payload = json.dumps(config or {}, ensure_ascii=False, sort_keys=True)
        digest = config_hash or sha256(config_payload.encode("utf-8")).hexdigest()
        with self.connection:
            self.connection.execute(
                """INSERT INTO processing_runs(
                       run_id,pipeline,command,config_hash,git_revision,config_json,
                       software_environment,status,project_id,started_at
                   ) VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (
                    run_id,
                    pipeline,
                    command_text,
                    digest,
                    git_revision,
                    config_payload,
                    software_environment,
                    status,
                    project_id,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
        return self.processing_run(run_id)

    def finish_processing_run(self, run_id: str, *, status: str = "completed") -> dict[str, Any]:
        with self.connection:
            self.connection.execute(
                "UPDATE processing_runs SET status=?, finished_at=? WHERE run_id=?",
                (status, datetime.now(timezone.utc).isoformat(), run_id),
            )
        run = self.processing_run(run_id)
        if run is None:
            raise ValueError(f"Unknown processing run: {run_id}")
        return run

    def add_asset_lineage(self, parent_asset_id: str, child_asset_id: str, *,
                          relation: str = "derived_from", run_id: str | None = None) -> None:
        with self.connection:
            self.connection.execute(
                """INSERT INTO asset_lineage(parent_asset_id,child_asset_id,run_id,relation)
                   VALUES(?,?,?,?)
                   ON CONFLICT(parent_asset_id,child_asset_id,relation)
                   DO UPDATE SET run_id=excluded.run_id""",
                (parent_asset_id, child_asset_id, run_id, relation),
            )

    def processing_run(self, run_id: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT * FROM processing_runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        return dict(row) if row else None

    def list_processing_runs(self, project_id: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        query = "SELECT * FROM processing_runs"
        params: list[Any] = []
        if project_id:
            query += " WHERE project_id = ?"
            params.append(project_id)
        query += " ORDER BY started_at DESC LIMIT ?"
        params.append(limit)
        return [dict(row) for row in self.connection.execute(query, params)]

    def lineage_for_asset(self, asset_id: str) -> dict[str, list[dict[str, Any]]]:
        parents = self.connection.execute(
            """SELECT asset_lineage.*, parent.uri AS parent_uri, child.uri AS child_uri,
                      processing_runs.pipeline AS pipeline, processing_runs.status AS run_status
               FROM asset_lineage
               JOIN data_assets AS parent ON parent.asset_id = asset_lineage.parent_asset_id
               JOIN data_assets AS child ON child.asset_id = asset_lineage.child_asset_id
               LEFT JOIN processing_runs USING(run_id)
               WHERE asset_lineage.child_asset_id = ?
               ORDER BY relation, parent.uri""",
            (asset_id,),
        )
        children = self.connection.execute(
            """SELECT asset_lineage.*, parent.uri AS parent_uri, child.uri AS child_uri,
                      processing_runs.pipeline AS pipeline, processing_runs.status AS run_status
               FROM asset_lineage
               JOIN data_assets AS parent ON parent.asset_id = asset_lineage.parent_asset_id
               JOIN data_assets AS child ON child.asset_id = asset_lineage.child_asset_id
               LEFT JOIN processing_runs USING(run_id)
               WHERE asset_lineage.parent_asset_id = ?
               ORDER BY relation, child.uri""",
            (asset_id,),
        )
        return {
            "parents": [dict(row) for row in parents],
            "children": [dict(row) for row in children],
        }

    def add_project(self, project_id: str, organization_id: str, name: str, repository_uri: str | None = None,
                    description: str | None = None) -> None:
        with self.connection:
            self.connection.execute(
                "INSERT OR IGNORE INTO organizations(organization_id,name,description) VALUES(?,?,?)",
                (organization_id, "Project 504", "Research institute shared data lake"),
            )
            self.connection.execute(
                """INSERT INTO projects(project_id,organization_id,name,repository_uri,description)
                   VALUES(?,?,?,?,?) ON CONFLICT(project_id) DO UPDATE SET name=excluded.name, repository_uri=excluded.repository_uri""",
                (project_id, organization_id, name, repository_uri, description),
            )

    def link_project_asset(self, project_id: str, asset_id: str, role: str, logical_path: str | None = None) -> None:
        logical_path = logical_path or asset_id
        with self.connection:
            self.connection.execute(
                """INSERT INTO project_assets(project_id,asset_id,role,logical_path) VALUES(?,?,?,?)
                   ON CONFLICT(project_id,asset_id,role) DO UPDATE SET logical_path=excluded.logical_path""",
                (project_id, asset_id, role, logical_path),
            )
            self.connection.execute(
                """INSERT INTO project_asset_paths(project_id,asset_id,role,logical_path) VALUES(?,?,?,?)
                   ON CONFLICT(project_id,logical_path) DO UPDATE SET asset_id=excluded.asset_id, role=excluded.role""",
                (project_id, asset_id, role, logical_path),
            )

    def project_assets(self, project_id: str) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """SELECT data_assets.*, project_asset_paths.role, project_asset_paths.logical_path
               FROM project_asset_paths JOIN data_assets USING(asset_id) WHERE project_asset_paths.project_id = ?
               ORDER BY project_asset_paths.role, project_asset_paths.logical_path""", (project_id,)
        )
        return [dict(row) for row in rows]

    def list_assets(self, kind: str | None = None) -> list[dict[str, Any]]:
        query, params = "SELECT * FROM data_assets", []
        if kind:
            query += " WHERE kind = ?"; params.append(kind)
        query += " ORDER BY created_at, uri"
        return [dict(row) for row in self.connection.execute(query, params)]

    def asset(self, asset_id: str) -> dict[str, Any] | None:
        row = self.connection.execute("SELECT * FROM data_assets WHERE asset_id = ?", (asset_id,)).fetchone()
        return dict(row) if row else None

    def verify(self, asset_id: str) -> dict[str, Any]:
        asset = self.asset(asset_id)
        if asset is None:
            raise ValueError(f"Unknown asset: {asset_id}")
        path = self.paths.resolve_uri(asset["uri"])
        exists = path.is_file()
        actual = sha256_file(path) if exists else None
        return {"asset_id": asset_id, "path": str(path), "exists": exists,
                "hash_matches": actual == asset["content_hash"] if actual else False}
