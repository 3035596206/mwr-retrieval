"""SQLite schema migrations for the local scientific-data catalog."""

from __future__ import annotations

import sqlite3

SCHEMA_VERSION = 3

DDL = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS stations (
    station_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    network TEXT,
    latitude REAL,
    longitude REAL,
    altitude_m REAL,
    timezone TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS instruments (
    instrument_id TEXT PRIMARY KEY,
    site_id TEXT REFERENCES stations(station_id),
    description TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS mwr_channels (
    instrument_id TEXT NOT NULL REFERENCES instruments(instrument_id),
    channel_index INTEGER NOT NULL,
    frequency_ghz REAL,
    polarization TEXT,
    elevation_deg REAL,
    calibration_version TEXT,
    valid_from TEXT,
    valid_to TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (instrument_id, channel_index, valid_from)
);
CREATE TABLE IF NOT EXISTS data_assets (
    asset_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    format TEXT NOT NULL,
    uri TEXT NOT NULL UNIQUE,
    content_hash TEXT NOT NULL UNIQUE,
    size_bytes INTEGER NOT NULL,
    source_name TEXT,
    source_url TEXT,
    source_request_json TEXT,
    status TEXT NOT NULL DEFAULT 'registered',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS asset_coverage (
    asset_id TEXT PRIMARY KEY REFERENCES data_assets(asset_id) ON DELETE CASCADE,
    station_id TEXT REFERENCES stations(station_id),
    start_time_utc TEXT,
    end_time_utc TEXT,
    north REAL, south REAL, east REAL, west REAL,
    latitude REAL, longitude REAL, altitude_m REAL,
    vertical_definition TEXT,
    time_resolution_s REAL,
    variables_json TEXT NOT NULL DEFAULT '[]',
    units_json TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS download_jobs (
    job_id TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    request_json TEXT NOT NULL,
    target_asset_id TEXT REFERENCES data_assets(asset_id),
    attempt_count INTEGER NOT NULL DEFAULT 0,
    state TEXT NOT NULL,
    last_error TEXT,
    started_at TEXT,
    finished_at TEXT
);
CREATE TABLE IF NOT EXISTS processing_runs (
    run_id TEXT PRIMARY KEY,
    pipeline TEXT NOT NULL,
    command TEXT,
    config_hash TEXT,
    git_revision TEXT,
    config_json TEXT NOT NULL DEFAULT '{}',
    software_environment TEXT,
    status TEXT NOT NULL,
    started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    finished_at TEXT
);
CREATE TABLE IF NOT EXISTS asset_lineage (
    parent_asset_id TEXT NOT NULL REFERENCES data_assets(asset_id),
    child_asset_id TEXT NOT NULL REFERENCES data_assets(asset_id),
    run_id TEXT REFERENCES processing_runs(run_id),
    relation TEXT NOT NULL,
    PRIMARY KEY (parent_asset_id, child_asset_id, relation)
);
CREATE TABLE IF NOT EXISTS organizations (
    organization_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS projects (
    project_id TEXT PRIMARY KEY,
    organization_id TEXT NOT NULL REFERENCES organizations(organization_id),
    name TEXT NOT NULL,
    repository_uri TEXT,
    description TEXT,
    status TEXT NOT NULL DEFAULT 'active',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS project_assets (
    project_id TEXT NOT NULL REFERENCES projects(project_id),
    asset_id TEXT NOT NULL REFERENCES data_assets(asset_id),
    role TEXT NOT NULL,
    logical_path TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (project_id, asset_id, role)
);
CREATE INDEX IF NOT EXISTS idx_project_assets_project ON project_assets(project_id, role);

CREATE TABLE IF NOT EXISTS project_asset_paths (
    project_id TEXT NOT NULL REFERENCES projects(project_id),
    asset_id TEXT NOT NULL REFERENCES data_assets(asset_id),
    role TEXT NOT NULL,
    logical_path TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (project_id, logical_path)
);
CREATE INDEX IF NOT EXISTS idx_project_asset_paths_asset ON project_asset_paths(asset_id);

CREATE TABLE IF NOT EXISTS matchups (
    matchup_id TEXT PRIMARY KEY,
    mwr_asset_id TEXT REFERENCES data_assets(asset_id),
    era5_asset_id TEXT REFERENCES data_assets(asset_id),
    radiosonde_asset_id TEXT REFERENCES data_assets(asset_id),
    gnss_asset_id TEXT REFERENCES data_assets(asset_id),
    reference_time_utc TEXT,
    max_time_delta_s REAL,
    spatial_distance_km REAL,
    qc_status TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_assets_kind ON data_assets(kind);
CREATE INDEX IF NOT EXISTS idx_coverage_time ON asset_coverage(start_time_utc, end_time_utc);
CREATE INDEX IF NOT EXISTS idx_lineage_child ON asset_lineage(child_asset_id);
"""


def migrate(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA foreign_keys = ON")
    connection.executescript(DDL)
    columns = {row[1] for row in connection.execute("PRAGMA table_info(processing_runs)")}
    if "project_id" not in columns:
        connection.execute("ALTER TABLE processing_runs ADD COLUMN project_id TEXT REFERENCES projects(project_id)")
    connection.execute(
        "INSERT OR IGNORE INTO schema_migrations(version) VALUES (?)", (SCHEMA_VERSION,)
    )
    connection.commit()
