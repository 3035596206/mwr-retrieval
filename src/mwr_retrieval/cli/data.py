"""Command line management for the local MWR data lake."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..catalog.migration import build_plan, execute_plan, write_plan
from ..catalog.register_existing import register_existing, write_report
from ..catalog.repository import Catalog
from ..catalog.workspace_ingest import (
    build_plan as build_workspace_plan,
    execute_plan as execute_workspace_plan,
    write_plan as write_workspace_plan,
    write_report as write_workspace_report,
)
from ..manifest import write_asset_manifest
from ..paths import DataPaths
from ..settings import load_instruments, load_sites


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="mwr-data")
    root.add_argument("--data-root", help="Overrides MWR_DATA_ROOT for this command")
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("init")
    register = commands.add_parser("register"); register.add_argument("path", type=Path)
    legacy = commands.add_parser("register-existing"); legacy.add_argument("--project-root", type=Path)
    plan = commands.add_parser("migration-plan"); plan.add_argument("--project-root", type=Path)
    migrate = commands.add_parser("migrate-project-brnn"); migrate.add_argument("--project-root", type=Path)
    workspace_plan = commands.add_parser("workspace-ingest-plan"); workspace_plan.add_argument("--workspace-root", type=Path)
    workspace_migrate = commands.add_parser("migrate-workspace-sources"); workspace_migrate.add_argument("--workspace-root", type=Path)
    project = commands.add_parser("project-assets"); project.add_argument("project_id", nargs="?", default="project-brnn")
    runs = commands.add_parser("runs"); runs.add_argument("--project-id"); runs.add_argument("--limit", type=int, default=50)
    show_run = commands.add_parser("show-run"); show_run.add_argument("run_id")
    lineage = commands.add_parser("lineage"); lineage.add_argument("asset_id")
    listing = commands.add_parser("list"); listing.add_argument("--kind")
    show = commands.add_parser("show"); show.add_argument("asset_id")
    verify = commands.add_parser("verify"); verify.add_argument("asset_id")
    return root


def seed_reference_metadata(catalog: Catalog) -> None:
    with catalog.connection:
        for site in load_sites().values():
            catalog.connection.execute(
                """INSERT INTO stations(station_id,name,network,latitude,longitude,altitude_m,timezone,metadata_json)
                VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(station_id) DO UPDATE SET name=excluded.name""",
                (site["site_id"], site["name"], site.get("network"), site.get("latitude"), site.get("longitude"),
                 site.get("altitude_m"), site.get("timezone"), json.dumps(site, ensure_ascii=False)),
            )
        for instrument in load_instruments().values():
            catalog.connection.execute(
                "INSERT INTO instruments(instrument_id,site_id,description,metadata_json) VALUES(?,?,?,?) ON CONFLICT(instrument_id) DO NOTHING",
                (instrument["instrument_id"], instrument.get("site_id"), instrument.get("description"), json.dumps(instrument, ensure_ascii=False)),
            )
            for channel in instrument["channels"]:
                catalog.connection.execute(
                    "INSERT OR IGNORE INTO mwr_channels(instrument_id,channel_index,frequency_ghz,metadata_json,valid_from) VALUES(?,?,?,?,?)",
                    (instrument["instrument_id"], channel["index"], channel["frequency_ghz"], json.dumps(channel), ""),
                )


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    paths = DataPaths.from_value(args.data_root)
    paths.init()
    catalog = Catalog(paths)
    try:
        seed_reference_metadata(catalog)
        if args.command == "init":
            print(json.dumps({"data_root": str(paths.root), "catalog": str(paths.catalog_path)}, ensure_ascii=False)); return 0
        if args.command == "register":
            asset = catalog.register(args.path); manifest = write_asset_manifest(paths, asset)
            print(json.dumps({"asset": asset, "manifest": str(manifest)}, ensure_ascii=False, indent=2)); return 0
        if args.command == "register-existing":
            roots = None if args.project_root is None else [args.project_root]
            report = register_existing(catalog, roots); reports = write_report(paths, report)
            print(json.dumps({"registered": len(report["registered"]), "skipped": len(report["skipped"]), "errors": len(report["errors"]), "reports": [str(item) for item in reports]}, ensure_ascii=False)); return 0
        if args.command == "migration-plan":
            plan = build_plan(args.project_root); reports = write_plan(paths, plan)
            print(json.dumps({"project_id": "project-brnn", "assets": len(plan), "reports": [str(item) for item in reports]}, ensure_ascii=False)); return 0
        if args.command == "migrate-project-brnn":
            plan = build_plan(args.project_root); plan_reports = write_plan(paths, plan)
            report = execute_plan(catalog, plan); report["plan_reports"] = [str(item) for item in plan_reports]
            report_paths = write_report(paths, report)
            print(json.dumps({key: len(value) if isinstance(value, list) else value for key, value in report.items() if key != "plan_reports"} | {"plan_reports": report["plan_reports"], "report_paths": [str(item) for item in report_paths]}, ensure_ascii=False)); return 0
        if args.command == "workspace-ingest-plan":
            plan = build_workspace_plan(args.workspace_root); reports = write_workspace_plan(paths, plan)
            print(json.dumps({"project_id": "project-brnn", "assets": len(plan), "reports": [str(item) for item in reports]}, ensure_ascii=False)); return 0
        if args.command == "migrate-workspace-sources":
            plan = build_workspace_plan(args.workspace_root); plan_reports = write_workspace_plan(paths, plan)
            report = execute_workspace_plan(catalog, plan); report["plan_reports"] = [str(item) for item in plan_reports]
            report_paths = write_workspace_report(paths, report)
            print(json.dumps({key: len(value) if isinstance(value, list) else value for key, value in report.items() if key != "plan_reports"} | {"plan_reports": report["plan_reports"], "report_paths": [str(item) for item in report_paths]}, ensure_ascii=False)); return 0
        if args.command == "project-assets":
            print(json.dumps(catalog.project_assets(args.project_id), ensure_ascii=False, indent=2)); return 0
        if args.command == "runs":
            print(json.dumps(catalog.list_processing_runs(args.project_id, args.limit), ensure_ascii=False, indent=2)); return 0
        if args.command == "show-run":
            print(json.dumps(catalog.processing_run(args.run_id), ensure_ascii=False, indent=2)); return 0
        if args.command == "lineage":
            print(json.dumps(catalog.lineage_for_asset(args.asset_id), ensure_ascii=False, indent=2)); return 0
        if args.command == "list": print(json.dumps(catalog.list_assets(args.kind), ensure_ascii=False, indent=2)); return 0
        if args.command == "show": print(json.dumps(catalog.asset(args.asset_id), ensure_ascii=False, indent=2)); return 0
        if args.command == "verify": print(json.dumps(catalog.verify(args.asset_id), ensure_ascii=False, indent=2)); return 0
    finally:
        catalog.close()
    return 1

if __name__ == "__main__":
    raise SystemExit(main())
