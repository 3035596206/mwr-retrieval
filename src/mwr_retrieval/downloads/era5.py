"""ERA5 download workflow with atomic placement and catalog registration."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from ..catalog.repository import Catalog
from ..manifest import write_asset_manifest
from ..paths import DataPaths

DATASETS = {
    "pressure": "reanalysis-era5-pressure-levels",
    "single": "reanalysis-era5-single-levels",
}


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def build_request(args: argparse.Namespace) -> dict:
    request = {
        "product_type": "reanalysis",
        "variable": parse_csv(args.variables),
        "year": str(args.year),
        "month": f"{args.month:02d}",
        "day": parse_csv(args.days),
        "time": parse_csv(args.times),
        "area": [args.north, args.west, args.south, args.east],
        "data_format": "netcdf",
    }
    if args.level_type == "pressure":
        request["pressure_level"] = parse_csv(args.pressure_levels)
    return request


def output_path(paths: DataPaths, args: argparse.Namespace) -> Path:
    scope = args.site_id or f"area={args.north:g}_{args.west:g}_{args.south:g}_{args.east:g}"
    filename = f"era5_{args.level_type}_{args.year}{args.month:02d}.nc"
    if args.request_tag:
        safe_tag = "".join(char if char.isalnum() or char in {"-", "_", "="} else "_" for char in args.request_tag)
        filename = f"era5_{args.level_type}_{args.year}{args.month:02d}_{safe_tag}.nc"
        return paths.root / "raw" / "era5" / "cds" / scope / f"{args.level_type}-levels" / f"year={args.year}" / f"month={args.month:02d}" / f"part={safe_tag}" / filename
    return paths.root / "raw" / "era5" / "cds" / scope / f"{args.level_type}-levels" / f"year={args.year}" / f"month={args.month:02d}" / filename


def validate_netcdf(path: Path) -> None:
    import xarray as xr
    with xr.open_dataset(path) as dataset:
        if not dataset.data_vars:
            raise ValueError("ERA5 download contains no data variables")
        time_name = "valid_time" if "valid_time" in dataset else "time"
        if time_name in dataset and dataset.sizes.get(time_name, 0) == 0:
            raise ValueError("ERA5 download contains no time records")


def normalize_downloaded_file(path: Path, temporary: Path) -> Path:
    """Return a NetCDF file path, extracting CDS ZIP responses when needed."""
    if not zipfile.is_zipfile(path):
        return path

    extract_root = temporary / "extracted"
    extract_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as archive:
        members = [name for name in archive.namelist() if not name.endswith("/")]
        netcdf_members = [
            name for name in members
            if Path(name).suffix.lower() in {".nc", ".nc4", ".cdf", ".netcdf"}
        ]
        if not netcdf_members:
            raise ValueError(
                "Expected at least one NetCDF file inside CDS ZIP response; found none"
            )
        extracted_paths = []
        for member in netcdf_members:
            extracted = extract_root / Path(member).name
            with archive.open(member) as source, extracted.open("wb") as target:
                shutil.copyfileobj(source, target)
            extracted_paths.append(extracted)
    if len(extracted_paths) == 1:
        return extracted_paths[0]

    import xarray as xr

    datasets = [xr.open_dataset(item) for item in extracted_paths]
    try:
        merged = xr.merge(datasets, compat="override", combine_attrs="drop_conflicts")
        merged_path = temporary / "merged_download.nc"
        merged.to_netcdf(merged_path, engine="netcdf4")
    finally:
        for dataset in datasets:
            dataset.close()
    return merged_path


def download(args: argparse.Namespace) -> dict:
    paths = DataPaths.from_value(args.data_root)
    catalog = Catalog(paths)
    request, job_id = build_request(args), str(uuid.uuid4())
    temporary = paths.root / "tmp" / "era5-download" / job_id
    temporary.mkdir(parents=True, exist_ok=False)
    target, staged = output_path(paths, args), temporary / "download.nc"
    with catalog.connection:
        catalog.connection.execute(
            "INSERT INTO download_jobs(job_id,provider,request_json,state,started_at) VALUES(?,?,?,?,?)",
            (job_id, "copernicus-cds", json.dumps(request), "running", datetime.now(timezone.utc).isoformat()),
        )
    run = catalog.create_processing_run(
        "era5_cds_download",
        command=[Path(sys.argv[0]).name, *sys.argv[1:]],
        config={
            "job_id": job_id,
            "dataset": DATASETS[args.level_type],
            "request": request,
            "target": str(target),
            "site_id": args.site_id,
            "request_tag": args.request_tag,
        },
        project_id=None,
    )
    try:
        import cdsapi
        cdsapi.Client().retrieve(DATASETS[args.level_type], request).download(str(staged))
        normalized = normalize_downloaded_file(staged, temporary)
        validate_netcdf(normalized)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and not args.overwrite:
            raise FileExistsError(f"Target already exists: {target}; use --overwrite only after reviewing it")
        os.replace(normalized, target)
        asset = catalog.register(target, source_name="Copernicus Climate Data Store", source_url="https://cds.climate.copernicus.eu/", source_request=request)
        manifest = write_asset_manifest(paths, asset)
        with catalog.connection:
            catalog.connection.execute("UPDATE download_jobs SET target_asset_id=?, state='completed', attempt_count=1, finished_at=? WHERE job_id=?", (asset["asset_id"], datetime.now(timezone.utc).isoformat(), job_id))
        catalog.finish_processing_run(run["run_id"], status="completed")
        return {"job_id": job_id, "run_id": run["run_id"], "asset_id": asset["asset_id"], "path": str(target), "manifest": str(manifest)}
    except Exception as error:
        with catalog.connection:
            catalog.connection.execute("UPDATE download_jobs SET state='failed', attempt_count=1, last_error=?, finished_at=? WHERE job_id=?", (str(error), datetime.now(timezone.utc).isoformat(), job_id))
        catalog.finish_processing_run(run["run_id"], status="failed")
        raise
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
        catalog.close()


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(prog="mwr-download-era5")
    command.add_argument("--data-root")
    command.add_argument("--level-type", choices=DATASETS, required=True)
    command.add_argument("--year", type=int, required=True); command.add_argument("--month", type=int, required=True)
    command.add_argument("--days", default="01"); command.add_argument("--times", default="00:00")
    command.add_argument("--variables", required=True)
    command.add_argument("--pressure-levels", default="1000,975,950,925,900,850,800,700,600,500,400,300,200,100")
    command.add_argument("--north", type=float, required=True); command.add_argument("--west", type=float, required=True)
    command.add_argument("--south", type=float, required=True); command.add_argument("--east", type=float, required=True)
    command.add_argument("--site-id"); command.add_argument("--request-tag")
    command.add_argument("--overwrite", action="store_true")
    return command


def main(argv: list[str] | None = None) -> int:
    result = download(parser().parse_args(argv))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
