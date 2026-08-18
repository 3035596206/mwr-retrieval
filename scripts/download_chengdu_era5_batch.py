"""Download Chengdu ERA5 monthly pressure/single-level files.

This is a thin batch wrapper around ``scripts/download_era5.py`` / the
catalog-backed downloader. It targets the Chengdu training range selected for
local retrieval improvement and skips files that are already present.
"""

from __future__ import annotations

import argparse
import calendar
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path(r"D:\project-504-data")
CHENGDU_AREA = {
    "north": 31.5,
    "west": 102.5,
    "south": 29.5,
    "east": 105.5,
}
PRESSURE_VARIABLES = ",".join(
    [
        "temperature",
        "relative_humidity",
        "specific_humidity",
        "geopotential",
        "u_component_of_wind",
        "v_component_of_wind",
    ]
)
SINGLE_VARIABLES = ",".join(
    [
        "2m_temperature",
        "2m_dewpoint_temperature",
        "surface_pressure",
        "total_column_water_vapour",
        "total_cloud_cover",
        "total_precipitation",
        "skin_temperature",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "boundary_layer_height",
    ]
)
PRESSURE_LEVELS = ",".join(
    [
        "1000",
        "975",
        "950",
        "925",
        "900",
        "875",
        "850",
        "825",
        "800",
        "775",
        "750",
        "700",
        "650",
        "600",
        "550",
        "500",
        "450",
        "400",
        "350",
        "300",
        "250",
        "225",
        "200",
        "175",
        "150",
        "125",
        "100",
    ]
)
ALL_HOURS = ",".join(f"{hour:02d}:00" for hour in range(24))


@dataclass(frozen=True)
class DownloadTask:
    year: int
    month: int
    level_type: str
    variables: str
    days: str | None = None
    request_tag: str | None = None
    pressure_levels: str | None = None

    def target_path(self, data_root: Path, site_id: str) -> Path:
        filename = f"era5_{self.level_type}_{self.year}{self.month:02d}.nc"
        if self.request_tag:
            filename = f"era5_{self.level_type}_{self.year}{self.month:02d}_{self.request_tag}.nc"
            return (
                data_root
                / "raw"
                / "era5"
                / "cds"
                / site_id
                / f"{self.level_type}-levels"
                / f"year={self.year}"
                / f"month={self.month:02d}"
                / f"part={self.request_tag}"
                / filename
            )
        return (
            data_root
            / "raw"
            / "era5"
            / "cds"
            / site_id
            / f"{self.level_type}-levels"
            / f"year={self.year}"
            / f"month={self.month:02d}"
            / filename
        )


def month_iter(start: str, end: str):
    start_year, start_month = map(int, start.split("-"))
    end_year, end_month = map(int, end.split("-"))
    year, month = start_year, start_month
    while (year, month) <= (end_year, end_month):
        yield year, month
        month += 1
        if month == 13:
            year += 1
            month = 1


def day_chunks(year: int, month: int, chunk_size: int | None) -> list[tuple[str, str | None]]:
    days_in_month = calendar.monthrange(year, month)[1]
    if not chunk_size:
        days = ",".join(f"{day:02d}" for day in range(1, days_in_month + 1))
        return [(days, None)]
    chunks = []
    for start_day in range(1, days_in_month + 1, chunk_size):
        end_day = min(start_day + chunk_size - 1, days_in_month)
        days = ",".join(f"{day:02d}" for day in range(start_day, end_day + 1))
        chunks.append((days, f"d{start_day:02d}-{end_day:02d}"))
    return chunks


def build_tasks(start: str, end: str, day_chunk_size: int | None) -> list[DownloadTask]:
    tasks: list[DownloadTask] = []
    for year, month in month_iter(start, end):
        for days, request_tag in day_chunks(year, month, day_chunk_size):
            tasks.append(
                DownloadTask(
                    year=year,
                    month=month,
                    level_type="pressure",
                    variables=PRESSURE_VARIABLES,
                    days=days,
                    request_tag=request_tag,
                    pressure_levels=PRESSURE_LEVELS,
                )
            )
            tasks.append(
                DownloadTask(
                    year=year,
                    month=month,
                    level_type="single",
                    variables=SINGLE_VARIABLES,
                    days=days,
                    request_tag=request_tag,
                )
            )
    return tasks


def command_for_task(
    task: DownloadTask,
    data_root: Path,
    site_id: str,
    python_exe: str,
    days: str | None,
    times: str,
) -> list[str]:
    task_days = days or task.days or ",".join(
        f"{day:02d}" for day in range(1, calendar.monthrange(task.year, task.month)[1] + 1)
    )
    command = [
        python_exe,
        str(PROJECT_ROOT / "scripts" / "download_era5.py"),
        "--data-root",
        str(data_root),
        "--level-type",
        task.level_type,
        "--year",
        str(task.year),
        "--month",
        str(task.month),
        "--days",
        task_days,
        "--times",
        times,
        "--variables",
        task.variables,
        "--north",
        str(CHENGDU_AREA["north"]),
        "--west",
        str(CHENGDU_AREA["west"]),
        "--south",
        str(CHENGDU_AREA["south"]),
        "--east",
        str(CHENGDU_AREA["east"]),
        "--site-id",
        site_id,
    ]
    if task.request_tag:
        command.extend(["--request-tag", task.request_tag])
    if task.pressure_levels:
        command.extend(["--pressure-levels", task.pressure_levels])
    return command


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--start", default="2023-01", help="first month, YYYY-MM")
    parser.add_argument("--end", default="2026-06", help="last month, YYYY-MM")
    parser.add_argument("--site-id", default="chengdu")
    parser.add_argument("--days", help="comma-separated day list; defaults to all valid days in each month")
    parser.add_argument("--times", default=ALL_HOURS)
    parser.add_argument("--day-chunk-size", type=int, default=1, help="split monthly requests into N-day chunks; use 0 for monthly")
    parser.add_argument("--python", default=sys.executable, help="Python executable used to run download_era5.py")
    parser.add_argument("--limit", type=int, help="run at most this many pending tasks")
    parser.add_argument("--task-retries", type=int, default=3, help="retry each failed task this many times")
    parser.add_argument("--task-retry-delay", type=float, default=120.0, help="seconds to wait before retrying a failed task")
    parser.add_argument("--plan-only", action="store_true", help="print pending tasks without downloading")
    parser.add_argument("--include-existing", action="store_true", help="include tasks whose target file already exists")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    tasks = build_tasks(args.start, args.end, args.day_chunk_size or None)
    pending: list[DownloadTask] = []
    for task in tasks:
        target = task.target_path(args.data_root, args.site_id)
        if target.exists() and not args.include_existing:
            continue
        pending.append(task)

    if args.limit is not None:
        pending = pending[: args.limit]

    plan = [
        {
            "year": task.year,
            "month": task.month,
            "level_type": task.level_type,
            "target": str(task.target_path(args.data_root, args.site_id)),
            "days": task.days,
            "request_tag": task.request_tag,
            "variables": task.variables.split(","),
        }
        for task in pending
    ]
    print(json.dumps({"n_pending": len(pending), "tasks": plan}, indent=2, ensure_ascii=False))
    if args.plan_only:
        return 0

    for index, task in enumerate(pending, start=1):
        print(f"[{index}/{len(pending)}] {task.level_type} {task.year}-{task.month:02d}", flush=True)
        command = command_for_task(task, args.data_root, args.site_id, args.python, args.days, args.times)
        for attempt in range(1, args.task_retries + 1):
            try:
                subprocess.run(command, cwd=PROJECT_ROOT, check=True)
                break
            except subprocess.CalledProcessError:
                if attempt >= args.task_retries:
                    raise
                print(
                    f"Task failed; retrying attempt {attempt + 1}/{args.task_retries} "
                    f"after {args.task_retry_delay:g}s",
                    flush=True,
                )
                time.sleep(args.task_retry_delay)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
