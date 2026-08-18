"""Filesystem layout and compatibility helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

_ENV_DATA_ROOT = "MWR_DATA_ROOT"
_DEFAULT_DATA_ROOT = Path(r"D:\project-504-data")


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def workspace_root() -> Path:
    return project_root().parent


def resolve_data_root(value: str | Path | None = None) -> Path:
    if value is not None:
        return Path(value).expanduser().resolve()
    configured = os.environ.get(_ENV_DATA_ROOT)
    if configured:
        return Path(configured).expanduser().resolve()
    return _DEFAULT_DATA_ROOT


@dataclass(frozen=True)
class DataPaths:
    """Canonical data-lake paths. `init()` creates them explicitly."""

    root: Path

    @classmethod
    def from_value(cls, value: str | Path | None = None) -> "DataPaths":
        return cls(resolve_data_root(value))

    @property
    def catalog_path(self) -> Path:
        return self.root / "metadata" / "catalog.sqlite3"

    @property
    def manifest_root(self) -> Path:
        return self.root / "metadata" / "manifests"

    @property
    def all_directories(self) -> tuple[Path, ...]:
        relative = (
            "raw/era5/cds", "raw/era5/arco", "raw/era5/grib",
            "raw/radiosonde/wyoming", "raw/radiosonde/wenjiang",
            "raw/mwr/chengdu", "raw/mwr/mp3000a", "raw/gnss", "raw/satellite",
            "raw/ancillary/monortm", "interim/era5/extracted",
            "interim/era5/grid48", "interim/radiosonde/parsed",
            "interim/radiosonde/layer48", "interim/mwr/hourly",
            "interim/monortm", "projects/project-brnn/curated/datasets",
            "projects/project-brnn/curated/models", "projects/project-brnn/curated/predictions",
            "projects/project-brnn/curated/reports", "projects/project-brnn/configs",
            "projects/project-brnn/manifests", "metadata/manifests/raw",
            "metadata/manifests/derived", "metadata/logs/downloads",
            "tmp/era5-download", "tmp/monortm",
        )
        return tuple(self.root / item for item in relative)

    def init(self) -> None:
        for directory in self.all_directories:
            directory.mkdir(parents=True, exist_ok=True)

    def relative_uri(self, path: Path) -> str:
        path = path.resolve()
        try:
            return path.relative_to(self.root.resolve()).as_posix()
        except ValueError:
            try:
                return f"legacy://{path.relative_to(project_root()).as_posix()}"
            except ValueError:
                return f"external://{path.as_posix()}"

    def resolve_uri(self, uri: str) -> Path:
        if uri.startswith("legacy://"):
            return project_root() / Path(uri.removeprefix("legacy://"))
        if uri.startswith("external://"):
            return Path(uri.removeprefix("external://"))
        return self.root / Path(uri)

    def compatible_candidates(self, *relative_parts: str) -> Iterable[Path]:
        yield self.root.joinpath(*relative_parts)
        yield project_root().joinpath("data", *relative_parts)
