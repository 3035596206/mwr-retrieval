from pathlib import Path

import numpy as np

from mwr_retrieval.catalog.repository import Catalog
from mwr_retrieval.catalog.workspace_ingest import (
    build_plan as build_workspace_ingest_plan,
    execute_plan as execute_workspace_ingest_plan,
)
from mwr_retrieval.grids import build_layer48_grid
from mwr_retrieval.paths import DataPaths
from mwr_retrieval.thermodynamics import rh_to_specific_humidity, specific_humidity_to_rh


def test_paths_catalog_and_asset_verification(tmp_path: Path):
    paths = DataPaths.from_value(tmp_path / "lake")
    catalog = Catalog(paths)
    source = tmp_path / "sample.txt"
    source.write_text("catalog fixture", encoding="utf-8")
    asset = catalog.register(source)
    assert asset["uri"].startswith("external://")
    assert catalog.register(source)["asset_id"] == asset["asset_id"]
    assert catalog.verify(asset["asset_id"])["hash_matches"]
    catalog.close()


def test_processing_runs_and_asset_lineage(tmp_path: Path):
    paths = DataPaths.from_value(tmp_path / "lake")
    catalog = Catalog(paths)
    parent = tmp_path / "parent.txt"
    child = paths.root / "projects" / "project-brnn" / "curated" / "datasets" / "child.txt"
    parent.write_text("input", encoding="utf-8")
    child.parent.mkdir(parents=True)
    child.write_text("output", encoding="utf-8")

    parent_asset = catalog.register(parent)
    child_asset = catalog.register(child)
    run = catalog.create_processing_run(
        "unit_test_pipeline",
        command=["pytest", "tests/test_data_foundation.py"],
        config={"alpha": 1},
        project_id="project-brnn",
    )
    catalog.add_asset_lineage(parent_asset["asset_id"], child_asset["asset_id"], run_id=run["run_id"])
    finished = catalog.finish_processing_run(run["run_id"])

    assert finished["status"] == "completed"
    assert catalog.list_processing_runs("project-brnn", limit=1)[0]["run_id"] == run["run_id"]
    lineage = catalog.lineage_for_asset(child_asset["asset_id"])
    assert lineage["parents"][0]["parent_asset_id"] == parent_asset["asset_id"]
    assert lineage["parents"][0]["pipeline"] == "unit_test_pipeline"
    catalog.close()


def test_layer48_grid_and_humidity_round_trip():
    edges, centers = build_layer48_grid()
    assert len(edges) == 49 and len(centers) == 48
    temperature = np.array([273.15, 293.15])
    pressure = np.array([700.0, 1000.0])
    rh = np.array([50.0, 80.0])
    recovered = specific_humidity_to_rh(temperature, rh_to_specific_humidity(temperature, rh, pressure), pressure)
    np.testing.assert_allclose(recovered, rh, atol=1e-5)


def test_workspace_ingest_plan_and_copy(tmp_path: Path):
    workspace = tmp_path / "workspace"
    era5_dir = workspace / "chengdu_era5" / "source"
    mwr_dir = workspace / "chengdu_obs_bt"
    sounding_dir = workspace / "wenjiang_sounding" / "station"
    era5_dir.mkdir(parents=True)
    mwr_dir.mkdir(parents=True)
    sounding_dir.mkdir(parents=True)
    (era5_dir / "202604.grib").write_bytes(b"grib fixture")
    (mwr_dir / "obs_bt_20260513.xlsx").write_bytes(b"xlsx fixture")
    (sounding_dir / "UPAR_WEA_CHN_MUL_FTM_SEC-56187-2026010100.txt").write_text("sounding", encoding="utf-8")

    plan = build_workspace_ingest_plan(workspace)
    targets = {item["relative_source"]: item["target_relative"] for item in plan}

    assert targets["chengdu_era5/source/202604.grib"] == "raw/era5/grib/site=chengdu/202604.grib"
    assert targets["chengdu_obs_bt/obs_bt_20260513.xlsx"] == "raw/mwr/chengdu/obs_bt/obs_bt_20260513.xlsx"
    assert (
        targets["wenjiang_sounding/station/UPAR_WEA_CHN_MUL_FTM_SEC-56187-2026010100.txt"]
        == "raw/radiosonde/wenjiang/station=56187/year=2026/month=01/UPAR_WEA_CHN_MUL_FTM_SEC-56187-2026010100.txt"
    )

    paths = DataPaths.from_value(tmp_path / "lake")
    catalog = Catalog(paths)
    report = execute_workspace_ingest_plan(catalog, plan)
    assert catalog.processing_run(report["run_id"])["status"] == "completed"
    assert len(report["copied"]) == 3
    assert not report["failed"]
    assert (paths.root / "raw/era5/grib/site=chengdu/202604.grib").is_file()
    assert len(catalog.project_assets("project-brnn")) == 3
    catalog.close()
