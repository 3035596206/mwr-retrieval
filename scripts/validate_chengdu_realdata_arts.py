#!/usr/bin/env python3
"""Validate the Chengdu real-data bridge with the ARTS forward model.

The validation compares observed 21-channel brightness temperatures with clear-sky
ARTS simulations driven by the matched ERA5 48-layer profiles.  The resulting
O-B residuals are an interface and consistency audit; they are not a cloudy-sky
retrieval accuracy score.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BRIDGE = PROJECT_ROOT / "results" / "chengdu_realdata_bridge" / "run_id=20260804T075728Z-d5b0fa9c" / "bridge_dataset.npz"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from forward_model import ForwardModel  # noqa: E402


def sample_indices(n_total: int, n_samples: int | None, seed: int) -> np.ndarray:
    if n_samples is None or n_samples >= n_total:
        return np.arange(n_total, dtype=int)
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    # Use an evenly distributed sample for temporal coverage, with seed only used
    # to break ties if later sampling modes are added.
    _ = np.random.default_rng(seed)
    return np.unique(np.linspace(0, n_total - 1, n_samples, dtype=int))


def rmse(values: np.ndarray, axis: int | None = None) -> np.ndarray:
    return np.sqrt(np.mean(np.square(values), axis=axis))


def band_masks(frequencies: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "K_22_31GHz": frequencies < 40.0,
        "V_51_58GHz": (frequencies >= 40.0) & (frequencies < 80.0),
        "W_89GHz": (frequencies >= 80.0) & (frequencies < 120.0),
        "G_183GHz": (frequencies >= 170.0) & (frequencies < 200.0),
        "window_229GHz": frequencies >= 200.0,
    }


def channel_stats(residual: np.ndarray, frequencies: np.ndarray) -> list[dict[str, Any]]:
    stats = []
    for index, frequency in enumerate(frequencies):
        values = residual[:, index]
        stats.append(
            {
                "channel_index": index,
                "frequency_ghz": float(frequency),
                "bias_obs_minus_arts_k": float(np.mean(values)),
                "rmse_obs_minus_arts_k": float(rmse(values)),
                "std_obs_minus_arts_k": float(np.std(values)),
                "median_abs_obs_minus_arts_k": float(np.median(np.abs(values))),
                "min_obs_minus_arts_k": float(np.min(values)),
                "max_obs_minus_arts_k": float(np.max(values)),
            }
        )
    return stats


def summarize_residuals(residual: np.ndarray, frequencies: np.ndarray) -> dict[str, Any]:
    bands = {}
    for name, mask in band_masks(frequencies).items():
        if np.any(mask):
            values = residual[:, mask]
            bands[name] = {
                "n_channels": int(mask.sum()),
                "bias_obs_minus_arts_k": float(np.mean(values)),
                "rmse_obs_minus_arts_k": float(rmse(values)),
                "std_obs_minus_arts_k": float(np.std(values)),
            }
    return {
        "overall": {
            "bias_obs_minus_arts_k": float(np.mean(residual)),
            "rmse_obs_minus_arts_k": float(rmse(residual)),
            "std_obs_minus_arts_k": float(np.std(residual)),
            "median_abs_obs_minus_arts_k": float(np.median(np.abs(residual))),
        },
        "bands": bands,
        "channels": channel_stats(residual, frequencies),
    }


def output_paths(output_dir: Path, run_id: str) -> tuple[Path, Path, Path]:
    run_dir = output_dir / f"run_id={run_id}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir / "arts_validation_predictions.npz", run_dir / "arts_validation_stats.json", run_dir / "manifest.json"


def register_catalog(
    *,
    data_root: Path,
    project_id: str,
    run_id: str,
    bridge_dataset: Path,
    predictions_path: Path,
    stats: dict[str, Any],
) -> dict[str, Any]:
    from mwr_retrieval.artifacts import register_run_output
    from mwr_retrieval.catalog.repository import Catalog
    from mwr_retrieval.paths import DataPaths

    catalog = Catalog(DataPaths.from_value(data_root))
    try:
        catalog.add_project(
            project_id,
            "project-504",
            "BRNN temperature and humidity retrieval",
            PROJECT_ROOT.as_uri(),
            "Current MWR BRNN retrieval project",
        )
        run = catalog.create_processing_run(
            "scripts/validate_chengdu_realdata_arts.py",
            command=sys.argv,
            config={
                "bridge_dataset": str(bridge_dataset),
                "n_requested": stats["dataset"]["n_requested"],
                "n_success": stats["dataset"]["n_success"],
                "forward_backend": "arts",
                "clear_sky": True,
            },
            project_id=project_id,
            run_id=run_id,
        )
        input_asset = catalog.register(bridge_dataset, source_name="chengdu_realdata_bridge")
        catalog.link_project_asset(project_id, input_asset["asset_id"], "input", catalog.paths.relative_uri(bridge_dataset))
        output_asset = register_run_output(
            catalog,
            predictions_path,
            run_id=run["run_id"],
            input_asset_ids=[input_asset["asset_id"]],
            relation="validated_by",
            role="validation",
            project_id=project_id,
            source_name="chengdu_realdata_arts_validation",
        )
        catalog.finish_processing_run(run["run_id"], status="completed")
        return {
            "status": "registered",
            "data_root": str(data_root),
            "project_id": project_id,
            "processing_run_id": run["run_id"],
            "input_asset_id": input_asset["asset_id"],
            "output_asset_id": output_asset["asset_id"],
            "output_asset_uri": output_asset["uri"],
        }
    finally:
        catalog.close()


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("--bridge-dataset", type=Path, default=DEFAULT_BRIDGE)
    root.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "chengdu_realdata_arts_validation")
    root.add_argument("--n-samples", type=int, default=20)
    root.add_argument("--all", action="store_true", help="validate all bridge samples")
    root.add_argument("--seed", type=int, default=42)
    root.add_argument("--register-catalog", action="store_true")
    root.add_argument("--catalog-data-root", type=Path, default=PROJECT_ROOT.parent.parent / "project-504-data")
    root.add_argument("--project-id", default="project-brnn")
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    bridge_dataset = args.bridge_dataset.resolve()
    run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
    predictions_path, stats_path, manifest_path = output_paths(args.output_dir, run_id)

    with np.load(bridge_dataset, allow_pickle=False) as data:
        obs_bt = np.asarray(data["X"], dtype=np.float64)
        temperature = np.asarray(data["T"], dtype=np.float64)
        relative_humidity = np.asarray(data["RH"], dtype=np.float64)
        pressure = np.asarray(data["P"], dtype=np.float64)
        heights = np.asarray(data["heights"], dtype=np.float64)
        frequencies = np.asarray(data["channel_frequencies_ghz"], dtype=np.float64)
        timestamps = np.asarray(data["timestamps"]).astype(str)
        bridge_run_id = str(data["run_id"]) if "run_id" in data.files else None

    selected = sample_indices(len(obs_bt), None if args.all else args.n_samples, args.seed)
    fm = ForwardModel(backend="arts", frequencies=frequencies, arts_persistent=True)

    simulated = np.full((len(selected), len(frequencies)), np.nan, dtype=np.float64)
    failures: list[dict[str, Any]] = []
    elapsed_by_sample = []
    started = time.time()
    try:
        for output_index, sample_index in enumerate(selected):
            profile = {
                "T": temperature[sample_index],
                "RH": relative_humidity[sample_index],
                "P_hPa": pressure[sample_index],
                "CLWC": np.zeros_like(heights),
                "height": heights,
            }
            sample_start = time.time()
            try:
                simulated[output_index] = fm.simulate(profile)
            except Exception as error:  # noqa: BLE001 - collect validation failures
                failures.append({"sample_index": int(sample_index), "timestamp": timestamps[sample_index], "error": str(error)})
            elapsed_by_sample.append(time.time() - sample_start)
    finally:
        backend = getattr(fm, "_backend", None)
        if backend is not None and hasattr(backend, "close"):
            backend.close()

    success_mask = np.all(np.isfinite(simulated), axis=1)
    success_indices = selected[success_mask]
    obs_selected = obs_bt[selected]
    residual = obs_selected[success_mask] - simulated[success_mask]
    stats = {
        "status": "chengdu_realdata_arts_validation_complete" if np.any(success_mask) else "chengdu_realdata_arts_validation_failed",
        "run_id": run_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "bridge_dataset": str(bridge_dataset),
        "bridge_run_id": bridge_run_id,
        "validation_scope": "clear_sky_arts_forward_observed_minus_background_audit",
        "dataset": {
            "n_bridge_samples": int(len(obs_bt)),
            "n_requested": int(len(selected)),
            "n_success": int(success_mask.sum()),
            "n_failed": int(len(failures)),
            "sample_indices": selected.astype(int).tolist(),
            "success_indices": success_indices.astype(int).tolist(),
            "first_timestamp": str(timestamps[selected[0]]) if len(selected) else None,
            "last_timestamp": str(timestamps[selected[-1]]) if len(selected) else None,
        },
        "forward_model": {
            "backend": "arts",
            "clear_sky": True,
            "cloud_liquid_water_g_m3": "zero profile",
            "n_channels": int(len(frequencies)),
            "channel_frequencies_ghz": frequencies.astype(float).tolist(),
            "height_grid_m": heights.astype(float).tolist(),
        },
        "performance": {
            "elapsed_sec": float(time.time() - started),
            "mean_sample_sec": float(np.mean(elapsed_by_sample)) if elapsed_by_sample else None,
            "first_sample_sec": float(elapsed_by_sample[0]) if elapsed_by_sample else None,
            "remaining_mean_sample_sec": float(np.mean(elapsed_by_sample[1:])) if len(elapsed_by_sample) > 1 else None,
        },
        "failures": failures,
        "residual_definition": "obs_minus_arts_k",
        "residual_stats": summarize_residuals(residual, frequencies) if np.any(success_mask) else None,
        "interpretation_note": "Large residuals are expected where clouds, calibration offsets, channel response, or site/ERA5 representativeness differ; this run validates the real-data-to-ARTS interface, not final retrieval accuracy.",
    }

    np.savez_compressed(
        predictions_path,
        run_id=np.asarray(run_id),
        bridge_run_id=np.asarray(bridge_run_id or ""),
        sample_indices=selected,
        success_mask=success_mask,
        timestamps=timestamps[selected],
        observed_bt_k=obs_selected,
        arts_bt_k=simulated,
        obs_minus_arts_k=obs_selected - simulated,
        channel_frequencies_ghz=frequencies,
        heights=heights,
    )
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "manifest_version": 1,
        "run_id": run_id,
        "pipeline": "scripts/validate_chengdu_realdata_arts.py",
        "generated_at_utc": stats["generated_at_utc"],
        "inputs": {"bridge_dataset": str(bridge_dataset), "bridge_run_id": bridge_run_id},
        "outputs": {"predictions": str(predictions_path), "stats": str(stats_path)},
        "summary": {
            "status": stats["status"],
            "n_success": stats["dataset"]["n_success"],
            "n_failed": stats["dataset"]["n_failed"],
            "overall_rmse_obs_minus_arts_k": stats["residual_stats"]["overall"]["rmse_obs_minus_arts_k"] if stats["residual_stats"] else None,
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.register_catalog:
        try:
            catalog_registration = register_catalog(
                data_root=args.catalog_data_root,
                project_id=args.project_id,
                run_id=run_id,
                bridge_dataset=bridge_dataset,
                predictions_path=predictions_path,
                stats=stats,
            )
        except Exception as error:  # noqa: BLE001
            catalog_registration = {"status": "failed", "error_type": type(error).__name__, "error": str(error)}
        stats["catalog_registration"] = catalog_registration
        stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest["catalog_registration"] = catalog_registration
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({"run_id": run_id, "outputs": {"predictions": str(predictions_path), "stats": str(stats_path), "manifest": str(manifest_path)}, "summary": manifest["summary"]}, ensure_ascii=False, indent=2))
    return 0 if np.any(success_mask) else 1


if __name__ == "__main__":
    raise SystemExit(main())
