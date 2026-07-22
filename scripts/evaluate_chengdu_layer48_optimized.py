#!/usr/bin/env python3
"""Evaluate the optimized physical 48-layer Chengdu retrieval."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mwr-retrieval-matplotlib"))

import matplotlib.pyplot as plt
from matplotlib import font_manager

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from build_chengdu_era5_layer48_dataset import (
    build_layer_grid,
    layer_average_from_interpolated,
    parse_sounding_layers,
)
from train_chengdu_brnn import sounding_time_from_name


COLORS = {"old": "#6a1b9a", "point": "#ef6c00", "optimized": "#1976d2", "qeof": "#2e7d32", "truth": "#202124"}


def configure_plotting() -> None:
    for path in [Path(r"C:\Windows\Fonts\msyh.ttc"), Path(r"C:\Windows\Fonts\simhei.ttf")]:
        if path.exists():
            font_manager.fontManager.addfont(path)
            plt.rcParams["font.sans-serif"] = [font_manager.FontProperties(fname=path).get_name()]
            break
    plt.rcParams.update(
        {
            "axes.unicode_minus": False,
            "figure.dpi": 130,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "legend.frameon": False,
        }
    )


def remap_to_layers(values: np.ndarray, heights: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.stack(
        [layer_average_from_interpolated(heights, row, edges) for row in values]
    ).astype(np.float32)


def metrics(pred: np.ndarray, truth: np.ndarray, weights: np.ndarray | None = None) -> dict[str, float]:
    error = pred - truth
    if weights is None:
        return {
            "rmse": float(np.sqrt(np.mean(error**2))),
            "mae": float(np.mean(np.abs(error))),
            "bias": float(np.mean(error)),
        }
    normalized = weights / weights.sum()
    return {
        "rmse": float(np.sqrt(np.mean(np.sum(error**2 * normalized, axis=1)))),
        "mae": float(np.mean(np.sum(np.abs(error) * normalized, axis=1))),
        "bias": float(np.mean(np.sum(error * normalized, axis=1))),
    }


def pair_soundings(sounding_dir: Path, timestamps: np.ndarray, edges: np.ndarray) -> list[dict]:
    observation_times = [datetime.strptime(str(item), "%Y_%m_%d %H:%M:%S") for item in timestamps]
    pairs = []
    for path in sorted(sounding_dir.rglob("*.txt")):
        launch = sounding_time_from_name(path)
        if launch.strftime("%Y%m%d") not in {"20260528", "20260529", "20260530"} or launch.hour not in {0, 12}:
            continue
        try:
            sounding = parse_sounding_layers(path, edges)
        except ValueError:
            continue
        index = min(
            range(len(observation_times)),
            key=lambda item: abs((observation_times[item] - launch).total_seconds()),
        )
        delta = abs((observation_times[index] - launch).total_seconds()) / 3600.0
        if delta <= 3.0:
            pairs.append(
                {
                    "launch": launch,
                    "observation": observation_times[index],
                    "index": index,
                    "delta_hours": delta,
                    "sounding": sounding,
                }
            )
    return pairs


def bar_panel(ax, labels, values, colors, title, ylabel) -> None:
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors, width=0.68)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=8)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, max(values) * 1.24)
    ax.grid(axis="x", visible=False)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2f}", ha="center", va="bottom")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate optimized physical 48-layer retrieval")
    parser.add_argument("--sounding-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_plotting()

    results = PROJECT_ROOT / "results"
    edges, centers = build_layer_grid()
    dataset = np.load(results / "chengdu_era5_layer48_dataset" / "chengdu_era5_layer48_dataset.npz")
    dataset_stats = json.loads(
        (results / "chengdu_era5_layer48_dataset" / "chengdu_era5_layer48_dataset_stats.json").read_text(encoding="utf-8")
    )
    old = np.load(results / "chengdu_era5_hybrid" / "chengdu_era5_hybrid_predictions.npz")
    point = np.load(results / "chengdu_era5_corrected48_hybrid" / "chengdu_corrected48_hybrid_predictions.npz")
    optimized = np.load(
        results / "chengdu_era5_layer48_selective_hybrid" / "chengdu_corrected48_hybrid_predictions.npz"
    )
    qeof = np.load(results / "chengdu_era5_layer48_qeof_all21" / "chengdu_layer48_qeof_predictions.npz")

    predictions = {
        "old93": {
            "T": remap_to_layers(old["T_pred"], old["heights"], edges),
            "RH": remap_to_layers(old["RH_pred"], old["heights"], edges),
        },
        "point48": {
            "T": remap_to_layers(point["T_pred"], point["heights"], edges),
            "RH": remap_to_layers(point["RH_pred"], point["heights"], edges),
        },
        "optimized48": {"T": optimized["T_pred"], "RH": optimized["RH_pred"]},
        "qeof48": {"T": qeof["T_pred"], "RH": qeof["RH_pred"]},
    }
    test_mask = dataset["test_mask"].astype(bool)
    raw_truth = {"T": dataset["T_raw"][test_mask], "RH": dataset["RH_raw"][test_mask]}
    selected_truth = {"T": dataset["T"][test_mask], "RH": dataset["RH_raw"][test_mask]}
    thickness = dataset["layer_thickness"]

    era5_metrics = {}
    for model_name, prediction in predictions.items():
        target = selected_truth if model_name in {"optimized48", "qeof48"} else raw_truth
        era5_metrics[model_name] = {
            variable: {
                "equal_layer": metrics(prediction[variable], target[variable]),
                "thickness_weighted": metrics(prediction[variable], target[variable], thickness),
            }
            for variable in ("T", "RH")
        }

    pairs = pair_soundings(args.sounding_dir, optimized["timestamps"], edges)
    sounding_metrics = {}
    for model_name, prediction in predictions.items():
        sounding_metrics[model_name] = {
            variable: metrics(
                np.stack([prediction[variable][item["index"]] for item in pairs]),
                np.stack([item["sounding"][variable] for item in pairs]),
            )
            for variable in ("T", "RH")
        }

    band_metrics = {}
    for model_name, prediction in predictions.items():
        band_metrics[model_name] = {}
        for low, high in [(0.0, 2.0), (2.0, 5.0), (5.0, 8.0), (8.0, 10.0)]:
            mask = (centers >= low * 1000.0) & (centers < high * 1000.0)
            band_metrics[model_name][f"{low}-{high}km"] = {}
            for variable in ("T", "RH"):
                pred = np.stack([prediction[variable][item["index"], mask] for item in pairs])
                truth = np.stack([item["sounding"][variable][mask] for item in pairs])
                band_metrics[model_name][f"{low}-{high}km"][variable] = metrics(pred, truth)

    summary = {
        "status": "chengdu_optimized_physical_48_layer_evaluation",
        "layer_edges_m": edges.tolist(),
        "layer_centers_m": centers.tolist(),
        "n_test_samples": int(test_mask.sum()),
        "n_independent_soundings": len(pairs),
        "era5_metrics": era5_metrics,
        "independent_sounding_metrics": sounding_metrics,
        "independent_sounding_band_metrics": band_metrics,
        "sounding_pairs": [
            {
                "launch": item["launch"].isoformat(),
                "observation": item["observation"].isoformat(),
                "delta_hours": item["delta_hours"],
            }
            for item in pairs
        ],
        "processing_decisions": {
            "physical_layer_averaging": True,
            "below_ground_era5_levels_removed": True,
            "temperature_bias_correction": True,
            "logq_bias_correction": False,
            "humidity_target": "raw physical-layer RH",
            "humidity_model": "validation-selected segment-wise top-2 BRNN ensemble",
            "qeof_promoted": False,
        },
        "correction_validation": dataset_stats["correction_evaluation"],
    }
    (args.output_dir / "chengdu_layer48_optimized_stats.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.5), sharey=True)
    thickness_colors = [COLORS["optimized"] if value <= 100 else COLORS["point"] for value in thickness]
    axes[0].barh(centers / 1000.0, thickness, height=np.minimum(thickness / 1000.0, 0.18), color=thickness_colors)
    axes[0].set_xlabel("层厚 (m)")
    axes[0].set_ylabel("层中心高度 (km)")
    axes[0].set_title("物理48层")
    axes[1].plot(dataset["bias_T"], centers / 1000.0, color=COLORS["optimized"], linewidth=2)
    axes[1].axvline(0, color="#5f6368", linestyle="--", linewidth=1)
    axes[1].set_xlabel("温度订正量 (K)")
    axes[1].set_title("验证集保留")
    axes[2].plot(dataset["bias_logq"], centers / 1000.0, color=COLORS["qeof"], linewidth=2)
    axes[2].axvline(0, color="#5f6368", linestyle="--", linewidth=1)
    axes[2].set_xlabel("log(q)订正量")
    axes[2].set_title("验证集拒绝")
    fig.suptitle("物理层平均与选择性偏差订正", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(args.output_dir / "01_layer_grid_processing.png", facecolor="white")
    plt.close(fig)

    labels = ["原93层\n层平均", "旧48点\n层平均", "优化物理48层", "log(q)-EOF"]
    names = ["old93", "point48", "optimized48", "qeof48"]
    colors = [COLORS["old"], COLORS["point"], COLORS["optimized"], COLORS["qeof"]]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6))
    for ax, variable, unit in [(axes[0], "T", "K"), (axes[1], "RH", "%")]:
        bar_panel(
            ax,
            labels,
            [sounding_metrics[name][variable]["rmse"] for name in names],
            colors,
            "温度" if variable == "T" else "相对湿度",
            f"独立探空RMSE ({unit})",
        )
    fig.suptitle(f"独立温江探空测试（{len(pairs)}个时次）", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(args.output_dir / "02_independent_sounding_comparison.png", facecolor="white")
    plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(15.5, 5.6), sharey=True)
    for ax, variable, kind, unit in [
        (axes[0], "T", "RMSE", "K"),
        (axes[1], "T", "Bias", "K"),
        (axes[2], "RH", "RMSE", "%"),
        (axes[3], "RH", "Bias", "%"),
    ]:
        for name, label, color in [
            ("old93", "原93层", COLORS["old"]),
            ("point48", "旧48点", COLORS["point"]),
            ("optimized48", "优化48层", COLORS["optimized"]),
        ]:
            pred = np.stack([predictions[name][variable][item["index"]] for item in pairs])
            truth = np.stack([item["sounding"][variable] for item in pairs])
            error = pred - truth
            value = np.sqrt(np.mean(error**2, axis=0)) if kind == "RMSE" else np.mean(error, axis=0)
            ax.plot(value, centers / 1000.0, color=color, linewidth=1.9, label=label)
        if kind == "Bias":
            ax.axvline(0, color="#5f6368", linestyle="--", linewidth=1)
        ax.set_xlabel(f"{kind} ({unit})")
        ax.set_title(f"{'温度' if variable == 'T' else '相对湿度'} {kind}")
    axes[0].set_ylabel("层中心高度 (km)")
    axes[0].legend(loc="best")
    fig.suptitle("独立探空分高度误差", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(args.output_dir / "03_sounding_height_errors.png", facecolor="white")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.0), sharey=True)
    for row, (variable, unit) in enumerate([("T", "K"), ("RH", "%")]):
        truth = np.stack([item["sounding"][variable] for item in pairs])
        axes[row, 0].plot(truth.mean(axis=0), centers / 1000.0, color=COLORS["truth"], linewidth=2.2, label="温江探空")
        for name, label, color in [
            ("old93", "原93层", COLORS["old"]),
            ("point48", "旧48点", COLORS["point"]),
            ("optimized48", "优化48层", COLORS["optimized"]),
        ]:
            pred = np.stack([predictions[name][variable][item["index"]] for item in pairs])
            axes[row, 0].plot(pred.mean(axis=0), centers / 1000.0, color=color, linewidth=1.8, label=label)
            axes[row, 1].plot(np.sqrt(np.mean((pred - truth) ** 2, axis=0)), centers / 1000.0, color=color, linewidth=1.8, label=label)
        axes[row, 0].set_xlabel(f"{'温度' if variable == 'T' else '相对湿度'} ({unit})")
        axes[row, 1].set_xlabel(f"RMSE ({unit})")
        axes[row, 0].set_ylabel("层中心高度 (km)")
    axes[0, 0].set_title("独立探空平均廓线")
    axes[0, 1].set_title("独立探空分高度RMSE")
    axes[0, 0].legend(loc="best")
    fig.suptitle("优化物理48层廓线表现", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(args.output_dir / "04_sounding_profiles.png", facecolor="white")
    plt.close(fig)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
