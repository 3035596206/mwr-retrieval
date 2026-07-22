#!/usr/bin/env python3
"""Evaluate the new 48-layer and radiosonde-bias-corrected Chengdu experiments."""

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

from train_chengdu_brnn import parse_sounding, sounding_time_from_name


COLORS = {"old": "#6a1b9a", "raw": "#ef6c00", "corrected": "#1976d2", "truth": "#202124"}


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


def metrics(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    error = pred - truth
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
    }


def interpolate_profiles(values: np.ndarray, source_height: np.ndarray, target_height: np.ndarray) -> np.ndarray:
    return np.stack([np.interp(target_height, source_height, row) for row in values]).astype(np.float32)


def pair_test_soundings(
    sounding_dir: Path,
    timestamps: np.ndarray,
    heights: np.ndarray,
    test_dates: set[str],
    max_delta_hours: float,
) -> list[dict]:
    observation_times = [datetime.strptime(str(item), "%Y_%m_%d %H:%M:%S") for item in timestamps]
    pairs = []
    for path in sorted(sounding_dir.rglob("*.txt")):
        launch_time = sounding_time_from_name(path)
        if launch_time.strftime("%Y%m%d") not in test_dates or launch_time.hour not in {0, 12}:
            continue
        try:
            sounding_t, sounding_rh, _ = parse_sounding(path, heights)
        except ValueError:
            continue
        index = min(
            range(len(observation_times)),
            key=lambda item: abs((observation_times[item] - launch_time).total_seconds()),
        )
        delta_hours = abs((observation_times[index] - launch_time).total_seconds()) / 3600.0
        if delta_hours <= max_delta_hours:
            pairs.append(
                {
                    "launch_time": launch_time,
                    "observation_time": observation_times[index],
                    "observation_index": index,
                    "delta_hours": delta_hours,
                    "T": sounding_t,
                    "RH": sounding_rh,
                }
            )
    return pairs


def bar_values(ax: plt.Axes, labels: list[str], values: list[float], colors: list[str], ylabel: str, title: str) -> None:
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors, width=0.68)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, max(values) * 1.24)
    ax.grid(axis="x", visible=False)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2f}", ha="center", va="bottom")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate corrected Chengdu 48-layer results")
    parser.add_argument("--sounding-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "chengdu_era5_corrected48_evaluation",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_plotting()

    results = PROJECT_ROOT / "results"
    dataset = np.load(
        results / "chengdu_era5_corrected48_dataset" / "chengdu_era5_corrected48_dataset.npz"
    )
    dataset_stats = json.loads(
        (
            results
            / "chengdu_era5_corrected48_dataset"
            / "chengdu_era5_corrected48_dataset_stats.json"
        ).read_text(encoding="utf-8")
    )
    old = np.load(results / "chengdu_era5_hybrid" / "chengdu_era5_hybrid_predictions.npz")
    raw48 = np.load(results / "chengdu_era5_raw48_hybrid" / "chengdu_corrected48_hybrid_predictions.npz")
    corrected48 = np.load(
        results / "chengdu_era5_corrected48_hybrid" / "chengdu_corrected48_hybrid_predictions.npz"
    )
    heights = corrected48["heights"]
    test_mask = dataset["test_mask"].astype(bool)
    raw_truth = {"T": dataset["T_raw"][test_mask], "RH": dataset["RH_raw"][test_mask]}
    corrected_truth = {"T": dataset["T"][test_mask], "RH": dataset["RH"][test_mask]}
    predictions = {
        "old93": {
            "T": interpolate_profiles(old["T_pred"], old["heights"], heights),
            "RH": interpolate_profiles(old["RH_pred"], old["heights"], heights),
        },
        "raw48": {"T": raw48["T_pred"], "RH": raw48["RH_pred"]},
        "corrected48": {"T": corrected48["T_pred"], "RH": corrected48["RH_pred"]},
    }

    era5_metrics = {
        "old93_interpolated_vs_raw48": {
            variable: metrics(predictions["old93"][variable], raw_truth[variable])
            for variable in ("T", "RH")
        },
        "raw48_vs_raw48": {
            variable: metrics(predictions["raw48"][variable], raw_truth[variable])
            for variable in ("T", "RH")
        },
        "corrected48_vs_corrected48": {
            variable: metrics(predictions["corrected48"][variable], corrected_truth[variable])
            for variable in ("T", "RH")
        },
        "corrected48_vs_raw48_reference": {
            variable: metrics(predictions["corrected48"][variable], raw_truth[variable])
            for variable in ("T", "RH")
        },
    }

    pairs = pair_test_soundings(
        args.sounding_dir,
        corrected48["timestamps"],
        heights,
        set(str(item) for item in corrected48["dates"]),
        max_delta_hours=3.0,
    )
    sounding_metrics = {}
    for model_name, model_prediction in predictions.items():
        sounding_metrics[model_name] = {}
        for variable in ("T", "RH"):
            pred = np.stack(
                [model_prediction[variable][item["observation_index"]] for item in pairs]
            )
            truth = np.stack([item[variable] for item in pairs])
            sounding_metrics[model_name][variable] = metrics(pred, truth)

    band_metrics = {}
    for model_name in ("old93", "raw48", "corrected48"):
        target = corrected_truth if model_name == "corrected48" else raw_truth
        band_metrics[model_name] = {}
        for low, high in [(0.5, 2.0), (2.25, 5.0), (5.25, 8.0), (8.25, 10.0)]:
            mask = (heights >= low * 1000.0) & (heights <= high * 1000.0)
            band_metrics[model_name][f"{low}-{high}km"] = {
                variable: metrics(
                    predictions[model_name][variable][:, mask], target[variable][:, mask]
                )
                for variable in ("T", "RH")
            }

    summary = {
        "status": "chengdu_48layer_bias_correction_evaluation",
        "height_grid_m": heights.tolist(),
        "n_era5_test_samples": int(test_mask.sum()),
        "n_independent_test_soundings": len(pairs),
        "sounding_pairs": [
            {
                "launch_time": item["launch_time"].isoformat(),
                "observation_time": item["observation_time"].isoformat(),
                "delta_hours": item["delta_hours"],
            }
            for item in pairs
        ],
        "era5_test_metrics": era5_metrics,
        "independent_sounding_metrics": sounding_metrics,
        "band_metrics": band_metrics,
        "era5_sounding_bias_correction": dataset_stats["sounding_evaluation"],
        "decision": {
            "temperature": "retain 48-layer radiosonde bias correction",
            "humidity": "do not promote radiosonde mean-bias correction yet",
            "reason": "temperature improves on independent soundings; humidity does not show stable validation/test gain",
        },
    }
    (args.output_dir / "chengdu_corrected48_comparison_stats.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    bias_t = dataset["bias_T"]
    bias_rh = dataset["bias_RH"]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.5), sharey=True)
    axes[0].scatter(np.zeros_like(heights), heights / 1000.0, color=COLORS["corrected"], s=22)
    for value in heights / 1000.0:
        axes[0].plot([-0.08, 0.08], [value, value], color=COLORS["corrected"], linewidth=0.8)
    axes[0].set_xlim(-0.25, 0.25)
    axes[0].set_xticks([])
    axes[0].set_ylabel("距地高度 (km)")
    axes[0].set_title("48层高度位置")
    axes[1].plot(bias_t, heights / 1000.0, color=COLORS["corrected"], linewidth=2.1)
    axes[1].axvline(0, color="#5f6368", linestyle="--", linewidth=1)
    axes[1].set_xlabel("T订正量 (K)")
    axes[1].set_title("探空 - ERA5温度偏差")
    axes[2].plot(bias_rh, heights / 1000.0, color=COLORS["raw"], linewidth=2.1)
    axes[2].axvline(0, color="#5f6368", linestyle="--", linewidth=1)
    axes[2].set_xlabel("RH订正量 (%)")
    axes[2].set_title("探空 - ERA5湿度偏差")
    fig.suptitle("成都新高度网格与训练期探空偏差订正", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(args.output_dir / "01_height_grid_bias_profiles.png", facecolor="white")
    plt.close(fig)

    eval_stats = dataset_stats["sounding_evaluation"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4))
    labels = ["训练", "验证", "测试"]
    x = np.arange(3)
    width = 0.34
    for ax, variable, unit in [(axes[0], "T", "K"), (axes[1], "RH", "%")]:
        raw_values = [eval_stats[name]["raw"][variable]["rmse"] for name in ("train", "val", "test")]
        corrected_values = [
            eval_stats[name]["corrected"][variable]["rmse"] for name in ("train", "val", "test")
        ]
        bars1 = ax.bar(x - width / 2, raw_values, width, color=COLORS["raw"], label="原始ERA5")
        bars2 = ax.bar(x + width / 2, corrected_values, width, color=COLORS["corrected"], label="订正ERA5")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel(f"RMSE ({unit})")
        ax.set_title("温度" if variable == "T" else "相对湿度")
        ax.grid(axis="x", visible=False)
        for bars in (bars1, bars2):
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{bar.get_height():.2f}", ha="center", va="bottom")
    axes[0].legend(loc="upper right")
    fig.suptitle("ERA5偏差订正在独立探空上的效果", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(args.output_dir / "02_era5_bias_correction_validation.png", facecolor="white")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.0))
    model_labels = ["原93层模型\n插值到48层", "原始ERA5\n训练48层", "偏差订正\n训练48层"]
    colors = [COLORS["old"], COLORS["raw"], COLORS["corrected"]]
    matched_era5 = [
        era5_metrics["old93_interpolated_vs_raw48"],
        era5_metrics["raw48_vs_raw48"],
        era5_metrics["corrected48_vs_corrected48"],
    ]
    for ax, variable, unit in [(axes[0, 0], "T", "K"), (axes[0, 1], "RH", "%")]:
        bar_values(
            ax,
            model_labels,
            [item[variable]["rmse"] for item in matched_era5],
            colors,
            f"RMSE ({unit})",
            f"同源ERA5测试：{'温度' if variable == 'T' else '相对湿度'}",
        )
    for ax, variable, unit in [(axes[1, 0], "T", "K"), (axes[1, 1], "RH", "%")]:
        bar_values(
            ax,
            model_labels,
            [sounding_metrics[name][variable]["rmse"] for name in ("old93", "raw48", "corrected48")],
            colors,
            f"RMSE ({unit})",
            f"独立探空测试：{'温度' if variable == 'T' else '相对湿度'}",
        )
    fig.suptitle("新分层与偏差订正模型对比", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(args.output_dir / "03_model_comparison.png", facecolor="white")
    plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(15.5, 5.6), sharey=True)
    comparisons = [
        ("T", "RMSE", "K"),
        ("T", "Bias", "K"),
        ("RH", "RMSE", "%"),
        ("RH", "Bias", "%"),
    ]
    for ax, (variable, kind, unit) in zip(axes, comparisons):
        for name, label, color, target in [
            ("old93", "原93层插值", COLORS["old"], raw_truth),
            ("raw48", "原始48层", COLORS["raw"], raw_truth),
            ("corrected48", "订正48层", COLORS["corrected"], corrected_truth),
        ]:
            error = predictions[name][variable] - target[variable]
            value = np.sqrt(np.mean(error**2, axis=0)) if kind == "RMSE" else np.mean(error, axis=0)
            ax.plot(value, heights / 1000.0, color=color, linewidth=1.9, label=label)
        if kind == "Bias":
            ax.axvline(0, color="#5f6368", linestyle="--", linewidth=1)
        ax.set_xlabel(f"{kind} ({unit})")
        ax.set_title(f"{'温度' if variable == 'T' else '相对湿度'} {kind}")
    axes[0].set_ylabel("距地高度 (km)")
    axes[0].legend(loc="best")
    fig.suptitle("测试集分高度误差对比", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(args.output_dir / "04_height_resolved_comparison.png", facecolor="white")
    plt.close(fig)

    sounding_truth = {
        variable: np.stack([item[variable] for item in pairs]) for variable in ("T", "RH")
    }
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.0), sharey=True)
    for row, (variable, unit) in enumerate([("T", "K"), ("RH", "%")]):
        axes[row, 0].plot(
            sounding_truth[variable].mean(axis=0), heights / 1000.0, color=COLORS["truth"], linewidth=2.2, label="温江探空"
        )
        for name, label, color in [
            ("old93", "原93层插值", COLORS["old"]),
            ("raw48", "原始48层", COLORS["raw"]),
            ("corrected48", "订正48层", COLORS["corrected"]),
        ]:
            selected = np.stack(
                [predictions[name][variable][item["observation_index"]] for item in pairs]
            )
            axes[row, 0].plot(selected.mean(axis=0), heights / 1000.0, color=color, linewidth=1.8, label=label)
            rmse_height = np.sqrt(np.mean((selected - sounding_truth[variable]) ** 2, axis=0))
            axes[row, 1].plot(rmse_height, heights / 1000.0, color=color, linewidth=1.8, label=label)
        axes[row, 0].set_xlabel(f"{'温度' if variable == 'T' else '相对湿度'} ({unit})")
        axes[row, 1].set_xlabel(f"RMSE ({unit})")
        axes[row, 0].set_ylabel("距地高度 (km)")
    axes[0, 0].set_title("独立探空平均廓线")
    axes[0, 1].set_title("独立探空分高度RMSE")
    axes[0, 0].legend(loc="best")
    fig.suptitle(f"独立温江探空验证（{len(pairs)}个时次）", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(args.output_dir / "05_independent_sounding_profiles.png", facecolor="white")
    plt.close(fig)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
