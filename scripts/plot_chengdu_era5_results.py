#!/usr/bin/env python3
"""Create presentation-ready figures for the Chengdu ERA5 retrieval experiment."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mwr-retrieval-matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


DEFAULT_RESULTS = PROJECT_ROOT / "results"
DEFAULT_OUTPUT = DEFAULT_RESULTS / "chengdu_era5_figures"

COLORS = {
    "truth": "#202124",
    "hybrid": "#1976d2",
    "brnn": "#ef6c00",
    "ridge": "#2e7d32",
    "baseline": "#7b1fa2",
    "train": "#1976d2",
    "val": "#d84315",
}


def configure_plotting() -> None:
    font_candidates = [
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
    ]
    for path in font_candidates:
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
            "grid.linewidth": 0.7,
            "legend.frameon": False,
        }
    )


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def rolling_mean(values: np.ndarray, width: int = 9) -> np.ndarray:
    if len(values) < width:
        return values
    kernel = np.ones(width, dtype=float) / width
    padded = np.pad(values, (width // 2, width - width // 2 - 1), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def metrics(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    error = pred - truth
    residual = np.sum(error**2)
    total = np.sum((truth - truth.mean()) ** 2)
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
        "r2": float(1.0 - residual / total),
    }


def save_figure(fig: plt.Figure, output_dir: Path, name: str) -> None:
    fig.savefig(output_dir / name, facecolor="white")
    plt.close(fig)


def plot_training_curves(results_dir: Path, output_dir: Path) -> None:
    history = load_json(results_dir / "chengdu_era5_all21" / "training_history.json")
    bands = ["0-2km", "2-8km", "8-10km"]
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.2), sharex=False)
    for row, variable in enumerate(["T", "RH"]):
        for col, band in enumerate(bands):
            ax = axes[row, col]
            item = history[f"brnn_{variable}_{band}"]
            records = item["history"]
            epochs = np.asarray([r["epoch"] for r in records])
            train = np.asarray([r["train_loss"] for r in records])
            val = np.asarray([r["val_loss"] for r in records])
            ax.plot(epochs, train, color=COLORS["train"], alpha=0.25, linewidth=0.8)
            ax.plot(epochs, rolling_mean(train), color=COLORS["train"], linewidth=1.8, label="训练损失")
            ax.plot(epochs, val, color=COLORS["val"], alpha=0.20, linewidth=0.8)
            ax.plot(epochs, rolling_mean(val), color=COLORS["val"], linewidth=1.8, label="验证损失")
            ax.axvline(item["best_epoch"], color="#5f6368", linestyle="--", linewidth=1.0)
            ax.scatter(
                [item["best_epoch"]],
                [item["best_val"]],
                color=COLORS["val"],
                marker="o",
                s=28,
                zorder=5,
            )
            ax.annotate(
                f"best={item['best_epoch']}",
                (item["best_epoch"], item["best_val"]),
                xytext=(5, 8),
                textcoords="offset points",
                fontsize=8,
            )
            ax.set_title(f"{'温度' if variable == 'T' else '相对湿度'}：{band}")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("标准化 MSE")
            ax.set_yscale("log")
            if row == 0 and col == 0:
                ax.legend(loc="upper right")
    fig.suptitle("成都21通道 BRNN 训练与验证损失", fontsize=16, fontweight="normal")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(fig, output_dir, "01_brnn_training_curves.png")


def plot_height_errors(results_dir: Path, output_dir: Path) -> None:
    hybrid = np.load(results_dir / "chengdu_era5_hybrid" / "chengdu_era5_hybrid_predictions.npz")
    brnn = np.load(results_dir / "chengdu_era5_all21" / "chengdu_era5_brnn_predictions.npz")
    height = hybrid["heights"] / 1000.0
    fig, axes = plt.subplots(1, 4, figsize=(15.5, 5.8), sharey=True)
    specs = [
        ("T", "RMSE", "K"),
        ("T", "Bias", "K"),
        ("RH", "RMSE", "%"),
        ("RH", "Bias", "%"),
    ]
    for ax, (variable, kind, unit) in zip(axes, specs):
        truth = hybrid[f"{variable}_true"]
        for label, source, color in [
            ("混合模型", hybrid, COLORS["hybrid"]),
            ("单次 BRNN", brnn, COLORS["brnn"]),
        ]:
            error = source[f"{variable}_pred"] - truth
            value = np.sqrt(np.mean(error**2, axis=0)) if kind == "RMSE" else np.mean(error, axis=0)
            ax.plot(value, height, color=color, linewidth=2.0, label=label)
        if kind == "Bias":
            ax.axvline(0, color="#5f6368", linewidth=1.0, linestyle="--")
        ax.set_xlabel(f"{kind} ({unit})")
        ax.set_title(f"{'温度' if variable == 'T' else '相对湿度'} {kind}")
        ax.set_ylim(0, 10)
    axes[0].set_ylabel("距地高度 (km)")
    axes[0].legend(loc="best")
    fig.suptitle("测试集分高度误差廓线", fontsize=16, fontweight="normal")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(fig, output_dir, "02_height_resolved_errors.png")


def plot_scatter(results_dir: Path, output_dir: Path) -> None:
    data = np.load(results_dir / "chengdu_era5_hybrid" / "chengdu_era5_hybrid_predictions.npz")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6))
    for ax, variable, label, unit in [
        (axes[0], "T", "温度", "K"),
        (axes[1], "RH", "相对湿度", "%"),
    ]:
        truth = data[f"{variable}_true"].ravel()
        pred = data[f"{variable}_pred"].ravel()
        score = metrics(pred, truth)
        lo = float(min(truth.min(), pred.min()))
        hi = float(max(truth.max(), pred.max()))
        ax.hexbin(truth, pred, gridsize=45, mincnt=1, cmap="viridis", linewidths=0)
        ax.plot([lo, hi], [lo, hi], color="#d32f2f", linestyle="--", linewidth=1.4, label="1:1线")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(f"ERA5 {label} ({unit})")
        ax.set_ylabel(f"反演 {label} ({unit})")
        ax.set_title(label)
        ax.text(
            0.04,
            0.96,
            f"RMSE = {score['rmse']:.3f} {unit}\nMAE = {score['mae']:.3f} {unit}\nBias = {score['bias']:+.3f} {unit}\n$R^2$ = {score['r2']:.3f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#dadce0", "alpha": 0.92},
        )
        ax.legend(loc="lower right")
    fig.suptitle("混合模型反演值与 ERA5 标签对比", fontsize=16, fontweight="normal")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(fig, output_dir, "03_prediction_scatter.png")


def plot_example_profiles(results_dir: Path, output_dir: Path) -> None:
    data = np.load(results_dir / "chengdu_era5_hybrid" / "chengdu_era5_hybrid_predictions.npz")
    height = data["heights"] / 1000.0
    t_rmse = np.sqrt(np.mean((data["T_pred"] - data["T_true"]) ** 2, axis=1))
    rh_rmse = np.sqrt(np.mean((data["RH_pred"] - data["RH_true"]) ** 2, axis=1))
    score = t_rmse / np.median(t_rmse) + rh_rmse / np.median(rh_rmse)
    order = np.argsort(score)
    indices = [int(order[len(order) // 8]), int(order[len(order) // 2]), int(order[-1])]
    case_labels = ["较好样本", "中位样本", "较难样本"]

    fig, axes = plt.subplots(2, 3, figsize=(13.8, 9.0), sharey=True)
    for col, (index, case_label) in enumerate(zip(indices, case_labels)):
        timestamp = str(data["timestamps"][index]).replace("_", " ")
        axes[0, col].plot(data["T_true"][index], height, color=COLORS["truth"], linewidth=2.1, label="ERA5")
        axes[0, col].plot(data["T_pred"][index], height, color=COLORS["hybrid"], linewidth=2.1, linestyle="--", label="反演")
        axes[0, col].set_xlabel("温度 (K)")
        axes[0, col].set_title(f"{case_label}\n{timestamp[:16]}")
        axes[0, col].text(0.04, 0.05, f"RMSE={t_rmse[index]:.2f} K", transform=axes[0, col].transAxes)

        axes[1, col].plot(data["RH_true"][index], height, color=COLORS["truth"], linewidth=2.1, label="ERA5")
        axes[1, col].plot(data["RH_pred"][index], height, color=COLORS["hybrid"], linewidth=2.1, linestyle="--", label="反演")
        axes[1, col].set_xlabel("相对湿度 (%)")
        axes[1, col].set_xlim(0, 100)
        axes[1, col].text(0.04, 0.05, f"RMSE={rh_rmse[index]:.2f}%", transform=axes[1, col].transAxes)
    axes[0, 0].set_ylabel("距地高度 (km)")
    axes[1, 0].set_ylabel("距地高度 (km)")
    axes[0, 0].legend(loc="best")
    axes[1, 0].legend(loc="best")
    fig.suptitle("测试集代表性反演廓线", fontsize=16, fontweight="normal")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(fig, output_dir, "04_example_profiles.png")


def plot_error_heatmaps(results_dir: Path, output_dir: Path) -> None:
    data = np.load(results_dir / "chengdu_era5_hybrid" / "chengdu_era5_hybrid_predictions.npz")
    height = data["heights"] / 1000.0
    timestamps = [str(value).replace("_", " ") for value in data["timestamps"]]
    fig, axes = plt.subplots(2, 1, figsize=(14.5, 8.2), sharex=True)
    for ax, variable, label, unit in [
        (axes[0], "T", "温度误差", "K"),
        (axes[1], "RH", "相对湿度误差", "%"),
    ]:
        error = (data[f"{variable}_pred"] - data[f"{variable}_true"]).T
        limit = float(np.percentile(np.abs(error), 98))
        image = ax.imshow(
            error,
            origin="lower",
            aspect="auto",
            extent=(-0.5, error.shape[1] - 0.5, height[0], height[-1]),
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            interpolation="nearest",
        )
        ax.set_ylabel("距地高度 (km)")
        ax.set_title(f"{label}（反演 - ERA5）")
        cbar = fig.colorbar(image, ax=ax, pad=0.012)
        cbar.set_label(unit)
    tick_idx = np.unique(np.linspace(0, len(timestamps) - 1, 8).astype(int))
    axes[1].set_xticks(tick_idx)
    axes[1].set_xticklabels([timestamps[i][5:16] for i in tick_idx], rotation=25, ha="right")
    axes[1].set_xlabel("测试样本时间 (UTC)")
    fig.suptitle("混合模型测试集时高误差演变", fontsize=16, fontweight="normal")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(fig, output_dir, "05_time_height_error_heatmap.png")


def plot_model_and_channel_comparison(results_dir: Path, output_dir: Path) -> None:
    hybrid_stats = load_json(results_dir / "chengdu_era5_hybrid" / "chengdu_era5_hybrid_stats.json")
    all21_stats = load_json(results_dir / "chengdu_era5_all21" / "chengdu_era5_brnn_stats.json")
    ridge_stats = load_json(results_dir / "chengdu_era5_ridge_all21" / "chengdu_era5_ridge_stats.json")
    channel_dirs = [
        ("全21通道", "chengdu_era5_all21"),
        ("前14通道", "chengdu_era5_hatpro14"),
        ("高频7通道", "chengdu_era5_high7"),
        ("183 GHz 5通道", "chengdu_era5_vapor183"),
    ]
    channel_t = []
    channel_rh = []
    for _, directory in channel_dirs:
        stats = load_json(results_dir / directory / "chengdu_era5_brnn_stats.json")
        channel_t.append(stats["metrics"]["model"]["T"]["rmse"])
        channel_rh.append(stats["metrics"]["model"]["RH"]["rmse"])

    method_names = ["气候态", "BRNN", "Ridge/EOF", "混合模型"]
    method_t = [
        all21_stats["metrics"]["climatology"]["T"]["rmse"],
        all21_stats["metrics"]["model"]["T"]["rmse"],
        ridge_stats["metrics"]["ridge"]["T"]["rmse"],
        hybrid_stats["metrics"]["T"]["rmse"],
    ]
    method_rh = [
        all21_stats["metrics"]["climatology"]["RH"]["rmse"],
        all21_stats["metrics"]["model"]["RH"]["rmse"],
        ridge_stats["metrics"]["ridge"]["RH"]["rmse"],
        hybrid_stats["metrics"]["RH"]["rmse"],
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13.8, 9.0))
    bar_colors = [COLORS["baseline"], COLORS["brnn"], COLORS["ridge"], COLORS["hybrid"]]
    panels = [
        (axes[0, 0], method_names, method_t, "方法对比：温度", "RMSE (K)", bar_colors),
        (axes[0, 1], method_names, method_rh, "方法对比：相对湿度", "RMSE (%)", bar_colors),
        (axes[1, 0], [x[0] for x in channel_dirs], channel_t, "BRNN通道消融：温度", "RMSE (K)", [COLORS["hybrid"]] * 4),
        (axes[1, 1], [x[0] for x in channel_dirs], channel_rh, "BRNN通道消融：相对湿度", "RMSE (%)", [COLORS["brnn"]] * 4),
    ]
    for ax, labels, values, title, ylabel, colors in panels:
        x = np.arange(len(labels))
        bars = ax.bar(x, values, color=colors, width=0.68)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=10)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_ylim(0, max(values) * 1.22)
        ax.grid(axis="x", visible=False)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2f}", ha="center", va="bottom")
    fig.suptitle("模型选择与通道消融结果", fontsize=16, fontweight="normal")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(fig, output_dir, "06_model_channel_comparison.png")


def write_readme(output_dir: Path) -> None:
    content = """# 成都 ERA5 反演训练效果图

这些图由 `scripts/plot_chengdu_era5_results.py` 从已保存的训练日志和测试预测中生成。

1. `01_brnn_training_curves.png`：21通道BRNN六个分层子模型的训练/验证损失。
2. `02_height_resolved_errors.png`：混合模型与单次BRNN的分高度RMSE和Bias。
3. `03_prediction_scatter.png`：全部测试样本、全部高度层的反演值与ERA5标签对比。
4. `04_example_profiles.png`：较好、中位和较难测试样本的温湿廓线。
5. `05_time_height_error_heatmap.png`：测试时段内误差随时间和高度的演变。
6. `06_model_channel_comparison.png`：方法选择和BRNN通道消融对比。

当前最佳方案为温度 `21ch Ridge + 5 EOF`、湿度 `21ch BRNN四种子集成`。测试集只有48个样本、3天，图片用于展示当前基线，不代表正式业务精度。

重新生成：

```powershell
$env:PYTHONPATH='D:\\project-504\\pydeps'
python scripts/plot_chengdu_era5_results.py
```
"""
    (output_dir / "README.md").write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Chengdu ERA5 retrieval results")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_plotting()
    plot_training_curves(args.results_dir, args.output_dir)
    plot_height_errors(args.results_dir, args.output_dir)
    plot_scatter(args.results_dir, args.output_dir)
    plot_example_profiles(args.results_dir, args.output_dir)
    plot_error_heatmaps(args.results_dir, args.output_dir)
    plot_model_and_channel_comparison(args.results_dir, args.output_dir)
    write_readme(args.output_dir)
    print(f"Wrote figures to {args.output_dir}")


if __name__ == "__main__":
    main()
