"""Evaluation and visualization for atmospheric profile retrieval results.

Implements the validation metrics and plots from Section 3.5:
- RMSE and bias profiles by height (Fig 3.14, 3.16, 3.17)
- Scatter density plots (Fig 3.15)
- Comparison with LV2 products and radiosonde retrieval
"""

import os
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config


def compute_rmse(y_pred, y_true, axis=0):
    """Compute root mean square error."""
    return np.sqrt(np.mean((y_pred - y_true) ** 2, axis=axis))


def compute_bias(y_pred, y_true, axis=0):
    """Compute mean bias (prediction - truth)."""
    return np.mean(y_pred - y_true, axis=axis)


def compute_std(y_pred, y_true, axis=0):
    """Compute standard deviation of error."""
    return np.std(y_pred - y_true, axis=axis)


def compute_iwv(rh_profile, t_profile, p_profile, height_profile):
    """Compute integrated water vapor [kg/m^2]."""
    es = 6.1078 * np.exp(17.2693882 * (t_profile - 273.16) / (t_profile - 35.86))
    e = rh_profile / 100.0 * es
    rho_v = e * 216.7 / t_profile

    dz = np.diff(height_profile, prepend=0)
    iwv = np.sum(rho_v * dz) / 1000.0

    return iwv


def plot_error_profiles(rmse1, bias1, rmse2, bias2, rmse3, bias3,
                        height_grid, labels, title, output_path):
    """Plot RMSE and bias profiles for up to 3 retrieval methods.

    Similar to Fig 3.16 in the paper.

    Args:
        rmse1, rmse2, rmse3: RMSE arrays for methods 1-3 (n_layers,)
        bias1, bias2, bias3: Bias arrays
        height_grid: height in meters
        labels: [label1, label2, label3]
        title: plot title
        output_path: path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8), sharey=True)

    height_km = np.array(height_grid) / 1000.0

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    linestyles = ["-", "--", "-."]

    for i, (rmse, bias, label, color, ls) in enumerate(
        zip([rmse1, rmse2, rmse3], [bias1, bias2, bias3], labels, colors, linestyles)
    ):
        if rmse is not None:
            ax1.plot(bias, height_km, color=color, linestyle=ls, linewidth=2, label=label)
            ax2.plot(rmse, height_km, color=color, linestyle=ls, linewidth=2, label=label)

    ax1.set_xlabel("Mean Bias [K] or [%]", fontsize=12)
    ax1.set_ylabel("Height [km]", fontsize=12)
    ax1.axvline(x=0, color="gray", linestyle=":", alpha=0.5)
    ax1.legend(loc="upper right", fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f"{title} - Bias", fontsize=13)

    ax2.set_xlabel("RMSE [K] or [%]", fontsize=12)
    ax2.legend(loc="upper right", fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_title(f"{title} - RMSE", fontsize=13)

    ax1.set_ylim(0, 10)
    ax2.set_ylim(0, 10)
    ax1.invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved error profile plot to {output_path}")


def plot_scatter_density(y_pred, y_true, xlabel, ylabel, title, output_path):
    """Plot scatter density of retrieval vs truth.

    Similar to Fig 3.15 in the paper.

    Args:
        y_pred: predicted values, flattened
        y_true: true values, flattened
        xlabel, ylabel: axis labels
        title: plot title
        output_path: path to save figure
    """
    fig, ax = plt.subplots(figsize=(8, 7))

    # Remove NaN
    valid = np.isfinite(y_pred) & np.isfinite(y_true)
    y_pred = y_pred[valid]
    y_true = y_true[valid]

    # Compute 2D histogram density
    xy = np.vstack([y_pred, y_true])
    z = stats.gaussian_kde(xy)(xy)

    # Sort by density
    idx = z.argsort()
    y_pred, y_true, z = y_pred[idx], y_true[idx], z[idx]

    sc = ax.scatter(y_pred, y_true, c=z, s=1, alpha=0.6, cmap="RdYlBu_r")

    # 1:1 line
    min_val = min(y_pred.min(), y_true.min())
    max_val = max(y_pred.max(), y_true.max())
    ax.plot([min_val, max_val], [min_val, max_val], "k--", linewidth=1.5,
            label="1:1 line")

    # Fitted line
    slope, intercept, r_value, p_value, std_err = stats.linregress(y_pred, y_true)
    x_fit = np.linspace(min_val, max_val, 100)
    ax.plot(x_fit, slope * x_fit + intercept, "r--", linewidth=1.5,
            label=f"y = {slope:.3f}x + {intercept:.1f}\nR = {r_value:.3f}")

    # RMSE annotation
    rmse = compute_rmse(y_pred, y_true)
    bias = compute_bias(y_pred, y_true)
    ax.text(0.05, 0.95,
            f"RMSE = {rmse:.2f}\nBias = {bias:.2f}\nN = {len(y_pred)}",
            transform=ax.transAxes, fontsize=11, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.colorbar(sc, ax=ax, label="Density")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved scatter plot to {output_path}")


def plot_bt_error_boxplot(tbs_sim, tbs_obs, channels, output_path,
                          title="Simulated vs Observed BT"):
    """Plot boxplot of BT difference before/after correction.

    Similar to Fig 3.8 in the paper.

    Args:
        tbs_sim: simulated BT, shape (n_samples, n_channels)
        tbs_obs: observed BT, shape (n_samples, n_channels)
        channels: list of channel frequencies
        output_path: output file path
        title: plot title
    """
    n_channels = tbs_sim.shape[1]

    fig, ax = plt.subplots(figsize=(14, 5))

    diff = tbs_obs - tbs_sim
    positions = np.arange(n_channels)
    bp = ax.boxplot([diff[:, i] for i in range(n_channels)],
                    positions=positions, widths=0.4,
                    patch_artist=True,
                    boxprops=dict(facecolor="lightblue", alpha=0.7),
                    medianprops=dict(color="red", linewidth=1.5))

    # Mean bias line
    mean_bias = np.mean(diff, axis=0)
    ax.plot(positions, mean_bias, "ks-", markersize=5, linewidth=1.5,
            label="Mean Bias")

    ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Channel Frequency [GHz]", fontsize=12)
    ax.set_ylabel("TBO - TBS [K]", fontsize=12)
    ax.set_title(title, fontsize=13)

    # Label x-axis with frequencies
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{f:.1f}" for f in channels], rotation=45, ha="right",
                       fontsize=9)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved BT error boxplot to {output_path}")


def evaluate_full_profile(T_retrieved, RH_retrieved, T_truth, RH_truth,
                          height_grid, output_dir, method_name="ERA5"):
    """Full evaluation of retrieved temperature and RH profiles.

    Args:
        T_retrieved: shape (n_samples, n_layers)
        RH_retrieved: shape (n_samples, n_layers)
        T_truth: shape (n_samples, n_layers)
        RH_truth: shape (n_samples, n_layers)
        height_grid: list of heights [m]
        output_dir: output directory for plots
        method_name: label for the method
    """
    os.makedirs(output_dir, exist_ok=True)
    height_km = np.array(height_grid) / 1000.0

    # Compute per-layer statistics
    T_rmse = compute_rmse(T_retrieved, T_truth)
    T_bias = compute_bias(T_retrieved, T_truth)
    T_std = compute_std(T_retrieved, T_truth)

    RH_rmse = compute_rmse(RH_retrieved, RH_truth)
    RH_bias = compute_bias(RH_retrieved, RH_truth)
    RH_std = compute_std(RH_retrieved, RH_truth)

    # Whole-atmosphere averages
    print(f"\n{'='*60}")
    print(f"Evaluation Results - {method_name}")
    print(f"{'='*60}")
    print(f"Temperature:")
    print(f"  Whole-layer RMSE: {np.mean(T_rmse):.2f} K")
    print(f"  Whole-layer Bias: {np.mean(T_bias):.2f} K")
    print(f"  Whole-layer Std:  {np.mean(T_std):.2f} K")
    print(f"  Surface (0-0.5km) RMSE: {np.mean(T_rmse[:config.IDX_2KM]):.2f} K")
    print(f"Relative Humidity:")
    print(f"  Whole-layer RMSE: {np.mean(RH_rmse):.2f} %")
    print(f"  Whole-layer Bias: {np.mean(RH_bias):.2f} %")
    print(f"  Whole-layer Std:  {np.mean(RH_std):.2f} %")

    # Per-layer summary
    print(f"\n  Height [km]  |  T RMSE  |  T Bias  |  RH RMSE  |  RH Bias")
    print(f"  " + "-" * 56)
    for idx in [config.IDX_0KM, config.IDX_2KM, config.IDX_8KM, config.IDX_10KM - 1]:
        h = height_grid[idx] / 1000.0
        print(f"  {h:6.1f}        |  {T_rmse[idx]:6.2f}  |  {T_bias[idx]:6.2f}  |  "
              f"{RH_rmse[idx]:7.2f}  |  {RH_bias[idx]:7.2f}")

    # Plot: Temperature and RH error profiles
    plot_error_profiles(
        T_rmse, T_bias, None, None, None, None,
        height_grid,
        [method_name, None, None],
        "Temperature Retrieval",
        os.path.join(output_dir, "T_error_profile.png")
    )

    plot_error_profiles(
        RH_rmse, RH_bias, None, None, None, None,
        height_grid,
        [method_name, None, None],
        "Relative Humidity Retrieval",
        os.path.join(output_dir, "RH_error_profile.png")
    )

    # Scatter density plots (all layers combined)
    plot_scatter_density(
        T_retrieved.flatten(), T_truth.flatten(),
        f"{method_name} Retrieval [K]", "Truth (Sounding) [K]",
        f"Temperature: {method_name} vs Truth",
        os.path.join(output_dir, "T_scatter.png")
    )

    plot_scatter_density(
        RH_retrieved.flatten(), RH_truth.flatten(),
        f"{method_name} Retrieval [%]", "Truth (Sounding) [%]",
        f"Relative Humidity: {method_name} vs Truth",
        os.path.join(output_dir, "RH_scatter.png")
    )

    # Save statistics to CSV
    csv_path = os.path.join(output_dir, "error_stats.csv")
    with open(csv_path, "w") as f:
        f.write("height_km,T_rmse,T_bias,T_std,RH_rmse,RH_bias,RH_std\n")
        for i, h in enumerate(height_km):
            f.write(f"{h:.3f},{T_rmse[i]:.4f},{T_bias[i]:.4f},{T_std[i]:.4f},"
                    f"{RH_rmse[i]:.4f},{RH_bias[i]:.4f},{RH_std[i]:.4f}\n")
    print(f"Saved error statistics to {csv_path}")

    return {
        "T_rmse": T_rmse,
        "T_bias": T_bias,
        "T_std": T_std,
        "RH_rmse": RH_rmse,
        "RH_bias": RH_bias,
        "RH_std": RH_std,
    }


def compare_methods(results_list, height_grid, output_dir):
    """Compare multiple retrieval methods (Fig 3.16).

    Args:
        results_list: list of dicts, each with 'T_rmse', 'T_bias', 'RH_rmse',
                      'RH_bias', 'label'
        height_grid: list of heights
        output_dir: output directory
    """
    os.makedirs(output_dir, exist_ok=True)

    n = len(results_list)
    # Pad to 3 methods
    while len(results_list) < 3:
        results_list.append({"T_rmse": None, "T_bias": None,
                             "RH_rmse": None, "RH_bias": None,
                             "label": None})

    r = results_list

    plot_error_profiles(
        r[0]["T_rmse"], r[0]["T_bias"],
        r[1]["T_rmse"] if n > 1 else None, r[1]["T_bias"] if n > 1 else None,
        r[2]["T_rmse"] if n > 2 else None, r[2]["T_bias"] if n > 2 else None,
        height_grid,
        [r[i]["label"] for i in range(3)],
        "Temperature Profile Comparison",
        os.path.join(output_dir, "T_comparison.png")
    )

    plot_error_profiles(
        r[0]["RH_rmse"], r[0]["RH_bias"],
        r[1]["RH_rmse"] if n > 1 else None, r[1]["RH_bias"] if n > 1 else None,
        r[2]["RH_rmse"] if n > 2 else None, r[2]["RH_bias"] if n > 2 else None,
        height_grid,
        [r[i]["label"] for i in range(3)],
        "Relative Humidity Profile Comparison",
        os.path.join(output_dir, "RH_comparison.png")
    )


if __name__ == "__main__":
    # Example: generate synthetic results for testing
    n_test = 100
    heights = np.array(config.HEIGHT_GRID)
    T_true = np.random.randn(n_test, config.N_LAYERS) * 10 + 270.0
    RH_true = np.random.rand(n_test, config.N_LAYERS) * 50 + 30.0

    T_pred = T_true + np.random.randn(n_test, config.N_LAYERS) * 1.5
    RH_pred = RH_true + np.random.randn(n_test, config.N_LAYERS) * 10.0

    os.makedirs(config.RESULT_DIR, exist_ok=True)
    evaluate_full_profile(T_pred, RH_pred, T_true, RH_true,
                          heights, config.RESULT_DIR, method_name="ERA5-BRNN")
