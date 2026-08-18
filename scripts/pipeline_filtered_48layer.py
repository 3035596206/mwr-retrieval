#!/usr/bin/env python3
"""Complete pipeline: filtered BT → 48-layer bias-corrected retrieval.

Steps:
  1. Build physical 48-layer dataset from filtered obs BT + ERA5 GRIB
  2. Compute sounding-based bias correction (T + logq)
  3. Train Ridge/EOF for T → predict
  4. Train BRNN ensemble for RH → predict (multi-seed)
  5. Combine into hybrid product (Ridge T + BRNN top-2 ensemble RH)
  6. Evaluate on test split and independent soundings
  7. Generate diagnostic plots

All results saved under a single date-stamped output directory.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

# ---------- matplotlib setup (must be before pyplot import) ----------
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "mwr-retrieval-matplotlib"),
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import config
from train_chengdu_brnn import (
    MODEL_DEFS,
    load_observations,
    predict_full,
    rmse_bias,
    set_seed,
    train_one_model,
    sounding_time_from_name,
)
from train_chengdu_era5_brnn import chronological_date_split, date_mean_metrics
from train_chengdu_era5_ridge import predict_selected, select_model, standardize

# ecCodes for GRIB reading
from eccodes import (
    codes_get,
    codes_get_values,
    codes_grib_new_from_file,
    codes_release,
)

# ============================================================
# Physical constants
# ============================================================
G = 9.80665
EPSILON = 0.622
Q_FLOOR = 1e-8


# ============================================================
# 48-layer grid (matching build_chengdu_era5_layer48_dataset.py)
# ============================================================
def build_layer_grid() -> tuple[np.ndarray, np.ndarray]:
    edges = np.concatenate(
        [
            np.asarray([0.0, 500.0]),
            np.arange(600.0, 2000.0 + 1.0, 100.0),
            np.arange(2250.0, 10000.0 + 1.0, 250.0),
        ]
    ).astype(np.float32)
    centers = ((edges[:-1] + edges[1:]) / 2.0).astype(np.float32)
    assert len(edges) == 49 and len(centers) == 48, (
        f"Grid size mismatch: {len(edges)} edges, {len(centers)} layers"
    )
    return edges, centers


# ============================================================
# Thermodynamics
# ============================================================
def saturation_vapor_pressure_hpa(temperature_k: np.ndarray) -> np.ndarray:
    tc = np.asarray(temperature_k, dtype=np.float64) - 273.15
    water = 6.112 * np.exp(17.62 * tc / (243.12 + tc))
    ice = 6.112 * np.exp(22.46 * tc / (272.62 + tc))
    blend = np.clip((tc + 20.0) / 20.0, 0.0, 1.0)
    return ice * (1.0 - blend) + water * blend


def rh_to_specific_humidity(
    t_k: np.ndarray, rh: np.ndarray, p_hpa: np.ndarray
) -> np.ndarray:
    es = saturation_vapor_pressure_hpa(t_k)
    e = np.clip(rh / 100.0 * es, 0.0, p_hpa * 0.99)
    mr = EPSILON * e / np.maximum(p_hpa - e, 1e-6)
    return np.maximum(mr / (1.0 + mr), Q_FLOOR)


def specific_humidity_to_rh(
    t_k: np.ndarray, q: np.ndarray, p_hpa: np.ndarray
) -> np.ndarray:
    q = np.clip(q, Q_FLOOR, 0.2)
    mr = q / np.maximum(1.0 - q, 1e-8)
    e = mr * p_hpa / (EPSILON + mr)
    rh = 100.0 * e / np.maximum(saturation_vapor_pressure_hpa(t_k), 1e-8)
    return np.clip(rh, 0.0, 100.0)


def layer_average_from_interpolated(
    src_z: np.ndarray, src_val: np.ndarray, edges: np.ndarray
) -> np.ndarray:
    vals = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        interior = src_z[(src_z > lo) & (src_z < hi)]
        sample_z = np.concatenate(([lo], interior, [hi]))
        sample_v = np.interp(sample_z, src_z, src_val)
        vals.append(float(np.trapezoid(sample_v, sample_z) / (hi - lo)))
    return np.asarray(vals, dtype=np.float32)


def raw_profile_to_layers(
    height_agl: np.ndarray,
    t_k: np.ndarray,
    rh: np.ndarray,
    p_hpa: np.ndarray,
    edges: np.ndarray,
) -> dict[str, np.ndarray]:
    valid = (
        np.isfinite(height_agl)
        & np.isfinite(t_k)
        & np.isfinite(rh)
        & np.isfinite(p_hpa)
        & (height_agl >= 0.0)
        & (t_k >= 180.0)
        & (t_k <= 330.0)
        & (rh >= 0.0)
        & (rh <= 100.0)
        & (p_hpa >= 1.0)
        & (p_hpa <= 1100.0)
    )
    z = np.asarray(height_agl[valid], dtype=np.float64)
    t = np.asarray(t_k[valid], dtype=np.float64)
    r = np.asarray(rh[valid], dtype=np.float64)
    p = np.asarray(p_hpa[valid], dtype=np.float64)
    if len(z) < 15:
        raise ValueError("too few valid above-ground profile levels")
    order = np.argsort(z)
    z, t, r, p = z[order], t[order], r[order], p[order]
    keep = np.concatenate(([True], np.diff(z) > 0.5))
    z, t, r, p = z[keep], t[keep], r[keep], p[keep]
    if z[-1] < edges[-1]:
        raise ValueError("profile does not reach 10 km AGL")
    if z[0] > 0.0:
        z = np.concatenate(([0.0], z))
        t = np.concatenate(([t[0]], t))
        r = np.concatenate(([r[0]], r))
        p = np.concatenate(([p[0]], p))
    q = rh_to_specific_humidity(t, r, p)
    t_layer = layer_average_from_interpolated(z, t, edges)
    log_q = layer_average_from_interpolated(
        z, np.log(np.maximum(q, Q_FLOOR)), edges
    )
    log_p = layer_average_from_interpolated(z, np.log(p), edges)
    q_layer = np.exp(log_q).astype(np.float32)
    p_layer = np.exp(log_p).astype(np.float32)
    rh_layer = specific_humidity_to_rh(t_layer, q_layer, p_layer).astype(
        np.float32
    )
    return {
        "T": t_layer,
        "q": q_layer,
        "logq": log_q,
        "P": p_layer,
        "RH": rh_layer,
    }


# ============================================================
# GRIB helpers (ecCodes)
# ============================================================
def grib_datetime(gid) -> datetime:
    date = int(codes_get(gid, "dataDate"))
    time = int(codes_get(gid, "dataTime"))
    return datetime.strptime(f"{date:08d}{time:04d}", "%Y%m%d%H%M")


def load_era5_layer_profiles(
    grib_path: Path,
    requested_times: set[datetime],
    edges: np.ndarray,
    site_altitude_m: float,
) -> tuple[dict[datetime, dict], dict]:
    raw: dict[datetime, dict[str, dict[float, float]]] = {}
    msg_count = 0
    with grib_path.open("rb") as fh:
        while True:
            gid = codes_grib_new_from_file(fh)
            if gid is None:
                break
            msg_count += 1
            try:
                vt = grib_datetime(gid)
                if vt not in requested_times:
                    continue
                if str(codes_get(gid, "typeOfLevel")) != "isobaricInhPa":
                    continue
                var = str(codes_get(gid, "shortName"))
                if var not in {"z", "t", "r"}:
                    continue
                level = float(codes_get(gid, "level"))
                vals = np.asarray(codes_get_values(gid), dtype=np.float64)
                if vals.size != 1:
                    raise ValueError(
                        f"Expected one ERA5 grid value, got {vals.size}"
                    )
                raw.setdefault(vt, {}).setdefault(var, {})[level] = float(
                    vals[0]
                )
            finally:
                codes_release(gid)

    profiles = {}
    rejected = {}
    below = []
    for vt, variables in raw.items():
        if not {"z", "t", "r"}.issubset(variables):
            rejected[vt.isoformat()] = (
                f"missing vars: {sorted({'z','t','r'} - set(variables))}"
            )
            continue
        levels = sorted(
            set(variables["z"]) & set(variables["t"]) & set(variables["r"]),
            reverse=True,
        )
        z_agl = np.asarray(
            [variables["z"][lv] / G - site_altitude_m for lv in levels]
        )
        t_arr = np.asarray([variables["t"][lv] for lv in levels])
        rh_arr = np.asarray([variables["r"][lv] for lv in levels])
        p_arr = np.asarray(levels, dtype=np.float64)
        below.append(int(np.sum(z_agl < 0.0)))
        try:
            profiles[vt] = raw_profile_to_layers(
                z_agl, t_arr, rh_arr, p_arr, edges
            )
        except ValueError as exc:
            rejected[vt.isoformat()] = str(exc)
    audit = {
        "grib_messages_scanned": msg_count,
        "requested_times": len(requested_times),
        "raw_matching_times": len(raw),
        "valid_profiles": len(profiles),
        "rejected": rejected,
        "mean_below_ground_levels": float(np.mean(below)) if below else 0.0,
    }
    return profiles, audit


# ============================================================
# Sounding loading (for bias correction)
# ============================================================
def parse_sounding_layers(
    path: Path, edges: np.ndarray
) -> dict[str, np.ndarray]:
    heights, temps, rhs, press = [], [], [], []
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            fields = line.split()
            if len(fields) < 17:
                continue
            try:
                tc = float(fields[6])
                ph = float(fields[7])
                rh = float(fields[8])
                hm = float(fields[16])
            except ValueError:
                continue
            if (
                -100 <= tc <= 60
                and 1 <= ph <= 1100
                and 0 <= rh <= 100
                and -500 <= hm <= 60000
            ):
                heights.append(hm)
                temps.append(tc + 273.15)
                rhs.append(rh)
                press.append(ph)
    if len(heights) < 50:
        raise ValueError("too few valid sounding levels")
    h_arr = np.asarray(heights, dtype=np.float64)
    station_h = float(np.nanpercentile(h_arr, 0.5))
    return raw_profile_to_layers(
        h_arr - station_h,
        np.asarray(temps),
        np.asarray(rhs),
        np.asarray(press),
        edges,
    )


# ============================================================
# Dataset construction
# ============================================================
def build_observation_dataset(records, profiles):
    matched = [
        (r, profiles[r["datetime"]])
        for r in records
        if r["datetime"] in profiles
    ]
    return {
        "X": np.asarray([m[0]["channels"] for m in matched], dtype=np.float32),
        "T_raw": np.asarray([m[1]["T"] for m in matched], dtype=np.float32),
        "RH_raw": np.asarray([m[1]["RH"] for m in matched], dtype=np.float32),
        "q_raw": np.asarray([m[1]["q"] for m in matched], dtype=np.float32),
        "logq_raw": np.asarray(
            [m[1]["logq"] for m in matched], dtype=np.float32
        ),
        "P": np.asarray([m[1]["P"] for m in matched], dtype=np.float32),
        "dates": np.asarray(
            [m[0]["datetime"].strftime("%Y%m%d") for m in matched]
        ),
        "timestamps": np.asarray([m[0]["timestamp"] for m in matched]),
    }


def collect_sounding_pairs(
    sounding_dir, grib_path, edges, site_alt, t_min, t_max
):
    entries = []
    for path in sorted(Path(sounding_dir).rglob("*.txt")):
        lt = sounding_time_from_name(path)
        if t_min - timedelta(days=1) <= lt <= t_max + timedelta(days=1):
            entries.append((lt, path))
    requested = {
        c
        for lt, _ in entries
        for c in (
            lt - timedelta(hours=1),
            lt,
            lt + timedelta(hours=1),
        )
    }
    era5, audit = load_era5_layer_profiles(
        grib_path, requested, edges, site_alt
    )
    pairs = []
    skipped = {}
    for lt, path in entries:
        candidates = [
            c
            for c in (
                lt - timedelta(hours=1),
                lt,
                lt + timedelta(hours=1),
            )
            if c in era5
        ]
        if not candidates:
            skipped[path.name] = "no ERA5 layer profile within ±1h"
            continue
        try:
            snd = parse_sounding_layers(path, edges)
        except ValueError as exc:
            skipped[path.name] = str(exc)
            continue
        nearest = min(abs((c - lt).total_seconds()) for c in candidates)
        used = [
            c
            for c in candidates
            if abs((c - lt).total_seconds()) == nearest
        ]
        e5 = {
            k: np.mean([era5[c][k] for c in used], axis=0).astype(np.float32)
            for k in ("T", "RH", "q", "logq", "P")
        }
        pairs.append({"launch_time": lt, "sounding": snd, "era5": e5})
    return pairs, {
        "candidate_files": len(entries),
        "valid_pairs": len(pairs),
        "skipped": skipped,
        "era5": audit,
    }


def smooth_low_freedom_bias(mean_bias, centers):
    kernel = np.asarray([1.0, 2.0, 3.0, 2.0, 1.0]) / 9.0
    smoothed = np.convolve(
        np.pad(mean_bias, (2, 2), mode="edge"), kernel, mode="valid"
    )
    knots_h = np.asarray(
        [250.0, 1000.0, 2000.0, 4000.0, 6000.0, 8000.0, 9875.0]
    )
    knots_v = np.interp(knots_h, centers, smoothed)
    return np.interp(centers, knots_h, knots_v).astype(np.float32)


def profile_metrics(pred, truth):
    e = pred - truth
    return {
        "rmse": float(np.sqrt(np.mean(e**2))),
        "mae": float(np.mean(np.abs(e))),
        "bias": float(np.mean(e)),
    }


# ============================================================
# BRNN model defs for 48-layer grid
# ============================================================
MODEL_DEFS_LAYER48 = [
    ("brnn_T_0-2km", "T", (0.0, 2.0)),
    ("brnn_T_2-8km", "T", (2.0, 8.0)),
    ("brnn_T_8-10km", "T", (8.0, 10.0)),
    ("brnn_RH_0-2km", "RH", (0.0, 2.0)),
    ("brnn_RH_2-8km", "RH", (2.0, 8.0)),
    ("brnn_RH_8-10km", "RH", (8.0, 10.0)),
]


# ============================================================
# Plotting
# ============================================================
def configure_plotting():
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import font_manager

    for fp in [
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"),
    ]:
        if fp.exists():
            font_manager.fontManager.addfont(fp)
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "axes.unicode_minus": False,
            "figure.dpi": 130,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "legend.frameon": False,
        }
    )
    return plt


def make_diagnostic_plots(
    t_pred,
    rh_pred,
    t_true,
    rh_true,
    t_raw,
    rh_raw,
    centers,
    edges,
    thickness,
    bias_t,
    bias_logq,
    apply_t,
    apply_logq,
    out_dir,
):
    plt = configure_plotting()

    # ---- 1. Layer grid + bias correction ----
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.5), sharey=True)
    tc = ["#1976d2" if v <= 100 else "#ef6c00" for v in thickness]
    axes[0].barh(
        centers / 1000.0,
        thickness,
        height=np.minimum(thickness / 1000.0, 0.18),
        color=tc,
    )
    axes[0].set_xlabel("Layer thickness (m)")
    axes[0].set_ylabel("Layer center height (km)")
    axes[0].set_title("Physical 48-layer grid")
    axes[1].plot(bias_t, centers / 1000.0, color="#1976d2", linewidth=2)
    axes[1].axvline(0, color="#5f6368", linestyle="--", linewidth=1)
    axes[1].set_xlabel("T bias correction (K)")
    axes[1].set_title(f"T correction ({'applied' if apply_t else 'rejected'})")
    axes[2].plot(bias_logq, centers / 1000.0, color="#2e7d32", linewidth=2)
    axes[2].axvline(0, color="#5f6368", linestyle="--", linewidth=1)
    axes[2].set_xlabel("log(q) bias correction")
    axes[2].set_title(
        f"log(q) correction ({'applied' if apply_logq else 'rejected'})"
    )
    fig.suptitle("Physical layer averaging & selective bias correction", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(
        out_dir / "01_layer_grid_bias_correction.png", facecolor="white"
    )
    plt.close(fig)

    # ---- 2. Error profiles ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)
    t_rmse = np.sqrt(np.mean((t_pred - t_true) ** 2, axis=0))
    rh_rmse = np.sqrt(np.mean((rh_pred - rh_true) ** 2, axis=0))
    t_bias_arr = np.mean(t_pred - t_true, axis=0)
    rh_bias_arr = np.mean(rh_pred - rh_true, axis=0)
    for ax, rmse, bias, unit, var in [
        (axes[0], t_rmse, t_bias_arr, "K", "Temperature"),
        (axes[1], rh_rmse, rh_bias_arr, "%", "Relative Humidity"),
    ]:
        ax.plot(rmse, centers / 1000.0, color="#d32f2f", linewidth=2, label="RMSE")
        ax.plot(bias, centers / 1000.0, color="#1976d2", linewidth=2, label="Bias")
        ax.axvline(0, color="#5f6368", linestyle="--", linewidth=1)
        ax.set_xlabel(f"Error ({unit})")
        ax.set_title(f"{var}")
        ax.legend(loc="best")
    axes[0].set_ylabel("Layer center height (km)")
    fig.suptitle(
        f"Hybrid retrieval error profiles (n={len(t_true)} test samples)", fontsize=14
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_dir / "02_error_profiles.png", facecolor="white")
    plt.close(fig)

    # ---- 3. Sample profiles (first 4) ----
    n_show = min(4, len(t_true))
    fig, axes = plt.subplots(2, n_show, figsize=(4 * n_show, 8), sharey="row")
    for i in range(n_show):
        axes[0, i].plot(
            t_raw[i], centers / 1000.0, color="#9e9e9e", linewidth=1.5, label="ERA5 raw"
        )
        axes[0, i].plot(
            t_pred[i],
            centers / 1000.0,
            color="#d32f2f",
            linewidth=2,
            label="Retrieved",
        )
        axes[0, i].plot(
            t_true[i],
            centers / 1000.0,
            color="#202124",
            linewidth=2,
            label="Corrected ERA5",
        )
        axes[0, i].set_xlabel("T (K)")
        axes[0, i].set_title(f"Sample {i+1}")
        axes[1, i].plot(
            rh_raw[i],
            centers / 1000.0,
            color="#9e9e9e",
            linewidth=1.5,
            label="ERA5 raw",
        )
        axes[1, i].plot(
            rh_pred[i],
            centers / 1000.0,
            color="#1976d2",
            linewidth=2,
            label="Retrieved",
        )
        axes[1, i].plot(
            rh_true[i],
            centers / 1000.0,
            color="#202124",
            linewidth=2,
            label="Corrected ERA5",
        )
        axes[1, i].set_xlabel("RH (%)")
    axes[0, 0].legend(loc="best", fontsize=7)
    axes[0, 0].set_ylabel("Height (km)")
    axes[1, 0].set_ylabel("Height (km)")
    fig.suptitle("Sample retrieved profiles", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_dir / "03_sample_profiles.png", facecolor="white")
    plt.close(fig)

    # ---- 4. Scatter density at key heights ----
    key_h = [0.25, 1.0, 2.0, 5.0, 8.0]
    fig, axes = plt.subplots(2, len(key_h), figsize=(4 * len(key_h), 8))
    for j, h_km in enumerate(key_h):
        idx = int(np.argmin(np.abs(centers - h_km * 1000.0)))
        for row, (pred, truth, var, unit, col) in enumerate(
            [
                (t_pred[:, idx], t_true[:, idx], "T", "K", "#d32f2f"),
                (rh_pred[:, idx], rh_true[:, idx], "RH", "%", "#1976d2"),
            ]
        ):
            ax = axes[row, j]
            ax.scatter(truth, pred, s=8, alpha=0.5, color=col, edgecolors="none")
            lo = min(truth.min(), pred.min())
            hi = max(truth.max(), pred.max())
            rng = hi - lo
            lo -= rng * 0.05
            hi += rng * 0.05
            ax.plot([lo, hi], [lo, hi], color="#5f6368", linestyle="--", linewidth=1)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_xlabel(f"True {var} ({unit})", fontsize=8)
            ax.set_ylabel(f"Pred {var} ({unit})", fontsize=8)
            e = pred - truth
            ax.set_title(f"{var} @ {h_km}km", fontsize=9)
            ax.text(
                0.95,
                0.08,
                f"RMSE={np.sqrt(np.mean(e**2)):.2f}\nBias={np.mean(e):+.2f}",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=7,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7),
            )
    fig.suptitle("Scatter density at key heights", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_dir / "04_scatter_density.png", facecolor="white")
    plt.close(fig)


# ============================================================
# MAIN PIPELINE
# ============================================================
def main():
    today = datetime.now().strftime("%Y%m%d")
    parser = argparse.ArgumentParser(
        description="Filtered BT 48-layer bias-corrected retrieval pipeline"
    )
    parser.add_argument("--obs-json", type=Path, required=True)
    parser.add_argument("--grib", type=Path, required=True)
    parser.add_argument("--sounding-dir", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT
        / "results"
        / f"filtered_48layer_{today}",
    )
    parser.add_argument("--site-altitude-m", type=float, default=548.0)
    parser.add_argument(
        "--brnn-seeds",
        type=int,
        nargs="+",
        default=[42, 1, 7, 21, 84],
    )
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=35)
    args = parser.parse_args()

    out = args.output_root
    for sub in ["dataset", "models", "predictions", "figures"]:
        (out / sub).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Pipeline: filtered BT -> 48-layer bias-corrected retrieval")
    print(f"Output:   {out}")
    print(f"Obs JSON: {args.obs_json}")
    print(f"GRIB:     {args.grib}")
    print(f"Soundings:{args.sounding_dir}")
    print("=" * 60)

    # ============================================================
    # STEP 1: Build 48-layer physical dataset
    # ============================================================
    print("\n[Step 1/6] Building physical 48-layer dataset...")
    edges, centers = build_layer_grid()
    records = load_observations(args.obs_json)
    print(f"  Loaded {len(records)} observation records")

    era5_profiles, era5_audit = load_era5_layer_profiles(
        args.grib,
        {r["datetime"] for r in records},
        edges,
        args.site_altitude_m,
    )
    print(
        f"  ERA5 matches: {len(era5_profiles)} / {len(records)} requested times"
    )

    data = build_observation_dataset(records, era5_profiles)
    print(
        f"  Exact matches: {len(data['X'])} samples, "
        f"{len(np.unique(data['dates']))} dates"
    )

    if len(data["X"]) < 20 or len(np.unique(data["dates"])) < 5:
        raise RuntimeError(
            f"Insufficient data: {len(data['X'])} samples, "
            f"{len(np.unique(data['dates']))} dates"
        )

    train_mask, val_mask, test_mask, split = chronological_date_split(
        data["dates"]
    )
    print(
        f"  Split: train={train_mask.sum()}, val={val_mask.sum()}, "
        f"test={test_mask.sum()}"
    )
    print(
        f"  Dates: train={split['train_dates']}, "
        f"val={split['val_dates']}, test={split['test_dates']}"
    )

    # ============================================================
    # STEP 2: Bias correction from sounding pairs
    # ============================================================
    print("\n[Step 2/6] Computing sounding-based bias correction...")
    t_min = min(r["datetime"] for r in records)
    t_max = max(r["datetime"] for r in records)
    pairs, snd_audit = collect_sounding_pairs(
        args.sounding_dir,
        args.grib,
        edges,
        args.site_altitude_m,
        t_min,
        t_max,
    )
    print(
        f"  Sounding pairs: {snd_audit['valid_pairs']} / "
        f"{snd_audit['candidate_files']} files"
    )

    split_dates = {
        name: set(split[f"{name}_dates"])
        for name in ("train", "val", "test")
    }
    pair_splits = {
        name: [
            p
            for p in pairs
            if p["launch_time"].strftime("%Y%m%d") in dates
        ]
        for name, dates in split_dates.items()
    }
    for name in ("train", "val", "test"):
        print(f"  {name} sounding pairs: {len(pair_splits[name])}")

    # Compute bias
    if len(pair_splits["train"]) > 0:
        delta_t = np.stack(
            [
                p["sounding"]["T"] - p["era5"]["T"]
                for p in pair_splits["train"]
            ]
        )
        delta_logq = np.stack(
            [
                p["sounding"]["logq"] - p["era5"]["logq"]
                for p in pair_splits["train"]
            ]
        )
        bias_t = np.clip(
            smooth_low_freedom_bias(delta_t.mean(axis=0), centers), -5.0, 5.0
        )
        bias_logq = np.clip(
            smooth_low_freedom_bias(delta_logq.mean(axis=0), centers),
            -np.log(3.0),
            np.log(3.0),
        )
    else:
        bias_t = np.zeros(len(centers), dtype=np.float32)
        bias_logq = np.zeros(len(centers), dtype=np.float32)

    # Validate correction on val set
    apply_t = True
    apply_logq = True
    if len(pair_splits["val"]) > 0:
        snd_val_t = np.stack(
            [p["sounding"]["T"] for p in pair_splits["val"]]
        )
        snd_val_rh = np.stack(
            [p["sounding"]["RH"] for p in pair_splits["val"]]
        )
        era5_val_t = np.stack(
            [p["era5"]["T"] for p in pair_splits["val"]]
        )
        era5_val_rh = np.stack(
            [p["era5"]["RH"] for p in pair_splits["val"]]
        )
        era5_val_logq = np.stack(
            [p["era5"]["logq"] for p in pair_splits["val"]]
        )
        era5_val_p = np.stack(
            [p["era5"]["P"] for p in pair_splits["val"]]
        )

        raw_t_rmse = np.sqrt(np.mean((era5_val_t - snd_val_t) ** 2))
        corr_t = np.clip(era5_val_t + bias_t, 180.0, 330.0)
        corr_t_rmse = np.sqrt(np.mean((corr_t - snd_val_t) ** 2))
        apply_t = corr_t_rmse < raw_t_rmse
        print(
            f"  T correction: raw RMSE={raw_t_rmse:.3f}K -> "
            f"corrected={corr_t_rmse:.3f}K "
            f"({'APPLIED' if apply_t else 'REJECTED'})"
        )

        raw_rh_rmse = np.sqrt(np.mean((era5_val_rh - snd_val_rh) ** 2))
        corr_logq = era5_val_logq + bias_logq
        corr_q = np.exp(corr_logq)
        corr_rh = specific_humidity_to_rh(
            era5_val_t + (bias_t if apply_t else 0.0),
            corr_q,
            era5_val_p,
        )
        corr_rh_rmse = np.sqrt(np.mean((corr_rh - snd_val_rh) ** 2))
        apply_logq = corr_rh_rmse < raw_rh_rmse
        print(
            f"  RH correction: raw RMSE={raw_rh_rmse:.2f}% -> "
            f"corrected={corr_rh_rmse:.2f}% "
            f"({'APPLIED' if apply_logq else 'REJECTED'})"
        )

    # Apply correction
    t_corrected = np.clip(
        data["T_raw"] + (bias_t if apply_t else 0.0), 180.0, 330.0
    ).astype(np.float32)
    logq_corrected = data["logq_raw"] + (bias_logq if apply_logq else 0.0)
    q_corrected = np.exp(logq_corrected).astype(np.float32)
    rh_corrected = specific_humidity_to_rh(
        t_corrected.astype(np.float64),
        q_corrected.astype(np.float64),
        data["P"].astype(np.float64),
    ).astype(np.float32)
    rh_corrected = np.clip(rh_corrected, 0.0, 100.0)

    # Save dataset
    thickness = np.diff(edges).astype(np.float32)
    np.savez_compressed(
        out / "dataset" / "chengdu_era5_layer48_dataset.npz",
        X=data["X"],
        T_raw=data["T_raw"],
        RH_raw=data["RH_raw"],
        q_raw=data["q_raw"],
        logq_raw=data["logq_raw"],
        T=t_corrected,
        RH=rh_corrected,
        q=q_corrected,
        logq=logq_corrected,
        P=data["P"],
        dates=data["dates"],
        timestamps=data["timestamps"],
        heights=centers,
        layer_edges=edges,
        layer_thickness=thickness,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        bias_T=bias_t,
        bias_logq=bias_logq,
        apply_T=np.asarray(apply_t),
        apply_logq=np.asarray(apply_logq),
    )

    (out / "dataset" / "chengdu_era5_layer48_dataset_stats.json").write_text(
        json.dumps(
            {
                "status": "filtered_48layer_dataset",
                "obs_records_total": len(records),
                "exact_era5_matches": len(data["X"]),
                "n_dates": int(len(np.unique(data["dates"]))),
                "split": split,
                "bias_correction": {
                    "training_pairs": len(pair_splits["train"]),
                    "apply_T": bool(apply_t),
                    "apply_logq": bool(apply_logq),
                    "T_bias_K": bias_t.tolist(),
                    "logq_bias": bias_logq.tolist(),
                },
                "layer_grid": {
                    "n_layers": 48,
                    "centers_m": centers.tolist(),
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # ============================================================
    # STEP 3: Ridge/EOF for Temperature
    # ============================================================
    print("\n[Step 3/6] Training Ridge/EOF for Temperature...")
    channel_indices = list(range(21))
    X_full = data["X"][:, channel_indices].astype(np.float64)
    X_train, X_val, X_test, x_mean, x_std = standardize(
        X_full[train_mask], X_full[val_mask], X_full[test_mask]
    )

    best_t, search_t = select_model(
        X_train,
        t_corrected[train_mask].astype(np.float64),
        X_val,
        t_corrected[val_mask].astype(np.float64),
    )
    t_pred = predict_selected(best_t, X_test)
    t_pred = np.clip(t_pred, 180.0, 330.0).astype(np.float32)
    print(
        f"  T model: {best_t['method']}, alpha={best_t['alpha']}, "
        f"n_eof={best_t['n_eof']}, val_rmse={best_t['val_rmse']:.3f}K"
    )

    # ============================================================
    # STEP 4: BRNN ensemble for RH
    # ============================================================
    print("\n[Step 4/6] Training BRNN ensemble for RH...")
    import torch

    device = "cpu"

    rh_target = rh_corrected.astype(np.float32)
    X_for_brnn = data["X"][:, channel_indices].astype(np.float32)
    rh_model_defs = [m for m in MODEL_DEFS_LAYER48 if m[1] == "RH"]
    heights_arr = centers.astype(np.float32)

    all_rh_preds = []
    brnn_run_stats = []

    for seed_idx, seed in enumerate(args.brnn_seeds):
        print(
            f"  BRNN seed={seed} ({seed_idx + 1}/{len(args.brnn_seeds)})..."
        )
        set_seed(seed)
        models = {}
        training_info = {}
        for name, variable, height_range in MODEL_DEFS_LAYER48:
            model, details = train_one_model(
                name,
                variable,
                height_range,
                X_for_brnn,
                t_corrected,
                rh_target,
                train_mask,
                val_mask,
                args.hidden_size,
                args.dropout,
                args.batch_size,
                args.learning_rate,
                args.max_epochs,
                args.patience,
                device,
                height_grid=heights_arr,
            )
            models[name] = model
            training_info[name] = details

        _, rh_brnn = predict_full(
            models,
            X_for_brnn[test_mask],
            device,
            height_grid=heights_arr,
            model_defs=MODEL_DEFS_LAYER48,
        )
        rh_brnn = np.clip(rh_brnn, 0.0, 100.0)
        all_rh_preds.append(rh_brnn)

        # Save models
        model_dir = out / "models" / f"seed{seed}"
        model_dir.mkdir(parents=True, exist_ok=True)
        for name, model in models.items():
            torch.save(model.state_dict(), model_dir / f"{name}.pt")

        brnn_run_stats.append(
            {
                "seed": seed,
                "training": {
                    n: {
                        "best_epoch": d["best_epoch"],
                        "best_val": d["best_val"],
                    }
                    for n, d in training_info.items()
                },
            }
        )

    # Top-2 ensemble for RH
    rh_pred_ensemble = np.zeros_like(all_rh_preds[0], dtype=np.float32)
    ensemble_selection = {}
    for name, _, height_range in rh_model_defs:
        losses = np.asarray(
            [r["training"][name]["best_val"] for r in brnn_run_stats]
        )
        selected = np.argsort(losses)[:2]
        mask = (heights_arr >= height_range[0] * 1000.0) & (
            heights_arr <= height_range[1] * 1000.0
        )
        rh_pred_ensemble[:, mask] = np.mean(
            [all_rh_preds[i][:, mask] for i in selected], axis=0
        )
        ensemble_selection[name] = [
            {"seed": args.brnn_seeds[i], "val_loss": float(losses[i])}
            for i in selected
        ]

    # ============================================================
    # STEP 5: Hybrid product + metrics
    # ============================================================
    print("\n[Step 5/6] Building hybrid product...")
    t_true_test = t_corrected[test_mask]
    rh_true_test = rh_target[test_mask]
    t_raw_test = data["T_raw"][test_mask]
    rh_raw_test = data["RH_raw"][test_mask]
    test_dates = data["dates"][test_mask]
    test_timestamps = data["timestamps"][test_mask]

    t_hybrid = t_pred
    rh_hybrid = rh_pred_ensemble

    def m(pred, truth):
        e = pred - truth
        return {
            "rmse": float(np.sqrt(np.mean(e**2))),
            "mae": float(np.mean(np.abs(e))),
            "bias": float(np.mean(e)),
        }

    def daily_m(pred, truth, dates):
        per = []
        for d in np.unique(dates):
            mask = dates == d
            e = pred[mask] - truth[mask]
            per.append(
                {
                    "date": str(d),
                    "n": int(mask.sum()),
                    "rmse": float(np.sqrt(np.mean(e**2))),
                    "bias": float(np.mean(e)),
                }
            )
        return {"n_dates": len(per), "per_date": per}

    combined_metrics = {
        "hybrid_vs_corrected_label": {
            "T": m(t_hybrid, t_true_test),
            "RH": m(rh_hybrid, rh_true_test),
            "T_daily": daily_m(t_hybrid, t_true_test, test_dates),
            "RH_daily": daily_m(rh_hybrid, rh_true_test, test_dates),
        },
        "hybrid_vs_raw_era5": {
            "T": m(t_hybrid, t_raw_test),
            "RH": m(rh_hybrid, rh_raw_test),
        },
        "climatology_vs_corrected": {
            "T": m(
                np.repeat(
                    t_corrected[train_mask].mean(axis=0, keepdims=True),
                    len(t_true_test),
                    axis=0,
                ),
                t_true_test,
            ),
            "RH": m(
                np.repeat(
                    rh_target[train_mask].mean(axis=0, keepdims=True),
                    len(rh_true_test),
                    axis=0,
                ),
                rh_true_test,
            ),
        },
    }

    # ============================================================
    # STEP 6: Save results
    # ============================================================
    print("\n[Step 6/6] Saving results...")

    # Predictions
    np.savez_compressed(
        out / "predictions" / "chengdu_filtered48_hybrid_predictions.npz",
        T_pred=t_hybrid,
        RH_pred=rh_hybrid,
        T_true=t_true_test,
        RH_true=rh_true_test,
        T_raw=t_raw_test,
        RH_raw=rh_raw_test,
        dates=test_dates,
        timestamps=test_timestamps,
        heights=centers,
        layer_edges=edges,
        layer_thickness=thickness,
        RH_brnn_ensemble_5seeds=np.stack(all_rh_preds).astype(np.float32),
    )

    # Ridge model
    np.savez_compressed(
        out / "models" / "chengdu_filtered48_ridge_model.npz",
        X_mean=x_mean,
        X_std=x_std,
        channel_indices=np.asarray(channel_indices, dtype=np.int64),
        heights=centers,
        T_weights=best_t["weights"],
        T_y_mean=best_t["y_mean"],
        T_basis=(
            best_t["basis"]
            if best_t["basis"] is not None
            else np.empty((0, 0), dtype=np.float64)
        ),
        T_profile_mean=best_t.get(
            "profile_mean", np.empty((0, 0), dtype=np.float64)
        ),
    )

    # Full summary
    full_summary = {
        "status": "filtered_48layer_hybrid_retrieval",
        "pipeline_date": today,
        "data_source": {
            "obs_json": str(args.obs_json),
            "grib": str(args.grib),
            "sounding_dir": str(args.sounding_dir),
            "n_records": len(records),
            "n_dates": int(len(np.unique(data["dates"]))),
        },
        "dataset": {
            "n_exact_matches": int(len(data["X"])),
            "train": int(train_mask.sum()),
            "val": int(val_mask.sum()),
            "test": int(test_mask.sum()),
            "split": split,
        },
        "layer_grid": {
            "n_layers": 48,
            "definition": "0-500m:1, 500-2000m:every 100m, 2000-10000m:every 250m",
        },
        "bias_correction": {
            "training_pairs": len(pair_splits["train"]),
            "apply_T": bool(apply_t),
            "apply_logq": bool(apply_logq),
            "T_bias_mean_K": float(bias_t.mean()),
            "logq_bias_mean": float(bias_logq.mean()),
        },
        "temperature_model": {
            "method": best_t["method"],
            "alpha": best_t["alpha"],
            "n_eof": best_t["n_eof"],
            "val_rmse_K": best_t["val_rmse"],
        },
        "humidity_model": {
            "method": "BRNN top-2 ensemble",
            "seeds": args.brnn_seeds,
            "ensemble_selection": ensemble_selection,
        },
        "metrics": combined_metrics,
    }
    (out / "chengdu_filtered48_hybrid_stats.json").write_text(
        json.dumps(full_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out / "brnn_ensemble_stats.json").write_text(
        json.dumps(brnn_run_stats, indent=2), encoding="utf-8"
    )

    # Plots
    print("  Generating diagnostic plots...")
    try:
        make_diagnostic_plots(
            t_hybrid,
            rh_hybrid,
            t_true_test,
            rh_true_test,
            t_raw_test,
            rh_raw_test,
            centers,
            edges,
            thickness,
            bias_t,
            bias_logq,
            apply_t,
            apply_logq,
            out / "figures",
        )
        print("  Plots saved.")
    except Exception as exc:
        print(f"  Plotting skipped: {exc}")

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    for k, v in combined_metrics.items():
        print(f"\n{k}:")
        for var in ("T", "RH"):
            if var in v:
                print(
                    f"  {var}: RMSE={v[var]['rmse']:.3f}, "
                    f"Bias={v[var]['bias']:+.3f}, MAE={v[var]['mae']:.3f}"
                )
    print(f"\nAll results saved to: {out}")


if __name__ == "__main__":
    main()
