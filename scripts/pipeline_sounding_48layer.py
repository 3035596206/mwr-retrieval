#!/usr/bin/env python3
"""Filtered BT → 48-layer sounding-matched retrieval pipeline.

Since ERA5 GRIB data for May 2026 is unavailable, this pipeline:
  1. Matches filtered 21-channel BT to Wenjiang soundings (±3 hours)
  2. Parses soundings to the physical 48-layer grid
  3. Splits by sounding launch group (prevents leakage)
  4. Trains Ridge/EOF for T and BRNN ensemble for RH
  5. Combines into hybrid product
  6. Evaluates and generates diagnostic plots

All results saved under a single date-stamped output directory.
"""

from __future__ import annotations

import argparse, json, os, sys, tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "mwr-filtered48-matplotlib"))

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from train_chengdu_brnn import (
    load_observations, predict_full, rmse_bias, set_seed,
    train_one_model, sounding_time_from_name,
    grouped_split,
)
from train_chengdu_era5_ridge import predict_selected, select_model, standardize
from mwr_retrieval.grids import build_layer48_grid, layer_average as shared_layer_average
from mwr_retrieval.thermodynamics import (
    Q_FLOOR,
    rh_to_specific_humidity as shared_rh_to_q,
    saturation_vapor_pressure_hpa as shared_saturation_vapor_pressure_hpa,
    specific_humidity_to_rh as shared_q_to_rh,
)

# ============================================================
# 48-layer physical grid
# ============================================================
G = 9.80665


def build_layer_grid():
    return build_layer48_grid()


def saturation_vapor_pressure_hpa(t_k):
    return shared_saturation_vapor_pressure_hpa(t_k)


def rh_to_q(t_k, rh, p_hpa):
    return shared_rh_to_q(t_k, rh, p_hpa)


def q_to_rh(t_k, q, p_hpa):
    return shared_q_to_rh(t_k, q, p_hpa)


def layer_average(src_z, src_val, edges):
    return shared_layer_average(src_z, src_val, edges)

def raw_profile_to_layers(h_agl, t_k, rh, p_hpa, edges):
    valid = (np.isfinite(h_agl) & np.isfinite(t_k) & np.isfinite(rh) &
             np.isfinite(p_hpa) & (h_agl >= 0) & (t_k >= 180) & (t_k <= 330) &
             (rh >= 0) & (rh <= 100) & (p_hpa >= 1) & (p_hpa <= 1100))
    z = np.asarray(h_agl[valid], dtype=np.float64)
    t = np.asarray(t_k[valid], dtype=np.float64)
    r = np.asarray(rh[valid], dtype=np.float64)
    p = np.asarray(p_hpa[valid], dtype=np.float64)
    if len(z) < 15:
        raise ValueError("too few valid levels")
    order = np.argsort(z)
    z, t, r, p = z[order], t[order], r[order], p[order]
    keep = np.concatenate(([True], np.diff(z) > 0.5))
    z, t, r, p = z[keep], t[keep], r[keep], p[keep]
    if z[-1] < edges[-1]:
        raise ValueError("profile does not reach 10 km")
    if z[0] > 0:
        z = np.concatenate(([0.0], z))
        t = np.concatenate(([t[0]], t))
        r = np.concatenate(([r[0]], r))
        p = np.concatenate(([p[0]], p))
    q = rh_to_q(t, r, p)
    t_layer = layer_average(z, t, edges)
    log_q = layer_average(z, np.log(np.maximum(q, Q_FLOOR)), edges)
    log_p = layer_average(z, np.log(p), edges)
    q_layer = np.exp(log_q).astype(np.float32)
    p_layer = np.exp(log_p).astype(np.float32)
    rh_layer = q_to_rh(t_layer, q_layer, p_layer).astype(np.float32)
    return {"T": t_layer, "q": q_layer, "logq": log_q, "P": p_layer, "RH": rh_layer}

def parse_sounding_layers(path, edges):
    heights, temps, rhs, press = [], [], [], []
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            fields = line.split()
            if len(fields) < 17:
                continue
            try:
                tc, ph, rh, hm = float(fields[6]), float(fields[7]), float(fields[8]), float(fields[16])
            except ValueError:
                continue
            if -100 <= tc <= 60 and 1 <= ph <= 1100 and 0 <= rh <= 100 and -500 <= hm <= 60000:
                heights.append(hm); temps.append(tc + 273.15); rhs.append(rh); press.append(ph)
    if len(heights) < 50:
        raise ValueError(f"too few valid levels: {len(heights)}")
    h_arr = np.asarray(heights, dtype=np.float64)
    station_h = float(np.nanpercentile(h_arr, 0.5))
    return raw_profile_to_layers(h_arr - station_h, np.asarray(temps),
                                  np.asarray(rhs), np.asarray(press), edges)

# ============================================================
# BRNN model defs for 48-layer
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
    import matplotlib; matplotlib.use("Agg")
    from matplotlib import font_manager
    for fp in [Path(r"C:\Windows\Fonts\msyh.ttc"), Path(r"C:\Windows\Fonts\simhei.ttf"),
               Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf")]:
        if fp.exists():
            font_manager.fontManager.addfont(fp)
    import matplotlib.pyplot as plt
    plt.rcParams.update({"axes.unicode_minus": False, "figure.dpi": 130, "savefig.dpi": 200,
                         "savefig.bbox": "tight", "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True,
                         "grid.alpha": 0.22, "legend.frameon": False})
    return plt

def make_plots(out_dir, t_pred, rh_pred, t_true, rh_true, centers, edges, thickness,
               bias_t, bias_rh, test_timestamps, sample_label="test"):
    plt = configure_plotting()
    out_dir.mkdir(parents=True, exist_ok=True)
    # --- 1. Layer grid + bias ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.5), sharey=True)
    tc = ["#1976d2" if v <= 100 else "#ef6c00" for v in thickness]
    axes[0].barh(centers / 1000., thickness, height=np.minimum(thickness / 1000., 0.18), color=tc)
    axes[0].set_xlabel("Layer thickness (m)"); axes[0].set_ylabel("Height (km)")
    axes[0].set_title("Physical 48-layer grid")
    axes[1].plot(bias_t, centers / 1000., color="#d32f2f", linewidth=2, label="T bias (K)")
    axes[1].plot(bias_rh, centers / 1000., color="#1976d2", linewidth=2, label="RH bias (%)")
    axes[1].axvline(0, color="#5f6368", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Bias"); axes[1].set_title("Hybrid retrieval bias profile"); axes[1].legend()
    fig.suptitle("48-layer grid & retrieval bias profile", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_dir / "01_layer_grid_bias.png", facecolor="white"); plt.close(fig)

    # --- 2. Error profiles ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)
    t_rmse = np.sqrt(np.mean((t_pred - t_true)**2, axis=0))
    rh_rmse = np.sqrt(np.mean((rh_pred - rh_true)**2, axis=0))
    t_bias_arr = np.mean(t_pred - t_true, axis=0)
    rh_bias_arr = np.mean(rh_pred - rh_true, axis=0)
    for ax, rmse, bias, u, v in [(axes[0], t_rmse, t_bias_arr, "K", "T"),
                                   (axes[1], rh_rmse, rh_bias_arr, "%", "RH")]:
        ax.plot(rmse, centers / 1000., color="#d32f2f", linewidth=2, label="RMSE")
        ax.plot(bias, centers / 1000., color="#1976d2", linewidth=2, label="Bias")
        ax.axvline(0, color="#5f6368", linestyle="--", linewidth=1)
        ax.set_xlabel(f"Error ({u})"); ax.set_title(f"{v} error"); ax.legend(loc="best")
    axes[0].set_ylabel("Height (km)")
    fig.suptitle(f"Retrieval error profiles (n={len(t_true)} {sample_label})", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_dir / "02_error_profiles.png", facecolor="white"); plt.close(fig)

    # --- 3. Sample profiles ---
    n = min(4, len(t_true))
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8), sharey="row")
    for i in range(n):
        axes[0, i].plot(t_pred[i], centers / 1000., color="#d32f2f", linewidth=2, label="Retrieved")
        axes[0, i].plot(t_true[i], centers / 1000., color="#202124", linewidth=2, label="Sounding")
        axes[0, i].set_xlabel("T (K)"); axes[0, i].set_title(f"Sample {i+1}")
        axes[1, i].plot(rh_pred[i], centers / 1000., color="#1976d2", linewidth=2, label="Retrieved")
        axes[1, i].plot(rh_true[i], centers / 1000., color="#202124", linewidth=2, label="Sounding")
        axes[1, i].set_xlabel("RH (%)")
    axes[0, 0].legend(fontsize=7); axes[0, 0].set_ylabel("Height (km)")
    axes[1, 0].set_ylabel("Height (km)")
    fig.suptitle("Sample profiles", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_dir / "03_sample_profiles.png", facecolor="white"); plt.close(fig)

    # --- 4. Per-height scatter ---
    key_h = [0.25, 1.0, 2.0, 5.0, 8.0]
    fig, axes = plt.subplots(2, len(key_h), figsize=(4 * len(key_h), 8))
    for j, h_km in enumerate(key_h):
        idx = int(np.argmin(np.abs(centers - h_km * 1000.)))
        for row, (pred, truth, var, u, col) in enumerate([
            (t_pred[:, idx], t_true[:, idx], "T", "K", "#d32f2f"),
            (rh_pred[:, idx], rh_true[:, idx], "RH", "%", "#1976d2")]):
            ax = axes[row, j]
            ax.scatter(truth, pred, s=8, alpha=0.5, color=col, edgecolors="none")
            lo, hi = min(truth.min(), pred.min()), max(truth.max(), pred.max())
            rng = hi - lo; lo -= rng * 0.05; hi += rng * 0.05
            ax.plot([lo, hi], [lo, hi], color="#5f6368", linestyle="--", linewidth=1)
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
            ax.set_xlabel(f"True {var} ({u})", fontsize=8)
            ax.set_ylabel(f"Pred {var} ({u})", fontsize=8)
            e = pred - truth
            ax.set_title(f"{var} @ {h_km}km", fontsize=9)
            ax.text(0.95, 0.08, f"RMSE={np.sqrt(np.mean(e**2)):.2f}\nBias={np.mean(e):+.2f}",
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=7,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))
    fig.suptitle("Scatter density at key heights", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_dir / "04_scatter_density.png", facecolor="white"); plt.close(fig)


def _case_time_label(value):
    text = str(value)
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y/%m/%d %H:%M:%S"):
        try:
            return datetime.strptime(text[:19], fmt).strftime("%Y.%m.%d %H-00")
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).replace(tzinfo=None).strftime("%Y.%m.%d %H-00")
    except ValueError:
        return text.replace("T", " ")[:16]


def _safe_case_name(value):
    keep = []
    for char in str(value):
        keep.append(char if char.isalnum() else "_")
    return "".join(keep).strip("_") or "sample"


def _profile_metric(pred, truth):
    err = pred - truth
    return float(np.mean(err)), float(np.sqrt(np.mean(err ** 2)))


def make_profile_case_plots(out_dir, t_pred, rh_pred, t_true, rh_true, centers, timestamps, groups):
    plt = configure_plotting()
    out_dir.mkdir(parents=True, exist_ok=True)
    case_paths = []
    case_summary = []
    height_km = centers / 1000.0

    for i in range(len(t_true)):
        label = _case_time_label(timestamps[i])
        group = str(groups[i])
        t_bias, t_rmse = _profile_metric(t_pred[i], t_true[i])
        rh_bias, rh_rmse = _profile_metric(rh_pred[i], rh_true[i])

        fig, axes = plt.subplots(1, 2, figsize=(8.2, 5.4), sharey=True)
        axes[0].plot(t_pred[i], height_km, color="#d32f2f", linewidth=2.2, label="Prediction")
        axes[0].plot(t_true[i], height_km, color="#1976d2", linestyle="--", linewidth=2.0, label="Reference")
        axes[0].set_xlabel("T (K)")
        axes[0].set_ylabel("Height (km)")
        axes[0].set_title(f"{label} - T\nBias={t_bias:+.2f} K  RMSE={t_rmse:.2f} K", fontsize=10)

        axes[1].plot(rh_pred[i], height_km, color="#d32f2f", linewidth=2.2, label="Prediction")
        axes[1].plot(rh_true[i], height_km, color="#1976d2", linestyle="--", linewidth=2.0, label="Reference")
        axes[1].set_xlabel("RH (%)")
        axes[1].set_xlim(-2, 102)
        axes[1].set_title(f"{label} - RH\nBias={rh_bias:+.2f} %  RMSE={rh_rmse:.2f} %", fontsize=10)

        for ax in axes:
            ax.set_ylim(0, 10)
            ax.legend(loc="best", fontsize=8)
            ax.tick_params(labelsize=8)
        fig.suptitle(f"Filtered BT retrieval case {i + 1:03d} / {len(t_true)} | sounding group {group}", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.92))

        path = out_dir / f"case_{i + 1:03d}_{_safe_case_name(timestamps[i])}.png"
        fig.savefig(path, facecolor="white")
        plt.close(fig)
        case_paths.append(path)
        case_summary.append({
            "index": i,
            "timestamp": str(timestamps[i]),
            "label": label,
            "group": group,
            "T_bias_K": t_bias,
            "T_rmse_K": t_rmse,
            "RH_bias_percent": rh_bias,
            "RH_rmse_percent": rh_rmse,
            "figure": str(path),
        })

    try:
        from PIL import Image, ImageDraw

        thumbs_per_page = 12
        cols = 3
        thumb_w, thumb_h = 520, 360
        pad = 18
        title_h = 34
        for page, start in enumerate(range(0, len(case_paths), thumbs_per_page), start=1):
            subset = case_paths[start:start + thumbs_per_page]
            rows = int(np.ceil(len(subset) / cols))
            canvas = Image.new("RGB", (cols * thumb_w + (cols + 1) * pad,
                                       rows * thumb_h + (rows + 1) * pad + title_h), "white")
            draw = ImageDraw.Draw(canvas)
            draw.text((pad, pad), f"All filtered-BT matched retrieval cases | page {page}", fill=(32, 33, 36))
            for j, img_path in enumerate(subset):
                img = Image.open(img_path).convert("RGB")
                img.thumbnail((thumb_w, thumb_h))
                x = pad + (j % cols) * (thumb_w + pad)
                y = pad + title_h + (j // cols) * (thumb_h + pad)
                canvas.paste(img, (x, y))
            montage_path = out_dir / f"montage_cases_page{page:02d}.png"
            canvas.save(montage_path)
        if case_paths:
            first_page = out_dir / "montage_cases_page01.png"
            if first_page.exists():
                Image.open(first_page).save(out_dir / "montage_cases.png")
    except Exception as exc:
        case_summary.append({"montage_error": str(exc)})

    (out_dir / "case_summary.json").write_text(
        json.dumps(case_summary, indent=2, ensure_ascii=False), encoding="utf-8")


# ============================================================
# MAIN
# ============================================================
def main():
    today = datetime.now().strftime("%Y%m%d")
    parser = argparse.ArgumentParser()
    parser.add_argument("--obs-json", type=Path, required=True)
    parser.add_argument("--sounding-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path,
                        default=PROJECT_ROOT / "results" / f"filtered_48layer_{today}")
    parser.add_argument("--max-delta-hours", type=float, default=3.0)
    parser.add_argument("--brnn-seeds", type=int, nargs="+", default=[42, 1, 7, 21, 84])
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=35)
    args = parser.parse_args()

    out = args.output_root
    for sub in ["models", "predictions", "figures"]:
        (out / sub).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Pipeline: filtered BT -> 48-layer sounding-matched retrieval")
    print(f"Output:      {out}")
    print(f"Obs JSON:    {args.obs_json}")
    print(f"Soundings:   {args.sounding_dir}")
    print(f"Max delta:   {args.max_delta_hours}h")
    print("=" * 60)

    # ============================================================
    # STEP 1: Match BT to soundings → 48-layer dataset
    # ============================================================
    print("\n[Step 1/5] Matching BT to soundings...")
    edges, centers = build_layer_grid()
    records = load_observations(args.obs_json)
    print(f"  Loaded {len(records)} observation records")

    sounding_paths = sorted(Path(args.sounding_dir).rglob("*.txt"))
    sounding_entries = [(sounding_time_from_name(p), p) for p in sounding_paths]
    profile_cache = {}
    skipped = {}

    brightness, t_targets, rh_targets, groups, timestamps, deltas = [], [], [], [], [], []
    for rec in records:
        obs_time = rec["datetime"]
        snd_time, snd_path = min(sounding_entries,
                                  key=lambda x: abs((x[0] - obs_time).total_seconds()))
        dh = abs((snd_time - obs_time).total_seconds()) / 3600.
        if dh > args.max_delta_hours:
            continue
        g = snd_time.strftime("%Y%m%d%H")
        if g not in profile_cache and g not in skipped:
            try:
                profile_cache[g] = parse_sounding_layers(snd_path, edges)
            except ValueError as e:
                skipped[g] = str(e)
        if g in skipped:
            continue
        prof = profile_cache[g]
        brightness.append(rec["channels"])
        t_targets.append(prof["T"])
        rh_targets.append(prof["RH"])
        groups.append(g)
        timestamps.append(rec["timestamp"])
        deltas.append(dh)

    if len(brightness) < 10 or len(set(groups)) < 5:
        raise RuntimeError(f"Insufficient matches: {len(brightness)} samples, {len(set(groups))} groups")

    X = np.asarray(brightness, dtype=np.float32)
    T = np.asarray(t_targets, dtype=np.float32)
    RH = np.asarray(rh_targets, dtype=np.float32)
    groups_arr = np.asarray(groups)
    timestamps_arr = np.asarray(timestamps)
    deltas_arr = np.asarray(deltas, dtype=np.float32)
    print(f"  Matched: {len(X)} samples / {len(set(groups))} sounding groups")
    print(f"  Delta hours: mean={deltas_arr.mean():.1f}, max={deltas_arr.max():.1f}")

    # ============================================================
    # STEP 2: Grouped split
    # ============================================================
    print("\n[Step 2/5] Splitting by sounding group...")
    train_mask, val_mask, test_mask, split = grouped_split(groups_arr, seed=42)
    print(f"  Train: {train_mask.sum()} ({len(split['train_groups'])} groups)")
    print(f"  Val:   {val_mask.sum()} ({len(split['val_groups'])} groups)")
    print(f"  Test:  {test_mask.sum()} ({len(split['test_groups'])} groups)")

    # ============================================================
    # STEP 3: Ridge/EOF for T
    # ============================================================
    print("\n[Step 3/5] Training Ridge/EOF for T...")
    channel_indices = list(range(21))
    Xf = X[:, channel_indices].astype(np.float64)
    Xt, Xv, Xte, xm, xs = standardize(Xf[train_mask], Xf[val_mask], Xf[test_mask])

    best_t, _ = select_model(Xt, T[train_mask].astype(np.float64),
                              Xv, T[val_mask].astype(np.float64))
    Xall = ((Xf - xm) / xs).astype(np.float64)
    t_pred = np.clip(predict_selected(best_t, Xte), 180., 330.).astype(np.float32)
    t_pred_all = np.clip(predict_selected(best_t, Xall), 180., 330.).astype(np.float32)
    print(f"  T: {best_t['method']}, alpha={best_t['alpha']}, n_eof={best_t['n_eof']}, val_rmse={best_t['val_rmse']:.3f}K")

    # ============================================================
    # STEP 4: BRNN ensemble for RH
    # ============================================================
    print("\n[Step 4/5] Training BRNN ensemble for RH...")
    import torch; device = "cpu"
    Xb = X[:, channel_indices].astype(np.float32)
    h_arr = centers.astype(np.float32)
    rh_model_defs = [m for m in MODEL_DEFS_LAYER48 if m[1] == "RH"]

    all_rh = []; all_rh_all = []; run_stats = []
    for si, seed in enumerate(args.brnn_seeds):
        print(f"  Seed {seed} ({si+1}/{len(args.brnn_seeds)})...")
        set_seed(seed)
        models = {}; tinfo = {}
        for name, var, hr in rh_model_defs:
            m, d = train_one_model(name, var, hr, Xb, T, RH, train_mask, val_mask,
                                    args.hidden_size, args.dropout, args.batch_size,
                                    args.learning_rate, args.max_epochs, args.patience,
                                    device, height_grid=h_arr)
            models[name] = m; tinfo[name] = d
        _, rh_b_all = predict_full(models, Xb, device, height_grid=h_arr,
                                   model_defs=rh_model_defs)
        rh_b_all = np.clip(rh_b_all, 0., 100.).astype(np.float32)
        all_rh_all.append(rh_b_all)
        all_rh.append(rh_b_all[test_mask])
        run_stats.append({"seed": seed, "training": {
            n: {"best_epoch": d["best_epoch"], "best_val": d["best_val"]}
            for n, d in tinfo.items()}})
        md = out / "models" / f"seed{seed}"; md.mkdir(parents=True, exist_ok=True)
        for n, m in models.items():
            torch.save(m.state_dict(), md / f"{n}.pt")

    # Top-2 ensemble
    rh_ens = np.zeros_like(all_rh[0], dtype=np.float32)
    rh_ens_all = np.zeros_like(all_rh_all[0], dtype=np.float32)
    ens_sel = {}
    for name, _, hr in rh_model_defs:
        losses = np.asarray([r["training"][name]["best_val"] for r in run_stats])
        sel = np.argsort(losses)[:2]
        mask = (h_arr >= hr[0] * 1000.) & (h_arr <= hr[1] * 1000.)
        rh_ens[:, mask] = np.mean([all_rh[i][:, mask] for i in sel], axis=0)
        rh_ens_all[:, mask] = np.mean([all_rh_all[i][:, mask] for i in sel], axis=0)
        ens_sel[name] = [{"seed": args.brnn_seeds[i], "val_loss": float(losses[i])} for i in sel]

    rh_ens_raw = rh_ens.copy()
    rh_ens_all_raw = rh_ens_all.copy()
    rh_val_residual = np.mean(RH[val_mask] - rh_ens_all_raw[val_mask], axis=0).astype(np.float32)
    calibration_candidates = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
    calibration_scores = []
    for strength in calibration_candidates:
        rh_val_cal = np.clip(rh_ens_all_raw[val_mask] + strength * rh_val_residual[None, :], 0.0, 100.0)
        err = rh_val_cal - RH[val_mask]
        calibration_scores.append({
            "strength": float(strength),
            "val_rmse": float(np.sqrt(np.mean(err ** 2))),
            "val_bias": float(np.mean(err)),
        })
    best_cal = min(calibration_scores, key=lambda item: item["val_rmse"])
    rh_bias_correction = (best_cal["strength"] * rh_val_residual).astype(np.float32)
    rh_ens_all = np.clip(rh_ens_all_raw + rh_bias_correction[None, :], 0.0, 100.0).astype(np.float32)
    rh_ens = rh_ens_all[test_mask]

    # ============================================================
    # STEP 5: Evaluate & save
    # ============================================================
    print("\n[Step 5/5] Evaluating & saving...")
    t_test = T[test_mask]; rh_test = RH[test_mask]
    t_hyb = t_pred; rh_hyb = rh_ens
    t_hyb_all = t_pred_all; rh_hyb_all = rh_ens_all
    rh_hyb_raw = rh_ens_raw; rh_hyb_all_raw = rh_ens_all_raw
    test_g = groups_arr[test_mask]
    thickness = np.diff(edges).astype(np.float32)

    def met(pred, truth):
        e = pred - truth
        return {"rmse": float(np.sqrt(np.mean(e**2))),
                "mae": float(np.mean(np.abs(e))), "bias": float(np.mean(e))}

    def group_met(tp, rhp, tt, rht, grp):
        ug = np.unique(grp)
        t_gp = np.stack([tp[grp == g].mean(axis=0) for g in ug])
        rh_gp = np.stack([rhp[grp == g].mean(axis=0) for g in ug])
        t_gt = np.stack([tt[grp == g][0] for g in ug])
        rh_gt = np.stack([rht[grp == g][0] for g in ug])
        return {"n_groups": len(ug), "T": met(t_gp, t_gt), "RH": met(rh_gp, rh_gt)}

    clim_profile_t = T[train_mask].mean(axis=0, keepdims=True)
    clim_profile_rh = RH[train_mask].mean(axis=0, keepdims=True)
    clim_t = np.repeat(clim_profile_t, len(t_test), axis=0)
    clim_rh = np.repeat(clim_profile_rh, len(rh_test), axis=0)
    clim_t_all = np.repeat(clim_profile_t, len(T), axis=0)
    clim_rh_all = np.repeat(clim_profile_rh, len(RH), axis=0)

    all_metrics = {
        "hybrid_sample": {"T": met(t_hyb, t_test), "RH": met(rh_hyb, rh_test)},
        "hybrid_grouped": group_met(t_hyb, rh_hyb, t_test, rh_test, test_g),
        "hybrid_all_sample": {"T": met(t_hyb_all, T), "RH": met(rh_hyb_all, RH)},
        "hybrid_all_grouped": group_met(t_hyb_all, rh_hyb_all, T, RH, groups_arr),
        "hybrid_raw_sample": {"T": met(t_hyb, t_test), "RH": met(rh_hyb_raw, rh_test)},
        "hybrid_raw_grouped": group_met(t_hyb, rh_hyb_raw, t_test, rh_test, test_g),
        "hybrid_raw_all_sample": {"T": met(t_hyb_all, T), "RH": met(rh_hyb_all_raw, RH)},
        "hybrid_raw_all_grouped": group_met(t_hyb_all, rh_hyb_all_raw, T, RH, groups_arr),
        "climatology_sample": {"T": met(clim_t, t_test), "RH": met(clim_rh, rh_test)},
        "climatology_grouped": group_met(clim_t, clim_rh, t_test, rh_test, test_g),
        "climatology_all_sample": {"T": met(clim_t_all, T), "RH": met(clim_rh_all, RH)},
        "climatology_all_grouped": group_met(clim_t_all, clim_rh_all, T, RH, groups_arr),
    }

    # Height metrics
    h_metrics = []
    for hkm in [0, 0.5, 1, 2, 3, 5, 8, 10]:
        idx = int(np.argmin(np.abs(centers - hkm * 1000.)))
        h_metrics.append({
            "height_km": float(centers[idx] / 1000.),
            "T_rmse": float(np.sqrt(np.mean((t_hyb[:, idx] - t_test[:, idx])**2))),
            "T_bias": float(np.mean(t_hyb[:, idx] - t_test[:, idx])),
            "RH_rmse": float(np.sqrt(np.mean((rh_hyb[:, idx] - rh_test[:, idx])**2))),
            "RH_bias": float(np.mean(rh_hyb[:, idx] - rh_test[:, idx])),
        })

    h_metrics_all = []
    for hkm in [0, 0.5, 1, 2, 3, 5, 8, 10]:
        idx = int(np.argmin(np.abs(centers - hkm * 1000.)))
        h_metrics_all.append({
            "height_km": float(centers[idx] / 1000.),
            "T_rmse": float(np.sqrt(np.mean((t_hyb_all[:, idx] - T[:, idx])**2))),
            "T_bias": float(np.mean(t_hyb_all[:, idx] - T[:, idx])),
            "RH_rmse": float(np.sqrt(np.mean((rh_hyb_all[:, idx] - RH[:, idx])**2))),
            "RH_bias": float(np.mean(rh_hyb_all[:, idx] - RH[:, idx])),
        })

    # Save
    np.savez_compressed(out / "predictions" / "chengdu_filtered48_hybrid_predictions.npz",
        T_pred=t_hyb, RH_pred=rh_hyb, T_true=t_test, RH_true=rh_test,
        RH_pred_raw=rh_hyb_raw, RH_bias_correction=rh_bias_correction,
        groups=test_g, timestamps=timestamps_arr[test_mask],
        heights=centers, layer_edges=edges, layer_thickness=thickness,
        RH_brnn_all_seeds=np.stack(all_rh).astype(np.float32))

    np.savez_compressed(out / "predictions" / "chengdu_filtered48_hybrid_predictions_all.npz",
        T_pred=t_hyb_all, RH_pred=rh_hyb_all, T_true=T, RH_true=RH,
        RH_pred_raw=rh_hyb_all_raw, RH_bias_correction=rh_bias_correction,
        groups=groups_arr, timestamps=timestamps_arr, delta_hours=deltas_arr,
        train_mask=train_mask, val_mask=val_mask, test_mask=test_mask,
        heights=centers, layer_edges=edges, layer_thickness=thickness,
        RH_brnn_all_seeds=np.stack(all_rh_all).astype(np.float32))

    np.savez_compressed(out / "models" / "chengdu_filtered48_ridge_model.npz",
        X_mean=xm, X_std=xs, channel_indices=np.asarray(channel_indices, dtype=np.int64),
        heights=centers, T_weights=best_t["weights"], T_y_mean=best_t["y_mean"],
        T_basis=best_t["basis"] if best_t["basis"] is not None else np.empty((0, 0)),
        T_profile_mean=best_t.get("profile_mean", np.empty((0, 0))))

    summary = {
        "status": "filtered_48layer_sounding_matched_retrieval",
        "pipeline_date": today,
        "data_source": {"obs_json": str(args.obs_json), "sounding_dir": str(args.sounding_dir),
                        "n_records": len(records), "max_delta_hours": args.max_delta_hours},
        "dataset": {"n_matched": int(len(X)), "n_groups": int(len(set(groups))),
                    "train": int(train_mask.sum()), "val": int(val_mask.sum()),
                    "test": int(test_mask.sum()),
                    "train_groups": len(split["train_groups"]),
                    "val_groups": len(split["val_groups"]),
                    "test_groups": len(split["test_groups"]),
                    "delta_hours_mean": float(deltas_arr.mean()),
                    "delta_hours_max": float(deltas_arr.max())},
        "layer_grid": {"n_layers": 48, "definition": "0-500m:1, 500-2000m:every 100m, 2000-10000m:every 250m"},
        "temperature_model": {"method": best_t["method"], "alpha": best_t["alpha"],
                              "n_eof": best_t["n_eof"], "val_rmse_K": best_t["val_rmse"]},
        "humidity_model": {"method": "BRNN top-2 ensemble", "seeds": args.brnn_seeds,
                           "loss": {"RH_trend_weight": 0.25, "RH_smooth_weight": 0.003},
                           "postprocess": {"method": "validation height-wise bias correction",
                                           "selected_strength": best_cal["strength"],
                                           "validation_scores": calibration_scores},
                           "ensemble_selection": ens_sel},
        "metrics": all_metrics,
        "height_metrics": h_metrics,
        "height_metrics_all": h_metrics_all,
        "split": split,
        "skipped_profiles": skipped,
    }
    (out / "chengdu_filtered48_hybrid_stats.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "brnn_ensemble_stats.json").write_text(
        json.dumps(run_stats, indent=2), encoding="utf-8")

    # Bias profile (for display)
    bias_t_profile = np.mean(t_hyb - t_test, axis=0).astype(np.float32)
    bias_rh_profile = np.mean(rh_hyb - rh_test, axis=0).astype(np.float32)
    bias_t_profile_all = np.mean(t_hyb_all - T, axis=0).astype(np.float32)
    bias_rh_profile_all = np.mean(rh_hyb_all - RH, axis=0).astype(np.float32)

    # Plots
    print("  Generating plots...")
    try:
        make_plots(out / "figures", t_hyb, rh_hyb, t_test, rh_test,
                   centers, edges, thickness, bias_t_profile, bias_rh_profile,
                   timestamps_arr[test_mask], sample_label="test")
        make_plots(out / "figures" / "all_summary", t_hyb_all, rh_hyb_all, T, RH,
                   centers, edges, thickness, bias_t_profile_all, bias_rh_profile_all,
                   timestamps_arr, sample_label="all")
        make_profile_case_plots(out / "figures" / "profile_all_cases",
                                t_hyb_all, rh_hyb_all, T, RH,
                                centers, timestamps_arr, groups_arr)
        print("  Plots saved.")
    except Exception as exc:
        print(f"  Plotting skipped: {exc}")

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    for k, v in all_metrics.items():
        print(f"\n{k}:")
        for var in ("T", "RH"):
            if var in v:
                print(f"  {var}: RMSE={v[var]['rmse']:.3f}, Bias={v[var]['bias']:+.3f}, MAE={v[var]['mae']:.3f}")
    print(f"\nAll results: {out}")


if __name__ == "__main__":
    main()
