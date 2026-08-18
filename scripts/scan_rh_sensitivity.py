#!/usr/bin/env python3
"""RH sensitivity scan: S_a / S_e parameter tuning for optimal RH retrieval.

Sweeps over:
  - S_a sigma_RH and length_RH
  - S_e K-band / V-band weight ratios
  - Channel subset experiments (K-only, V-only, full)

Uses the configured forward backend. ARTS is the project default; pass
``--forward simple`` for quick algorithm-only scans.

Usage:
  python scripts/scan_rh_sensitivity.py --n-samples 30
"""

import sys, os, itertools, csv, time
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.join(_SCRIPT_DIR, "..")
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))

import numpy as np
import config
from forward_model import ForwardModel
from oem_state import make_default_packer
from oem_covariance import (build_sa_exponential, build_se_diagonal,
                             build_se_from_channels)
from oem import OEMSolver


def load_data():
    import pickle
    profiles_path = os.path.join(_PROJECT_ROOT, "data", "era5",
                                  "era5_profiles_201301_poc.pkl")
    with open(profiles_path, "rb") as f:
        profiles = pickle.load(f)
    return profiles


def run_one_config(profiles, indices, fm, packer, S_a, S_e, cfg_label):
    """Run OEM on a fixed config and return aggregate RH/T stats."""
    solver = OEMSolver(fm, packer)
    n = len(indices)
    t_start = time.time()

    T_prior_rmse = np.zeros(n)
    T_post_rmse = np.zeros(n)
    RH_prior_rmse = np.zeros(n)
    RH_post_rmse = np.zeros(n)
    bt_prior_rms = np.zeros(n)
    bt_post_rms = np.zeros(n)
    dofs = np.zeros(n)
    converged = np.zeros(n, dtype=bool)

    heights = profiles["height"]
    for count, idx in enumerate(indices):
        profile_true = {
            "T": profiles["T"][idx],
            "RH": profiles["RH"][idx],
            "CLWC": profiles["CLWC"][idx],
            "P_hPa": 1013.25 * np.exp(-heights / 8000.0),
            "height": heights,
        }
        x_true = packer.pack(profile_true)

        np.random.seed(idx * 3 + 1)
        pert_T = np.random.randn(packer.n_T) * 3.0
        pert_RH = np.random.randn(packer.n_RH) * 10.0
        x_a = x_true + np.concatenate([pert_T, pert_RH])
        x_a[packer.n_T:] = np.clip(x_a[packer.n_T:], 2.0, 98.0)

        y_true = fm.simulate(profile_true)
        np.random.seed(idx * 7 + 3)
        y_obs = y_true + np.random.randn(fm.n_channels) * 0.5

        try:
            result = solver.retrieve(y_obs=y_obs, x_a=x_a, S_a=S_a, S_e=S_e,
                                      max_iter=15, verbose=False)
        except Exception:
            converged[count] = False
            continue

        prof_ret = packer.unpack(result["x_retrieved"])
        prof_bg = packer.unpack(x_a)

        T_prior_rmse[count] = np.sqrt(np.mean((prof_bg["T"] - profile_true["T"]) ** 2))
        T_post_rmse[count] = np.sqrt(np.mean((prof_ret["T"] - profile_true["T"]) ** 2))
        RH_prior_rmse[count] = np.sqrt(np.mean((prof_bg["RH"] - profile_true["RH"]) ** 2))
        RH_post_rmse[count] = np.sqrt(np.mean((prof_ret["RH"] - profile_true["RH"]) ** 2))
        bt_prior_rms[count] = np.sqrt(np.mean((result["y_sim_background"] - y_obs) ** 2))
        bt_post_rms[count] = np.sqrt(np.mean((result["y_sim_retrieved"] - y_obs) ** 2))
        dofs[count] = result["dofs"]
        converged[count] = result["converged"]

    elapsed = time.time() - t_start
    conv_mask = converged

    return {
        "config": cfg_label,
        "n": n,
        "n_conv": int(conv_mask.sum()),
        "conv_rate": float(conv_mask.mean()),
        "elapsed": elapsed,
        "T_prior_mean": float(np.mean(T_prior_rmse[conv_mask])),
        "T_post_mean": float(np.mean(T_post_rmse[conv_mask])),
        "RH_prior_mean": float(np.mean(RH_prior_rmse[conv_mask])),
        "RH_post_mean": float(np.mean(RH_post_rmse[conv_mask])),
        "BT_prior_mean": float(np.mean(bt_prior_rms[conv_mask])),
        "BT_post_mean": float(np.mean(bt_post_rms[conv_mask])),
        "DOFS_mean": float(np.mean(dofs[conv_mask])),
        "RH_impr_pct": float((1 - np.mean(RH_post_rmse[conv_mask]) /
                              max(np.mean(RH_prior_rmse[conv_mask]), 0.01)) * 100),
        "T_impr_pct": float((1 - np.mean(T_post_rmse[conv_mask]) /
                             max(np.mean(T_prior_rmse[conv_mask]), 0.01)) * 100),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--forward", default=config.DEFAULT_FORWARD_BACKEND,
                        choices=["arts", "simple", "monortm"])
    parser.add_argument("--arts-command", default=None,
                        help="External ARTS runner command; reads profile JSON on stdin")
    parser.add_argument("--elevation-angle", type=float, default=90.0,
                        help="Ground-based elevation angle for ARTS [deg]")
    args = parser.parse_args()

    np.random.seed(args.seed)
    profiles = load_data()
    n_total = profiles["T"].shape[0]
    indices = sorted(np.random.choice(n_total, min(args.n_samples, n_total), replace=False))

    fm = ForwardModel(
        backend=args.forward,
        arts_command=args.arts_command,
        elevation_angle_deg=args.elevation_angle,
    )
    packer = make_default_packer()
    all_results = []

    # ================================================================
    # Experiment 1: S_a sigma_RH and length_RH sweep
    # ================================================================
    print("=" * 70)
    print("Experiment 1: S_a (sigma_RH, length_RH) sweep")
    print("=" * 70)

    sigma_RH_vals = [4.0, 6.0, 8.0, 10.0, 12.0, 15.0]
    length_RH_vals = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    S_e_base = build_se_diagonal(fm, sigma_K=1.5, sigma_V=0.5)

    for sigma_RH, len_RH in itertools.product(sigma_RH_vals, length_RH_vals):
        S_a = build_sa_exponential(packer, sigma_T=2.0, sigma_RH=sigma_RH,
                                    length_T=1.5, length_RH=len_RH)
        label = f"sa_sigRH={sigma_RH}_lenRH={len_RH}"
        r = run_one_config(profiles, indices, fm, packer, S_a, S_e_base, label)
        all_results.append(r)
        print(f"  {label:30s} | RH: {r['RH_prior_mean']:.2f}->{r['RH_post_mean']:.2f}% "
              f"({r['RH_impr_pct']:+.1f}%) | conv={r['conv_rate']:.0%} | "
              f"T: {r['T_post_mean']:.2f}K")

    # ================================================================
    # Experiment 2: S_e K/V band weight sweep
    # ================================================================
    print(f"\n{'='*70}")
    print("Experiment 2: S_e K/V band weight sweep")
    print("=" * 70)

    S_a_best = build_sa_exponential(packer, sigma_T=2.0, sigma_RH=8.0,
                                     length_T=1.5, length_RH=0.75)

    k_weights = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
    v_weights = [0.5, 1.0, 1.5, 2.0]

    for kw, vw in itertools.product(k_weights, v_weights):
        sigmas = []
        for f in config.ALL_CHANNELS:
            sigmas.append(1.5 * kw if f < 40 else 0.5 * vw)
        S_e = build_se_from_channels(sigmas)
        label = f"se_Kx{kw}_Vx{vw}"
        r = run_one_config(profiles, indices, fm, packer, S_a_best, S_e, label)
        all_results.append(r)
        print(f"  {label:30s} | RH: {r['RH_prior_mean']:.2f}->{r['RH_post_mean']:.2f}% "
              f"({r['RH_impr_pct']:+.1f}%) | conv={r['conv_rate']:.0%}")

    # ================================================================
    # Experiment 3: Channel subset (K-only, V-only, full)
    # ================================================================
    print(f"\n{'='*70}")
    print("Experiment 3: Channel subset sensitivity")
    print("=" * 70)

    channel_configs = {
        "full_14ch": config.ALL_CHANNELS,
        "k_only_7ch": config.MWR_CHANNELS["K_band"],
        "v_only_7ch": config.MWR_CHANNELS["V_band"],
    }

    for name, freqs in channel_configs.items():
        fm_sub = ForwardModel(
            backend=args.forward,
            frequencies=freqs,
            arts_command=args.arts_command,
            elevation_angle_deg=args.elevation_angle,
        )
        S_e_sub = build_se_diagonal(fm_sub, sigma_K=1.5, sigma_V=0.5)
        label = f"ch_{name}"
        r = run_one_config(profiles, indices, fm_sub, packer, S_a_best, S_e_sub, label)
        all_results.append(r)
        print(f"  {label:30s} | RH: {r['RH_prior_mean']:.2f}->{r['RH_post_mean']:.2f}% "
              f"({r['RH_impr_pct']:+.1f}%) | conv={r['conv_rate']:.0%}")

    # ================================================================
    # Save results
    # ================================================================
    out_dir = os.path.join(_PROJECT_ROOT, "results", "oem_sensitivity_rh")
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "sensitivity_scan.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nResults saved to: {csv_path}")

    # Best configs
    best_rh = max(all_results, key=lambda r: r["RH_impr_pct"])
    best_t = max(all_results, key=lambda r: r["T_impr_pct"])
    print(f"\nBest RH: {best_rh['config']} (RH impr={best_rh['RH_impr_pct']:+.1f}%)")
    print(f"Best T:  {best_t['config']} (T impr={best_t['T_impr_pct']:+.1f}%)")
    print("Done.")


if __name__ == "__main__":
    main()
