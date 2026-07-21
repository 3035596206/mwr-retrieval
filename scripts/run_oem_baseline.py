#!/usr/bin/env python3
"""OEM baseline script — fixed config, reproducible, scalable.

Purpose:
  - Freeze all OEM parameters (seed, S_a, S_e, convergence thresholds)
  - Run self-consistent MonoRTM OEM on ERA5 2013-01
  - Scale to n=100/200/500/744
  - Save per-sample diagnostics (prior/posterior, BT, AK, DOFS, cost)
  - Track failure samples explicitly

Usage:
  python scripts/run_oem_baseline.py --n-samples 100
  python scripts/run_oem_baseline.py --n-samples 500 --seed 42
"""

import sys, os, argparse, pickle, time, json, warnings
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.join(_SCRIPT_DIR, "..")
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import config
from forward_model import ForwardModel
from oem_state import make_default_packer
from oem_covariance import build_sa_exponential, build_se_diagonal
from oem import OEMSolver

warnings.filterwarnings("ignore")

# ============================================================
# Frozen baseline configuration
# ============================================================
BASELINE_CONFIG = {
    "random_seed": 42,
    "state_mode": "coarse",
    "coarse_T": [500, 1000, 2000, 3000, 5000, 8000, 10000],
    "coarse_RH": [500, 1000, 2000, 3000, 5000, 8000, 10000],
    "sa_type": "exponential",
    "sigma_T": 2.0,
    "sigma_RH": 8.0,
    "se_type": "diagonal",
    "sigma_K": 1.5,
    "sigma_V": 0.5,
    "noise_std": 0.5,
    "max_iter": 15,
    "cost_tol": 1e-3,
    "dx_tol": 1e-4,
    "gamma_init": 1.0,
    "forward_backend": "simple",
    "self_consistent": True,
    "perturb_T_std": 3.0,
    "perturb_RH_std": 10.0,
}


def load_data():
    profiles_path = os.path.join(_PROJECT_ROOT, "data", "era5", "era5_profiles_201301_poc.pkl")
    bt_path = os.path.join(_PROJECT_ROOT, "data", "era5", "era5_bt_sim_201301_poc.pkl")
    with open(profiles_path, "rb") as f:
        profiles = pickle.load(f)
    with open(bt_path, "rb") as f:
        bt_sim = pickle.load(f)
    return profiles, bt_sim


def run_baseline(n_samples, cfg, out_dir, verbose=False):
    """Run OEM baseline with frozen configuration."""
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "baseline_config.json"), "w") as f:
        json.dump(cfg, f, indent=2, default=str)

    profiles, bt_sim = load_data()
    np.random.seed(cfg["random_seed"])
    n_total = profiles["T"].shape[0]
    n_samples = min(n_samples, n_total)
    indices = sorted(np.random.choice(n_total, n_samples, replace=False))

    fm = ForwardModel(backend=cfg["forward_backend"], monortm_path=cfg.get("monortm_path"), tape3_path=cfg.get("tape3_path"))
    packer = make_default_packer()
    S_a = build_sa_exponential(packer, sigma_T=cfg["sigma_T"], sigma_RH=cfg["sigma_RH"])
    S_e = build_se_diagonal(fm, sigma_K=cfg["sigma_K"], sigma_V=cfg["sigma_V"])
    solver = OEMSolver(fm, packer)

    samples = []
    failures = []
    t_start = time.time()

    for count, idx in enumerate(indices):
        heights = profiles["height"]
        profile_true = {
            "T": profiles["T"][idx],
            "RH": profiles["RH"][idx],
            "CLWC": profiles["CLWC"][idx],
            "P_hPa": 1013.25 * np.exp(-heights / 8000.0),
            "height": heights,
        }

        x_true = packer.pack(profile_true)

        np.random.seed(idx * 3 + 1)
        pert_T = np.random.randn(packer.n_T) * cfg["perturb_T_std"]
        pert_RH = np.random.randn(packer.n_RH) * cfg["perturb_RH_std"]
        x_a = x_true + np.concatenate([pert_T, pert_RH])
        x_a[packer.n_T:] = np.clip(x_a[packer.n_T:], 2.0, 98.0)

        y_true = fm.simulate(profile_true)
        np.random.seed(idx * 7 + 3)
        y_obs = y_true + np.random.randn(fm.n_channels) * cfg["noise_std"]

        try:
            result = solver.retrieve(
                y_obs=y_obs, x_a=x_a, S_a=S_a, S_e=S_e,
                max_iter=cfg["max_iter"], cost_tol=cfg["cost_tol"],
                dx_tol=cfg["dx_tol"], gamma_init=cfg["gamma_init"],
                verbose=verbose,
            )
        except Exception as e:
            failures.append({"index": int(idx), "error": str(e)})
            continue

        prof_ret = packer.unpack(result["x_retrieved"])
        prof_bg = packer.unpack(x_a)

        sample = {
            "index": int(idx),
            "converged": bool(result["converged"]),
            "n_iter": int(result["n_iter"]),
            "exit_reason": result.get("exit_reason", "unknown"),
            "dofs": float(result["dofs"]),
            "cost_final": float(result["cost_history"][-1]),
            "cost_initial": float(result["cost_history"][0]),
            "bt_rms_prior": float(np.sqrt(np.mean(
                (result["y_sim_background"] - y_obs) ** 2))),
            "bt_rms_post": float(np.sqrt(np.mean(
                (result["y_sim_retrieved"] - y_obs) ** 2))),
            "T_rmse_prior": float(np.sqrt(np.mean(
                (prof_bg["T"] - profile_true["T"]) ** 2))),
            "T_rmse_post": float(np.sqrt(np.mean(
                (prof_ret["T"] - profile_true["T"]) ** 2))),
            "RH_rmse_prior": float(np.sqrt(np.mean(
                (prof_bg["RH"] - profile_true["RH"]) ** 2))),
            "RH_rmse_post": float(np.sqrt(np.mean(
                (prof_ret["RH"] - profile_true["RH"]) ** 2))),
            "x_retrieved": result["x_retrieved"],
            "x_background": x_a,
            "x_true": x_true,
            "averaging_kernel": result["averaging_kernel"],
            "posterior_covariance": result["posterior_covariance"],
            "jacobian": result["jacobian"],
        }
        samples.append(sample)

        if (count + 1) % max(1, n_samples // 10) == 0 or count == n_samples - 1:
            elapsed = time.time() - t_start
            rate = (count + 1) / elapsed if elapsed > 0 else 0
            conv = sum(1 for s in samples if s["converged"]) / len(samples)
            print(f"  [{count + 1:4d}/{n_samples}] "
                  f"{rate:.1f} prof/s | conv={conv:.1%} | {elapsed:.0f}s")

    elapsed = time.time() - t_start
    conv_samples = [s for s in samples if s["converged"]]

    stats = {
        "n_total": len(samples),
        "n_converged": len(conv_samples),
        "n_failed": len(failures),
        "converged_rate": len(conv_samples) / max(len(samples), 1),
        "elapsed_sec": elapsed,
        "rate_pp_sec": len(samples) / elapsed if elapsed > 0 else 0,
    }

    if conv_samples:
        for key in ["T_rmse_prior", "T_rmse_post", "RH_rmse_prior",
                     "RH_rmse_post", "bt_rms_prior", "bt_rms_post",
                     "dofs", "n_iter"]:
            vals = [s[key] for s in conv_samples]
            stats[f"{key}_mean"] = float(np.mean(vals))
            stats[f"{key}_std"] = float(np.std(vals))

    nan_count = 0
    for s in samples:
        if np.any(~np.isfinite(s["x_retrieved"])):
            nan_count += 1
    stats["nan_inf_count"] = nan_count

    print(f"\n{'='*65}")
    print(f"  Baseline Results (n={n_samples}, {cfg['forward_backend']})")
    print(f"{'='*65}")
    print(f"  Samples:      {stats['n_total']} ({stats['n_failed']} failed)")
    print(f"  Converged:    {stats['n_converged']} ({stats['converged_rate']:.1%})")
    if nan_count:
        print(f"  NaN/Inf:      {nan_count} (WARNING)")
    print(f"  Elapsed:      {elapsed:.0f}s ({stats['rate_pp_sec']:.2f} prof/s)")
    if conv_samples:
        print(f"  T  RMSE: prior={stats['T_rmse_prior_mean']:.4f}K "
              f"-> post={stats['T_rmse_post_mean']:.4f}K")
        print(f"  RH RMSE: prior={stats['RH_rmse_prior_mean']:.2f}% "
              f"-> post={stats['RH_rmse_post_mean']:.2f}%")
        print(f"  BT RMS:  prior={stats['bt_rms_prior_mean']:.4f}K "
              f"-> post={stats['bt_rms_post_mean']:.4f}K")
        print(f"  DOFS:    {stats['dofs_mean']:.2f} +/- {stats['dofs_std']:.2f}")
        print(f"  Avg iter:{stats['n_iter_mean']:.1f}")

        t_impr = (1 - stats['T_rmse_post_mean'] /
                  max(stats['T_rmse_prior_mean'], 0.001)) * 100
        rh_impr = (1 - stats['RH_rmse_post_mean'] /
                   max(stats['RH_rmse_prior_mean'], 0.001)) * 100
        bt_impr = (1 - stats['bt_rms_post_mean'] /
                   max(stats['bt_rms_prior_mean'], 0.001)) * 100
        print(f"  Improvement: T={t_impr:+.1f}%  RH={rh_impr:+.1f}%  BT={bt_impr:+.1f}%")

    with open(os.path.join(out_dir, "baseline_stats.json"), "w") as f:
        json.dump(stats, f, indent=2, default=str)
    with open(os.path.join(out_dir, "baseline_samples.pkl"), "wb") as f:
        pickle.dump(samples, f)
    if failures:
        with open(os.path.join(out_dir, "baseline_failures.json"), "w") as f:
            json.dump(failures, f, indent=2, default=str)

    print(f"\nResults saved to: {out_dir}/")
    return stats, samples, failures


def main():
    parser = argparse.ArgumentParser(description="OEM Baseline (fixed config)")
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--forward", type=str, default="simple",
                        choices=["simple", "monortm"])
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--monortm-path", type=str, default=None,
                        help="Path to MonoRTM executable")
    parser.add_argument("--tape3-path", type=str, default=None,
                        help="Path to TAPE3 binary file")
    args = parser.parse_args()

    cfg = dict(BASELINE_CONFIG)
    cfg["random_seed"] = args.seed
    cfg["forward_backend"] = args.forward
    cfg["monortm_path"] = args.monortm_path
    cfg["tape3_path"] = args.tape3_path
    if args.forward != "simple":
        cfg["self_consistent"] = True

    exp_name = f"oem_baseline_{args.forward}_n{args.n_samples}_seed{args.seed}"
    out_dir = os.path.join(_PROJECT_ROOT, "results", exp_name)

    print(f"OEM Baseline: {args.forward}, n={args.n_samples}, seed={args.seed}")
    print(f"Output: {out_dir}\n")

    run_baseline(args.n_samples, cfg, out_dir, verbose=args.verbose)
    print("Done.")


if __name__ == "__main__":
    main()
