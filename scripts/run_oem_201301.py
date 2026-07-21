#!/usr/bin/env python
"""OEM 2013-01 ERA5 POC — first real-data OEM experiment (Plan §12.2).

Uses ERA5 profiles as truth, simulated BT as observation, and perturbed
profiles as background.  Demonstrates OEM on real atmospheric variability.

Usage:
    py scripts/run_oem_201301.py
    py scripts/run_oem_201301.py --n-samples 50 --noise 1.0
"""

import sys, os
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.join(_SCRIPT_DIR, "..")
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))

import argparse
import pickle
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from forward_model import ForwardModel
from oem_state import make_default_packer
from oem_covariance import build_sa_exponential, build_se_diagonal
from oem import OEMSolver


def load_data():
    """Load ERA5 profiles and simulated BT for January 2013."""
    profiles_path = os.path.join(
        _PROJECT_ROOT, "data", "era5", "era5_profiles_201301_poc.pkl"
    )
    bt_path = os.path.join(
        _PROJECT_ROOT, "data", "era5", "era5_bt_sim_201301_poc.pkl"
    )

    with open(profiles_path, "rb") as f:
        profiles = pickle.load(f)
    with open(bt_path, "rb") as f:
        bt_sim = pickle.load(f)

    print(f"Loaded {profiles['T'].shape[0]} profiles, {bt_sim.shape[0]} BT samples")
    print(f"BT shape: {bt_sim.shape}")
    return profiles, bt_sim


def select_samples(profiles, bt_sim, n_samples, seed=42):
    """Select a subset of profiles for the POC run."""
    np.random.seed(seed)
    n_total = profiles["T"].shape[0]
    if n_samples >= n_total:
        return list(range(n_total))
    indices = np.random.choice(n_total, n_samples, replace=False)
    return sorted(indices)


def run_retrievals(profiles, bt_sim, indices, noise_std=0.5, self_consistent=False, max_iter=15, verbose=False,
                   forward_backend="simple", monortm_path=None, tape3_path=None):
    """Run OEM retrieval on selected profiles.

    Args:
        profiles: dict with T, RH, CLWC, height arrays
        bt_sim: (n_total, 14) simulated BT (observations)
        indices: list of indices to process
        noise_std: observation noise std to add [K]
        max_iter: max OEM iterations
        verbose: print per-profile diagnostics

    Returns:
        results: list of result dicts
        x_a_list: list of background state vectors
        x_true_list: list of truth state vectors
    """
    if forward_backend == "monortm":
        fm = ForwardModel(backend="monortm", monortm_path=monortm_path, tape3_path=tape3_path)
    else:
        fm = ForwardModel(backend="simple")
    packer = make_default_packer()
    S_a = build_sa_exponential(packer, sigma_T=2.0, sigma_RH=8.0)
    S_e = build_se_diagonal(fm, sigma_K=1.5, sigma_V=0.5)

    solver = OEMSolver(fm, packer)

    results = []
    x_a_list = []
    x_true_list = []

    n_total = len(indices)
    t_start = time.time()

    for count, idx in enumerate(indices):
        # Build truth profile
        heights = profiles["height"]
        profile_true = {
            "T": profiles["T"][idx],
            "RH": profiles["RH"][idx],
            "CLWC": profiles["CLWC"][idx],
            "P_hPa": 1013.25 * np.exp(-heights / 8000.0),
            "height": heights,
        }

        # True state vector
        x_true = packer.pack(profile_true)
        x_true_list.append(x_true)

        # Perturbed background: add 3K T noise, 10% RH noise
        np.random.seed(idx * 3 + 1)
        pert_T = np.random.randn(packer.n_T) * 3.0
        pert_RH = np.random.randn(packer.n_RH) * 10.0
        x_a = x_true + np.concatenate([pert_T, pert_RH])
        x_a[packer.n_T:] = np.clip(x_a[packer.n_T:], 2.0, 98.0)
        x_a_list.append(x_a)

        # Observation: self-consistent or stored BT + noise
        if self_consistent:
            y_true = fm.simulate(profile_true)
        else:
            y_true = bt_sim[idx]
        np.random.seed(idx * 7 + 3)
        y_obs = y_true + np.random.randn(fm.n_channels) * noise_std

        # Run retrieval
        result = solver.retrieve(
            y_obs=y_obs, x_a=x_a, S_a=S_a, S_e=S_e,
            max_iter=max_iter, verbose=verbose,
        )
        results.append(result)

        if (count + 1) % max(1, n_total // 10) == 0 or count == n_total - 1:
            elapsed = time.time() - t_start
            rate = (count + 1) / elapsed if elapsed > 0 else 0
            conv_rate = sum(1 for r in results if r["converged"]) / len(results)
            print(
                f"  [{count + 1:4d}/{n_total}] "
                f"{rate:.1f} profiles/s | "
                f"conv={conv_rate:.1%} | "
                f"elapsed={elapsed:.0f}s"
            )

    elapsed = time.time() - t_start
    conv_count = sum(1 for r in results if r["converged"])
    print(f"\n  Total: {len(results)} profiles in {elapsed:.1f}s")
    print(f"  Converged: {conv_count}/{len(results)} ({conv_count/len(results):.1%})")

    return results, x_a_list, x_true_list


def compute_statistics(results, x_a_list, x_true_list, packer):
    """Compute prior vs posterior statistics in physical and control space."""
    heights = np.array(config.HEIGHT_GRID)
    n = len(results)
    n_layers = config.N_LAYERS

    # Accumulators for profile-space RMSE
    T_prior_err = np.zeros((n, n_layers))
    T_post_err = np.zeros((n, n_layers))
    RH_prior_err = np.zeros((n, n_layers))
    RH_post_err = np.zeros((n, n_layers))

    # Control-space errors
    dx_prior = np.zeros(n)
    dx_post = np.zeros(n)

    # BT residuals
    bt_prior_rms = np.zeros(n)
    bt_post_rms = np.zeros(n)

    dofs_list = np.zeros(n)
    converged_list = np.zeros(n, dtype=bool)
    n_iter_list = np.zeros(n, dtype=int)

    for i in range(n):
        r = results[i]
        x_true = x_true_list[i]
        x_a = x_a_list[i]

        # Profile-space
        prof_true = packer.unpack(x_true)
        prof_bg = packer.unpack(x_a)
        prof_ret = packer.unpack(r["x_retrieved"])

        T_prior_err[i] = prof_bg["T"] - prof_true["T"]
        T_post_err[i] = prof_ret["T"] - prof_true["T"]
        RH_prior_err[i] = prof_bg["RH"] - prof_true["RH"]
        RH_post_err[i] = prof_ret["RH"] - prof_true["RH"]

        # Control-space
        dx_prior[i] = np.linalg.norm(x_a - x_true)
        dx_post[i] = np.linalg.norm(r["x_retrieved"] - x_true)

        # BT residuals
        bt_prior_rms[i] = np.sqrt(np.mean(
            (r["y_sim_background"] - r["y_obs"]) ** 2
        ))
        bt_post_rms[i] = np.sqrt(np.mean(
            (r["y_sim_retrieved"] - r["y_obs"]) ** 2
        ))

        dofs_list[i] = r["dofs"]
        converged_list[i] = r["converged"]
        n_iter_list[i] = r["n_iter"]

    stats = {
        "T_prior_rmse": np.sqrt(np.mean(T_prior_err ** 2, axis=0)),   # (93,)
        "T_post_rmse": np.sqrt(np.mean(T_post_err ** 2, axis=0)),
        "RH_prior_rmse": np.sqrt(np.mean(RH_prior_err ** 2, axis=0)),
        "RH_post_rmse": np.sqrt(np.mean(RH_post_err ** 2, axis=0)),
        "T_prior_rmse_mean": float(np.sqrt(np.mean(T_prior_err ** 2))),
        "T_post_rmse_mean": float(np.sqrt(np.mean(T_post_err ** 2))),
        "RH_prior_rmse_mean": float(np.sqrt(np.mean(RH_prior_err ** 2))),
        "RH_post_rmse_mean": float(np.sqrt(np.mean(RH_post_err ** 2))),
        "dx_prior_mean": float(np.mean(dx_prior)),
        "dx_post_mean": float(np.mean(dx_post)),
        "bt_prior_rms_mean": float(np.mean(bt_prior_rms)),
        "bt_post_rms_mean": float(np.mean(bt_post_rms)),
        "dofs_mean": float(np.mean(dofs_list)),
        "converged_rate": float(np.mean(converged_list)),
        "n_iter_mean": float(np.mean(n_iter_list)),
        "heights": heights,
        # Per-profile data for scatter
        "dx_prior": dx_prior,
        "dx_post": dx_post,
        "bt_prior_rms": bt_prior_rms,
        "bt_post_rms": bt_post_rms,
        "dofs_list": dofs_list,
    }

    return stats


def make_poc_plots(stats, out_dir, noise_std):
    """Generate POC diagnostic figures."""
    os.makedirs(out_dir, exist_ok=True)
    heights = stats["heights"]

    # --- Figure 1: RMSE profiles (prior vs posterior) ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 8))

    ax = axes[0]
    ax.plot(stats["T_prior_rmse"], heights / 1000, "r--", lw=1.5, label="Prior")
    ax.plot(stats["T_post_rmse"], heights / 1000, "b-", lw=1.5, label="Posterior")
    ax.set_xlabel("T RMSE [K]")
    ax.set_ylabel("Height [km]")
    ax.set_title(f"T RMSE (prior={stats['T_prior_rmse_mean']:.2f}K, post={stats['T_post_rmse_mean']:.2f}K)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(stats["RH_prior_rmse"], heights / 1000, "r--", lw=1.5, label="Prior")
    ax.plot(stats["RH_post_rmse"], heights / 1000, "b-", lw=1.5, label="Posterior")
    ax.set_xlabel("RH RMSE [%]")
    ax.set_ylabel("Height [km]")
    ax.set_title(f"RH RMSE (prior={stats['RH_prior_rmse_mean']:.1f}%, post={stats['RH_post_rmse_mean']:.1f}%)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"OEM 2013-01 POC: Prior vs Posterior RMSE (n={len(stats['dx_prior'])}, noise={noise_std}K)", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "rmse_profiles.png"), dpi=150)
    plt.close(fig)

    # --- Figure 2: BT residual scatter ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.scatter(stats["bt_prior_rms"], stats["bt_post_rms"], alpha=0.3, s=10)
    lim = max(stats["bt_prior_rms"].max(), stats["bt_post_rms"].max()) * 1.1
    ax.plot([0, lim], [0, lim], "k--", lw=0.5)
    ax.set_xlabel("Prior BT RMS [K]")
    ax.set_ylabel("Posterior BT RMS [K]")
    ax.set_title(f"BT Residual Improvement")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)

    ax = axes[1]
    ax.hist(stats["dofs_list"], bins=30, color="steelblue", edgecolor="white")
    ax.axvline(stats["dofs_mean"], color="red", linestyle="--",
               label=f"mean={stats['dofs_mean']:.2f}")
    ax.set_xlabel("DOFS")
    ax.set_ylabel("Count")
    ax.set_title("Degrees of Freedom for Signal")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "bt_dofs.png"), dpi=150)
    plt.close(fig)

    # --- Figure 3: Averaging kernel of first converged profile ---
    fig, ax = plt.subplots(figsize=(8, 7))
    packer = make_default_packer()
    A = np.eye(packer.n_state)  # fallback
    for r_idx in range(len(stats["dofs_list"])):
        # We need to access results directly; use stats to reconstruct
        pass
    # Use a dummy averaging kernel since we don't have results here
    # This is a placeholder — real AK needs access to results list
    labels = packer.state_labels()
    ax.text(0.5, 0.5, "Averaging kernel data not stored in stats\n"
            "Use evaluate_oem.py for full diagnostics",
            transform=ax.transAxes, ha="center", va="center", fontsize=12)
    ax.set_title("Averaging Kernel (placeholder)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "averaging_kernel.png"), dpi=150)
    plt.close(fig)

    print(f"POC figures saved to: {out_dir}/")


def main():
    parser = argparse.ArgumentParser(description="OEM 2013-01 ERA5 POC")
    parser.add_argument("--n-samples", type=int, default=100,
                        help="Number of profiles to process (default 100, max 744)")
    parser.add_argument("--noise", type=float, default=0.5,
                        help="Observation noise std [K] (default 0.5)")
    parser.add_argument("--max-iter", type=int, default=15,
                        help="Max OEM iterations (default 15)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-profile diagnostics")
    parser.add_argument("--self-consistent", action="store_true",
                        help="Use self-consistent BT (simple RTM) instead of stored BT")
    parser.add_argument("--forward", type=str, default="simple",
                        choices=["simple", "monortm"],
                        help="Forward model backend (default: simple)")
    parser.add_argument("--monortm-path", type=str,
                        default="/mnt/d/project-504/mwr-retrieval-main/bin/monortm_linux",
                        help="Path to MonoRTM executable")
    parser.add_argument("--tape3-path", type=str,
                        default="/mnt/d/project-504/mwr-retrieval-main/data/TAPE3/TAPE3_bin",
                        help="Path to TAPE3 binary file")
    args = parser.parse_args()

    exp_name = "oem_201301"
    if args.self_consistent:
        exp_name += "_self_consistent"
    else:
        exp_name += "_forward_mismatch"
    if args.forward != "simple":
        exp_name += f"_{args.forward}"
    exp_name += f"_n{args.n_samples}"
    out_dir = os.path.join(_PROJECT_ROOT, "results", exp_name)

    print("=" * 65)
    print("  OEM 2013-01 ERA5 POC (Proof of Concept)")
    print("=" * 65)
    print(f"  n_samples={args.n_samples}  noise={args.noise}K  max_iter={args.max_iter}  self_consistent={args.self_consistent}")
    print(f"  forward={args.forward}  out_dir={out_dir}")
    print()

    # Load data
    profiles, bt_sim = load_data()
    indices = select_samples(profiles, bt_sim, args.n_samples)

    # Run retrievals
    print(f"Running OEM on {len(indices)} profiles...")
    results, x_a_list, x_true_list = run_retrievals(
        profiles, bt_sim, indices,
        noise_std=args.noise, max_iter=args.max_iter, verbose=args.verbose,
        self_consistent=args.self_consistent,
        forward_backend=args.forward,
        monortm_path=args.monortm_path, tape3_path=args.tape3_path,
    )

    # Compute statistics
    packer = make_default_packer()
    print("\nComputing statistics...")
    stats = compute_statistics(results, x_a_list, x_true_list, packer)

    # Print summary
    print("\n" + "=" * 65)
    print("  POC Results Summary")
    print("=" * 65)
    print(f"  Samples:          {len(results)}")
    print(f"  Converged:        {stats['converged_rate']:.1%}")
    print(f"  Avg iterations:   {stats['n_iter_mean']:.1f}")
    print(f"  Avg DOFS:         {stats['dofs_mean']:.2f} / {packer.n_state}")
    print()
    print(f"  T  RMSE:  prior={stats['T_prior_rmse_mean']:.4f} K  ->  post={stats['T_post_rmse_mean']:.4f} K")
    print(f"  RH RMSE:  prior={stats['RH_prior_rmse_mean']:.2f} %  ->  post={stats['RH_post_rmse_mean']:.2f} %")
    print(f"  BT RMS:   prior={stats['bt_prior_rms_mean']:.4f} K  ->  post={stats['bt_post_rms_mean']:.4f} K")
    t_impr = (1 - stats['T_post_rmse_mean'] / stats['T_prior_rmse_mean']) * 100
    rh_impr = (1 - stats['RH_post_rmse_mean'] / stats['RH_prior_rmse_mean']) * 100
    bt_impr = (1 - stats['bt_post_rms_mean'] / stats['bt_prior_rms_mean']) * 100
    print(f"  Improvement: T={t_impr:+.1f}%  RH={rh_impr:+.1f}%  BT={bt_impr:+.1f}%")

    # Generate plots
    print("\nGenerating POC figures...")
    make_poc_plots(stats, out_dir, args.noise)

    # Save stats
    stats_path = os.path.join(out_dir, "poc_stats.pkl")
    os.makedirs(out_dir, exist_ok=True)
    with open(stats_path, "wb") as f:
        pickle.dump(stats, f)
    print(f"Statistics saved to: {stats_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
