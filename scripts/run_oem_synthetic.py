#!/usr/bin/env python
"""OEM synthetic closure test — verify algorithm correctness (Plan §12.1).

Generates a synthetic truth profile, computes H(x_true), adds noise, perturbs
the background, then runs OEM retrieval.  Produces diagnostic figures.

Usage:
    py scripts/run_oem_synthetic.py --forward arts --arts-command "python run_arts_profile.py"
    py scripts/run_oem_synthetic.py --forward simple
"""

import sys, os
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.join(_SCRIPT_DIR, "..")
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from forward_model import ForwardModel
from oem_state import make_default_packer
from oem_covariance import build_sa_exponential, build_se_diagonal
from oem import OEMSolver


def generate_synthetic_truth(seed=42):
    """Create a realistic synthetic atmospheric profile."""
    np.random.seed(seed)
    heights = np.array(config.HEIGHT_GRID)

    # Realistic temperature: surface ~290K, lapse ~6.5 K/km + noise
    T = 290.0 - 6.5 * heights / 1000.0 + np.random.randn(93) * 0.5

    # Realistic RH: 70% at surface, exponential decay + noise
    RH = 70.0 * np.exp(-heights / 3000.0) + np.random.randn(93) * 2.0
    RH = np.clip(RH, 5.0, 100.0)

    # Standard atmosphere pressure
    P = 1013.25 * np.exp(-heights / 8000.0)

    return {
        "T": T, "P_hPa": P, "RH": RH,
        "CLWC": np.zeros(93), "height": heights,
    }


def perturb_background(x_true, packer, T_noise=3.0, RH_noise=10.0, seed=123):
    """Add Gaussian noise to create a perturbed background."""
    np.random.seed(seed)
    pert_T = np.random.randn(packer.n_T) * T_noise
    pert_RH = np.random.randn(packer.n_RH) * RH_noise
    x_a = x_true + np.concatenate([pert_T, pert_RH])
    x_a[packer.n_T:] = np.clip(x_a[packer.n_T:], 2.0, 98.0)
    return x_a


def run_closure_test(noise_levels=None, forward_backend=None, arts_command=None,
                     elevation_angle_deg=90.0):
    """Run closure test at multiple noise levels and report results."""
    if noise_levels is None:
        noise_levels = [0.1, 0.5, 1.0, 2.0]

    fm = ForwardModel(
        backend=forward_backend or config.DEFAULT_FORWARD_BACKEND,
        arts_command=arts_command,
        elevation_angle_deg=elevation_angle_deg,
    )
    packer = make_default_packer()
    profile_true = generate_synthetic_truth()
    x_true = packer.pack(profile_true)
    y_true = fm.simulate(profile_true)

    x_a = perturb_background(x_true, packer)

    S_a = build_sa_exponential(packer, sigma_T=2.0, sigma_RH=8.0)

    results = {}
    for noise_std in noise_levels:
        np.random.seed(999)
        y_obs = y_true + np.random.randn(fm.n_channels) * noise_std
        S_e = build_se_diagonal(fm, sigma_K=noise_std * 3, sigma_V=noise_std)

        solver = OEMSolver(fm, packer)
        result = solver.retrieve(
            y_obs=y_obs, x_a=x_a, S_a=S_a, S_e=S_e, verbose=False,
        )
        results[noise_std] = result

        d_ret = np.linalg.norm(result["x_retrieved"] - x_true)
        d_pri = np.linalg.norm(x_a - x_true)
        impr = (1 - d_ret / d_pri) * 100 if d_pri > 0 else 0

        print(
            f"noise={noise_std:.1f}K | "
            f"conv={result['converged']} iter={result['n_iter']:2d} | "
            f"DOFS={result['dofs']:.2f} | "
            f"|dx|: prior={d_pri:.2f} post={d_ret:.2f} ({impr:+.1f}%) | "
            f"BT RMS: {np.sqrt(np.mean((result['y_sim_retrieved']-y_obs)**2)):.4f} K"
        )

    return results, packer, profile_true, x_true, x_a


def make_diagnostic_plots(results, packer, profile_true, x_true, x_a, out_dir):
    """Generate diagnostic figures per Plan §11."""
    os.makedirs(out_dir, exist_ok=True)
    heights = np.array(config.HEIGHT_GRID)

    # Pick the 0.5 K noise case for detailed plotting
    key = 0.5
    if key not in results:
        key = list(results.keys())[0]
    r = results[key]
    y_obs = r["y_obs"]

    # --- Figure 1: Profile comparison ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 8))

    prof_true = profile_true
    prof_bg = packer.unpack(x_a)
    prof_ret = packer.unpack(r["x_retrieved"])

    # Temperature
    ax = axes[0]
    ax.plot(prof_true["T"], heights / 1000, "k-", lw=2, label="Truth")
    ax.plot(prof_bg["T"], heights / 1000, "r--", lw=1.5, label="Background")
    ax.plot(prof_ret["T"], heights / 1000, "b-", lw=1.5, label="Retrieved")
    ax.set_xlabel("Temperature [K]")
    ax.set_ylabel("Height [km]")
    ax.set_title("Temperature Profile")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # RH
    ax = axes[1]
    ax.plot(prof_true["RH"], heights / 1000, "k-", lw=2, label="Truth")
    ax.plot(prof_bg["RH"], heights / 1000, "r--", lw=1.5, label="Background")
    ax.plot(prof_ret["RH"], heights / 1000, "b-", lw=1.5, label="Retrieved")
    ax.set_xlabel("RH [%]")
    ax.set_ylabel("Height [km]")
    ax.set_title("Relative Humidity Profile")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"OEM Closure Test (noise={key}K)", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "profile_comparison.png"), dpi=150)
    plt.close(fig)

    # --- Figure 2: BT residual ---
    fig, ax = plt.subplots(figsize=(10, 5))
    channels = np.arange(len(config.ALL_CHANNELS))
    ax.bar(channels - 0.2, r["y_sim_background"] - y_obs, 0.35,
           label="Background - Obs", color="red", alpha=0.7)
    ax.bar(channels + 0.2, r["y_sim_retrieved"] - y_obs, 0.35,
           label="Retrieved - Obs", color="blue", alpha=0.7)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("Channel Index")
    ax.set_ylabel("BT Residual [K]")
    ax.set_title(f"Brightness Temperature Residuals (noise={key}K)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "bt_residual.png"), dpi=150)
    plt.close(fig)

    # --- Figure 3: Cost history ---
    n_plots = len(results)
    fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]
    for ax, (noise, r) in zip(axes, results.items()):
        ax.plot(r["cost_history"], "o-", markersize=4)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("J_total")
        ax.set_title(f"Cost (noise={noise}K)")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Cost Function History", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "cost_history.png"), dpi=150)
    plt.close(fig)

    # --- Figure 4: Averaging kernel ---
    fig, ax = plt.subplots(figsize=(8, 7))
    A = r["averaging_kernel"]
    im = ax.imshow(A, cmap="RdBu_r", aspect="auto", vmin=-0.5, vmax=1.0)
    labels = packer.state_labels()
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_title(f"Averaging Kernel (DOFS={r['dofs']:.2f})")
    plt.colorbar(im, ax=ax, label="A[i,j]")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "averaging_kernel.png"), dpi=150)
    plt.close(fig)

    # --- Figure 5: Posterior uncertainty ---
    fig, ax = plt.subplots(figsize=(8, 4))
    post_std = np.sqrt(np.diag(r["posterior_covariance"]))
    prior_std = np.sqrt(np.diag(
        build_sa_exponential(packer, sigma_T=2.0, sigma_RH=8.0)
    ))
    x_idx = np.arange(len(post_std))
    ax.bar(x_idx - 0.15, prior_std, 0.3, label="Prior", color="gray", alpha=0.7)
    ax.bar(x_idx + 0.15, post_std, 0.3, label="Posterior", color="green", alpha=0.7)
    ax.set_xticks(x_idx)
    ax.set_xticklabels(packer.state_labels(), rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Std Dev")
    ax.set_title(f"Prior vs Posterior Uncertainty (noise={key}K)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "uncertainty.png"), dpi=150)
    plt.close(fig)

    print(f"\nDiagnostic figures saved to: {out_dir}/")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="OEM synthetic closure test")
    parser.add_argument("--forward", default=config.DEFAULT_FORWARD_BACKEND,
                        choices=["arts", "simple", "monortm"])
    parser.add_argument("--arts-command", default=None,
                        help="External ARTS runner command; reads profile JSON on stdin")
    parser.add_argument("--elevation-angle", type=float, default=90.0,
                        help="Ground-based elevation angle for ARTS [deg]")
    args = parser.parse_args()

    out_dir = os.path.join(_PROJECT_ROOT, "results", "oem_synthetic")

    print("=" * 65)
    print("  OEM Synthetic Closure Test")
    print("=" * 65)
    print()

    results, packer, profile_true, x_true, x_a = run_closure_test(
        noise_levels=[0.1, 0.5, 1.0, 2.0],
        forward_backend=args.forward,
        arts_command=args.arts_command,
        elevation_angle_deg=args.elevation_angle,
    )

    print("\nGenerating diagnostic figures...")
    make_diagnostic_plots(results, packer, profile_true, x_true, x_a, out_dir)

    # Summary
    print("\n" + "=" * 65)
    print("  Closure Test Summary")
    print("=" * 65)
    for noise, r in results.items():
        d_ret = np.linalg.norm(r["x_retrieved"] - x_true)
        d_pri = np.linalg.norm(x_a - x_true)
        impr = (1 - d_ret / d_pri) * 100
        bt_ret = np.sqrt(np.mean((r["y_sim_retrieved"] - r["y_obs"]) ** 2))
        bt_bg = np.sqrt(np.mean((r["y_sim_background"] - r["y_obs"]) ** 2))
        print(
            f"  noise={noise:.1f}K: "
            f"conv={str(r['converged']):5s} "
            f"iter={r['n_iter']:2d} "
            f"DOFS={r['dofs']:5.2f} "
            f"state_impr={impr:+5.1f}% "
            f"BT_RMS: {bt_bg:.3f}->{bt_ret:.3f}K"
        )

    # Success criteria check
    print("\n  Success criteria (Plan 12.1):")
    sample = results[0.5]
    d_ret = np.linalg.norm(sample["x_retrieved"] - x_true)
    d_pri = np.linalg.norm(x_a - x_true)
    bt_ret = np.sqrt(np.mean((sample["y_sim_retrieved"] - sample["y_obs"]) ** 2))
    bt_bg = np.sqrt(np.mean((sample["y_sim_background"] - sample["y_obs"]) ** 2))
    checks = [
        ("posterior RMSE < prior RMSE", d_ret < d_pri),
        ("BT residual improved", bt_ret < bt_bg),
        ("cost monotonic decreasing",
         all(np.diff(sample["cost_history"]) <= 0)),
        ("converged", sample["converged"]),
    ]
    all_pass = True
    for desc, ok in checks:
        mark = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"    [{mark}] {desc}")
    print(f"\n  Overall: {'ALL PASSED' if all_pass else 'SOME FAILED'}")
    print()


if __name__ == "__main__":
    main()
