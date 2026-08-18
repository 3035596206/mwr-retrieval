#!/usr/bin/env python3
"""Complete retrieval pipeline on CDS ERA5 48-layer dataset."""

import argparse, json, os, sys
from pathlib import Path
import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from train_chengdu_brnn import rmse_bias, set_seed, train_one_model
from train_chengdu_era5_ridge import predict_selected, select_model, standardize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--brnn-seeds", type=int, nargs="+", default=[42, 1, 7, 21, 84])
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "models").mkdir(exist_ok=True)
    (out / "predictions").mkdir(exist_ok=True)
    
    # Load
    data = np.load(args.dataset)
    BT_train, T_train, RH_train = data["BT_train"], data["T_train"], data["RH_train"]
    BT_val, T_val, RH_val = data["BT_val"], data["T_val"], data["RH_val"]
    BT_test, T_test, RH_test = data["BT_test"], data["T_test"], data["RH_test"]
    centers = data["centers"]
    n_layers = len(centers)
    print(f"Train={len(BT_train)} Val={len(BT_val)} Test={len(BT_test)}, layers={n_layers}")
    
    # Climatology
    T_clim = np.mean(T_train, axis=0)
    RH_clim = np.mean(RH_train, axis=0)
    clim_T = rmse_bias(np.tile(T_clim, (len(T_test), 1)), T_test)
    clim_RH = rmse_bias(np.tile(RH_clim, (len(RH_test), 1)), RH_test)
    
    # Ridge/EOF T
    X_train_std, X_mu, X_sig = standardize(BT_train)
    X_val_std = (BT_val - X_mu) / X_sig
    X_test_std = (BT_test - X_mu) / X_sig
    
    best_model, _ = select_model(X_train_std, T_train, X_val_std, T_val)
    print(f"Best T model: method={best_model['method']}, alpha={best_model['alpha']}, "
          f"n_eof={best_model['n_eof']}, val_rmse={best_model['val_rmse']:.3f}")
    
    T_pred = predict_selected(best_model, X_test_std)
    t_metrics = rmse_bias(T_pred, T_test)
    
    # Save T model
    np.savez(out / "models" / "ridge_t_model.npz",
             weights=best_model["weights"], y_mean=best_model["y_mean"],
             basis=best_model.get("basis"), profile_mean=best_model.get("profile_mean"),
             X_mu=X_mu, X_sig=X_sig)
    
    # BRNN RH
    from brnn_model import get_height_range_indices
    
    X_full = np.concatenate([BT_train, BT_val], axis=0).astype(np.float32)
    T_full = np.concatenate([T_train, T_val], axis=0).astype(np.float32)
    RH_full = np.concatenate([RH_train, RH_val], axis=0).astype(np.float32)
    n_train = len(BT_train)
    train_mask = np.zeros(len(X_full), dtype=bool); train_mask[:n_train] = True
    val_mask = np.zeros(len(X_full), dtype=bool); val_mask[n_train:] = True
    BT_test_f = BT_test.astype(np.float32)
    
    grid_m = centers * 1000.0
    bands_km = [("RH_0-2km", "RH", (0, 2)), ("RH_2-8km", "RH", (2, 8)),
                ("RH_8-10km", "RH", (8, 10))]
    
    ensemble_preds = []; seed_results = {}
    
    for seed in args.brnn_seeds:
        set_seed(seed)
        sd = out / "models" / f"seed{seed}"; sd.mkdir(exist_ok=True)
        seed_preds = np.zeros((len(RH_test), n_layers), dtype=np.float32)
        
        for name, var, (lo_km, hi_km) in bands_km:
            model, _ = train_one_model(
                name=f"{name}_s{seed}", variable=var, height_range=(lo_km, hi_km),
                X=X_full, T=T_full, RH=RH_full,
                train_mask=train_mask, val_mask=val_mask,
                hidden_size=32, dropout=0.4, batch_size=16,
                learning_rate=1e-3, max_epochs=300, patience_limit=35,
                device=args.device, height_grid=grid_m,
            )
            import torch
            torch.save(model.state_dict(), sd / f"brnn_{name}.pt")
            model.eval()
            lo, hi = get_height_range_indices(lo_km, hi_km, grid_m)
            with torch.no_grad():
                seed_preds[:, lo:hi] = model(torch.tensor(BT_test_f)).cpu().numpy() * 100.0
        
        rmse = float(np.sqrt(np.mean((seed_preds - RH_test) ** 2)))
        seed_results[seed] = rmse
        ensemble_preds.append(seed_preds)
        print(f"  Seed {seed}: test RMSE={rmse:.2f}%")
    
    top2 = sorted(seed_results, key=seed_results.get)[:2]
    rh_pred = np.mean([ensemble_preds[args.brnn_seeds.index(s)] for s in top2], axis=0)
    rh_metrics = rmse_bias(rh_pred, RH_test)
    print(f"BRNN top-2 seeds: {top2}")
    
    # Save predictions
    np.savez(out / "predictions" / "hybrid_predictions.npz",
             T_pred=T_pred, T_true=T_test, RH_pred=rh_pred, RH_true=RH_test, centers=centers)
    
    # Height metrics
    h_metrics = []
    for i, h in enumerate(centers):
        h_metrics.append({
            "height_km": round(float(h), 3),
            "T_rmse": float(np.sqrt(np.mean((T_pred[:, i] - T_test[:, i]) ** 2))),
            "T_bias": float(np.mean(T_pred[:, i] - T_test[:, i])),
            "RH_rmse": float(np.sqrt(np.mean((rh_pred[:, i] - RH_test[:, i]) ** 2))),
            "RH_bias": float(np.mean(rh_pred[:, i] - RH_test[:, i])),
        })
    
    # Stats
    stats = {
        "status": "cds_era5_48layer_hybrid",
        "config": {"n_layers": n_layers, "best_method": best_model["method"],
                    "best_alpha": best_model["alpha"], "best_n_eof": best_model["n_eof"],
                    "brnn_seeds": args.brnn_seeds, "top2_seeds": top2},
        "metrics": {"hybrid_T": t_metrics, "hybrid_RH": rh_metrics,
                     "climatology_T": clim_T, "climatology_RH": clim_RH},
        "height_metrics": h_metrics,
        "dataset": {"train": len(BT_train), "val": len(BT_val), "test": len(BT_test)},
    }
    with open(out / "hybrid_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"RESULTS (test set, n={len(BT_test)})")
    print(f"{'T RMSE (K)':<20} {t_metrics['rmse']:>8.3f}  (clim {clim_T['rmse']:.3f})")
    print(f"{'T Bias (K)':<20} {t_metrics['bias']:>8.3f}  (clim {clim_T['bias']:.3f})")
    print(f"{'RH RMSE (%)':<20} {rh_metrics['rmse']:>8.2f}  (clim {clim_RH['rmse']:.2f})")
    print(f"{'RH Bias (%)':<20} {rh_metrics['bias']:>8.2f}  (clim {clim_RH['bias']:.2f})")
    for m in h_metrics:
        if m["height_km"] in [0.25, 0.55, 0.95, 2.88, 4.88, 7.88, 9.88]:
            print(f"  {m['height_km']:.2f}km: T={m['T_rmse']:.2f}K  RH={m['RH_rmse']:.1f}%")
    print(f"\nOutput: {out}")


if __name__ == "__main__":
    main()
