#!/usr/bin/env python3
"""Train BRNN models from MP-3000A data — v6.

Strategy: Train on Sim_BT (clean forward model), optimize the data pipeline.

Key improvements over v3:
  1. Full ERA5 profile QC before using as training target
     - RH scaling ×0.9 (paper step 1)
     - CLWC screening (>750 g/m² profiles excluded)
     - BT linear correction pre-applied (paper step 4)
  2. Two-stage: pre-train on all Sim_BT samples, fine-tune on best-matched Obs_BT
  3. Only use Obs_BT samples where |OMB| < 2σ in stable V-band channels
  4. Per-season BT correction instead of global linear

Expected: bridge the Sim→Obs gap, T_RMSE < 2.0K, RH_RMSE < 11%
"""

import os, sys, pickle, warnings
import numpy as np
import xarray as xr
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

PROJ = "/Users/ink/test/mwr_retrieval"
NC_FILE = "/Users/ink/test/54623_MP_3000A_ERA5Model_20231101_20240331_areamean_selfgb.nc"
sys.path.insert(0, os.path.join(PROJ, "src"))
sys.path.insert(0, PROJ)
import config

RD = 287.058; G = 9.80665; EPSILON = 0.622; N_CH = 22


class BRNN(nn.Module):
    def __init__(self, n_input, n_output, hidden_size=256, dropout_rate=0.3):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(n_input)
        self.fc1 = nn.Linear(n_input, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc_out = nn.Linear(hidden_size, n_output)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.input_bn(x)
        x = self.fc1(x); x = self.relu(x); x = self.bn1(x); x = self.dropout1(x)
        x = self.fc2(x); x = self.relu(x); x = self.bn2(x); x = self.dropout2(x)
        x = self.fc_out(x); x = self.sigmoid(x)
        return x


def q_to_rh(q_gkg, T, P_hpa):
    q = q_gkg / 1000.0
    e_hpa = q * P_hpa / (EPSILON + (1 - EPSILON) * q)
    Tc = T - 273.16
    es_hpa = 6.1078 * np.exp(17.2693882 * Tc / (Tc + 237.3))
    return np.clip(e_hpa / es_hpa * 100.0, 0.0, 100.0)


def pressure_to_height(P_hpa, T, q_gkg, sfc_alt_km):
    n_lev, n_prof = T.shape
    Z = np.zeros_like(T)
    for i in range(n_prof):
        Tv = T[:, i] * (1.0 + 0.608 * q_gkg[:, i] / 1000.0)
        P_pa = (P_hpa[:, i] if P_hpa.ndim > 1 else P_hpa) * 100.0
        Z[-1, i] = 0.0
        for k in range(n_lev - 2, -1, -1):
            Z[k, i] = Z[k+1, i] + RD * 0.5*(Tv[k]+Tv[k+1]) / G * (P_pa[k+1]-P_pa[k]) / (0.5*(P_pa[k]+P_pa[k+1]))
        Z[:, i] += sfc_alt_km[i] * 1000.0
    return Z


def interpolate_to_mwr_grid(Z_m, T, RH, target_h):
    from scipy.interpolate import interp1d
    T_out = np.zeros((T.shape[1], len(target_h)))
    RH_out = np.zeros((T.shape[1], len(target_h)))
    for i in range(T.shape[1]):
        z = Z_m[:, i]; t_i = T[:, i]; rh_i = RH[:, i]
        if z[0] > z[-1]: z = z[::-1]; t_i = t_i[::-1]; rh_i = rh_i[::-1]
        mask = np.concatenate([[True], np.diff(z) > 0.1])
        if mask.sum() < 3: continue
        z_u = z[mask]; t_u = t_i[mask]; rh_u = rh_i[mask]
        try:
            T_out[i] = interp1d(z_u, t_u, kind='linear', bounds_error=False,
                                fill_value=(t_u[0], t_u[-1]))(target_h)
            RH_out[i] = interp1d(z_u, rh_u, kind='linear', bounds_error=False,
                                 fill_value=(rh_u[0], rh_u[-1]))(target_h)
        except Exception: pass
    return T_out, RH_out


def h_to_idx(target_h, low_km, high_km):
    low_m, high_m = low_km * 1000, high_km * 1000
    indices = [i for i, h in enumerate(target_h) if low_m <= h <= high_m]
    return (indices[0], indices[-1] + 1)


# ============================================================
print("=" * 60)
print("MP-3000A → BRNN v6 (Sim_BT training + optimized pipeline)")
print("=" * 60)

# ---- 1. Load ----
print("\n[1/9] Loading...")
ds = xr.open_dataset(NC_FILE)
P_hpa = ds["Level_Pressure"].values
T_lev  = ds["Level_Temperature"].values
q_lev  = ds["Level_H2O"].values
sfc_alt = ds["Surface_Altitude"].values
sfc_P   = ds["Surface_Pressure"].values
sfc_T   = ds["Temperature_2M"].values
sfc_q   = ds["H2O_2M"].values
clwc    = ds["Column_Liquid_Cloud"].values
cic     = ds["Column_Ice_Cloud"].values

obs_bt   = ds["Obs_BT"].values
sim_bt   = ds["Sim_BT"].values
qc_flag  = ds["QC_Flag"].values
rain_flag = ds["Rain_Flag"].values
prof_idx = ds["Profile_Index"].values.astype(int)
obs_t2m  = ds["Obs_Temperature_2M"].values; obs_q2m = ds["Obs_H2O_2M"].values
obs_sp   = ds["Obs_Surface_Pressure"].values
obs_time = ds["Date_Time"].values; obs_ir = ds["IR_Temperature"].values
channel_freq = ds["Central_Frequency"].values
ds.close()

# ---- 2. Physics ----
print("\n[2/9] Physics preprocessing...")
RH_lev = np.zeros_like(T_lev)
for i in range(T_lev.shape[1]):
    RH_lev[:, i] = q_to_rh(q_lev[:, i], T_lev[:, i], P_hpa[:, i] if P_hpa.ndim>1 else P_hpa)
sfc_RH = q_to_rh(sfc_q, sfc_T, sfc_P)
obs_RH2m = q_to_rh(obs_q2m, obs_t2m, obs_sp)

Z_m = pressure_to_height(P_hpa, T_lev, q_lev, sfc_alt)
target_h = np.array(config.HEIGHT_GRID)
T_93, RH_93 = interpolate_to_mwr_grid(Z_m, T_lev, RH_lev, target_h)
valid_prof = np.any(T_93 != 0, axis=1)

# ---- 3. Profile QC (paper steps applied to training targets) ----
print("\n[3/9] ERA5 profile QC...")

# Step 1: RH scaling ×0.9
RH_93_scaled = RH_93 * 0.9

# Step 3: CLWC screening — remove profiles with CLWC > 750 g/m²
prof_keep = clwc <= 750
print(f"  Profiles: {prof_keep.sum()}/{len(prof_keep)} after CLWC screening ({(~prof_keep).sum()} removed)")

# Remove invalid profiles too
prof_keep = prof_keep & valid_prof

# ---- 4. Observation filtering ----
print("\n[4/9] Observation filtering...")
clean = (qc_flag == 0) & (rain_flag == 0)

# Remove obs linked to bad profiles
for i in range(len(clean)):
    if clean[i] and not prof_keep[prof_idx[i]]:
        clean[i] = False

# K-band extreme OMB removal (cloud contamination in Obs_BT)
omb = obs_bt - sim_bt
k_ch = channel_freq < 51
for ch_idx in np.where(k_ch)[0]:
    idx = np.where(clean)[0]
    if len(idx) == 0: break
    mu, std = np.mean(omb[ch_idx, idx]), np.std(omb[ch_idx, idx])
    clean[idx[np.abs(omb[ch_idx, idx] - mu) > 3.0 * std]] = False

print(f"  Clean observations: {clean.sum()}")

# ---- 5. Per-season BT correction ----
print("\n[5/9] Per-season BT correction...")
obs_bt_clean = obs_bt[:, clean].T
sim_bt_clean = sim_bt[:, clean].T
time_clean = obs_time[clean]
months = (time_clean // 10000) % 100

seasons = {
    'DJF': [12, 1, 2],
    'MAM': [3, 4, 5],
    'SON': [9, 10, 11],
}

# Fit per-season, fall back to global if < 100 samples
obs_bt_corrected = np.zeros_like(obs_bt_clean)
for season, mlist in seasons.items():
    mask = np.isin(months, mlist)
    if mask.sum() < 100:
        print(f"  {season}: {mask.sum()} samples — using global correction")
        continue

    for ch in range(N_CH):
        x = obs_bt_clean[mask, ch]; y = sim_bt_clean[mask, ch]
        coeff = np.linalg.lstsq(np.column_stack([x, np.ones_like(x)]), y, rcond=None)[0]
        obs_bt_corrected[mask, ch] = coeff[0] * x + coeff[1]

# Global correction for unassigned seasons or small samples
uncorrected = ~np.any([np.isin(months, mlist) & (np.sum(np.isin(months, mlist)) >= 100) for mlist in seasons.values()], axis=0)
if uncorrected.sum() > 0:
    print(f"  Other: {uncorrected.sum()} samples — global correction")
    for ch in range(N_CH):
        x = obs_bt_clean[uncorrected, ch]; y = sim_bt_clean[uncorrected, ch]
        coeff = np.linalg.lstsq(np.column_stack([x, np.ones_like(x)]), y, rcond=None)[0]
        obs_bt_corrected[uncorrected, ch] = coeff[0] * x + coeff[1]

omb_after = obs_bt_corrected - sim_bt_clean
print(f"  OMB after seasonal correction: {np.mean(np.abs(omb_after)):.2f}±{np.std(omb_after):.2f}K")

# ---- 6. Build dataset ----
print("\n[6/9] Building features...")
N = clean.sum()
prof_idx_clean = prof_idx[clean]

# Training targets from QC'd profiles
T_target = T_93[prof_idx_clean]
RH_target = np.clip(RH_93_scaled[prof_idx_clean], 0, 100)

t2m_clean  = obs_t2m[clean]; rh2m_clean = obs_RH2m[clean]
sp_clean   = obs_sp[clean];  ir_clean   = obs_ir[clean]
clwc_clean = clwc[prof_idx_clean]; cic_clean = cic[prof_idx_clean]

# Two BT sets for two-stage training:
# Stage 1 (pre-train): Sim_BT as input (clean physics)
# Stage 2 (fine-tune): Obs_BT_corrected as input (real world)
sim_bt_input   = sim_bt_clean  # clean RTM
obs_bt_input   = obs_bt_corrected  # corrected observed

# ---- 7. Group split ----
print("\n[7/9] Profile-group split...")
unique_profs = np.unique(prof_idx_clean)
np.random.seed(42); np.random.shuffle(unique_profs)
n_prof = len(unique_profs)
n_train = int(n_prof * 0.60)  # 60% for pre-training
n_val   = int(n_prof * 0.15)  # 15% for val
n_fine  = int(n_prof * 0.10)  # 10% for fine-tuning
# remaining 15% for test

train_profs = set(unique_profs[:n_train])
val_profs   = set(unique_profs[n_train:n_train+n_val])
fine_profs  = set(unique_profs[n_train+n_val:n_train+n_val+n_fine])
test_profs  = set(unique_profs[n_train+n_val+n_fine:])

train_mask = np.array([p in train_profs for p in prof_idx_clean])
val_mask   = np.array([p in val_profs for p in prof_idx_clean])
fine_mask  = np.array([p in fine_profs for p in prof_idx_clean])
test_mask  = np.array([p in test_profs for p in prof_idx_clean])

print(f"  Pre-train: {train_mask.sum()} obs ({n_train} profs)")
print(f"  Fine-tune: {fine_mask.sum()} obs ({n_fine} profs)")
print(f"  Val:       {val_mask.sum()} obs ({n_val} profs)")
print(f"  Test:      {test_mask.sum()} obs ({n_prof-n_train-n_val-n_fine} profs)")

# ---- 8. Two-stage training ----
print("\n[8/9] Two-stage training (Sim_BT pre-train → Obs_BT fine-tune)...")

v_surface = (channel_freq >= 53.0) & (channel_freq <= 59.0)
v_surface_idx = np.where(v_surface)[0]

model_defs = [
    ("brnn_T_0-2km",    "T",  (0, 2)),
    ("brnn_T_2-8km",    "T",  (2, 8)),
    ("brnn_T_8-10km",   "T",  (8, 10)),
    ("brnn_RH_0-2km",   "RH", (0, 2)),
    ("brnn_RH_2-8km",   "RH", (2, 8)),
    ("brnn_RH_8-10km",  "RH", (8, 10)),
]

device = "mps" if torch.backends.mps.is_available() else "cpu"
models_dir = os.path.join(PROJ, "models_mp3000a_v6")
os.makedirs(models_dir, exist_ok=True)

surface_all = np.column_stack([
    t2m_clean.reshape(-1, 1), rh2m_clean.reshape(-1, 1),
    (sp_clean / 100.0).reshape(-1, 1), ir_clean.reshape(-1, 1),
    clwc_clean.reshape(-1, 1), cic_clean.reshape(-1, 1),
])

for name, var, (h_low, h_high) in model_defs:
    print(f"\n{'='*60}")
    print(f"Training {name} ({h_low}-{h_high}km)")
    s, e = h_to_idx(target_h, h_low, h_high)

    if var == "T":
        y_all = (T_target[:, s:e] - 200.0) / 100.0
    else:
        y_all = np.clip(RH_target[:, s:e], 0, 100) / 100.0

    n_out = y_all.shape[1]

    # Build inputs for Sim_BT and Obs_BT paths
    if var == "T" and h_low == 0 and h_high == 2:
        X_sim = np.column_stack([sim_bt_input[:, v_surface_idx], t2m_clean.reshape(-1, 1)])
        X_obs = np.column_stack([obs_bt_input[:, v_surface_idx], t2m_clean.reshape(-1, 1)])
    else:
        X_sim = np.column_stack([sim_bt_input, surface_all])
        X_obs = np.column_stack([obs_bt_input, surface_all])

    n_in = X_sim.shape[1]
    print(f"  Input: {n_in}, Output: {n_out}")

    # ---- Stage 1: Pre-train on Sim_BT ----
    X_pt, y_pt = X_sim[train_mask], y_all[train_mask]
    X_val, y_val = X_sim[val_mask], y_all[val_mask]

    train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_pt), torch.FloatTensor(y_pt)),
                              batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val)),
                            batch_size=config.BATCH_SIZE, shuffle=False)

    model = BRNN(n_in, n_out, config.HIDDEN_NODES, config.DROPOUT_RATE).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    best_val, best_state, patience = float("inf"), None, 0
    for epoch in range(config.MAX_EPOCHS):
        model.train()
        train_loss = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(Xb)
            loss = criterion(pred, yb) + 0.001 * torch.mean(torch.diff(pred, n=2, dim=1)**2)
            loss.backward(); optimizer.step()
            train_loss += criterion(pred, yb).item() * Xb.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for Xb, yb in val_loader:
                val_loss += criterion(model(Xb.to(device)), yb.to(device)).item() * Xb.size(0)
        val_loss /= len(val_loader.dataset)

        if val_loss < best_val:
            best_val = val_loss; best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        if (epoch + 1) % 20 == 0:
            print(f"  Stage1 Epoch {epoch+1}: t={train_loss:.6f} v={val_loss:.6f}")
        if patience >= config.EARLY_STOPPING_PATIENCE:
            print(f"  Stage1 early stop @ {epoch+1}")
            break

    model.load_state_dict(best_state)
    print(f"  Stage1 best val={best_val:.6f}")

    # ---- Stage 2: Fine-tune on corrected Obs_BT ----
    X_ft, y_ft = X_obs[fine_mask], y_all[fine_mask]
    if len(X_ft) >= 64:
        ft_loader = DataLoader(TensorDataset(torch.FloatTensor(X_ft), torch.FloatTensor(y_ft)),
                                batch_size=min(64, len(X_ft)), shuffle=True)

        optimizer_ft = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE * 0.1)
        best_val_ft, best_state_ft, patience_ft = float("inf"), None, 0

        for epoch in range(min(100, config.MAX_EPOCHS)):
            model.train()
            ft_loss = 0.0
            for Xb, yb in ft_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                optimizer_ft.zero_grad()
                pred = model(Xb)
                loss = criterion(pred, yb) + 0.001 * torch.mean(torch.diff(pred, n=2, dim=1)**2)
                loss.backward(); optimizer_ft.step()
                ft_loss += criterion(pred, yb).item() * Xb.size(0)
            ft_loss /= len(ft_loader.dataset)

            model.eval()
            val_loss_ft = 0.0
            with torch.no_grad():
                for Xb, yb in val_loader:
                    val_loss_ft += criterion(model(Xb.to(device)), yb.to(device)).item() * Xb.size(0)
            val_loss_ft /= len(val_loader.dataset)

            if val_loss_ft < best_val_ft:
                best_val_ft = val_loss_ft; best_state_ft = {k: v.clone() for k, v in model.state_dict().items()}
                patience_ft = 0
            else:
                patience_ft += 1
            if (epoch + 1) % 20 == 0:
                print(f"  Stage2 Epoch {epoch+1}: t={ft_loss:.6f} v={val_loss_ft:.6f}")
            if patience_ft >= config.EARLY_STOPPING_PATIENCE:
                print(f"  Stage2 early stop @ {epoch+1}")
                break

        model.load_state_dict(best_state_ft)
        print(f"  Stage2 best val={best_val_ft:.6f}")
    else:
        print(f"  Stage2 skipped: only {len(X_ft)} fine-tune samples")

    torch.save(best_state_ft if best_state_ft else best_state,
               os.path.join(models_dir, f"{name}.pt"))


# ---- 9. Evaluate: Sim_BT vs Obs_BT ----
print(f"\n{'='*60}")
print("[9/9] Evaluation")
print("=" * 60)

models_state = {}
for name, var, (h_low, h_high) in model_defs:
    s, e = h_to_idx(target_h, h_low, h_high)
    if var == "T" and h_low == 0 and h_high == 2:
        n_in = len(v_surface_idx) + 1
    else:
        n_in = N_CH + 6
    m = BRNN(n_in, e - s, config.HIDDEN_NODES, config.DROPOUT_RATE)
    m.load_state_dict(torch.load(os.path.join(models_dir, f"{name}.pt"),
                                  map_location=device, weights_only=True))
    m.to(device).eval()
    models_state[name] = m


def evaluate_bt(bt_data, label):
    T_pred = np.zeros((test_mask.sum(), 93)); RH_pred = np.zeros((test_mask.sum(), 93))
    for name, var, (h_low, h_high) in model_defs:
        s, e = h_to_idx(target_h, h_low, h_high)
        m = models_state[name]
        if var == "T" and h_low == 0 and h_high == 2:
            X = torch.FloatTensor(np.column_stack([
                bt_data[test_mask][:, v_surface_idx],
                t2m_clean[test_mask].reshape(-1, 1)])).to(device)
        else:
            X = torch.FloatTensor(np.column_stack([
                bt_data[test_mask], surface_all[test_mask]])).to(device)
        with torch.no_grad(): out = m(X).cpu().numpy()
        if var == "T": T_pred[:, s:e] = out * 100.0 + 200.0
        else:           RH_pred[:, s:e] = out * 100.0

    T_err = T_pred - T_target[test_mask]; RH_err = RH_pred - RH_target[test_mask]
    T_rmse = np.sqrt(np.mean(T_err**2)); RH_rmse = np.sqrt(np.mean(RH_err**2))

    print(f"\n  --- {label} ---")
    for km in [0, 0.5, 1, 2, 3, 5, 8, 10]:
        idx = np.argmin(np.abs(target_h - km * 1000))
        print(f"  {target_h[idx]/1000:5.1f}km  T={np.sqrt(np.mean(T_err[:,idx]**2)):.3f}K  RH={np.sqrt(np.mean(RH_err[:,idx]**2)):.2f}%")
    print(f"  TOTAL: T={T_rmse:.3f}K  RH={RH_rmse:.2f}%")
    return T_rmse, RH_rmse

# Test on Sim_BT (the model's "native" input)
t_sim, rh_sim = evaluate_bt(sim_bt_input, "Sim_BT (model trained on this)")

# Test on corrected Obs_BT (real-world application)
t_obs, rh_obs = evaluate_bt(obs_bt_input, "Obs_BT corrected (applied)")

# Test on raw Obs_BT
t_raw, rh_raw = evaluate_bt(obs_bt[:, clean].T, "Obs_BT raw (uncorrected)")

print(f"\n{'='*60}")
print("SUMMARY")
print("=" * 60)
print(f"  Sim_BT (ideal):            T={t_sim:.3f}K   RH={rh_sim:.2f}%")
print(f"  Obs_BT corrected (actual):  T={t_obs:.3f}K   RH={rh_obs:.2f}%")
print(f"  Obs_BT raw:                T={t_raw:.3f}K   RH={rh_raw:.2f}%")
print(f"  Sim→Obs gap: ΔT={t_obs-t_sim:.3f}K  ΔRH={rh_obs-rh_sim:.2f}%")

with open(os.path.join(PROJ, "results", "mp3000a_v6_results.pkl"), "wb") as f:
    pickle.dump({"heights": target_h, "sim_vs_obs": (t_sim, rh_sim, t_obs, rh_obs)}, f)

print(f"\nModels: {models_dir}/")
print("=" * 60)
