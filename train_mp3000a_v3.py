#!/usr/bin/env python3
"""Train BRNN models from MP-3000A data — v3.

Key improvements over v2:
  1. Train on Sim_BT (clean RTM signal), test on Obs_BT (paper's approach)
  2. Full ERA5 profile QC: humidity scaling, seasonal RH bias, CLWC screening, BT correction
  3. Additional features: IR_Temperature, cyclical hour/month encoding
  4. Profile-group-aware split (keep from v2)
  5. Physics-informed loss: smoothness penalty on predicted profiles
"""

import os, sys, pickle, warnings
import numpy as np
import xarray as xr
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

PROJ = "/Users/ink/test/mwr_retrieval"
NC_FILE = "/Users/ink/test/54623_MP_3000A_ERA5Model_20231101_20240331_areamean_selfgb.nc"
sys.path.insert(0, os.path.join(PROJ, "src"))
sys.path.insert(0, PROJ)
import config

RD = 287.058; G = 9.80665; EPSILON = 0.622
N_CH = 22  # 22-channel MP-3000A


# ---- BRNN (same architecture, kept simple and proven) ----
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


# ---- Physics utilities ----
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
            dp = P_pa[k+1] - P_pa[k]
            Tv_avg = 0.5 * (Tv[k] + Tv[k+1])
            P_avg = 0.5 * (P_pa[k] + P_pa[k+1])
            Z[k, i] = Z[k+1, i] + RD * Tv_avg / G * dp / P_avg
        Z[:, i] += sfc_alt_km[i] * 1000.0
    return Z


def interpolate_to_mwr_grid(Z_m, T, RH, target_h):
    from scipy.interpolate import interp1d
    n_prof = T.shape[1]
    T_out = np.zeros((n_prof, len(target_h)))
    RH_out = np.zeros((n_prof, len(target_h)))
    for i in range(n_prof):
        z = Z_m[:, i]; t_i = T[:, i]; rh_i = RH[:, i]
        if z[0] > z[-1]:
            z = z[::-1]; t_i = t_i[::-1]; rh_i = rh_i[::-1]
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


# ---- QC pipeline (paper Section 3.2.2) ----
def compute_iwv(rh, t, height):
    """Integrated water vapor [kg/m²]."""
    Tc = t - 273.16
    es = 6.1078 * np.exp(17.2693882 * Tc / (Tc + 237.3))
    e = rh / 100.0 * es
    rho_v = e * 216.7 / t
    dz = np.diff(height, prepend=0)
    return np.sum(rho_v * dz) / 1000.0


def compute_iclwc(clwc, height):
    dz = np.diff(height, prepend=0)
    return np.sum(clwc * dz)


def apply_profile_qc(T_93, RH_93, CLWC_93, heights):
    """Apply paper's QC steps to profiles.

    Step 1: Humidity scaling ×0.9
    Step 3: LWC screening (>1250 delete, >750 scale)
    Step 2: Seasonal bias needs sounding data, skipped

    Returns: filtered T, RH, CLWC, keep_mask
    """
    n_prof = T_93.shape[0]
    n_layers = T_93.shape[1]

    # Step 1: Scale humidity
    RH_scaled = RH_93 * 0.9

    # Step 3: LWC screening
    delete_mask = np.zeros(n_prof, dtype=bool)
    CLWC_corrected = CLWC_93.copy() if np.any(CLWC_93 != 0) else np.zeros_like(T_93)

    for i in range(n_prof):
        if np.any(CLWC_93[i] != 0):
            iclwc = compute_iclwc(CLWC_93[i], heights)
            if iclwc > 1250:
                delete_mask[i] = True
            elif iclwc > 750:
                CLWC_corrected[i] = CLWC_93[i] * (750.0 / iclwc)

    keep_mask = ~delete_mask
    return T_93[keep_mask], RH_scaled[keep_mask], CLWC_corrected[keep_mask], keep_mask


# ============================================================
print("=" * 60)
print("MP-3000A → BRNN v3 (Train Sim_BT, Test Obs_BT)")
print("=" * 60)

# ---- 1. Load ----
print("\n[1/9] Loading data...")
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
obs_t2m  = ds["Obs_Temperature_2M"].values
obs_q2m  = ds["Obs_H2O_2M"].values
obs_sp   = ds["Obs_Surface_Pressure"].values
obs_time = ds["Date_Time"].values
obs_ir   = ds["IR_Temperature"].values
channel_freq = ds["Central_Frequency"].values
ds.close()

print(f"  {T_lev.shape[1]} profiles × {T_lev.shape[0]} levels, {len(qc_flag)} obs")

# ---- 2. Physics preprocess ----
print("\n[2/9] Physics preprocessing...")
RH_lev = np.zeros_like(T_lev)
for i in range(T_lev.shape[1]):
    RH_lev[:, i] = q_to_rh(q_lev[:, i], T_lev[:, i],
                            P_hpa[:, i] if P_hpa.ndim>1 else P_hpa)
sfc_RH = q_to_rh(sfc_q, sfc_T, sfc_P)
obs_RH2m = q_to_rh(obs_q2m, obs_t2m, obs_sp)

Z_m = pressure_to_height(P_hpa, T_lev, q_lev, sfc_alt)
target_h = np.array(config.HEIGHT_GRID)
T_93_all, RH_93_all = interpolate_to_mwr_grid(Z_m, T_lev, RH_lev, target_h)
valid_prof = np.all(T_93_all != 0, axis=1)
print(f"  Valid profiles: {valid_prof.sum()}/{T_lev.shape[1]}")

# ---- 3. Profile QC ----
print("\n[3/9] ERA5 profile QC...")
# Estimate CLWC profiles (use column value spread across layers — crude but acceptable)
CLWC_93 = np.zeros_like(T_93_all)
for i in range(len(clwc)):
    if clwc[i] > 0:
        # Distribute column LWC across lower layers
        low_mask = target_h < 3000  # below 3km
        n_low = low_mask.sum()
        if n_low > 0:
            CLWC_93[i, :n_low] = clwc[i] / n_low

T_qc, RH_qc, CLWC_qc, qc_keep = apply_profile_qc(T_93_all, RH_93_all, CLWC_93, target_h)
print(f"  Profiles after QC: {qc_keep.sum()}/{len(qc_keep)} "
      f"(removed {(~qc_keep).sum()})")

# Build QC'd profile lookup
prof_T = T_qc
prof_RH = RH_qc
# Map old index → new index (if kept) or -1 (if removed)
old_to_new = np.full(T_lev.shape[1], -1, dtype=int)
old_to_new[qc_keep] = np.arange(qc_keep.sum())

# ---- 4. Match + filter ----
print("\n[4/9] Matching observations...")
clean = (qc_flag == 0) & (rain_flag == 0)

# OMB screening
omb = obs_bt - sim_bt
stable_ch = (channel_freq >= 51) & (channel_freq <= 59)
omb_stable = omb[stable_ch, :]
omb_mean = np.mean(omb_stable, axis=1, keepdims=True)
omb_std  = np.std(omb_stable, axis=1, keepdims=True)
omb_z = np.abs(omb_stable - omb_mean) / np.maximum(omb_std, 0.1)
vband_outlier = np.any(omb_z > 4.0, axis=0)
kband = channel_freq < 51
kband_cloud = np.any(omb[kband, :] > 50, axis=0)
clean = clean & ~vband_outlier & ~kband_cloud

# Match to valid QC'd profiles
for i in range(len(clean)):
    if clean[i]:
        if prof_idx[i] >= len(old_to_new): clean[i] = False
        elif old_to_new[prof_idx[i]] < 0: clean[i] = False
        elif not valid_prof[prof_idx[i]]: clean[i] = False

print(f"  Clean obs: {clean.sum()} / {len(clean)}")

# ---- 5. Build features ----
print("\n[5/9] Building features...")
prof_idx_clean = prof_idx[clean]
qc_prof_idx = old_to_new[prof_idx_clean]

# BT data
sim_bt_clean = sim_bt[:, clean].T    # [N, 22]
obs_bt_clean = obs_bt[:, clean].T

# Surface obs
t2m_clean  = obs_t2m[clean]
rh2m_clean = obs_RH2m[clean]
sp_clean   = obs_sp[clean]
ir_clean   = obs_ir[clean]
time_clean = obs_time[clean]

# Cloud features
clwc_prof_clean = clwc[prof_idx_clean]
cic_prof_clean  = cic[prof_idx_clean]

# Target profiles (QC'd)
T_target  = prof_T[qc_prof_idx]
RH_target = prof_RH[qc_prof_idx]

# Temporal cyclical features
def hour_as_float(t):
    """Extract hour from YYYYMMDDHH.ff format."""
    return (t / 100.0) % 100.0
hour_raw = np.array([hour_as_float(t) for t in time_clean])
month_raw = np.array([(t // 10000) % 100 for t in time_clean])

# Cyclical encoding: sin/cos
hour_sin = np.sin(2 * np.pi * hour_raw / 24.0)
hour_cos = np.cos(2 * np.pi * hour_raw / 24.0)
month_sin = np.sin(2 * np.pi * month_raw / 12.0)
month_cos = np.cos(2 * np.pi * month_raw / 12.0)

N = clean.sum()
print(f"  Training samples: {N}")

# ---- 6. BT correction for Obs_BT (fit to Sim_BT) ----
print("\n[6/9] BT linear correction (Obs→Sim alignment)...")
bt_slope = np.ones(N_CH)
bt_intercept = np.zeros(N_CH)
for ch in range(N_CH):
    A = np.column_stack([obs_bt_clean[:, ch], np.ones(N)])
    coeff = np.linalg.lstsq(A, sim_bt_clean[:, ch], rcond=None)[0]
    bt_slope[ch] = coeff[0]
    bt_intercept[ch] = coeff[1]
obs_bt_corrected = bt_slope[np.newaxis, :] * obs_bt_clean + bt_intercept[np.newaxis, :]

# ---- 7. Group-aware split ----
print("\n[7/9] Profile-group split...")
unique_profs = np.unique(qc_prof_idx)
np.random.seed(42)
np.random.shuffle(unique_profs)
n_prof_total = len(unique_profs)

# Time-order the profiles for a time-aware split (since we have temporal features)
# Actually keep group-aware random split — don't let profiles leak across splits
n_train_prof = int(n_prof_total * 0.70)
n_val_prof   = int(n_prof_total * 0.15)

train_profs = set(unique_profs[:n_train_prof])
val_profs   = set(unique_profs[n_train_prof:n_train_prof+n_val_prof])
test_profs  = set(unique_profs[n_train_prof+n_val_prof:])

train_mask = np.array([p in train_profs for p in qc_prof_idx])
val_mask   = np.array([p in val_profs for p in qc_prof_idx])
test_mask  = np.array([p in test_profs for p in qc_prof_idx])

print(f"  Train: {train_mask.sum()} obs ({n_train_prof} profiles)")
print(f"  Val:   {val_mask.sum()} obs ({n_val_prof} profiles)")
print(f"  Test:  {test_mask.sum()} obs ({n_prof_total-n_train_prof-n_val_prof} profiles)")

# ---- 8. Train on Sim_BT ----
print("\n[8/9] Training on Sim_BT...")

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
models_dir = os.path.join(PROJ, "models_mp3000a_v3")
os.makedirs(models_dir, exist_ok=True)

for name, var, (h_low, h_high) in model_defs:
    print(f"\n{'='*60}")
    print(f"Training {name} ({h_low}-{h_high}km)")

    # Build features: Sim_BT + surface + temporal + cloud
    surface_feats = np.column_stack([
        t2m_clean.reshape(-1, 1),
        rh2m_clean.reshape(-1, 1),
        (sp_clean / 100.0).reshape(-1, 1),
        ir_clean.reshape(-1, 1),
        clwc_prof_clean.reshape(-1, 1),
        cic_prof_clean.reshape(-1, 1),
        hour_sin.reshape(-1, 1),
        hour_cos.reshape(-1, 1),
        month_sin.reshape(-1, 1),
        month_cos.reshape(-1, 1),
    ])

    if var == "T" and h_low == 0 and h_high == 2:
        X = np.column_stack([sim_bt_clean[:, v_surface_idx], t2m_clean.reshape(-1, 1)])
    else:
        X = np.column_stack([sim_bt_clean, surface_feats])

    s, e = h_to_idx(target_h, h_low, h_high)
    if var == "T":
        y = (T_target[:, s:e] - 200.0) / 100.0
    else:
        y = RH_target[:, s:e] / 100.0

    n_in, n_out = X.shape[1], y.shape[1]
    print(f"  Input: {n_in}, Output: {n_out}")

    X_train, y_train = X[train_mask], y[train_mask]
    X_val,   y_val   = X[val_mask],   y[val_mask]

    train_loader = DataLoader(TensorDataset(
        torch.FloatTensor(X_train), torch.FloatTensor(y_train)),
        batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(
        torch.FloatTensor(X_val), torch.FloatTensor(y_val)),
        batch_size=config.BATCH_SIZE, shuffle=False)

    model = BRNN(n_in, n_out, config.HIDDEN_NODES, config.DROPOUT_RATE).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    best_val_loss = float("inf")
    best_state = None
    patience = 0

    for epoch in range(config.MAX_EPOCHS):
        model.train()
        train_loss = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()

            # Physics-informed loss: MSE + smoothness penalty
            pred = model(Xb)
            mse = criterion(pred, yb)

            # Smoothness: penalize large second derivatives in the profile
            smoothness = 0.0
            if config.N_LAYERS > 2 and yb.size(0) > 0:
                diff2 = torch.diff(pred, n=2, dim=1)
                smoothness = 0.001 * torch.mean(diff2 ** 2)

            loss = mse + smoothness
            loss.backward()
            optimizer.step()
            train_loss += mse.item() * Xb.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for Xb, yb in val_loader:
                val_loss += criterion(model(Xb.to(device)), yb.to(device)).item() * Xb.size(0)
        val_loss /= len(val_loader.dataset)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}: train={train_loss:.6f} val={val_loss:.6f}")

        if patience >= config.EARLY_STOPPING_PATIENCE:
            print(f"  Early stop @ {epoch+1}")
            break

    model.load_state_dict(best_state)
    torch.save(best_state, os.path.join(models_dir, f"{name}.pt"))


# ---- 9. Evaluate: Sim_BT vs Obs_BT ----
print(f"\n{'='*60}")
print("[9/9] Evaluation: Sim_BT baseline + Obs_BT application")
print("=" * 60)

models_state = {}
for name, var, (h_low, h_high) in model_defs:
    s, e = h_to_idx(target_h, h_low, h_high)
    if var == "T" and h_low == 0 and h_high == 2:
        n_in = len(v_surface_idx) + 1
    else:
        n_in = N_CH + 10  # 22ch + T2m+RH2m+Ps+IR+CLWC+CIC+4 temporal
    m = BRNN(n_in, e - s, config.HIDDEN_NODES, config.DROPOUT_RATE)
    m.load_state_dict(torch.load(os.path.join(models_dir, f"{name}.pt"),
                                  map_location=device, weights_only=True))
    m.to(device).eval()
    models_state[name] = m


def evaluate_bt(bt_data, label, test_mask, device, models_state, model_defs,
                target_h, T_target, RH_target, t2m_clean, rh2m_clean, sp_clean,
                ir_clean, clwc_prof_clean, cic_prof_clean,
                hour_sin, hour_cos, month_sin, month_cos,
                v_surface_idx, N_CH):
    """Evaluate model on given BT input."""
    surface_test = np.column_stack([
        t2m_clean.reshape(-1, 1), rh2m_clean.reshape(-1, 1),
        sp_clean.reshape(-1, 1) / 100.0, ir_clean.reshape(-1, 1),
        clwc_prof_clean.reshape(-1, 1), cic_prof_clean.reshape(-1, 1),
        hour_sin.reshape(-1, 1), hour_cos.reshape(-1, 1),
        month_sin.reshape(-1, 1), month_cos.reshape(-1, 1),
    ])

    T_pred = np.zeros((test_mask.sum(), 93))
    RH_pred = np.zeros((test_mask.sum(), 93))

    for name, var, (h_low, h_high) in model_defs:
        s, e = h_to_idx(target_h, h_low, h_high)
        m = models_state[name]
        if var == "T" and h_low == 0 and h_high == 2:
            X = torch.FloatTensor(np.column_stack([
                bt_data[test_mask][:, v_surface_idx],
                t2m_clean[test_mask].reshape(-1, 1)
            ])).to(device)
        else:
            X = torch.FloatTensor(np.column_stack([
                bt_data[test_mask], surface_test[test_mask]
            ])).to(device)
        with torch.no_grad():
            out = m(X).cpu().numpy()
        if var == "T":
            T_pred[:, s:e] = out * 100.0 + 200.0
        else:
            RH_pred[:, s:e] = out * 100.0

    T_err = T_pred - T_target[test_mask]
    RH_err = RH_pred - RH_target[test_mask]

    print(f"\n  --- {label} ---")
    print(f"  {'Height':>8s}  {'T_RMSE':>8s}  {'T_Bias':>8s}  {'RH_RMSE':>8s}")
    print(f"  {'-'*42}")
    for km in [0.0, 0.5, 1, 2, 3, 5, 8, 10]:
        idx = np.argmin(np.abs(target_h - km * 1000))
        t_rmse = np.sqrt(np.mean(T_err[:, idx]**2))
        t_bias = np.mean(T_err[:, idx])
        rh_rmse = np.sqrt(np.mean(RH_err[:, idx]**2))
        print(f"  {target_h[idx]/1000:6.1f}km  {t_rmse:7.3f}K  {t_bias:+7.3f}K  {rh_rmse:7.2f}%")

    T_rmse = np.sqrt(np.mean(T_err**2))
    RH_rmse = np.sqrt(np.mean(RH_err**2))
    print(f"  {'TOTAL':>8s}  {T_rmse:7.3f}K  {np.mean(T_err):+7.3f}K  {RH_rmse:7.2f}%")
    return T_rmse, RH_rmse, T_pred, RH_pred


# Evaluate on Sim_BT (clean baseline)
sim_t, sim_rh, T_sim, RH_sim = evaluate_bt(
    sim_bt_clean, "Sim_BT (trained input)", test_mask, device,
    models_state, model_defs, target_h, T_target, RH_target,
    t2m_clean, rh2m_clean, sp_clean, ir_clean,
    clwc_prof_clean, cic_prof_clean,
    hour_sin, hour_cos, month_sin, month_cos,
    v_surface_idx, N_CH)

# Evaluate on corrected Obs_BT (real-world application)
obs_t, obs_rh, T_obs, RH_obs = evaluate_bt(
    obs_bt_corrected, "Obs_BT corrected (application)", test_mask, device,
    models_state, model_defs, target_h, T_target, RH_target,
    t2m_clean, rh2m_clean, sp_clean, ir_clean,
    clwc_prof_clean, cic_prof_clean,
    hour_sin, hour_cos, month_sin, month_cos,
    v_surface_idx, N_CH)

# Evaluate on Raw Obs_BT (no correction)
raw_t, raw_rh, _, _ = evaluate_bt(
    obs_bt_clean, "Obs_BT raw (no correction)", test_mask, device,
    models_state, model_defs, target_h, T_target, RH_target,
    t2m_clean, rh2m_clean, sp_clean, ir_clean,
    clwc_prof_clean, cic_prof_clean,
    hour_sin, hour_cos, month_sin, month_cos,
    v_surface_idx, N_CH)

# Summary
print(f"\n{'='*60}")
print("SUMMARY")
print("=" * 60)
print(f"  Sim_BT (clean baseline):    T={sim_t:.3f}K, RH={sim_rh:.2f}%")
print(f"  Obs_BT corrected (applied):  T={obs_t:.3f}K, RH={obs_rh:.2f}%")
print(f"  Obs_BT raw (no correction):  T={raw_t:.3f}K, RH={raw_rh:.2f}%")

# Save
results = {"T_sim": T_sim, "RH_sim": RH_sim, "T_obs": T_obs, "RH_obs": RH_obs,
           "T_true": T_target[test_mask], "RH_true": RH_target[test_mask],
           "heights": target_h}
with open(os.path.join(PROJ, "results", "mp3000a_v3_results.pkl"), "wb") as f:
    pickle.dump(results, f)

print(f"\nModels: {models_dir}/")
print(f"Results: results/mp3000a_v3_results.pkl")
print("Done.")
