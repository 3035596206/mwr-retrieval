#!/usr/bin/env python3
"""Train BRNN models from MP-3000A data — v2 with improvements.

Improvements over v1:
  1. OMB-based outlier filter: remove samples where OMB > 3σ in any channel
  2. Per-channel BT linear correction (align Obs_BT → Sim_BT baseline)
  3. Profile-group-aware train/val/test split (prevent leakage)
  4. Additional inputs: Column_Liquid_Cloud, Column_Ice_Cloud
  5. Quality control on ERA5 profiles (CLWC screening like paper)

Expected: T RMSE < 2.5K, RH RMSE < 14%
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


# ---- BRNN model (same as paper) ----
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
    """Hydrostatic integration upward from surface."""
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
        z = Z_m[:, i]
        t_i = T[:, i]; rh_i = RH[:, i]

        # Z[0] = top (highest altitude) — reverse for interp1d
        if z[0] > z[-1]:
            z = z[::-1]; t_i = t_i[::-1]; rh_i = rh_i[::-1]

        # Drop near-duplicate/reversed Z
        mask = np.concatenate([[True], np.diff(z) > 0.1])
        if mask.sum() < 3:
            continue
        z_u = z[mask]; t_u = t_i[mask]; rh_u = rh_i[mask]

        try:
            T_out[i] = interp1d(z_u, t_u, kind='linear', bounds_error=False,
                                fill_value=(t_u[0], t_u[-1]))(target_h)
            RH_out[i] = interp1d(z_u, rh_u, kind='linear', bounds_error=False,
                                 fill_value=(rh_u[0], rh_u[-1]))(target_h)
        except Exception:
            pass
    return T_out, RH_out


def h_to_idx(target_h, low_km, high_km):
    low_m, high_m = low_km * 1000, high_km * 1000
    indices = [i for i, h in enumerate(target_h) if low_m <= h <= high_m]
    return (indices[0], indices[-1] + 1)


# ============================================================
print("=" * 60)
print("MP-3000A → BRNN v2 (Improved)")
print("=" * 60)

# ---- 1. Load ----
print("\n[1/8] Loading...")
ds = xr.open_dataset(NC_FILE)

P_hpa = ds["Level_Pressure"].values
# P_hpa shape: (137, 3453) — per-profile pressure grid
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
channel_freq = ds["Central_Frequency"].values
ds.close()

print(f"  Loaded: {T_lev.shape[1]} profiles × {T_lev.shape[0]} levels, {len(qc_flag)} obs")

# ---- 2. Convert q → RH, compute heights, interpolate ----
print("\n[2/8] Physics preprocess...")
RH_lev = np.zeros_like(T_lev)
for i in range(T_lev.shape[1]):
    RH_lev[:, i] = q_to_rh(q_lev[:, i], T_lev[:, i],
                            P_hpa if P_hpa.ndim == 1 else P_hpa[:, i])
sfc_RH = q_to_rh(sfc_q, sfc_T, sfc_P)
obs_RH2m = q_to_rh(obs_q2m, obs_t2m, obs_sp)

Z_m = pressure_to_height(P_hpa, T_lev, q_lev, sfc_alt)
target_h = np.array(config.HEIGHT_GRID)
T_93, RH_93 = interpolate_to_mwr_grid(Z_m, T_lev, RH_lev, target_h)
valid_prof = np.all(T_93 != 0, axis=1)
print(f"  Valid profiles: {valid_prof.sum()}/{T_lev.shape[1]}")

# ---- 3. IMPROVED filtering ----
print("\n[3/8] Quality filtering...")

# 3a. Basic QC
clean = (qc_flag == 0) & (rain_flag == 0)

# 3b. OMB-based outlier: flag samples where OMB > 3σ in any stable channel
omb = obs_bt - sim_bt           # [22, N]
# V-band 52-59 GHz are most stable — use these for screening
stable_ch = (channel_freq >= 51) & (channel_freq <= 59)
omb_stable = omb[stable_ch, :]  # [14, N]

# Per-channel mean and std
omb_mean = np.mean(omb_stable, axis=1, keepdims=True)   # [14, 1]
omb_std  = np.std(omb_stable, axis=1, keepdims=True)     # [14, 1]
omb_z = np.abs(omb_stable - omb_mean) / omb_std           # [14, N]

# Flag if any V-band channel deviates > 3.5σ
vband_outlier = np.any(omb_z > 3.5, axis=0)

# Also flag extreme Obs_BT (physically unreasonable)
bt_too_low  = np.any(obs_bt < 5, axis=0)     # BT < 5K impossible
bt_too_high = np.any(obs_bt > 350, axis=0)   # BT > 350K impossible

# Also flag K-band heavy cloud (OMB > 50K in any K-band channel)
kband = channel_freq < 51
kband_cloud = np.any(omb[kband, :] > 50, axis=0)

extra_bad = vband_outlier | bt_too_low | bt_too_high | kband_cloud
clean = clean & (~extra_bad)

n_removed = np.sum(extra_bad)
print(f"  Removed by OMB screening: {n_removed} obs")
print(f"  Basic QC+Rain: {(qc_flag==0)&(rain_flag==0)} → {np.sum(clean)} после screening")

# 3c. Profile QC: remove profiles with CLWC > 1250 g/m² (paper threshold)
bad_clwc = clwc > config.ICLWC_DELETE_THRESHOLD
prof_clean = ~bad_clwc

# Remove obs pointing to bad profiles
for i in range(len(clean)):
    if clean[i] and not prof_clean[prof_idx[i]]:
        clean[i] = False

# Remove obs pointing to invalid profiles
for i in range(len(clean)):
    if clean[i] and not valid_prof[prof_idx[i]]:
        clean[i] = False

print(f"  After profile QC: {clean.sum()} observations")

# ---- 4. BT linear correction (align Obs_BT → Sim_BT baseline) ----
print("\n[4/8] BT linear correction...")
# Fit: Obs_BT_corrected = slope * Obs_BT + intercept to match Sim_BT
slope = np.ones(22)
intercept = np.zeros(22)
for ch in range(22):
    obs_ch = obs_bt[ch, clean]
    sim_ch = sim_bt[ch, clean]
    A = np.column_stack([obs_ch, np.ones_like(obs_ch)])
    coeff = np.linalg.lstsq(A, sim_ch, rcond=None)[0]
    slope[ch] = coeff[0]
    intercept[ch] = coeff[1]

# Apply correction
obs_bt_corrected = slope[:, np.newaxis] * obs_bt + intercept[:, np.newaxis]
print(f"  Slope range: {slope.min():.3f} – {slope.max():.3f}")
print(f"  Intercept range: {intercept.min():.2f} – {intercept.max():.2f} K")

# Verify correction
omb_corrected = obs_bt_corrected - sim_bt
print(f"  OMB before: mean={np.mean(omb[:,clean]):+.2f}±{np.std(omb[:,clean]):.2f}K")
print(f"  OMB after:  mean={np.mean(omb_corrected[:,clean]):+.4f}±{np.std(omb_corrected[:,clean]):.2f}K")

# ---- 5. Build dataset ----
print("\n[5/8] Building dataset...")
prof_idx_clean = prof_idx[clean]
bt_clean  = obs_bt_corrected[:, clean].T    # [N, 22]
t2m_clean = obs_t2m[clean]
rh2m_clean = obs_RH2m[clean]
sp_clean  = obs_sp[clean]
time_clean = obs_time[clean]

T_target  = T_93[prof_idx_clean]
RH_target = RH_93[prof_idx_clean]

# Add cloud column features
clwc_prof = clwc[prof_idx_clean]
cic_prof  = cic[prof_idx_clean]

N = clean.sum()
print(f"  Training samples: {N}")

# ---- 6. Group-aware split (by unique profile) ----
print("\n[6/8] Group-aware split...")
unique_profs = np.unique(prof_idx_clean)
np.random.seed(42)
np.random.shuffle(unique_profs)

n_prof_total = len(unique_profs)
n_train_prof = int(n_prof_total * 0.70)
n_val_prof   = int(n_prof_total * 0.15)

train_profs = set(unique_profs[:n_train_prof])
val_profs   = set(unique_profs[n_train_prof:n_train_prof + n_val_prof])
test_profs  = set(unique_profs[n_train_prof + n_val_prof:])

train_mask = np.array([p in train_profs for p in prof_idx_clean])
val_mask   = np.array([p in val_profs for p in prof_idx_clean])
test_mask  = np.array([p in test_profs for p in prof_idx_clean])

print(f"  Train: {train_mask.sum()} obs from {n_train_prof} profiles")
print(f"  Val:   {val_mask.sum()} obs from {n_val_prof} profiles")
print(f"  Test:  {test_mask.sum()} obs from {n_prof_total - n_train_prof - n_val_prof} profiles")

# ---- 7. Train models ----
print("\n[7/8] Training BRNN models...")
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

device = "cpu"
if torch.backends.mps.is_available():
    device = "mps"

models_dir = os.path.join(PROJ, "models_mp3000a_v2")
os.makedirs(models_dir, exist_ok=True)

for name, var, (h_low, h_high) in model_defs:
    print(f"\n{'='*60}")
    print(f"Training {name} ({h_low}-{h_high}km)")

    # Build input features
    # All models get: 22ch corrected BT + T2m + RH2m + Ps(hPa) + CLWC + CIC
    surface_inputs = np.column_stack([
        t2m_clean.reshape(-1, 1),
        rh2m_clean.reshape(-1, 1),
        (sp_clean / 100.0).reshape(-1, 1),  # Pa → hPa
        clwc_prof.reshape(-1, 1),
        cic_prof.reshape(-1, 1),
    ])

    if var == "T" and h_low == 0 and h_high == 2:
        # T 0-2km: paper uses only V-band surface channels + T2m
        X = np.column_stack([bt_clean[:, v_surface_idx], t2m_clean.reshape(-1, 1)])
    else:
        X = np.column_stack([bt_clean, surface_inputs])

    # Target
    s, e = h_to_idx(target_h, h_low, h_high)
    if var == "T":
        y = (T_target[:, s:e] - 200.0) / 100.0   # [200,300]K → [0,1]
    else:
        y = RH_target[:, s:e] / 100.0

    n_in, n_out = X.shape[1], y.shape[1]
    print(f"  Input: {n_in}, Output: {n_out} (slice {s}:{e})")

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
            loss = criterion(model(Xb), yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * Xb.size(0)
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
    print(f"  Saved: best_val_loss={best_val_loss:.6f}")


# ---- 8. Evaluation ----
print(f"\n{'='*60}")
print("[8/8] Test evaluation")
print("=" * 60)

models_state = {}
for name, var, (h_low, h_high) in model_defs:
    s, e = h_to_idx(target_h, h_low, h_high)
    if var == "T" and h_low == 0 and h_high == 2:
        n_in = len(v_surface_idx) + 1
    else:
        n_in = 22 + 5  # 22ch + T2m + RH2m + Ps + CLWC + CIC
    m = BRNN(n_in, e - s, config.HIDDEN_NODES, config.DROPOUT_RATE)
    m.load_state_dict(torch.load(os.path.join(models_dir, f"{name}.pt"),
                                  map_location=device, weights_only=True))
    m.to(device).eval()
    models_state[name] = m

# Predict
surface_test = np.column_stack([
    t2m_clean.reshape(-1, 1), rh2m_clean.reshape(-1, 1),
    sp_clean.reshape(-1, 1) / 100.0, clwc_prof.reshape(-1, 1), cic_prof.reshape(-1, 1)
])

T_pred = np.zeros((test_mask.sum(), 93))
RH_pred = np.zeros((test_mask.sum(), 93))

for name, var, (h_low, h_high) in model_defs:
    s, e = h_to_idx(target_h, h_low, h_high)
    m = models_state[name]
    if var == "T" and h_low == 0 and h_high == 2:
        X = torch.FloatTensor(np.column_stack([bt_clean[test_mask][:, v_surface_idx],
                                                t2m_clean[test_mask].reshape(-1, 1)])).to(device)
    else:
        X = torch.FloatTensor(np.column_stack([bt_clean[test_mask], surface_test[test_mask]])).to(device)
    with torch.no_grad():
        out = m(X).cpu().numpy()
    if var == "T":
        T_pred[:, s:e] = out * 100.0 + 200.0
    else:
        RH_pred[:, s:e] = out * 100.0

# Metrics
T_err = T_pred - T_target[test_mask]
RH_err = RH_pred - RH_target[test_mask]

print(f"\n{'Height':>8s}  {'T_RMSE':>8s}  {'T_Bias':>8s}  {'RH_RMSE':>8s}  {'RH_Bias':>8s}")
print("-" * 52)
for km in [0.0, 0.5, 1, 2, 3, 5, 8, 10]:
    idx = np.argmin(np.abs(target_h - km * 1000))
    t_rmse = np.sqrt(np.mean(T_err[:, idx]**2))
    t_bias = np.mean(T_err[:, idx])
    rh_rmse = np.sqrt(np.mean(RH_err[:, idx]**2))
    rh_bias = np.mean(RH_err[:, idx])
    print(f"{target_h[idx]/1000:6.1f}km  {t_rmse:7.3f}K  {t_bias:+7.3f}K  {rh_rmse:7.2f}%  {rh_bias:+7.2f}%")

T_total_rmse = np.sqrt(np.mean(T_err**2))
RH_total_rmse = np.sqrt(np.mean(RH_err**2))
print(f"\n  T  total: RMSE={T_total_rmse:.3f}K  Bias={np.mean(T_err):+.3f}K  MAE={np.mean(np.abs(T_err)):.3f}K")
print(f"  RH total: RMSE={RH_total_rmse:.2f}%  Bias={np.mean(RH_err):+.2f}%  MAE={np.mean(np.abs(RH_err)):.2f}%")

# Per-channel OMB after correction
print(f"\n  Per-channel OMB after correction (stable V-band):")
for ch in v_surface_idx:
    omb_ch = omb_corrected[ch, clean][test_mask]
    print(f"    ch{ch:2d} {channel_freq[ch]:7.3f}GHz: {np.mean(omb_ch):+.3f}±{np.std(omb_ch):.3f}K")

# Save
results = {
    "T_pred": T_pred, "RH_pred": RH_pred,
    "T_true": T_target[test_mask], "RH_true": RH_target[test_mask],
    "heights": target_h, "test_mask": test_mask,
}
with open(os.path.join(PROJ, "results", "mp3000a_v2_test_results.pkl"), "wb") as f:
    pickle.dump(results, f)

print(f"\nModels: {models_dir}/")
print(f"Results: results/mp3000a_v2_test_results.pkl")
print("=" * 60)
