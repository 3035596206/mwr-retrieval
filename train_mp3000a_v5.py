#!/usr/bin/env python3
"""Train BRNN models from MP-3000A data — v5.

Improvements over v4:
  1. Channel engineering: K-band differentials (water vapor shape), V-band ratios (T profile)
  2. Height-weighted MSE loss: 2× weight on 6-8km (weakest microwave sensitivity)
  3. All v4 improvements retained (K-band 2.5σ filter, Winsorize, IR temp)
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


class WeightedMSELoss(nn.Module):
    """MSE with per-output-layer weights. Higher weight = more emphasis."""
    def __init__(self, weights):
        super().__init__()
        self.weights = torch.tensor(weights, dtype=torch.float32)

    def forward(self, pred, target):
        w = self.weights.to(pred.device)
        return torch.mean(w * (pred - target) ** 2)


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
            Z[k, i] = Z[k+1, i] + RD * 0.5*(Tv[k]+Tv[k+1]) / G * dp / (0.5*(P_pa[k]+P_pa[k+1]))
        Z[:, i] += sfc_alt_km[i] * 1000.0
    return Z


def interpolate_to_mwr_grid(Z_m, T, RH, target_h):
    from scipy.interpolate import interp1d
    n_prof = T.shape[1]
    T_out = np.zeros((n_prof, len(target_h)))
    RH_out = np.zeros((n_prof, len(target_h)))
    for i in range(n_prof):
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


def engineer_bt_features(bt_22ch, freqs):
    """Add physics-informed channel features.

    K-band (ch0-7,  22-30 GHz): water vapor sensitive
    V-band (ch8-21, 51-59 GHz): temperature profile sensitive

    Returns augmented BT array with channel engineering.
    """
    k = np.arange(8)   # K-band indices
    v = np.arange(8, 22)  # V-band indices

    feats = [bt_22ch]  # original 22 channels

    # 1. K-band inter-channel differences (water vapor profile shape)
    for i in range(7):
        feats.append((bt_22ch[:, i] - bt_22ch[:, i+1]).reshape(-1, 1))

    # 2. V-band "wing" differences (temperature weighting function gradient)
    #    ch14 (54.4 GHz) is the peak — differences from peak encode T profile
    peak = 14
    for offset in [1, 2, 3, 4]:
        if peak + offset < 22:
            feats.append((bt_22ch[:, peak] - bt_22ch[:, peak+offset]).reshape(-1, 1))
        if peak - offset >= 8:
            feats.append((bt_22ch[:, peak] - bt_22ch[:, peak-offset]).reshape(-1, 1))

    # 3. K-V band ratio (water vapor indicator — K bright / V bright)
    k_mean = np.mean(bt_22ch[:, k], axis=1, keepdims=True)
    v_mean = np.mean(bt_22ch[:, v], axis=1, keepdims=True)
    feats.append(k_mean / np.maximum(v_mean, 1.0))

    # 4. V-band slope (high-vs-low channel difference → lapse rate indicator)
    v_high = np.mean(bt_22ch[:, 18:22], axis=1, keepdims=True)  # 56-59 GHz
    v_low  = np.mean(bt_22ch[:, 8:12], axis=1, keepdims=True)   # 51-53 GHz
    feats.append(v_high - v_low)

    return np.column_stack(feats)


# ============================================================
print("=" * 60)
print("MP-3000A → BRNN v5 (Channel engineering + Weighted loss)")
print("=" * 60)

# ---- 1. Load ----
print("\n[1/8] Loading...")
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

# ---- 2. Physics ----
print("\n[2/8] Physics preprocessing...")
RH_lev = np.zeros_like(T_lev)
for i in range(T_lev.shape[1]):
    RH_lev[:, i] = q_to_rh(q_lev[:, i], T_lev[:, i],
                            P_hpa[:, i] if P_hpa.ndim>1 else P_hpa)
sfc_RH = q_to_rh(sfc_q, sfc_T, sfc_P)
obs_RH2m = q_to_rh(obs_q2m, obs_t2m, obs_sp)

Z_m = pressure_to_height(P_hpa, T_lev, q_lev, sfc_alt)
target_h = np.array(config.HEIGHT_GRID)
T_93, RH_93 = interpolate_to_mwr_grid(Z_m, T_lev, RH_lev, target_h)
valid_prof = np.any(T_93 != 0, axis=1)

# ---- 3. Filtering (same as v4) ----
print("\n[3/8] Filtering...")
clean = (qc_flag == 0) & (rain_flag == 0)
omb = obs_bt - sim_bt

# CLWC screening
clwc_danger = clwc > 750
for i in range(len(clean)):
    if clean[i] and clwc_danger[prof_idx[i]]:
        clean[i] = False

# K-band OMB 2.5σ screening
k_ch = channel_freq < 51
for ch_idx in np.where(k_ch)[0]:
    clean_idx = np.where(clean)[0]
    if len(clean_idx) == 0: break
    omb_ch = omb[ch_idx, clean_idx]
    mu, std = np.mean(omb_ch), np.std(omb_ch)
    bad = np.abs(omb_ch - mu) > 2.5 * std
    clean[clean_idx[bad]] = False

# Profile matching
for i in range(len(clean)):
    if clean[i] and (prof_idx[i] >= T_93.shape[0] or not valid_prof[prof_idx[i]]):
        clean[i] = False

print(f"  Clean: {clean.sum()} obs")

# ---- 4. BT correction (Winsorized, same as v4) ----
print("\n[4/8] BT correction (Winsorized)...")
obs_bt_clean_raw = obs_bt[:, clean].T
sim_bt_clean_raw = sim_bt[:, clean].T

slope = np.ones(N_CH); intercept = np.zeros(N_CH)
for ch in range(N_CH):
    x = obs_bt_clean_raw[:, ch]; y = sim_bt_clean_raw[:, ch]
    coeff_init = np.linalg.lstsq(np.column_stack([x, np.ones(len(x))]), y, rcond=None)[0]
    resid = y - (coeff_init[0]*x + coeff_init[1])
    clip = 10.0 if ch < 8 else 5.0
    y_clipped = coeff_init[0]*x + coeff_init[1] + np.clip(resid, -clip, clip)
    coeff = np.linalg.lstsq(np.column_stack([x, np.ones(len(x))]), y_clipped, rcond=None)[0]
    slope[ch] = coeff[0]; intercept[ch] = coeff[1]

obs_bt_corrected = slope[np.newaxis, :] * obs_bt_clean_raw + intercept[np.newaxis, :]

# ---- 5. Build features (IMPROVED: channel engineering) ----
print("\n[5/8] Building features with channel engineering...")
N = clean.sum()
prof_idx_clean = prof_idx[clean]
T_target  = T_93[prof_idx_clean]
RH_target = RH_93[prof_idx_clean]

t2m_clean  = obs_t2m[clean]; rh2m_clean = obs_RH2m[clean]
sp_clean   = obs_sp[clean];  ir_clean   = obs_ir[clean]
clwc_clean = clwc[prof_idx_clean]; cic_clean = cic[prof_idx_clean]

# Engineered BT features
bt_engineered = engineer_bt_features(obs_bt_corrected, channel_freq)
N_CH_ENG = bt_engineered.shape[1]
print(f"  BT features: {N_CH} channels → {N_CH_ENG} engineered ({N_CH_ENG - N_CH} derived)")

# Profile-group split
unique_profs = np.unique(prof_idx_clean)
np.random.seed(42); np.random.shuffle(unique_profs)
n_prof = len(unique_profs)
n_train = int(n_prof * 0.70); n_val = int(n_prof * 0.15)
train_set = set(unique_profs[:n_train])
val_set   = set(unique_profs[n_train:n_train+n_val])
test_set  = set(unique_profs[n_train+n_val:])
train_mask = np.array([p in train_set for p in prof_idx_clean])
val_mask   = np.array([p in val_set for p in prof_idx_clean])
test_mask  = np.array([p in test_set for p in prof_idx_clean])
print(f"  Train: {train_mask.sum()} obs, Val: {val_mask.sum()}, Test: {test_mask.sum()}")

# ---- 6. Build height weights ----
print("\n[6/8] Building height weights...")
# Weight = 2.0 for 5-8km, 1.0 elsewhere
h_weights = np.ones(93)
for i, h in enumerate(target_h):
    if 5000 <= h <= 8000:
        h_weights[i] = 2.0  # double weight on weak-sensitivity region

# Per-model weight slices
def get_model_weights(s, e):
    return h_weights[s:e]

# ---- 7. Train ----
print("\n[7/8] Training BRNN models...")

# For T_0-2km model, use V-band surface channels (but engineered)
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
models_dir = os.path.join(PROJ, "models_mp3000a_v5")
os.makedirs(models_dir, exist_ok=True)

for name, var, (h_low, h_high) in model_defs:
    print(f"\n{'='*60}")
    print(f"Training {name} ({h_low}-{h_high}km)")

    surface_feats = np.column_stack([
        t2m_clean.reshape(-1, 1), rh2m_clean.reshape(-1, 1),
        (sp_clean / 100.0).reshape(-1, 1), ir_clean.reshape(-1, 1),
        clwc_clean.reshape(-1, 1), cic_clean.reshape(-1, 1),
    ])

    if var == "T" and h_low == 0 and h_high == 2:
        # Original V-band surface channels only (paper-specified for T_0-2km)
        X = np.column_stack([obs_bt_corrected[:, v_surface_idx], t2m_clean.reshape(-1, 1)])
    else:
        X = np.column_stack([bt_engineered, surface_feats])

    s, e = h_to_idx(target_h, h_low, h_high)
    if var == "T":
        y = (T_target[:, s:e] - 200.0) / 100.0
    else:
        y = RH_target[:, s:e] / 100.0

    n_in, n_out = X.shape[1], y.shape[1]
    w = get_model_weights(s, e)
    print(f"  Input: {n_in}, Output: {n_out}, Height weights: [{w[0]:.1f}..{w[-1]:.1f}]")

    X_train, y_train = X[train_mask], y[train_mask]
    X_val,   y_val   = X[val_mask],   y[val_mask]

    train_loader = DataLoader(TensorDataset(
        torch.FloatTensor(X_train), torch.FloatTensor(y_train)),
        batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(
        torch.FloatTensor(X_val), torch.FloatTensor(y_val)),
        batch_size=config.BATCH_SIZE, shuffle=False)

    model = BRNN(n_in, n_out, config.HIDDEN_NODES, config.DROPOUT_RATE).to(device)
    criterion = WeightedMSELoss(w)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    best_val, best_state, patience = float("inf"), None, 0
    for epoch in range(config.MAX_EPOCHS):
        model.train()
        train_loss = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(Xb)
            loss = criterion(pred, yb) + 0.001 * torch.mean(torch.diff(pred, n=2, dim=1) ** 2)
            loss.backward()
            optimizer.step()
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
            print(f"  Epoch {epoch+1}: train={train_loss:.6f} val={val_loss:.6f}")
        if patience >= config.EARLY_STOPPING_PATIENCE:
            print(f"  Early stop @ {epoch+1}")
            break

    model.load_state_dict(best_state)
    torch.save(best_state, os.path.join(models_dir, f"{name}.pt"))
    print(f"  Saved: best_val={best_val:.6f}")


# ---- 8. Evaluate ----
print(f"\n{'='*60}")
print("[8/8] Test evaluation")
print("=" * 60)

models_state = {}
for name, var, (h_low, h_high) in model_defs:
    s, e = h_to_idx(target_h, h_low, h_high)
    if var == "T" and h_low == 0 and h_high == 2:
        n_in = len(v_surface_idx) + 1
    else:
        n_in = N_CH_ENG + 6
    m = BRNN(n_in, e - s, config.HIDDEN_NODES, config.DROPOUT_RATE)
    m.load_state_dict(torch.load(os.path.join(models_dir, f"{name}.pt"),
                                  map_location=device, weights_only=True))
    m.to(device).eval()
    models_state[name] = m

surface_test = np.column_stack([
    t2m_clean.reshape(-1, 1), rh2m_clean.reshape(-1, 1),
    sp_clean.reshape(-1, 1) / 100.0, ir_clean.reshape(-1, 1),
    clwc_clean.reshape(-1, 1), cic_clean.reshape(-1, 1),
])
bt_test_all = bt_engineered[test_mask]
bt_test_raw = obs_bt_corrected[test_mask]

T_pred = np.zeros((test_mask.sum(), 93))
RH_pred = np.zeros((test_mask.sum(), 93))

for name, var, (h_low, h_high) in model_defs:
    s, e = h_to_idx(target_h, h_low, h_high)
    m = models_state[name]
    if var == "T" and h_low == 0 and h_high == 2:
        X = torch.FloatTensor(np.column_stack([
            bt_test_raw[:, v_surface_idx],
            t2m_clean[test_mask].reshape(-1, 1)])).to(device)
    else:
        X = torch.FloatTensor(np.column_stack([
            bt_test_all, surface_test[test_mask]])).to(device)
    with torch.no_grad():
        out = m(X).cpu().numpy()
    if var == "T":
        T_pred[:, s:e] = out * 100.0 + 200.0
    else:
        RH_pred[:, s:e] = out * 100.0

T_err = T_pred - T_target[test_mask]
RH_err = RH_pred - RH_target[test_mask]

print(f"\n{'Height':>8s}  {'T_RMSE':>8s}  {'T_Bias':>8s}  {'RH_RMSE':>8s}")
print("-" * 42)
for km in [0, 0.5, 1, 2, 3, 5, 6, 7, 8, 10]:
    idx = np.argmin(np.abs(target_h - km * 1000))
    print(f"{target_h[idx]/1000:6.1f}km  "
          f"{np.sqrt(np.mean(T_err[:,idx]**2)):7.3f}K  "
          f"{np.mean(T_err[:,idx]):+7.3f}K  "
          f"{np.sqrt(np.mean(RH_err[:,idx]**2)):7.2f}%")

T_total = np.sqrt(np.mean(T_err**2))
RH_total = np.sqrt(np.mean(RH_err**2))
print(f"\n  T  total: RMSE={T_total:.3f}K  Bias={np.mean(T_err):+.3f}K  MAE={np.mean(np.abs(T_err)):.3f}K")
print(f"  RH total: RMSE={RH_total:.2f}%  Bias={np.mean(RH_err):+.2f}%  MAE={np.mean(np.abs(RH_err)):.2f}%")

# Compare with v4
print(f"\n  v4 comparison: T_RMSE=1.264K, RH_RMSE=7.76%")
print(f"  v5 vs v4:      T_RMSE={T_total:.3f}K ({100*(T_total/1.264-1):+.1f}%), "
      f"RH_RMSE={RH_total:.2f}% ({100*(RH_total/7.76-1):+.1f}%)")

with open(os.path.join(PROJ, "results", "mp3000a_v5_results.pkl"), "wb") as f:
    pickle.dump({"T_pred": T_pred, "RH_pred": RH_pred,
                 "T_true": T_target[test_mask], "RH_true": RH_target[test_mask],
                 "heights": target_h}, f)

print(f"\nModels: {models_dir}/")
print("=" * 60)
