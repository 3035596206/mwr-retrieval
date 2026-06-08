#!/usr/bin/env python3
"""Train BRNN models from MP-3000A radiometer + ERA5 profile data.

Input:  54623_MP_3000A_ERA5Model_20231101_20240331_areamean_selfgb.nc
        - 3,453 atmospheric profiles (137 levels T + q)
        - 20,142 observations, 22-channel Obs_BT
        - QC_Flag + Rain_Flag for filtering

Pipeline:
  1. Load profiles → convert q(g/kg)→RH(%), compute geopotential height
  2. Interpolate 137 pressure levels → 93-layer MWR height grid
  3. Match Obs_BT to profiles via Profile_Index
  4. QC filter (QC=0, Rain=0)
  5. Train 6 BRNN models: Obs_BT + surface → profile(T,RH)
"""

import os, sys, pickle, warnings
import numpy as np
import xarray as xr
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

# ---- paths ----
PROJ = "/Users/ink/test/mwr_retrieval"
NC_FILE = "/Users/ink/test/54623_MP_3000A_ERA5Model_20231101_20240331_areamean_selfgb.nc"
sys.path.insert(0, os.path.join(PROJ, "src"))
sys.path.insert(0, PROJ)
import config

# ---- constants ----
RD = 287.058      # J/(kg·K) dry air gas constant
G  = 9.80665      # m/s²
EPSILON = 0.622   # R_dry / R_vapor

# ---- BRNN model (same architecture as paper) ----
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
    """Convert specific humidity [g/kg] to RH [%] using Magnus formula."""
    q = q_gkg / 1000.0                                    # g/kg → kg/kg
    e_hpa = q * P_hpa / (EPSILON + (1 - EPSILON) * q)     # vapor pressure [hPa]
    Tc = T - 273.16
    es_hpa = 6.1078 * np.exp(17.2693882 * Tc / (Tc + 237.3))
    return np.clip(e_hpa / es_hpa * 100.0, 0.0, 100.0)


def pressure_to_height(P_hpa, T, q_gkg, sfc_alt_km):
    """Compute geopotential height [m] from pressure using hydrostatic equation.

    P_hpa:     [n_levels] or [n_levels, n_profiles]
    T:         [n_levels, n_profiles]
    q_gkg:     [n_levels, n_profiles]
    sfc_alt_km: [n_profiles] in km

    Returns Z_m: [n_levels, n_profiles] in meters
    """
    n_lev, n_prof = T.shape
    Z = np.zeros_like(T)

    for i in range(n_prof):
        Tv = T[:, i] * (1.0 + 0.608 * q_gkg[:, i] / 1000.0)  # virtual T
        P_pa = P_hpa[:, i] * 100.0 if P_hpa.ndim > 1 else P_hpa * 100.0

        # Integrate upward from surface (index -1, highest P) to top (index 0)
        Z[-1, i] = 0.0                             # surface anchor
        for k in range(n_lev - 2, -1, -1):
            dp = P_pa[k+1] - P_pa[k]               # >0 since P increases w/ index
            Tv_avg = 0.5 * (Tv[k] + Tv[k+1])
            P_avg = 0.5 * (P_pa[k] + P_pa[k+1])
            dZ = RD * Tv_avg / G * dp / P_avg      # hydrostatic: ΔZ > 0 going upward
            Z[k, i] = Z[k+1, i] + dZ               # Z increases as k→0 (higher altitude)

        # Shift all heights by surface altitude (convert km→m)
        Z[:, i] += sfc_alt_km[i] * 1000.0

    return Z


def interpolate_to_mwr_grid(Z_m, T, RH, target_h):
    """Interpolate profiles from native levels to 93-layer MWR grid.

    Z_m:     [n_levels, n_profiles]
    T:       [n_levels, n_profiles]
    RH:      [n_levels, n_profiles]
    target_h: [93] in meters

    Returns T_out, RH_out: [n_profiles, 93]
    """
    from scipy.interpolate import interp1d

    n_prof = T.shape[1]
    n_target = len(target_h)
    T_out = np.zeros((n_prof, n_target))
    RH_out = np.zeros((n_prof, n_target))

    for i in range(n_prof):
        z = Z_m[:, i]
        # Ensure monotonic increasing
        if z[0] > z[-1]:
            z = z[::-1]
            t_i = T[::-1, i]
            rh_i = RH[::-1, i]
        else:
            t_i = T[:, i]
            rh_i = RH[:, i]

        # Remove duplicate z values
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


# ============================================================
# MAIN PIPELINE
# ============================================================
print("=" * 60)
print("MP-3000A → BRNN Training Pipeline")
print("=" * 60)

# ---- 1. Load data ----
print("\n[1/6] Loading NC file...")
ds = xr.open_dataset(NC_FILE)

# Profile data (137 levels × 3453 profiles)
P_hpa = ds["Level_Pressure"].values      # [137, 3453] or [137,]
T_lev  = ds["Level_Temperature"].values   # [137, 3453]
q_lev  = ds["Level_H2O"].values          # [137, 3453] g/kg
sfc_alt = ds["Surface_Altitude"].values  # [3453] km
sfc_P   = ds["Surface_Pressure"].values  # [3453] hPa
sfc_T   = ds["Temperature_2M"].values    # [3453] K
sfc_q   = ds["H2O_2M"].values            # [3453] g/kg
clwc    = ds["Column_Liquid_Cloud"].values  # [3453] g/m²

# Handle P_hpa shape
if P_hpa.ndim == 1:
    P_hpa = np.tile(P_hpa[:, np.newaxis], (1, T_lev.shape[1]))

# Observation data
obs_bt   = ds["Obs_BT"].values           # [22, 20142]
sim_bt   = ds["Sim_BT"].values           # [22, 20142]
qc_flag  = ds["QC_Flag"].values          # [20142]
rain_flag = ds["Rain_Flag"].values       # [20142]
prof_idx = ds["Profile_Index"].values.astype(int)  # [20142]
obs_time = ds["Date_Time"].values        # [20142] YYYYMMDDHH format
obs_t2m  = ds["Obs_Temperature_2M"].values  # [20142] K
obs_q2m  = ds["Obs_H2O_2M"].values       # [20142] g/kg
obs_sp   = ds["Obs_Surface_Pressure"].values  # [20142] hPa
channel_freq = ds["Central_Frequency"].values  # [22]

ds.close()

n_obs = len(qc_flag)
n_prof = T_lev.shape[1]
n_lev  = T_lev.shape[0]
print(f"  Profiles: {n_prof} (×{n_lev} levels), Observations: {n_obs}")

# ---- 2. Convert q → RH ----
print("\n[2/6] Converting specific humidity → RH...")
RH_lev = np.zeros_like(T_lev)
for i in range(n_prof):
    RH_lev[:, i] = q_to_rh(q_lev[:, i], T_lev[:, i],
                            P_hpa[:, i] if P_hpa.ndim > 1 else P_hpa)

sfc_RH = q_to_rh(sfc_q, sfc_T, sfc_P)
obs_RH2m = q_to_rh(obs_q2m, obs_t2m, obs_sp)

print(f"  Profile RH: {RH_lev.min():.1f} – {RH_lev.max():.1f}%")
print(f"  2m RH: {sfc_RH.min():.1f} – {sfc_RH.max():.1f}%")

# ---- 3. Compute height + interpolate to 93-layer grid ----
print("\n[3/6] Computing geopotential height + interpolating to 93 layers...")
Z_m = pressure_to_height(P_hpa, T_lev, q_lev, sfc_alt)
target_h = np.array(config.HEIGHT_GRID)

T_93, RH_93 = interpolate_to_mwr_grid(Z_m, T_lev, RH_lev, target_h)

# Check for failed profiles (all zeros)
valid_prof = np.any(T_93 != 0, axis=1)
print(f"  Valid profiles: {valid_prof.sum()} / {n_prof}")
print(f"  T_93 range: {T_93[valid_prof].min():.1f} – {T_93[valid_prof].max():.1f} K")
print(f"  RH_93 range: {RH_93[valid_prof].min():.1f} – {RH_93[valid_prof].max():.1f}%")

# ---- 4. Match observations to profiles ----
print("\n[4/6] Matching observations to profiles...")
clean_mask = (qc_flag == 0) & (rain_flag == 0) & \
             (prof_idx < n_prof)  # safety

# Check profile is valid for each observation
for i in range(n_obs):
    if clean_mask[i] and not valid_prof[prof_idx[i]]:
        clean_mask[i] = False

print(f"  Clean observations: {clean_mask.sum()} / {n_obs}")

# Build training dataset
prof_idx_clean = prof_idx[clean_mask]
obs_bt_clean   = obs_bt[:, clean_mask].T      # [N, 22]
obs_t2m_clean  = obs_t2m[clean_mask]           # [N]
obs_rh2m_clean = obs_RH2m[clean_mask]          # [N]
obs_sp_clean   = obs_sp[clean_mask]            # [N]
obs_time_clean = obs_time[clean_mask]

T_target  = T_93[prof_idx_clean]    # [N, 93]
RH_target = RH_93[prof_idx_clean]   # [N, 93]

N = clean_mask.sum()
print(f"  Training samples: {N}")

# ---- 5. Train/Val/Test split (by time) ----
print("\n[5/6] Splitting data...")
time_sorted_idx = np.argsort(obs_time_clean)
train_n = int(N * 0.70)
val_n   = int(N * 0.15)

train_idx = time_sorted_idx[:train_n]
val_idx   = time_sorted_idx[train_n:train_n + val_n]
test_idx  = time_sorted_idx[train_n + val_n:]

print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

# ---- 6. Define models ----
# Identify V-band surface channels (53-58 GHz) for T_0-2km model
v_surface = (channel_freq >= 53.0) & (channel_freq <= 59.0)
v_surface_idx = np.where(v_surface)[0]
print(f"  V-band surface channels (53-59 GHz): {len(v_surface_idx)} channels")
print(f"    Frequencies: {channel_freq[v_surface_idx]}")

# Model definitions
model_defs = [
    # (name, variable, (h_low_km, h_high_km))
    ("brnn_T_0-2km",    "T",  (0, 2)),
    ("brnn_T_2-8km",    "T",  (2, 8)),
    ("brnn_T_8-10km",   "T",  (8, 10)),
    ("brnn_RH_0-2km",   "RH", (0, 2)),
    ("brnn_RH_2-8km",   "RH", (2, 8)),
    ("brnn_RH_8-10km",  "RH", (8, 10)),
]

# Height range → grid indices
def h_to_idx(target_h, low_km, high_km):
    low_m, high_m = low_km * 1000, high_km * 1000
    indices = [i for i, h in enumerate(target_h) if low_m <= h <= high_m]
    return (indices[0], indices[-1] + 1)

# ---- 7. Train each model ----
print("\n[6/6] Training BRNN models...")
device = "cpu"
if torch.backends.mps.is_available():
    device = "mps"

models_dir = os.path.join(PROJ, "models_mp3000a")
os.makedirs(models_dir, exist_ok=True)

for name, var, (h_low, h_high) in model_defs:
    print(f"\n{'='*60}")
    print(f"Training {name} (variable={var}, {h_low}-{h_high}km)")

    # Build input
    if var == "T" and h_low == 0 and h_high == 2:
        X = np.column_stack([obs_bt_clean[:, v_surface_idx], obs_t2m_clean.reshape(-1, 1)])
    else:
        X = np.column_stack([obs_bt_clean,
                             obs_t2m_clean.reshape(-1, 1),
                             obs_rh2m_clean.reshape(-1, 1),
                             (obs_sp_clean / 100.0).reshape(-1, 1)])  # Pa→hPa

    # Build target
    s, e = h_to_idx(target_h, h_low, h_high)
    if var == "T":
        y_target = T_target[:, s:e]         # K
        y = (y_target - 200.0) / 100.0      # normalize [200,300]K → [0,1] for sigmoid
    else:
        y_target = RH_target[:, s:e]        # %
        y = y_target / 100.0                # normalize to [0,1]

    n_in, n_out = X.shape[1], y.shape[1]
    print(f"  Input: {n_in}, Output: {n_out} (slice {s}:{e})")

    # Split
    X_train, y_train = X[train_idx], y[train_idx]
    X_val,   y_val   = X[val_idx],   y[val_idx]

    # DataLoaders
    train_loader = DataLoader(TensorDataset(
        torch.FloatTensor(X_train), torch.FloatTensor(y_train)),
        batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(
        torch.FloatTensor(X_val), torch.FloatTensor(y_val)),
        batch_size=config.BATCH_SIZE, shuffle=False)

    # Create model
    model = BRNN(n_in, n_out, hidden_size=config.HIDDEN_NODES,
                  dropout_rate=config.DROPOUT_RATE).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

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
                Xb, yb = Xb.to(device), yb.to(device)
                val_loss += criterion(model(Xb), yb).item() * Xb.size(0)
        val_loss /= len(val_loader.dataset)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{config.MAX_EPOCHS}: "
                  f"train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(best_state)
    torch.save(best_state, os.path.join(models_dir, f"{name}.pt"))
    print(f"  Saved: {name}.pt (val_loss={best_val_loss:.6f})")

# ---- 8. Quick test set evaluation ----
print(f"\n{'='*60}")
print("Test set evaluation")
print("=" * 60)

# Load all models and run prediction
models_state = {}
for name, var, (h_low, h_high) in model_defs:
    path = os.path.join(models_dir, f"{name}.pt")
    if os.path.exists(path):
        # Rebuild same architecture
        s, e = h_to_idx(target_h, h_low, h_high)
        if var == "T" and h_low == 0 and h_high == 2:
            n_in = len(v_surface_idx) + 1
        else:
            n_in = 22 + 3
        m = BRNN(n_in, e - s, config.HIDDEN_NODES, config.DROPOUT_RATE)
        m.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        m.to(device)
        m.eval()
        models_state[name] = m

# Predict on test set
X_test = np.column_stack([obs_bt_clean, obs_t2m_clean.reshape(-1, 1),
                          obs_rh2m_clean.reshape(-1, 1),
                          (obs_sp_clean / 100.0).reshape(-1, 1)])
X_test_v = np.column_stack([obs_bt_clean[:, v_surface_idx], obs_t2m_clean.reshape(-1, 1)])

T_pred = np.zeros((len(test_idx), 93))
RH_pred = np.zeros((len(test_idx), 93))

for name, var, (h_low, h_high) in model_defs:
    s, e = h_to_idx(target_h, h_low, h_high)
    m = models_state[name]
    if var == "T" and h_low == 0 and h_high == 2:
        X = torch.FloatTensor(X_test_v[test_idx]).to(device)
    else:
        X = torch.FloatTensor(X_test[test_idx]).to(device)
    with torch.no_grad():
        out = m(X).cpu().numpy()
    if var == "T":
        T_pred[:, s:e] = out * 100.0 + 200.0
    else:
        RH_pred[:, s:e] = out * 100.0

# Compute metrics
T_err = T_pred - T_target[test_idx]
RH_err = RH_pred - RH_target[test_idx]

T_rmse_per_layer = np.sqrt(np.mean(T_err**2, axis=0))
RH_rmse_per_layer = np.sqrt(np.mean(RH_err**2, axis=0))
T_bias_per_layer = np.mean(T_err, axis=0)

heights = np.array(target_h)

print(f"\n{'Height':>8s}  {'T_RMSE':>8s}  {'T_Bias':>8s}  {'RH_RMSE':>8s}")
print("-" * 42)
milestones = [0.5, 1, 2, 3, 5, 8, 10]
for km in milestones:
    idx = np.argmin(np.abs(heights - km * 1000))
    print(f"{heights[idx]/1000:6.2f}km  {T_rmse_per_layer[idx]:7.3f}K  "
          f"{T_bias_per_layer[idx]:+7.3f}K  {RH_rmse_per_layer[idx]:7.2f}%")

print(f"\n  T  total RMSE: {np.sqrt(np.mean(T_err**2)):.3f} K")
print(f"  T  total Bias: {np.mean(T_err):+.3f} K")
print(f"  RH total RMSE: {np.sqrt(np.mean(RH_err**2)):.2f} %")
print(f"  RH total Bias: {np.mean(RH_err):+.2f} %")

# Save predictions for later use
results = {
    "T_pred": T_pred, "RH_pred": RH_pred,
    "T_true": T_target[test_idx], "RH_true": RH_target[test_idx],
    "heights": heights, "test_idx": test_idx,
    "T_rmse_per_layer": T_rmse_per_layer,
    "RH_rmse_per_layer": RH_rmse_per_layer,
}
with open(os.path.join(PROJ, "results", "mp3000a_test_results.pkl"), "wb") as f:
    pickle.dump(results, f)

print(f"\nModels saved: {models_dir}/")
print(f"Results saved: results/mp3000a_test_results.pkl")
print("\n" + "=" * 60)
print("Training complete!")
print("=" * 60)
