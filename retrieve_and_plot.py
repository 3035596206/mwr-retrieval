#!/usr/bin/env python3
"""Retrieval pipeline: MP-3000A Obs_BT → BRNN v4 → T(z), RH(z) profiles.
Replicates exact v4 preprocessing and plots results vs ERA5 truth.

Usage: python3 retrieve_and_plot.py
"""

import os, sys, warnings
import numpy as np
import xarray as xr
import torch, torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

warnings.filterwarnings("ignore")

PROJ = "/Users/ink/test/mwr_retrieval"
NC_FILE = "/Users/ink/test/54623_MP_3000A_ERA5Model_20231101_20240331_areamean_selfgb.nc"
sys.path.insert(0, os.path.join(PROJ, "src")); sys.path.insert(0, PROJ)
import config

RD = 287.058; G = 9.80665; EPSILON = 0.622; N_CH = 22
DEVICE = "cpu"

# ── Model definition (must match training) ──
class BRNN(nn.Module):
    def __init__(self, n_in, n_out, h=256, dr=0.3):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(n_in)
        self.fc1 = nn.Linear(n_in, h); self.bn1 = nn.BatchNorm1d(h); self.drop1 = nn.Dropout(dr)
        self.fc2 = nn.Linear(h, h);     self.bn2 = nn.BatchNorm1d(h); self.drop2 = nn.Dropout(dr)
        self.fc_out = nn.Linear(h, n_out); self.relu = nn.ReLU(inplace=True); self.sig = nn.Sigmoid()
    def forward(self, x):
        x = self.input_bn(x)
        x = self.relu(self.fc1(x)); x = self.bn1(x); x = self.drop1(x)
        x = self.relu(self.fc2(x)); x = self.bn2(x); x = self.drop2(x)
        x = self.sig(self.fc_out(x))
        return x

# ── Physics helpers (exact copy from training) ──
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
    n_prof = T.shape[1]
    T_out = np.zeros((n_prof, len(target_h))); RH_out = np.zeros((n_prof, len(target_h)))
    for i in range(n_prof):
        z, ti, ri = Z_m[:, i], T[:, i], RH[:, i]
        if z[0] > z[-1]: z = z[::-1]; ti = ti[::-1]; ri = ri[::-1]
        mask = np.concatenate([[True], np.diff(z) > 0.1])
        if mask.sum() < 3: continue
        zu, tu, ru = z[mask], ti[mask], ri[mask]
        try:
            T_out[i] = interp1d(zu, tu, kind='linear', bounds_error=False, fill_value=(tu[0], tu[-1]))(target_h)
            RH_out[i] = interp1d(zu, ru, kind='linear', bounds_error=False, fill_value=(ru[0], ru[-1]))(target_h)
        except: pass
    return T_out, RH_out

def h_to_idx(target_h, low_km, high_km):
    lo, hi = low_km*1000, high_km*1000
    idxs = [i for i, h in enumerate(target_h) if lo <= h <= hi]
    return (idxs[0], idxs[-1]+1)

# ── 1. Load data ──
print("="*60)
print("MP-3000A BRNN v4 Retrieval Pipeline")
print("="*60)
print("\n[1/6] Loading data...")
ds = xr.open_dataset(NC_FILE)
P_hpa = ds["Level_Pressure"].values; T_lev = ds["Level_Temperature"].values
q_lev = ds["Level_H2O"].values; sfc_alt = ds["Surface_Altitude"].values
sfc_P = ds["Surface_Pressure"].values; sfc_T = ds["Temperature_2M"].values
sfc_q = ds["H2O_2M"].values; clwc = ds["Column_Liquid_Cloud"].values
cic = ds["Column_Ice_Cloud"].values
obs_bt = ds["Obs_BT"].values; sim_bt = ds["Sim_BT"].values
qc_flag = ds["QC_Flag"].values; rain_flag = ds["Rain_Flag"].values
prof_idx = ds["Profile_Index"].values.astype(int)
obs_t2m = ds["Obs_Temperature_2M"].values; obs_q2m = ds["Obs_H2O_2M"].values
obs_sp = ds["Obs_Surface_Pressure"].values; obs_time = ds["Date_Time"].values
obs_ir = ds["IR_Temperature"].values; channel_freq = ds["Central_Frequency"].values
ds.close()
print(f"  {T_lev.shape[1]} profiles, {len(qc_flag)} obs")

# ── 2. Physics ──
print("\n[2/6] Physics preprocessing...")
RH_lev = np.zeros_like(T_lev)
for i in range(T_lev.shape[1]):
    RH_lev[:, i] = q_to_rh(q_lev[:, i], T_lev[:, i], P_hpa[:, i] if P_hpa.ndim>1 else P_hpa)
sfc_RH = q_to_rh(sfc_q, sfc_T, sfc_P)
obs_RH2m = q_to_rh(obs_q2m, obs_t2m, obs_sp)
Z_m = pressure_to_height(P_hpa, T_lev, q_lev, sfc_alt)
target_h = np.array(config.HEIGHT_GRID)
T_93, RH_93 = interpolate_to_mwr_grid(Z_m, T_lev, RH_lev, target_h)
valid_prof = np.any(T_93 != 0, axis=1)
print(f"  Valid profiles: {valid_prof.sum()}/{T_lev.shape[1]}")

# ── 3. Filtering (same as v4) ──
print("\n[3/6] Advanced filtering (v4 criteria)...")
clean = (qc_flag == 0) & (rain_flag == 0)
omb = obs_bt - sim_bt

# CLWC
clwc_danger = clwc > 750
for i in range(len(clean)):
    if clean[i] and clwc_danger[prof_idx[i]]: clean[i] = False

# K-band OMB screening
k_ch = channel_freq < 51
for ch_idx in np.where(k_ch)[0]:
    clean_idx = np.where(clean)[0]
    if len(clean_idx) == 0: break
    omb_ch = omb[ch_idx, clean_idx]
    mu, std = np.mean(omb_ch), np.std(omb_ch)
    clean[clean_idx[np.abs(omb_ch - mu) > 2.5 * std]] = False

# Profile matching
for i in range(len(clean)):
    if clean[i] and (prof_idx[i] >= T_93.shape[0] or not valid_prof[prof_idx[i]]):
        clean[i] = False
print(f"  Clean obs: {clean.sum()} ({100*clean.sum()/len(clean):.1f}%)")

# ── 4. BT correction + Features ──
print("\n[4/6] BT correction (Winsorized) + features...")
obs_bt_clean = obs_bt[:, clean].T
sim_bt_clean = sim_bt[:, clean].T
slope = np.ones(N_CH); intercept = np.zeros(N_CH)

for ch in range(N_CH):
    x, y = obs_bt_clean[:, ch], sim_bt_clean[:, ch]
    A = np.column_stack([x, np.ones(len(x))])
    c0 = np.linalg.lstsq(A, y, rcond=None)[0]
    resid = y - (c0[0]*x + c0[1])
    clip = 10.0 if ch < 8 else 5.0
    yc = c0[0]*x + c0[1] + np.clip(resid, -clip, clip)
    coeff = np.linalg.lstsq(np.column_stack([x, np.ones(len(x))]), yc, rcond=None)[0]
    slope[ch], intercept[ch] = coeff[0], coeff[1]

obs_bt_corr = slope[np.newaxis,:] * obs_bt_clean + intercept[np.newaxis,:]
print(f"  OMB before: {np.mean(np.abs(omb[:,clean])):.2f}±{np.std(omb[:,clean]):.2f}K")
print(f"  OMB after:  {np.mean(np.abs(obs_bt_corr - sim_bt_clean)):.2f}±{np.std(obs_bt_corr - sim_bt_clean):.2f}K")

# Build samples
N = clean.sum()
prof_idx_c = prof_idx[clean]
T_target = T_93[prof_idx_c]; RH_target = RH_93[prof_idx_c]
t2m_c = obs_t2m[clean]; rh2m_c = obs_RH2m[clean]; sp_c = obs_sp[clean]
ir_c = obs_ir[clean]; time_c = obs_time[clean]
clwc_c = clwc[prof_idx_c]; cic_c = cic[prof_idx_c]

hour_raw = np.array([(t / 100.0) % 100.0 for t in time_c])
month_raw = np.array([(t // 10000) % 100 for t in time_c])
hour_sin = np.sin(2*np.pi*hour_raw/24.0); hour_cos = np.cos(2*np.pi*hour_raw/24.0)
month_sin = np.sin(2*np.pi*month_raw/12.0); month_cos = np.cos(2*np.pi*month_raw/12.0)

surface_feats = np.column_stack([
    t2m_c.reshape(-1,1), rh2m_c.reshape(-1,1), (sp_c/100.0).reshape(-1,1),
    ir_c.reshape(-1,1), clwc_c.reshape(-1,1), cic_c.reshape(-1,1),
])

# Profile-group split (same seed, same shuffle)
unique_profs = np.unique(prof_idx_c)
np.random.seed(42); np.random.shuffle(unique_profs)
n_prof = len(unique_profs)
n_tr, n_va = int(n_prof*0.70), int(n_prof*0.15)
test_set = set(unique_profs[n_tr+n_va:])
test_mask = np.array([p in test_set for p in prof_idx_c])
print(f"  Test set: {test_mask.sum()} obs from {len(test_set)} profiles")

v_surface = (channel_freq >= 53.0) & (channel_freq <= 59.0)
v_surface_idx = np.where(v_surface)[0]

# ── 5. Load models + predict ──
print("\n[5/6] Loading BRNN v4 models...")
models_dir = os.path.join(PROJ, "models_mp3000a_v4")
model_defs = [
    ("brnn_T_0-2km",  "T",  (0, 2)),
    ("brnn_T_2-8km",  "T",  (2, 8)),
    ("brnn_T_8-10km", "T",  (8, 10)),
    ("brnn_RH_0-2km", "RH", (0, 2)),
    ("brnn_RH_2-8km", "RH", (2, 8)),
    ("brnn_RH_8-10km","RH", (8, 10)),
]

predictions = {}
for name, var, (lo, hi) in model_defs:
    if var == "T" and lo == 0 and hi == 2:
        X_test = np.column_stack([obs_bt_corr[test_mask][:, v_surface_idx], t2m_c[test_mask].reshape(-1,1)])
    else:
        X_test = np.column_stack([obs_bt_corr[test_mask], surface_feats[test_mask]])

    n_in, s, e = X_test.shape[1], *h_to_idx(target_h, lo, hi)
    n_out = e - s
    print(f"  {name}: {n_in}→{n_out} [{lo}-{hi}km]")

    model = BRNN(n_in, n_out, config.HIDDEN_NODES, config.DROPOUT_RATE).to(DEVICE)
    state = torch.load(os.path.join(models_dir, f"{name}.pt"), map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        y_pred = model(torch.FloatTensor(X_test)).cpu().numpy()

    if var == "T":
        y_pred = y_pred * 100.0 + 200.0  # denormalize: [0,1] → [200,300]K
    else:
        y_pred = y_pred * 100.0           # denormalize: [0,1] → [0,100]%

    predictions[name] = (s, e, y_pred)

# ── 6. Assemble full profiles ──
print("\n[6/6] Assembling & plotting...")
n_test = test_mask.sum()
T_pred  = np.zeros((n_test, len(target_h)))
RH_pred = np.zeros((n_test, len(target_h)))
T_true  = T_target[test_mask]
RH_true = RH_target[test_mask]

for name, var, (lo, hi) in model_defs:
    s, e, y_pred = predictions[name]
    if var == "T":
        T_pred[:, s:e] = y_pred
    else:
        RH_pred[:, s:e] = y_pred

# Metrics
t_rmse = np.sqrt(np.mean((T_pred - T_true)**2))
rh_rmse = np.sqrt(np.mean((RH_pred - RH_true)**2))
t_bias = np.mean(T_pred - T_true)
rh_bias = np.mean(RH_pred - RH_true)
print(f"  T RMSE: {t_rmse:.2f}K  Bias: {t_bias:+.2f}K")
print(f"  RH RMSE: {rh_rmse:.2f}%  Bias: {rh_bias:+.2f}%")

# Per-layer metrics
layer_rmse_T = np.sqrt(np.mean((T_pred - T_true)**2, axis=0))
layer_rmse_RH = np.sqrt(np.mean((RH_pred - RH_true)**2, axis=0))

# ── PLOTS ──
os.makedirs(os.path.join(PROJ, "results"), exist_ok=True)

# Figure 1: Error profiles
fig, axes = plt.subplots(1, 2, figsize=(10, 8))
ax = axes[0]
ax.plot(layer_rmse_T, target_h/1000, 'b-', linewidth=2)
ax.fill_betweenx(target_h/1000, 0, layer_rmse_T, alpha=0.2, color='b')
ax.set_xlabel("T RMSE [K]"); ax.set_ylabel("Height [km]"); ax.set_title("Temperature Error Profile")
ax.grid(True, alpha=0.3); ax.set_ylim(0, 10)

ax = axes[1]
ax.plot(layer_rmse_RH, target_h/1000, 'r-', linewidth=2)
ax.fill_betweenx(target_h/1000, 0, layer_rmse_RH, alpha=0.2, color='r')
ax.set_xlabel("RH RMSE [%]"); ax.set_title("Humidity Error Profile")
ax.grid(True, alpha=0.3); ax.set_ylim(0, 10)

fig.suptitle(f"BRNN v4 Retrieval Errors — Test Set ({n_test} obs)\n"
             f"T RMSE={t_rmse:.2f}K Bias={t_bias:+.2f}K | RH RMSE={rh_rmse:.2f}% Bias={rh_bias:+.2f}%",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PROJ, "results/error_profiles.png"), dpi=150)
print("  Saved: results/error_profiles.png")

# Figure 2: Sample profiles (best, median, worst T RMSE)
per_sample_rmse_T = np.sqrt(np.mean((T_pred - T_true)**2, axis=1))
ranked = np.argsort(per_sample_rmse_T)
idxs = [ranked[0], ranked[len(ranked)//2], ranked[-1]]
labels = ["Best", "Median", "Worst"]

fig, axes = plt.subplots(1, 3, figsize=(14, 8))
for k, (idx, label) in enumerate(zip(idxs, labels)):
    ax = axes[k]
    ax.plot(T_true[idx], target_h/1000, 'k-', linewidth=2, label='ERA5 Truth')
    ax.plot(T_pred[idx], target_h/1000, 'b--', linewidth=2, label='BRNN v4')
    ax.set_xlabel("T [K]"); ax.set_ylabel("Height [km]")
    ax.set_title(f"{label} (T RMSE={per_sample_rmse_T[idx]:.2f}K)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.set_ylim(0, 10)

fig.suptitle("BRNN v4 Temperature Retrieval — Sample Profiles", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PROJ, "results/sample_T_profiles.png"), dpi=150)
print("  Saved: results/sample_T_profiles.png")

# Figure 3: RH sample profiles
per_sample_rmse_RH = np.sqrt(np.mean((RH_pred - RH_true)**2, axis=1))
ranked_rh = np.argsort(per_sample_rmse_RH)
idxs_rh = [ranked_rh[0], ranked_rh[len(ranked_rh)//2], ranked_rh[-1]]

fig, axes = plt.subplots(1, 3, figsize=(14, 8))
for k, (idx, label) in enumerate(zip(idxs_rh, labels)):
    ax = axes[k]
    ax.plot(RH_true[idx], target_h/1000, 'k-', linewidth=2, label='ERA5 Truth')
    ax.plot(RH_pred[idx], target_h/1000, 'r--', linewidth=2, label='BRNN v4')
    ax.set_xlabel("RH [%]"); ax.set_ylabel("Height [km]")
    ax.set_title(f"{label} (RH RMSE={per_sample_rmse_RH[idx]:.1f}%)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.set_ylim(0, 10)

fig.suptitle("BRNN v4 Relative Humidity Retrieval — Sample Profiles", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PROJ, "results/sample_RH_profiles.png"), dpi=150)
print("  Saved: results/sample_RH_profiles.png")

# Figure 4: Scatter density
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
ax = axes[0]; hb = ax.hexbin(T_true.flatten(), T_pred.flatten(), gridsize=60, cmap='Blues', mincnt=1)
ax.plot([200, 300], [200, 300], 'k--', linewidth=1); ax.set_xlabel("ERA5 T [K]"); ax.set_ylabel("BRNN T [K]")
ax.set_title(f"Temperature (N={n_test*len(target_h):,})"); plt.colorbar(hb, ax=ax, label='count')

ax = axes[1]; hb = ax.hexbin(RH_true.flatten(), RH_pred.flatten(), gridsize=60, cmap='Reds', mincnt=1)
ax.plot([0, 100], [0, 100], 'k--', linewidth=1); ax.set_xlabel("ERA5 RH [%]"); ax.set_ylabel("BRNN RH [%]")
ax.set_title(f"Relative Humidity"); plt.colorbar(hb, ax=ax, label='count')

fig.suptitle("BRNN v4: Predicted vs ERA5 Truth", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PROJ, "results/scatter_density.png"), dpi=150)
print("  Saved: results/scatter_density.png")

print(f"\n{'='*60}")
print(f"DONE — 4 plots in {PROJ}/results/")
print(f"  T RMSE={t_rmse:.2f}K  RH RMSE={rh_rmse:.2f}%")
print(f"  (Paper reference: T=1.5K, RH=12-13%)")
print(f"{'='*60}")
