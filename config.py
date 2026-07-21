"""Configuration for MWR atmospheric profile retrieval (Chapter 3)."""

# ============================================================
# RPG HATPRO 14-channel frequencies (GHz)
# ============================================================
MWR_CHANNELS = {
    "K_band": [22.24, 23.04, 23.84, 25.44, 26.24, 27.84, 31.40],
    "V_band": [51.26, 52.28, 53.86, 54.94, 56.66, 57.30, 58.00],
}

ALL_CHANNELS = MWR_CHANNELS["K_band"] + MWR_CHANNELS["V_band"]
N_CHANNELS = 14

# 53-58 GHz channels (for 0-2km temperature model)
V_SURFACE_CHANNELS = [53.86, 54.94, 56.66, 57.30, 58.00]
V_SURFACE_IDX = [ALL_CHANNELS.index(f) for f in V_SURFACE_CHANNELS]

# ============================================================
# Height grid: 93 layers from 0 to 10 km (Fig 3.10)
# ============================================================
def build_height_grid():
    """Build 93-layer height grid (0-10km, finer near surface).

    RPG HATPRO standard grid per Table 2.1 and Fig 3.10.
    Eight segments with contiguous boundaries, exactly 93 unique layers.
    Resolution ranges are approximate; actual spacing computed to fill
    each segment with the target layer count while keeping boundaries clean.
    """
    import numpy as np

    # (start_m, end_m, layers_incl_boundaries)
    # Segment boundaries: 0, 500, 1000, 1500, 2000, 3000, 5000, 6000, 10000
    # Sum of per-segment layer counts must equal 93 + 7 = 100
    # to yield 93 unique layers after removing duplicate boundary points.
    segments = [
        (0,    500,   19),   # ~27.8m  (nominal 30m)
        (500,  1000,  15),   # ~35.7m  (nominal 40m)
        (1000, 1500,  11),   # ~50.0m  (nominal 60m)
        (1500, 2000,   8),   # ~71.4m  (nominal 90m)
        (2000, 3000,  11),   # 100.0m  (nominal 120m)
        (3000, 5000,  14),   # ~153.8m (nominal 160m)
        (5000, 6000,   7),   # ~166.7m (nominal 200m)
        (6000, 10000, 15),   # ~285.7m (nominal 250m)
    ]

    layers = []
    for i, (start, end, n) in enumerate(segments):
        pts = np.linspace(start, end, n)
        if i > 0:
            pts = pts[1:]  # drop duplicate at shared segment boundary
        layers.extend(round(float(h), 1) for h in pts)

    return layers


HEIGHT_GRID = build_height_grid()
N_LAYERS = len(HEIGHT_GRID)

# Height interval indices
def height_to_idx(height_m):
    """Return the nearest grid index for a given height in meters."""
    return min(range(N_LAYERS), key=lambda i: abs(HEIGHT_GRID[i] - height_m))


IDX_0KM = height_to_idx(0)
IDX_2KM = height_to_idx(2000)
IDX_8KM = height_to_idx(8000)
IDX_10KM = height_to_idx(10000)

# ============================================================
# ERA5 configuration
# ============================================================
ERA5_PRESSURE_LEVELS = [
    1, 2, 3, 5, 7, 10, 20, 30, 50, 70,
    100, 125, 150, 175, 200, 225, 250, 300, 350, 400,
    450, 500, 550, 600, 650, 700, 750, 775, 800, 825,
    850, 875, 900, 925, 950, 975, 1000,
]

# Beijing Nanjiao site (approx coords)
NANJIAO_LAT = 39.80
NANJIAO_LON = 116.47

# Other Beijing MWR sites (Fig 3.1)
BEIJING_SITES = {
    "nanjiao": (39.80, 116.47, 32),
    "haidian": (39.99, 116.29, 50),
    "huairou": (40.37, 116.63, 60),
    "pinggu": (40.17, 117.10, 30),
    "shangdianzi": (40.65, 117.12, 290),
    "xiayunling": (39.73, 115.73, 410),
    "yanqing": (40.45, 115.97, 490),
}

# Zhangjiakou radiosonde site (for cross-site comparison)
ZHANGJIAKOU_LAT = 40.78
ZHANGJIAKOU_LON = 114.88
ZHANGJIAKOU_ALT = 774

# ============================================================
# Time ranges
# ============================================================
TRAIN_YEARS = (2013, 2016)
VALIDATION_YEARS = (2017, 2019)

# ============================================================
# QC thresholds (Section 3.2.2)
# ============================================================
# Humidity scaling factor from ERA5-vs-sounding IWV regression
RH_SCALING_FACTOR = 0.9

# LWC thresholds
ICLWC_DELETE_THRESHOLD = 1250  # g/m^2, delete sample if ICLWC > this
ICLWC_SCALE_THRESHOLD = 750    # g/m^2, scale LWC if ICLWC > this
ICLWC_SCALE_FACTOR = 750       # numerator in scaling formula

# ============================================================
# BRNN model hyperparameters
# ============================================================
HIDDEN_NODES = 256
DROPOUT_RATE = 0.3
LEARNING_RATE = 0.001
BATCH_SIZE = 128
MAX_EPOCHS = 200
EARLY_STOPPING_PATIENCE = 20

# ============================================================
# Data paths
# ============================================================
DATA_DIR = "data"
ERA5_DIR = f"{DATA_DIR}/era5"
RADIOSONDE_DIR = f"{DATA_DIR}/radiosonde"
MWR_DIR = f"{DATA_DIR}/mwr"
MODEL_DIR = "models"
RESULT_DIR = "results"
