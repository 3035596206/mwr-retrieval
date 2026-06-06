#!/usr/bin/env python3
"""Bulk download ERA5 for MWR project from ARCO Zarr on GCS.
Downloads all single-level data (2013-2019, Beijing area, needed vars only).
No CDS, no rate limit. Speed: ~3 min/month.
"""
import xarray as xr, os, sys, time, calendar

DATA_DIR = '/Users/ink/test/mwr_retrieval/data/era5'
ARCO_URL = "https://storage.googleapis.com/gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
YEARS = range(2013, 2020)
SL_VARS = ['2m_temperature', '2m_dewpoint_temperature', 'surface_pressure']

os.makedirs(DATA_DIR, exist_ok=True)

print(f"Opening ARCO Zarr (one-time metadata fetch)...", flush=True)
t0 = time.time()
ds = xr.open_zarr(ARCO_URL, consolidated=True)
print(f"  Opened in {time.time()-t0:.0f}s", flush=True)

total_ok, total_fail = 0, 0
for year in YEARS:
    for month in range(1, 13):
        fname = f'{DATA_DIR}/sl_{year}_{month:02d}.nc'
        if os.path.exists(fname) and os.path.getsize(fname) > 10000:
            print(f"  sl_{year}_{month:02d}: exists, skip")
            total_ok += 1
            continue

        last_day = calendar.monthrange(year, month)[1]
        t0 = time.time()
        print(f"  sl_{year}_{month:02d}: loading...", end="", flush=True)
        try:
            data = ds[SL_VARS].sel(
                time=slice(f'{year}-{month:02d}-01', f'{year}-{month:02d}-{last_day}T23:00'),
                latitude=slice(41, 39),
                longitude=slice(115, 118)
            ).load()
            data.to_netcdf(fname)
            sz = os.path.getsize(fname)
            print(f" {sz/1024:.0f}KB ({time.time()-t0:.0f}s)")
            total_ok += 1
        except Exception as e:
            print(f" FAIL ({time.time()-t0:.0f}s): {str(e)[:100]}")
            total_fail += 1

print(f"\nDone: {total_ok} ok, {total_fail} failed")
