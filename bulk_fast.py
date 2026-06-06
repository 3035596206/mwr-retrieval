#!/usr/bin/env python3
"""Download ERA5 single-level 2013-2019 from ARCO. One month per xarray session,
with 30s cooldown between months to avoid GCS rate limiting.
Only saves Beijing subset (39-41N, 115-118E), target variables only.
"""
import xarray as xr, os, sys, time, calendar

DATA_DIR = '/Users/ink/test/mwr_retrieval/data/era5'
ARCO_URL = "https://storage.googleapis.com/gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
SL_VARS = ['2m_temperature', '2m_dewpoint_temperature', 'surface_pressure']

os.makedirs(DATA_DIR, exist_ok=True)

years = list(range(2013, 2020))
total = 0

for y in years:
    for m in range(1, 13):
        fname = f'{DATA_DIR}/sl_{y}_{m:02d}.nc'
        if os.path.exists(fname) and os.path.getsize(fname) > 10000:
            print(f'{y}-{m:02d}: SKIP (exists)')
            total += 1
            continue

        last_day = calendar.monthrange(y, m)[1]
        t0 = time.time()
        print(f'{y}-{m:02d}: ', end='', flush=True)
        try:
            ds = xr.open_zarr(ARCO_URL, consolidated=True)
            data = ds[SL_VARS].sel(
                time=slice(f'{y}-{m:02d}-01', f'{y}-{m:02d}-{last_day}T23:00'),
                latitude=slice(41, 39), longitude=slice(115, 118)
            ).load()
            data.to_netcdf(fname)
            sz = os.path.getsize(fname)
            elapsed = time.time() - t0
            print(f'OK {sz/1024:.0f}KB {elapsed:.0f}s')
            total += 1
        except Exception as e:
            print(f'FAIL {time.time()-t0:.0f}s: {str(e)[:80]}')

        # Cooldown between months to avoid GCS throttling
        if not (y == years[-1] and m == 12):
            time.sleep(30)

print(f'\nDone: {total}/{84} months')
