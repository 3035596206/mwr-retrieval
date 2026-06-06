#!/usr/bin/env python3
"""Download ONE month of ERA5 single-level from ARCO and exit.
Fresh process = fresh HTTP connection pool. Called by bash loop.
Usage: python3 dl_one_month.py <year> <month>
"""
import xarray as xr, os, sys, time, calendar

year = int(sys.argv[1])
month = int(sys.argv[2])
DATA_DIR = '/Users/ink/test/mwr_retrieval/data/era5'
ARCO_URL = "https://storage.googleapis.com/gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
SL_VARS = ['2m_temperature', '2m_dewpoint_temperature', 'surface_pressure']

fname = f'{DATA_DIR}/sl_{year}_{month:02d}.nc'
if os.path.exists(fname) and os.path.getsize(fname) > 10000:
    print(f"EXISTS {os.path.getsize(fname)/1024:.0f}KB")
    sys.exit(0)

last_day = calendar.monthrange(year, month)[1]
t0 = time.time()
try:
    ds = xr.open_zarr(ARCO_URL, consolidated=True)
    data = ds[SL_VARS].sel(
        time=slice(f'{year}-{month:02d}-01', f'{year}-{month:02d}-{last_day}T23:00'),
        latitude=slice(41, 39),
        longitude=slice(115, 118)
    ).load()
    data.to_netcdf(fname)
    sz = os.path.getsize(fname)
    print(f"DONE {sz/1024:.0f}KB {time.time()-t0:.0f}s")
except Exception as e:
    print(f"FAIL {time.time()-t0:.0f}s {str(e)[:100]}")
    sys.exit(1)
