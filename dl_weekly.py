#!/usr/bin/env python3
"""Reliable ERA5 single-level download: 1 week at a time to avoid GCS throttle.
4 weeks → merge → 1 month NetCDF. 84 months × 4 weeks = 336 downloads.
Estimate: ~17 hours total.
"""
import xarray as xr, os, sys, time, calendar

DATA_DIR = '/Users/ink/test/mwr_retrieval/data/era5'
ARCO = 'https://storage.googleapis.com/gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3'
VARS = ['2m_temperature', '2m_dewpoint_temperature', 'surface_pressure']
os.makedirs(DATA_DIR, exist_ok=True)

def download_month(year, month):
    fname = f'{DATA_DIR}/sl_{year}_{month:02d}.nc'
    if os.path.exists(fname) and os.path.getsize(fname) > 10000:
        return f'SKIP {os.path.getsize(fname)//1024}KB'

    last_day = calendar.monthrange(year, month)[1]
    weeks = []
    t0 = time.time()

    # Download in 7-day chunks
    for start_day in range(1, last_day + 1, 7):
        end_day = min(start_day + 6, last_day)
        start_str = f'{year}-{month:02d}-{start_day:02d}'
        end_str = f'{year}-{month:02d}-{end_day:02d}T23:00'

        # Open fresh per week for connection hygiene
        ds = xr.open_zarr(ARCO, consolidated=True)
        chunk = ds[VARS].sel(
            time=slice(start_str, end_str),
            latitude=slice(41, 39), longitude=slice(115, 118)
        ).load()
        weeks.append(chunk)
        ds.close()
        del ds

    # Merge weeks into month
    month_data = xr.concat(weeks, dim='time')
    month_data.to_netcdf(fname)
    elapsed = time.time() - t0
    return f'OK {os.path.getsize(fname)//1024}KB {elapsed:.0f}s'

ok, fail, skip = 0, 0, 0
for year in range(2013, 2020):
    for month in range(1, 13):
        try:
            result = download_month(year, month)
            print(f'{year}-{month:02d}: {result}', flush=True)
            if 'OK' in result: ok += 1
            elif 'SKIP' in result: skip += 1
            else: fail += 1
        except Exception as e:
            print(f'{year}-{month:02d}: FAIL {str(e)[:100]}', flush=True)
            fail += 1

print(f'\nDONE: {ok} ok, {fail} fail, {skip} skip')
