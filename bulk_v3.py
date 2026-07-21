#!/usr/bin/env python3
"""Download ERA5 single-level 2013-2019 from ARCO Zarr.
Fresh xarray+fsspec session per month. Explicit cleanup after each month.
"""
import xarray as xr, os, sys, time, calendar, gc

DATA_DIR = '/Users/ink/test/mwr_retrieval/data/era5'
ARCO_URL = "https://storage.googleapis.com/gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
SL_VARS = ['2m_temperature', '2m_dewpoint_temperature', 'surface_pressure']
os.makedirs(DATA_DIR, exist_ok=True)

def download_one_month(year, month):
    fname = f'{DATA_DIR}/sl_{year}_{month:02d}.nc'
    if os.path.exists(fname) and os.path.getsize(fname) > 10000:
        return 'SKIP', os.path.getsize(fname)

    last_day = calendar.monthrange(year, month)[1]
    t0 = time.time()

    ds = None
    try:
        ds = xr.open_zarr(ARCO_URL, consolidated=True)
        data = ds[SL_VARS].sel(
            time=slice(f'{year}-{month:02d}-01', f'{year}-{month:02d}-{last_day}T23:00'),
            latitude=slice(41, 39), longitude=slice(115, 118)
        ).load()
        data.to_netcdf(fname)
        return 'OK', os.path.getsize(fname), time.time() - t0
    except Exception as e:
        return 'FAIL', str(e)[:100], time.time() - t0
    finally:
        if ds is not None:
            try:
                if hasattr(ds, '_store'):
                    ds._store.close()
            except:
                pass
            del ds
        # Aggressive cleanup
        gc.collect()
        # Also clean up fsspec caches
        try:
            import fsspec
            fsspec.filesystem('http').clear_instance_cache()
        except:
            pass

ok, fail, skip = 0, 0, 0
for year in range(2013, 2020):
    for month in range(1, 13):
        label = f'{year}-{month:02d}'
        result = download_one_month(year, month)

        if result[0] == 'SKIP':
            print(f'{label}: SKIP ({result[1]/1024:.0f}KB)')
            skip += 1
        elif result[0] == 'OK':
            print(f'{label}: OK {result[1]/1024:.0f}KB {result[2]:.0f}s')
            ok += 1
        else:
            print(f'{label}: FAIL {result[2]:.0f}s - {result[1]}')
            fail += 1

print(f'\nDone: {ok} ok, {fail} fail, {skip} skip')
