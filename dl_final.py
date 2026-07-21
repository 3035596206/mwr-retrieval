#!/usr/bin/env python3
"""Final download script: subprocess per month, 2-connection limit, 15min timeout.
Download all single-level ERA5 2013-2019 from ARCO GCS.
"""
import os, sys, time, calendar, subprocess

DATA_DIR = '/Users/ink/test/mwr_retrieval/data/era5'
TIMEOUT = 900  # 15 min per month
VENV_PY = '/Users/ink/test/mwr_retrieval/.venv/bin/python3'
ARCO = 'https://storage.googleapis.com/gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3'

os.makedirs(DATA_DIR, exist_ok=True)

ok, fail, skip = 0, 0, 0
total = 84

for year in range(2013, 2020):
    for month in range(1, 13):
        fname = f'{DATA_DIR}/sl_{year}_{month:02d}.nc'
        label = f'{year}-{month:02d}'

        if os.path.exists(fname) and os.path.getsize(fname) > 10000:
            print(f'{label}: SKIP ({os.path.getsize(fname)//1024}KB)')
            skip += 1
            continue

        last = calendar.monthrange(year, month)[1]
        code = f"""
import os, time
os.environ['AIOHTTP_MAX_CONNECTIONS'] = '2'
import xarray as xr
ds = xr.open_zarr('{ARCO}', consolidated=True)
d = ds[['2m_temperature','2m_dewpoint_temperature','surface_pressure']].sel(
    time=slice('{year}-{month:02d}-01','{year}-{month:02d}-{last}T23:00'),
    latitude=slice(41,39), longitude=slice(115,118)
).load()
d.to_netcdf('{fname}')
print(f'OK {{os.path.getsize(\"{fname}\")//1024}}KB')
"""

        t0 = time.time()
        print(f'{label}: ', end='', flush=True)
        try:
            r = subprocess.run(
                [VENV_PY, '-c', code],
                capture_output=True, text=True, timeout=TIMEOUT,
                cwd=os.path.dirname(__file__)
            )
            elapsed = time.time() - t0
            out = r.stdout.strip() or r.stderr.strip()[:120]
            if r.returncode == 0 and 'OK' in out:
                print(f'OK {elapsed:.0f}s {out}')
                ok += 1
            else:
                print(f'FAIL {elapsed:.0f}s {out}')
                fail += 1
        except subprocess.TimeoutExpired:
            print(f'TIMEOUT {TIMEOUT}s')
            fail += 1

        # Cooldown
        if not (year == 2019 and month == 12):
            time.sleep(3)

print(f'\n=== DONE: {ok} ok, {fail} fail, {skip} skip of {total} ===')
