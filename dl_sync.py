#!/usr/bin/env python3
"""Download ERA5 from ARCO using synchronous requests + curl.
No aiohttp, no async. One month per process, 5min timeout.
"""
import xarray as xr, os, sys, time, calendar, subprocess, signal

DATA_DIR = '/Users/ink/test/mwr_retrieval/data/era5'
ARCO = "https://storage.googleapis.com/gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
VARS = ['2m_temperature', '2m_dewpoint_temperature', 'surface_pressure']
TIMEOUT = 300  # 5 min per month
os.makedirs(DATA_DIR, exist_ok=True)

def dl(year, month):
    fname = f'{DATA_DIR}/sl_{year}_{month:02d}.nc'
    if os.path.exists(fname) and os.path.getsize(fname) > 10000:
        return f'sl_{year}_{month:02d}: SKIP {os.path.getsize(fname)//1024}KB'

    last = calendar.monthrange(year, month)[1]
    code = f"""
import xarray as xr, os
ds = xr.open_zarr("{ARCO}", consolidated=True)
d = ds[['2m_temperature','2m_dewpoint_temperature','surface_pressure']].sel(
    time=slice('{year}-{month:02d}-01','{year}-{month:02d}-{last}T23:00'),
    latitude=slice(41,39), longitude=slice(115,118)
).load()
d.to_netcdf("{fname}")
print(f"OK {{os.path.getsize('{fname}')//1024}}KB")
"""

    t0 = time.time()
    try:
        venv_python = os.path.join(os.path.dirname(__file__), '.venv', 'bin', 'python3')
        r = subprocess.run(
            [venv_python, '-c', code],
            capture_output=True, text=True,
            timeout=TIMEOUT, cwd=os.path.dirname(__file__),
            env={**os.environ, 'PYTHONUNBUFFERED': '1'}
        )
        elapsed = time.time() - t0
        out = r.stdout.strip() or r.stderr.strip()[:100]
        status = 'OK' if r.returncode == 0 and 'OK' in out else 'FAIL'
        return f'sl_{year}_{month:02d}: {status} {elapsed:.0f}s {out}'
    except subprocess.TimeoutExpired:
        return f'sl_{year}_{month:02d}: TIMEOUT {TIMEOUT}s'

# Main
print(f"Starting bulk download with {TIMEOUT}s timeout per month")
for y in range(2013, 2020):
    for m in range(1, 13):
        print(dl(y, m), flush=True)
print("DONE")
