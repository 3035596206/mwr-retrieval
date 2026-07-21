#!/usr/bin/env python3
"""Download ERA5 for MWR project: 2013-2019, Beijing area.
Single-level: ARCO Zarr on GCS (fast, no rate limit)
Pressure-level: CDS API (fast when not rate-limited, falls back to ARCO)
"""
import cdsapi, os, sys, time, argparse
import xarray as xr
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data', 'era5')
os.makedirs(DATA_DIR, exist_ok=True)

# Beijing area
LAT_SLICE = slice(41, 39)
LON_SLICE = slice(115, 118)

# Variables needed
SL_VARS_CDS = ['2m_temperature', '2m_dewpoint_temperature', 'surface_pressure']
SL_VARS_ARCO = ['2m_temperature', '2m_dewpoint_temperature', 'surface_pressure']
PL_VARS_CDS = ['temperature', 'relative_humidity', 'geopotential']
PL_VARS_ARCO = ['temperature', 'specific_humidity', 'geopotential']

PRESSURE_LEVELS = [
    '1','2','3','5','7','10','20','30','50','70',
    '100','125','150','175','200','225','250','300','350','400',
    '450','500','550','600','650','700','750','775','800','825',
    '850','875','900','925','950','975','1000',
]

# ARCO Zarr store
ARCO_URL = "https://storage.googleapis.com/gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"


def download_sl_arco(year, month):
    """Download single-level from ARCO GCS (fast)."""
    fname = f'{DATA_DIR}/sl_{year}_{month:02d}.nc'
    if os.path.exists(fname) and os.path.getsize(fname) > 1000:
        print(f"  sl_{year}_{month:02d}: exists ({os.path.getsize(fname)/1024:.0f}KB)")
        return True

    t0 = time.time()
    print(f"  sl_{year}_{month:02d}: ARCO loading...", end="", flush=True)
    try:
        ds = xr.open_zarr(ARCO_URL, consolidated=True)
        start = f'{year}-{month:02d}-01'
        # End of month
        import calendar
        last_day = calendar.monthrange(year, month)[1]
        end = f'{year}-{month:02d}-{last_day}T23:00'

        data = ds[SL_VARS_ARCO].sel(
            time=slice(start, end),
            latitude=LAT_SLICE,
            longitude=LON_SLICE
        ).load()
        data.to_netcdf(fname)
        sz = os.path.getsize(fname)
        print(f" {sz/1024:.0f}KB ({time.time()-t0:.0f}s)")
        return True
    except Exception as e:
        print(f" FAIL ({time.time()-t0:.0f}s): {str(e)[:80]}")
        return False


def download_pl_cds(year, month):
    """Download pressure-level from CDS (fast but may be rate-limited)."""
    fname = f'{DATA_DIR}/pl_{year}_{month:02d}.nc'
    if os.path.exists(fname) and os.path.getsize(fname) > 1000:
        print(f"  pl_{year}_{month:02d}: exists ({os.path.getsize(fname)/1024/1024:.1f}MB)")
        return True

    t0 = time.time()
    print(f"  pl_{year}_{month:02d}: CDS submitting...", end="", flush=True)
    try:
        c = cdsapi.Client(quiet=True)
        c.sleep_max = 600
        c.retry_max = 50
        c.timeout = 120
        c.retrieve(
            'reanalysis-era5-pressure-levels',
            {
                'product_type': 'reanalysis', 'format': 'netcdf',
                'variable': PL_VARS_CDS,
                'pressure_level': PRESSURE_LEVELS,
                'year': str(year), 'month': f'{month:02d}',
                'day': [f'{d:02d}' for d in range(1, 32)],
                'time': [f'{h:02d}:00' for h in range(24)],
                'area': [41, 115, 39, 118],
            },
            fname
        )
        sz = os.path.getsize(fname)
        print(f" {sz/1024/1024:.1f}MB ({time.time()-t0:.0f}s)")
        return True
    except Exception as e:
        msg = str(e)
        if 'rejected' in msg.lower():
            print(f" CDS_REJECTED ({time.time()-t0:.0f}s)")
        else:
            print(f" FAIL ({time.time()-t0:.0f}s): {msg[:80]}")
        return False


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--year', type=int, required=True)
    p.add_argument('--month', type=int, default=0)
    p.add_argument('--type', choices=['sl', 'pl', 'both'], default='both')
    p.add_argument('--source', choices=['arco', 'cds', 'auto'], default='auto',
                   help='auto: SL from ARCO, PL from CDS')
    args = p.parse_args()

    year = args.year
    months = [args.month] if args.month else range(1, 13)

    print(f"Download: {year}, {len(months)} months, type={args.type}, source={args.source}")
    try:
        print(f"Start: {time.strftime('%H:%M')}")
    except:
        pass
    print("=" * 60)

    ok, fail, skip = 0, 0, 0
    for month in months:
        label = f"{year}-{month:02d}"
        print(f"\n{label}:")

        if args.type in ('sl', 'both'):
            if args.source in ('arco', 'auto'):
                if download_sl_arco(year, month):
                    ok += 1
                else:
                    fail += 1

        if args.type in ('pl', 'both'):
            if args.source in ('cds', 'auto'):
                if download_pl_cds(year, month):
                    ok += 1
                else:
                    fail += 1

    print(f"\n{'='*60}")
    try:
        print(f"Done: {ok} ok, {fail} failed. End: {time.strftime('%H:%M')}")
    except:
        print(f"Done: {ok} ok, {fail} failed.")
