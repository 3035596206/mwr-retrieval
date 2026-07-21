#!/usr/bin/env python3
"""ERA5 download via CDS with proxy. CDS queue is ~3-5 hours — be patient."""
import cdsapi, os, time, argparse

# os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7897'
# os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7897'

DATA_DIR = '/Users/ink/test/mwr_retrieval/data/era5'
os.makedirs(DATA_DIR, exist_ok=True)

PRESSURE_LEVELS = [
    '1','2','3','5','7','10','20','30','50','70',
    '100','125','150','175','200','225','250','300','350','400',
    '450','500','550','600','650','700','750','775','800','825',
    '850','875','900','925','950','975','1000',
]

def download_single_level(year, month):
    fname = f'{DATA_DIR}/sl_{year}_{month:02d}.nc'
    if os.path.exists(fname) and os.path.getsize(fname) > 1000:
        print(f"  sl_{year}_{month:02d}: exists ({os.path.getsize(fname)/1024:.0f} KB), skip")
        return True

    c = cdsapi.Client(quiet=True)
    c.sleep_max = 900
    c.retry_max = 500

    print(f"  sl_{year}_{month:02d}: submitting ({time.strftime('%H:%M')})...")
    try:
        c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis', 'format': 'netcdf',
                'variable': ['2m_temperature', '2m_dewpoint_temperature', 'surface_pressure'],
                'year': str(year), 'month': f'{month:02d}',
                'day': [f'{d:02d}' for d in range(1, 32)],
                'time': [f'{h:02d}:00' for h in range(24)],
                'area': [41, 115, 39, 118],
            },
            fname
        )
        sz = os.path.getsize(fname)
        print(f"  sl_{year}_{month:02d}: DONE {sz/1024:.0f} KB ({time.strftime('%H:%M')})")
        return True
    except Exception as e:
        print(f"  sl_{year}_{month:02d}: FAILED ({time.strftime('%H:%M')}) - {e}")
        return False


def download_pressure_level(year, month):
    fname = f'{DATA_DIR}/pl_{year}_{month:02d}.nc'
    if os.path.exists(fname) and os.path.getsize(fname) > 1000:
        print(f"  pl_{year}_{month:02d}: exists ({os.path.getsize(fname)/1024/1024:.1f} MB), skip")
        return True

    c = cdsapi.Client(quiet=True)
    c.sleep_max = 900
    c.retry_max = 500

    print(f"  pl_{year}_{month:02d}: submitting ({time.strftime('%H:%M')})...")
    try:
        c.retrieve(
            'reanalysis-era5-pressure-levels',
            {
                'product_type': 'reanalysis', 'format': 'netcdf',
                'variable': ['temperature', 'relative_humidity', 'geopotential'],
                'pressure_level': PRESSURE_LEVELS,
                'year': str(year), 'month': f'{month:02d}',
                'day': [f'{d:02d}' for d in range(1, 32)],
                'time': [f'{h:02d}:00' for h in range(24)],
                'area': [41, 115, 39, 118],
            },
            fname
        )
        sz = os.path.getsize(fname)
        print(f"  pl_{year}_{month:02d}: DONE {sz/1024/1024:.1f} MB ({time.strftime('%H:%M')})")
        return True
    except Exception as e:
        print(f"  pl_{year}_{month:02d}: FAILED ({time.strftime('%H:%M')}) - {e}")
        return False


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--year', type=int, required=True)
    p.add_argument('--month', type=int, default=0)
    p.add_argument('--type', choices=['sl', 'pl', 'both'], default='both')
    args = p.parse_args()

    year = args.year
    months = [args.month] if args.month else range(1, 13)

    print(f"Downloading year {year}, {len(months)} months, type={args.type}")
    print(f"CDS queue ~3-5 hours per month. Total estimate: ~{len(months)*5}h")
    print(f"Start: {time.strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    ok = 0; fail = 0
    for month in months:
        print(f"\n{year}-{month:02d}:")
        if args.type in ('sl', 'both'):
            if download_single_level(year, month): ok += 1
            else: fail += 1
        if args.type in ('pl', 'both'):
            if download_pressure_level(year, month): ok += 1
            else: fail += 1

    print(f"\n{'='*60}")
    print(f"Done: {ok} ok, {fail} failed. End: {time.strftime('%Y-%m-%d %H:%M')}")
