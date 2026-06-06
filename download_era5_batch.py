#!/usr/bin/env python3
"""Batch ERA5 download matching paper Section 3.1.1 requirements.

Data specs (paper §3.1.1):
  Pressure levels: T, RH, Geopotential · 37 levels (1000-1 hPa) · hourly
  Single levels:   2mT, 2m dewpoint, surface pressure · hourly
  Area:            Beijing [41N, 115E, 39N, 118E]
  Period:          2013-2019 (2013-2016 train, 2017-2019 validation)

Strategy: 1 month per request, variables split to avoid CDS cost limits.
  - Single-level: 3 vars × 1 month = ~500 KB/request
  - Pressure T+RH: 2 vars × 37 levels × 1 month = ~15-30 MB/request
  - Pressure Geop:  1 var  × 37 levels × 1 month = ~8-15 MB/request

Usage:
  python download_era5_batch.py                  # full 2013-2019
  python download_era5_batch.py --year 2013      # single year
  python download_era5_batch.py --year 2013 --month 1  # single month
"""
import cdsapi, os, sys, time, argparse

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "era5")
os.makedirs(OUT, exist_ok=True)

PRESSURE_LEVELS = [
    '1','2','3','5','7','10','20','30','50','70',
    '100','125','150','175','200','225','250','300',
    '350','400','450','500','550','600','650','700',
    '750','775','800','825','850','875','900','925',
    '950','975','1000',
]
AREA = [41, 115, 39, 118]  # N, W, S, E


def clear_proxy():
    for k in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']:
        os.environ.pop(k, None)


def download_one(client, dataset, request, outpath, label):
    """Download a single file with retry on CDS queue timeout."""
    if os.path.exists(outpath) and os.path.getsize(outpath) > 1000:
        print(f"  [{label}] exists ({os.path.getsize(outpath)/1024:.0f} KB), skip")
        return True

    print(f"  [{label}] submitting ({time.strftime('%H:%M')})...")
    try:
        client.retrieve(dataset, request, outpath)
        sz = os.path.getsize(outpath)
        print(f"  [{label}] DONE {sz/1024/1024:.1f} MB ({time.strftime('%H:%M')})")
        return True
    except Exception as e:
        print(f"  [{label}] FAILED: {e}")
        # Clean up partial file
        if os.path.exists(outpath):
            os.remove(outpath)
        return False


def main():
    parser = argparse.ArgumentParser(description="Batch ERA5 download for BRNN project")
    parser.add_argument("--year", type=int, help="Single year (default: all 2013-2019)")
    parser.add_argument("--month", type=int, help="Single month (requires --year)")
    args = parser.parse_args()

    clear_proxy()
    client = cdsapi.Client(quiet=True)

    years = [args.year] if args.year else range(2013, 2020)
    months = [args.month] if args.month else range(1, 13)

    total = len(years) * len(months) * 3  # 3 requests per month
    ok, fail, skipped = 0, 0, 0

    print(f"ERA5 batch download: {len(years)} years × {len(months)} months = ~{total} requests")
    print(f"Output: {OUT}")
    print(f"Start: {time.strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    for year in years:
        for month in months:
            tag = f"{year}-{month:02d}"
            already = 0

            # --- Single-level (3 vars: 2mT + dewpoint + surface pressure) ---
            f_sl = f"{OUT}/era5_single_{year}_{month:02d}.nc"
            if os.path.exists(f_sl) and os.path.getsize(f_sl) > 1000:
                skipped += 1; already += 1
            else:
                if download_one(client, 'reanalysis-era5-single-levels', {
                    'product_type': 'reanalysis', 'format': 'netcdf',
                    'variable': ['2m_temperature', '2m_dewpoint_temperature',
                                 'surface_pressure'],
                    'year': str(year), 'month': f'{month:02d}',
                    'day': [f'{d:02d}' for d in range(1, 32)],
                    'time': [f'{h:02d}:00' for h in range(24)],
                    'area': AREA,
                }, f_sl, f"{tag} sl"):
                    ok += 1
                else:
                    fail += 1

            # --- Pressure T + RH (2 vars × 37 levels) ---
            f_trh = f"{OUT}/era5_pressure_TRH_{year}_{month:02d}.nc"
            if os.path.exists(f_trh) and os.path.getsize(f_trh) > 1000:
                skipped += 1; already += 1
            else:
                if download_one(client, 'reanalysis-era5-pressure-levels', {
                    'product_type': 'reanalysis', 'format': 'netcdf',
                    'variable': ['temperature', 'relative_humidity'],
                    'pressure_level': PRESSURE_LEVELS,
                    'year': str(year), 'month': f'{month:02d}',
                    'day': [f'{d:02d}' for d in range(1, 32)],
                    'time': [f'{h:02d}:00' for h in range(24)],
                    'area': AREA,
                }, f_trh, f"{tag} TRH"):
                    ok += 1
                else:
                    fail += 1

            # --- Pressure Geopotential (1 var × 37 levels) ---
            f_z = f"{OUT}/era5_pressure_Z_{year}_{month:02d}.nc"
            if os.path.exists(f_z) and os.path.getsize(f_z) > 1000:
                skipped += 1; already += 1
            else:
                if download_one(client, 'reanalysis-era5-pressure-levels', {
                    'product_type': 'reanalysis', 'format': 'netcdf',
                    'variable': ['geopotential'],
                    'pressure_level': PRESSURE_LEVELS,
                    'year': str(year), 'month': f'{month:02d}',
                    'day': [f'{d:02d}' for d in range(1, 32)],
                    'time': [f'{h:02d}:00' for h in range(24)],
                    'area': AREA,
                }, f_z, f"{tag} Z"):
                    ok += 1
                else:
                    fail += 1

            done = ok + fail + skipped
            print(f"\n  Progress: {done}/{total} | OK={ok} fail={fail} skip={skipped}"
                  f" ({time.strftime('%H:%M')})\n")

    print("=" * 60)
    print(f"Done: {ok} ok, {fail} failed, {skipped} skipped")
    print(f"End: {time.strftime('%Y-%m-%d %H:%M')}")


if __name__ == "__main__":
    main()
