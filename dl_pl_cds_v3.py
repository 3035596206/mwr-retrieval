#!/usr/bin/env python3
"""CDS ERA5 PL: 3-day windows, 37 levels, new key, strictly serial.

1 month ≈ 10 windows × 30min each ≈ 5 hours/month.
Resume-safe: skips days already in concat'd monthly file.
"""

import os, time, calendar
import cdsapi
import xarray as xr
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "data", "era5")
os.makedirs(OUT, exist_ok=True)

LEVELS = [1,2,3,5,7,10,20,30,50,70,100,125,150,175,200,225,250,
          300,350,400,450,500,550,600,650,700,750,775,800,825,850,
          875,900,925,950,975,1000]
VARIABLES = ['temperature', 'geopotential', 'specific_humidity']
AREA = [41, 115, 39, 118]
WINDOW = 3  # days
MIN_SIZE = 200000

DAILY_DIR = os.path.join(OUT, "_daily_v3")
os.makedirs(DAILY_DIR, exist_ok=True)


def df(y, m, d):
    return os.path.join(DAILY_DIR, f"pl_{y}_{m:02d}_{d:02d}.nc")


def mf(y, m):
    return os.path.join(OUT, f"pl_{y}_{m:02d}.nc")


def is_valid(p):
    if not os.path.exists(p) or os.path.getsize(p) < MIN_SIZE:
        return False
    try:
        ds = xr.open_dataset(p)
        for v in ds.data_vars:
            a = ds[v].values
            if a.size > 0 and not np.all(np.isnan(a)):
                ds.close()
                return True
        ds.close()
    except Exception:
        pass
    return False


def concat_month(y, m):
    """Merge daily chunks into pl_YYYY_MM.nc."""
    mfname = mf(y, m)
    if is_valid(mfname):
        return True
    days = calendar.monthrange(y, m)[1]
    chunks = []
    for d in range(1, days + 1):
        f = df(y, m, d)
        if is_valid(f):
            chunks.append(xr.open_dataset(f))
    if not chunks:
        return False
    merged = xr.concat(chunks, dim='valid_time')
    merged.to_netcdf(mfname)
    sz = os.path.getsize(mfname)
    for c in chunks:
        c.close()
    print(f"  → merged {len(chunks)}d → {os.path.basename(mfname)} ({sz//1024}KB)")
    return True


def main():
    c = cdsapi.Client()
    all_ok, all_fail, all_skip = 0, 0, 0

    for yr in range(2013, 2020):
        for mo in range(1, 13):
            mfname = mf(yr, mo)
            if is_valid(mfname):
                print(f"SKIP {yr}-{mo:02d} (monthly exists)")
                all_skip += 1
                continue

            days_in_month = calendar.monthrange(yr, mo)[1]

            # Count already-done days
            done = sum(1 for d in range(1, days_in_month + 1) if is_valid(df(yr, mo, d)))

            # Submit in 3-day windows
            for start in range(1, days_in_month + 1, WINDOW):
                end = min(start + WINDOW - 1, days_in_month)
                pending = [d for d in range(start, end + 1) if not is_valid(df(yr, mo, d))]
                if not pending:
                    continue

                days_list = [f"{d:02d}" for d in range(start, end + 1)]
                label = f"{yr}-{mo:02d} d{start}-{end}"

                # Submit + wait
                print(f"  {label}...", end=" ", flush=True)
                try:
                    r = c.retrieve('reanalysis-era5-pressure-levels', {
                        'product_type': 'reanalysis',
                        'variable': VARIABLES,
                        'pressure_level': LEVELS,
                        'year': str(yr), 'month': f"{mo:02d}",
                        'day': days_list,
                        'time': [f"{h:02d}:00" for h in range(24)],
                        'area': AREA, 'format': 'netcdf',
                    })

                    t0 = time.time()
                    tmp = os.path.join(DAILY_DIR, f"_tmp_{yr}{mo:02d}_{start}.nc")
                    r.download(tmp)
                    elapsed = time.time() - t0

                    # Split into per-day files
                    ds = xr.open_dataset(tmp)
                    times = ds.valid_time.values if 'valid_time' in ds else ds.time.values
                    day_groups = {}
                    for i, t in enumerate(times):
                        d = t.astype('datetime64[D]').astype(int) % 100  # day of month
                        if d not in day_groups:
                            day_groups[d] = []
                        day_groups[d].append(i)

                    for d_idx, idxs in day_groups.items():
                        d_ds = ds.isel(
                            {('valid_time' if 'valid_time' in ds else 'time'): idxs}
                        )
                        d_ds.to_netcdf(df(yr, mo, int(d_idx)))
                        d_ds.close()

                    ds.close()
                    os.remove(tmp)

                    print(f"✓ {elapsed/60:.0f}min → {len(day_groups)} files", flush=True)
                    all_ok += 1

                except Exception as e:
                    err = str(e)[:100]
                    print(f"✗ {err}", flush=True)
                    all_fail += 1
                    if "cost" in err.lower() or "too large" in err.lower():
                        print(f"  WINDOW TOO LARGE FOR THIS KEY, SKIPPING MONTH", flush=True)
                        break

                # Pause between windows
                time.sleep(10)

            # Try concat
            if is_valid(mf(yr, mo)) or concat_month(yr, mo):
                all_skip += 1

            # Pause between months
            time.sleep(30)

    print(f"\nDONE: {all_ok} windows ok, {all_fail} fail, {all_skip} months done")
    print(f"Files: {OUT}/pl_*_*.nc")


if __name__ == "__main__":
    main()
