#!/usr/bin/env python3
"""CDS pressure-level ERA5 download: 3-day windows, 37 levels, 84 months.
Resumes automatically — skips existing valid files.
"""

import os
import sys
# No proxy — CDS direct connection works when DNS resolves
# Proxy (127.0.0.1:7897) causes SSL EOF errors on CDS polling

import cdsapi
import time
import calendar

OUT = "data/era5"
os.makedirs(OUT, exist_ok=True)

LEVELS = [1,2,3,5,7,10,20,30,50,70,100,125,150,175,200,225,250,
          300,350,400,450,500,550,600,650,700,750,775,800,825,850,
          875,900,925,950,975,1000]
VARIABLES = ['temperature', 'geopotential', 'specific_humidity']
AREA = [41, 115, 39, 118]  # N, W, S, E — Beijing region

c = cdsapi.Client()

for year in range(2013, 2020):
    for month in range(1, 13):
        fname = f"{OUT}/pl_{year}_{month:02d}.nc"
        if os.path.exists(fname) and os.path.getsize(fname) > 100000:
            print(f"SKIP {year}-{month:02d} (exists, {os.path.getsize(fname)//1024}KB)")
            continue

        last_day = calendar.monthrange(year, month)[1]
        chunks = []
        ok = True

        for start_day in range(1, last_day + 1, 3):
            end_day = min(start_day + 2, last_day)
            days = [f"{d:02d}" for d in range(start_day, end_day + 1)]

            label = f"{year}-{month:02d} d{start_day}-{end_day}"
            print(f"{label} ...", end=" ", flush=True)
            t1 = time.time()

            try:
                r = c.retrieve('reanalysis-era5-pressure-levels', {
                    'product_type': 'reanalysis',
                    'variable': VARIABLES,
                    'pressure_level': LEVELS,
                    'year': str(year),
                    'month': f"{month:02d}",
                    'day': days,
                    'time': [f"{h:02d}:00" for h in range(24)],
                    'area': AREA,
                    'format': 'netcdf',
                })
                tmp = f"{OUT}/_tmp_{year}{month:02d}_{start_day}.nc"
                r.download(tmp)

                import xarray as xr
                ds = xr.open_dataset(tmp)
                chunks.append(ds)
                os.remove(tmp)
                elapsed = time.time() - t1
                print(f"{elapsed:.0f}s", flush=True)

            except Exception as e:
                print(f"FAIL: {e}", flush=True)
                ok = False
                break

            time.sleep(5)

        if ok and chunks:
            import xarray as xr
            month_data = xr.concat(chunks, dim='time')
            month_data.to_netcdf(fname)
            for ds in chunks:
                ds.close()
            print(f"  -> OK {os.path.getsize(fname)//1024}KB", flush=True)
        elif not ok:
            print(f"  -> ABORTED (will retry next run)", flush=True)
            for ds in chunks:
                try: ds.close()
                except: pass

        time.sleep(10)

print("ALL DONE")
