#!/bin/bash
# Download one ERA5 single-level month via curl + proxy to avoid Python aiohttp issues
# Usage: ./dl_curl.sh YEAR MONTH
set -e

YEAR=$1
MONTH=$2
MONTH_PAD=$(printf '%02d' $MONTH)
DATA_DIR=/Users/ink/test/mwr_retrieval/data/era5
mkdir -p "$DATA_DIR"

OUTFILE="$DATA_DIR/sl_${YEAR}_${MONTH_PAD}.nc"
if [ -f "$OUTFILE" ] && [ $(stat -f%z "$OUTFILE" 2>/dev/null || echo 0) -gt 10000 ]; then
    echo "SKIP: $OUTFILE exists"
    exit 0
fi

PROXY="http://127.0.0.1:7897"
BASE="https://storage.googleapis.com/gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"

echo "Downloading ERA5 single-level $YEAR-$MONTH_PAD via proxy..."
echo "Using proxy: $PROXY"

# Use Python for the complex part (Zarr indexing, subset, NetCDF save)
# but with proxy for all HTTP traffic
export HTTP_PROXY="$PROXY"
export HTTPS_PROXY="$PROXY"
export SSL_CERT_FILE=""  # skip SSL verify for proxy

python3 -u -c "
import xarray as xr, os, sys, calendar
url = '$BASE'
ds = xr.open_zarr(url, consolidated=True)
last = calendar.monthrange($YEAR, $MONTH)[1]
data = ds[['2m_temperature', '2m_dewpoint_temperature', 'surface_pressure']].sel(
    time=slice(f'$YEAR-$MONTH_PAD-01', f'$YEAR-$MONTH_PAD-{last}T23:00'),
    latitude=slice(41,39), longitude=slice(115,118)
).load()
data.to_netcdf('$OUTFILE')
print(f'OK: {os.path.getsize(\"$OUTFILE\")/1024:.0f}KB')
"
