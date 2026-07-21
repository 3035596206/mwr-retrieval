"""ERA5 data download and preprocessing.

Uses CDS API to download ERA5 hourly data on pressure levels and single levels,
extracts the nearest grid point, and interpolates to the 93-layer MWR height grid.

CDS API setup:
    pip install cdsapi
    Create account at https://cds.climate.copernicus.eu/
    Set up ~/.cdsapirc with your key
"""

import os
import numpy as np
import xarray as xr
from scipy.interpolate import interp1d
import config


def download_era5_pressure_levels(years, months, area, output_path):
    """Download ERA5 hourly data on pressure levels from CDS.

    Args:
        years: list of years
        months: list of months
        area: [north, west, south, east]
        output_path: path for output netCDF file
    """
    import cdsapi
    c = cdsapi.Client()

    request = {
        "product_type": "reanalysis",
        "format": "netcdf",
        "variable": [
            "temperature", "relative_humidity", "geopotential",
        ],
        "pressure_level": config.ERA5_PRESSURE_LEVELS,
        "year": [str(y) for y in years],
        "month": [f"{m:02d}" for m in months],
        "day": [f"{d:02d}" for d in range(1, 32)],
        "time": [f"{h:02d}:00" for h in range(24)],
        "area": area,
    }

    c.retrieve("reanalysis-era5-pressure-levels", request, output_path)
    print(f"Downloaded ERA5 pressure-level data to {output_path}")


def download_era5_single_levels(years, months, area, output_path):
    """Download ERA5 hourly single-level data from CDS."""
    import cdsapi
    c = cdsapi.Client()

    request = {
        "product_type": "reanalysis",
        "format": "netcdf",
        "variable": [
            "2m_temperature", "2m_dewpoint_temperature", "surface_pressure",
        ],
        "year": [str(y) for y in years],
        "month": [f"{m:02d}" for m in months],
        "day": [f"{d:02d}" for d in range(1, 32)],
        "time": [f"{h:02d}:00" for h in range(24)],
        "area": area,
    }

    c.retrieve("reanalysis-era5-single-levels", request, output_path)
    print(f"Downloaded ERA5 single-level data to {output_path}")


def calc_2m_rh(t2m, d2m, sp):
    """Calculate 2m relative humidity from temperature, dewpoint, and surface pressure.

    Uses Magnus formula (equations 3-1 to 3-3 in the paper).

    Args:
        t2m: 2m temperature [K]
        d2m: 2m dewpoint temperature [K]
        sp: surface pressure [Pa]
    Returns:
        2m relative humidity [%]
    """
    # Saturation vapor pressure at T2m (eq 3-1)
    t2m_c = t2m - 273.16
    es = 6.1078 * np.exp(17.2693882 * t2m_c / (t2m_c + 237.3))

    # Actual vapor pressure at Td (eq 3-2)
    d2m_c = d2m - 273.16
    e = 6.1078 * np.exp(17.2693882 * d2m_c / (d2m_c + 237.3))

    rh = (e / es) * 100.0
    return np.clip(rh, 0, 100)


def extract_era5_profiles(pressure_file, single_file, lat, lon):
    """Extract atmospheric profiles from ERA5 netCDF files at nearest grid point.

    Args:
        pressure_file: path to ERA5 pressure-level netCDF
        single_file: path to ERA5 single-level netCDF
        lat, lon: target coordinates
    Returns:
        dict with keys: time, z_agl (height above ground), T, RH, P,
                        t2m, rh2m, sp, clwc
    """
    ds_p = xr.open_dataset(pressure_file)
    ds_s = xr.open_dataset(single_file)

    # Find nearest grid point
    ds_p = ds_p.sel(latitude=lat, longitude=lon, method="nearest")
    ds_s = ds_s.sel(latitude=lat, longitude=lon, method="nearest")

    # Extract variables
    times = ds_p["time"].values
    T = ds_p["t"].values          # temperature [K]
    RH = ds_p["r"].values         # relative humidity [%]
    Z = ds_p["z"].values          # geopotential [m^2/s^2]
    P = np.array(config.ERA5_PRESSURE_LEVELS) * 100  # hPa -> Pa

    t2m = ds_s["t2m"].values      # 2m temperature [K]
    d2m = ds_s["d2m"].values      # 2m dewpoint temperature [K]
    sp = ds_s["sp"].values        # surface pressure [Pa]

    # Calculate 2m RH
    rh2m = calc_2m_rh(t2m, d2m, sp)

    # Geopotential height in meters above ground level
    Z_m = Z / 9.80665

    # ERA5 doesn't directly provide CLWC on pressure levels.
    # We'll estimate it from RH during the sounding processing step.
    CLWC = np.zeros_like(T)

    ds_p.close()
    ds_s.close()

    return {
        "time": times,
        "P_hPa": np.array(config.ERA5_PRESSURE_LEVELS),
        "Z_m": Z_m,
        "T": T,                # [time, level]
        "RH": RH,              # [time, level]
        "CLWC": CLWC,          # [time, level], placeholder
        "t2m": t2m,            # [time]
        "rh2m": rh2m,          # [time]
        "sp": sp,              # [time]
    }


def interpolate_to_mwr_grid(profiles):
    """Interpolate ERA5 profiles to the MWR height grid.

    Args:
        profiles: dict from extract_era5_profiles()
    Returns:
        dict with profiles interpolated to config.HEIGHT_GRID
    """
    n_time = len(profiles["time"])
    target_h = np.array(config.HEIGHT_GRID)

    T_interp = np.zeros((n_time, config.N_LAYERS))
    RH_interp = np.zeros((n_time, config.N_LAYERS))
    CLWC_interp = np.zeros((n_time, config.N_LAYERS))

    for t in range(n_time):
        z_era5 = profiles["Z_m"][t, :]  # [level]
        T_era5 = profiles["T"][t, :]
        RH_era5 = profiles["RH"][t, :]
        CLWC_era5 = profiles["CLWC"][t, :]

        # For points above ground, interpolate
        # ERA5 Z values are height above ground at each pressure level
        above_ground = z_era5 > 0

        if np.sum(above_ground) > 1:
            T_interp[t, :] = interp1d(
                z_era5[above_ground], T_era5[above_ground],
                kind="linear", bounds_error=False,
                fill_value=(T_era5[above_ground][0], T_era5[above_ground][-1])
            )(target_h)

            RH_interp[t, :] = interp1d(
                z_era5[above_ground], RH_era5[above_ground],
                kind="linear", bounds_error=False,
                fill_value=(RH_era5[above_ground][0], RH_era5[above_ground][-1])
            )(target_h)

            CLWC_interp[t, :] = interp1d(
                z_era5[above_ground], CLWC_era5[above_ground],
                kind="linear", bounds_error=False, fill_value=0.0
            )(target_h)

    return {
        "time": profiles["time"],
        "height": target_h,
        "T": T_interp,
        "RH": np.clip(RH_interp, 0, 100),
        "CLWC": np.clip(CLWC_interp, 0, None),
        "t2m": profiles["t2m"],
        "rh2m": profiles["rh2m"],
        "sp": profiles["sp"],
    }


if __name__ == "__main__":
    # Example: download 2013 data for Beijing area
    area = [40.5, 115.0, 39.0, 118.0]  # [N, W, S, E]

    os.makedirs(config.ERA5_DIR, exist_ok=True)

    download_era5_pressure_levels(
        [2013], list(range(1, 13)), area,
        os.path.join(config.ERA5_DIR, "era5_pressure_2013.nc")
    )
    download_era5_single_levels(
        [2013], list(range(1, 13)), area,
        os.path.join(config.ERA5_DIR, "era5_single_2013.nc")
    )
