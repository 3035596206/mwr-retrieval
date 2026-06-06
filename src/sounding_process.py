"""Radiosonde data download and processing.

Downloads sounding data from the University of Wyoming archive,
extracts temperature/RH profiles, estimates LWC, and interpolates
to the MWR height grid.
"""

import os
import re
import numpy as np
import requests
from scipy.interpolate import interp1d
from datetime import datetime
import config

# Wyoming sounding URL format
WYOMING_URL = (
    "http://weather.uwyo.edu/cgi-bin/sounding?"
    "region={region}&TYPE=TEXT%3ALIST&YEAR={year}&MONTH={month:02d}"
    "&FROM={from_day:02d}12&TO={to_day:02d}12&STNM={station}"
)


def download_sounding(station, year, month, output_dir=None):
    """Download monthly radiosonde data from Wyoming website.

    Args:
        station: WMO station number (e.g., 54511 for Beijing/Nanjiao)
        year: year
        month: month (1-12)
        output_dir: directory to save raw text files
    Returns:
        path to downloaded file, or None if download failed
    """
    if output_dir is None:
        output_dir = config.RADIOSONDE_DIR

    os.makedirs(output_dir, exist_ok=True)

    # Beijing (Nanjiao) uses region=naconf (North America/China config)
    # Nanjiao station: 54511
    url = WYOMING_URL.format(
        region="naconf", year=year, month=month,
        from_day=1, to_day=28, station=station
    )
    url2 = WYOMING_URL.format(
        region="naconf", year=year, month=month,
        from_day=29, to_day=31, station=station
    )

    all_text = []
    for url in [url, url2]:
        try:
            resp = requests.get(url, timeout=60)
            if resp.status_code == 200 and "Can't get" not in resp.text:
                all_text.append(resp.text)
        except Exception as e:
            print(f"  Warning: failed to download {url}: {e}")

    if not all_text:
        print(f"No data for {year}-{month:02d}")
        return None

    output_path = os.path.join(output_dir, f"sounding_{station}_{year}{month:02d}.txt")
    with open(output_path, "w") as f:
        f.write("\n".join(all_text))

    print(f"Downloaded sounding data for {year}-{month:02d} to {output_path}")
    return output_path


def parse_sounding_file(filepath):
    """Parse a Wyoming-formatted sounding file.

    Returns:
        list of dicts, each containing:
            time: datetime
            height: array [m]
            pressure: array [hPa]
            temp: array [C]
            rh: array [%]
    """
    with open(filepath, "r") as f:
        content = f.read()

    # Split into individual soundings
    soundings = []
    sections = re.split(r"<H2>|</H2>", content)

    # Process raw data sections
    data_sections = re.findall(
        r"PRE>([\s\S]*?)</PRE>",
        content,
        re.IGNORECASE
    )

    header_sections = re.findall(
        r"<H2>(.*?)</H2>",
        content,
        re.IGNORECASE
    )

    for header, data_text in zip(header_sections, data_sections):
        # Parse observation time from header
        time_match = re.search(r"(\d{2}Z \d{1,2} \w+ \d{4})", header)
        if not time_match:
            continue

        try:
            obs_time = datetime.strptime(time_match.group(1), "%HZ %d %b %Y")
        except ValueError:
            continue

        # Parse data lines
        lines = data_text.strip().split("\n")
        pressure = []
        height = []
        temp = []
        rh = []

        for line in lines:
            parts = line.split()
            if len(parts) < 7:
                continue
            try:
                pres = float(parts[0])
                hght = float(parts[1])
                tmp = float(parts[2])
                relh = float(parts[4])
                pressure.append(pres)
                height.append(hght)
                temp.append(tmp)
                rh.append(relh)
            except (ValueError, IndexError):
                continue

        if len(pressure) > 5:
            soundings.append({
                "time": obs_time,
                "pressure": np.array(pressure),    # [hPa]
                "height": np.array(height),        # [m]
                "temp": np.array(temp),            # [C]
                "rh": np.array(rh),                # [%]
            })

    return soundings


def estimate_lwc(height, rh):
    """Estimate liquid water content from relative humidity.

    Method from Liu Yaya (2011) and Zhang Xi (2012):
    - Above 600m: RH < 85% -> LWC = 0
    - Above 600m: RH > 95% -> LWC = 0.5 g/m^3
    - Between: linear interpolation
    - Below 600m: LWC = 0

    Args:
        height: height array [m]
        rh: relative humidity [%]
    Returns:
        lwc: liquid water content [g/m^3]
    """
    lwc = np.zeros_like(rh)

    above_600 = height > 600
    if not np.any(above_600):
        return lwc

    rh_above = rh[above_600]
    lwc_above = np.zeros_like(rh_above)

    # RH < 85 -> 0
    mask_low = rh_above < 85
    lwc_above[mask_low] = 0.0

    # RH > 95 -> 0.5 g/m^3
    mask_high = rh_above > 95
    lwc_above[mask_high] = 0.5

    # Linear interpolation between 85% and 95%
    mask_mid = ~mask_low & ~mask_high
    lwc_above[mask_mid] = 0.5 * (rh_above[mask_mid] - 85) / (95 - 85)

    lwc[above_600] = lwc_above
    return lwc


def interpolate_sounding_to_grid(sounding):
    """Interpolate single sounding to MWR 93-layer height grid.

    Also estimates LWC and computes 2m surrogate values.
    """
    target_h = np.array(config.HEIGHT_GRID)
    n = len(target_h)

    h = sounding["height"]
    T_C = sounding["temp"]
    RH = np.clip(sounding["rh"], 0, 100)
    P = sounding["pressure"]

    # Temperature to Kelvin
    T_K = T_C + 273.15

    # Sort by height
    sort_idx = np.argsort(h)
    h = h[sort_idx]
    T_K = T_K[sort_idx]
    RH = RH[sort_idx]
    P = P[sort_idx]

    # Remove duplicates
    unique_idx = np.unique(h, return_index=True)[1]
    h = h[unique_idx]
    T_K = T_K[unique_idx]
    RH = RH[unique_idx]
    P = P[unique_idx]

    # Interpolate to target grid
    if len(h) < 2:
        return None

    T_interp = interp1d(h, T_K, kind="linear", bounds_error=False,
                        fill_value=(T_K[0], T_K[-1]))(target_h)
    RH_interp = interp1d(h, RH, kind="linear", bounds_error=False,
                         fill_value=(RH[0], RH[-1]))(target_h)
    P_interp = interp1d(h, P, kind="linear", bounds_error=False,
                        fill_value=(P[0], P[-1]))(target_h)

    # Estimate LWC
    LWC = estimate_lwc(target_h, RH_interp)

    # 2m approximations (first grid level ~ 0m)
    t2m = float(T_interp[0])
    rh2m = float(RH_interp[0])

    return {
        "time": sounding["time"],
        "height": target_h,
        "T": T_interp,
        "RH": RH_interp,
        "P": P_interp,
        "CLWC": LWC,
        "t2m": t2m,
        "rh2m": rh2m,
    }


def process_all_soundings(filepath):
    """Parse a sounding file and interpolate all profiles to MWR grid."""
    soundings = parse_sounding_file(filepath)
    profiles = []
    for s in soundings:
        prof = interpolate_sounding_to_grid(s)
        if prof is not None:
            profiles.append(prof)
    return profiles


if __name__ == "__main__":
    # Example: download Beijing Nanjiao (54511) data for January 2018
    os.makedirs(config.RADIOSONDE_DIR, exist_ok=True)

    path = download_sounding(station=54511, year=2018, month=1)
    if path:
        profiles = process_all_soundings(path)
        print(f"Processed {len(profiles)} soundings")
        if profiles:
            print(f"  Temperature range: {profiles[0]['T'].min():.1f} - "
                  f"{profiles[0]['T'].max():.1f} K")
            print(f"  RH range: {profiles[0]['RH'].min():.1f} - "
                  f"{profiles[0]['RH'].max():.1f} %")
