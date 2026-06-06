"""Python wrapper for MonoRTM (MONOchromatic Radiative Transfer Model).

MonoRTM is a Fortran radiative transfer model from AER Inc. that computes
atmospheric microwave brightness temperatures.

This wrapper supports MonoRTM v5.x from https://github.com/AER-RC/monoRTM

Usage:
    rtm = MonoRTM(monortm_path='/usr/local/bin/monortm')
    tb = rtm.simulate(profile)  # profile is dict with T, RH, CLWC, height

Input format:
    MonoRTM reads two files in the working directory:
    1. MONORTM.IN - control file (TAPE5 format)
    2. MONORTM_PROF.IN - atmospheric profile data (for IATM=0)

    Plus TAPE3 - spectral line data (symlinked from installation).
"""

import os
import subprocess
import tempfile
import shutil
import numpy as np
import config


class MonoRTM:
    """Python interface to the MonoRTM Fortran executable.

    Example:
        rtm = MonoRTM(monortm_path='/usr/local/bin/monortm',
                       tape3_path='/usr/local/share/monortm/TAPE3')
        tb = rtm.simulate(profile)
        tb_batch = rtm.simulate_batch(profiles_dict)
    """

    # Default RPG HATPRO frequencies in wavenumber [cm^-1]
    # Conversion: wavenumber = frequency_GHz / 0.0299792458
    FREQ_GHZ_TO_CM1 = 1.0 / 0.0299792458

    def __init__(self, monortm_path=None, tape3_path=None):
        """
        Args:
            monortm_path: path to monoRTM executable
            tape3_path: path to TAPE3 spectral line data file
        """
        if monortm_path is None:
            monortm_path = self._find_executable()
        self.monortm_path = monortm_path
        self._check_executable()

        if tape3_path is None:
            tape3_path = self._find_tape3()
        self.tape3_path = tape3_path

        self.frequencies = config.ALL_CHANNELS  # GHz

    def _find_executable(self):
        """Find MonoRTM executable."""
        candidates = [
            os.environ.get("MONORTM_PATH", ""),
            os.path.join(os.path.dirname(__file__), "..", "bin", "monortm"),
            "/usr/local/bin/monortm",
            "/opt/homebrew/bin/monortm",
        ]
        for path in candidates:
            if path and os.path.isfile(path):
                return path
        return "monortm"

    def _find_tape3(self):
        """Find TAPE3 spectral line data file."""
        candidates = [
            os.environ.get("MONORTM_TAPE3", ""),
            # Project data directory (setup_monortm.sh installs here)
            os.path.join(os.path.dirname(__file__), "..", "data", "TAPE3"),
            os.path.join(os.path.dirname(self.monortm_path), "..", "run", "in",
                         "TAPE3_spectral_lines.dat.0_55.v5.0_fast"),
            "/tmp/monortm_src/run/in/TAPE3_spectral_lines.dat.0_55.v5.0_fast",
        ]
        for path in candidates:
            if path and os.path.isfile(path):
                return path
        # Search relative to executable
        exe_dir = os.path.dirname(os.path.abspath(self.monortm_path))
        for root, dirs, files in os.walk(os.path.dirname(exe_dir)):
            for f in files:
                if "TAPE3" in f and "spectral" in f:
                    return os.path.join(root, f)
        return None

    def _check_executable(self):
        if not os.path.isfile(self.monortm_path):
            raise FileNotFoundError(
                f"MonoRTM executable not found: '{self.monortm_path}'\n"
                f"Build it: build_monortm.sh\n"
                f"Or set MONORTM_PATH environment variable."
            )

    def _make_monortm_in(self, frequencies_ghz, tmpdir):
        """Generate MONORTM.IN control file for IATM=0 (user profile).

        Record layout follows the standard MonoRTM TAPE5 format.
        """
        # Convert frequencies to wavenumbers [cm^-1]
        wn = [f / 0.0299792458 for f in frequencies_ghz]

        lines = []
        lines.append(" MWR retrieval: downwelling brightness temperature simulation")
        lines.append("HIRAC LBLF CNTM AERS EMIT SCAN FLTR PLOT TEST IATM IMRG ILAS  IOD XSCT MPTS NPTS      ISPD")
        lines.append("$ Rundeck")

        # Record 1.1: flags. IATM=0 means user-defined atmosphere
        # fmt: HI=1 LBLFLG=1 CNTMFLG=1 AERSFLG=0 EMIT=1 SCAN=0 FLTR=1 PLOT=1
        #      TEST=0 IATM=0 IMRG=0 ILAS=0 IOD=0 XSCT=0 MPTS=0 NPTS=0 ISPD=0
        # PLOT=1 is required to output brightness temperatures
        lines.append("    1    1    1    0    1    0    1    1    0    0    0    0    0    0    0    0    0    0")

        # Record 1.2: bounds
        lines.append("-0.200E+00 8.800E+00 0.000E+00 0.100E-00 0.000E+00 0.000E+00 0.000E+00 0.000E+00    0      0.000E+00    0")

        # Record 1.3: frequency list
        lines.append(str(len(wn)))
        for w in wn:
            lines.append(f"{w:.6f}")

        # Record 1.4: additional parameters
        lines.append("     0.    1.0       0.000E+00 0.000E+00 0.000E+00 0.000E+00 0.000E+00")

        # End record
        lines.append("    6    2    0    1    1    7    1")
        lines.append("     0.000    30.000       0.000")
        lines.append("     0.000     3.000     3.000     0.000     0.000")
        lines.append("-1")
        lines.append("%%%%%%%%%%%%%%%%")
        lines.append("     0.000     0.000     0.000     0.000     0.000")

        return "\n".join(lines)

    def _make_monortm_prof_in(self, profile):
        """Generate MONORTM_PROF.IN with atmospheric profile data.

        The format for IATM=0 (user-defined atmosphere) is:
        Line 1:    NLAY NMOL, header
        Each layer: pressure, temperature, boundary info
                    molecular column amounts (23 values for v5.6)

        For our purpose, we use a simplified approach: provide P, T, and the
        standard molecular column amounts from a reference atmosphere,
        then scale H2O based on the actual RH profile.

        Args:
            profile: dict with T, RH, P_hPa, CLWC, height arrays
        Returns:
            content string for MONORTM_PROF.IN
        """
        n_layers = len(profile["height"])
        n_mol = 22  # number of molecules in the standard format

        # Standard mole fractions (from U.S. Standard Atmosphere)
        # H2O will be scaled per layer
        lines = []
        lines.append(f" 1 {n_layers}   {n_mol}  1.000000"
                     f"USER ATMOSPHERE  H1=    0.00 H2=   20.00 "
                     f"ANG=   0.000 LEN= 0")

        for i in range(n_layers):
            p_hpa = profile.get("P_hPa", np.ones(n_layers) * 1013.0)[i] if isinstance(profile.get("P_hPa"), np.ndarray) else (1013.25 * np.exp(-profile["height"][i] / 8000.0))
            if isinstance(p_hpa, np.ndarray):
                p_hpa = float(p_hpa)
            t_k = float(profile["T"][i])
            rh = float(profile["RH"][i])

            # Surface altitude reference
            alt_km = profile["height"][i] / 1000.0
            p_ref = float(p_hpa)

            # Layer boundary data line
            # Format: PRESSURE(mb) TEMPERATURE(K) BOUNDARY_FLAG ALT_START ALT_END P_REF T_REF
            lines.append(f"{p_ref:12.4f} {t_k:12.2f}              3"
                         f"                         {alt_km:.3f} "
                         f"{alt_km + 0.1:.3f} {p_ref:.2f} {t_k:.2f}")

            # Molecular column amounts
            # We use approximate standard column values scaled to layer pressure
            p_frac = p_ref / 1013.25

            # Species: H2O, CO2, O3, N2O, CO, CH4, O2, NO, SO2, NO2, NH3, HNO3,
            #          OH, HF, HCL, HBR, HI, ClO, OCS, H2CO, HOCl, N2, HCN, etc.
            # For microwave, the key species are H2O and O2.
            # Standard column amounts per layer (molec/cm^2) - approximate values

            # H2O column: scale by RH
            es = 6.1078 * np.exp(17.2693882 * (t_k - 273.16) / (t_k - 35.86))
            e = rh / 100.0 * es
            h2o_mmr = 0.622 * e / (p_ref - e)  # mass mixing ratio
            # Column: scale from standard ~ 1e22 molec/cm^2 at surface
            h2o_col = h2o_mmr * p_frac * 3.0e22
            h2o_col = max(h2o_col, 1.0e10)

            # Standard column values (molec/cm^2) - scaled to this layer
            cols = [
                h2o_col,                                 # 1: H2O
                6.0e21 * p_frac,                         # 2: CO2
                4.7e16 * p_frac,                         # 3: O3
                5.5e17 * p_frac,                         # 4: N2O
                2.5e17 * p_frac,                         # 5: CO
                2.9e18 * p_frac,                         # 6: CH4
                3.6e23 * p_frac,                         # 7: O2
                1.5e22 * p_frac,                         # 8: NO
                5.1e14 * p_frac,                         # 9: SO2
                5.0e14 * p_frac,                         # 10: NO2
                3.9e13 * p_frac,                         # 11: NH3
                8.6e14 * p_frac,                         # 12: HNO3
                9.1e13 * p_frac,                         # 13: OH
                7.5e10 * p_frac,                         # 14: HF
                1.7e10 * p_frac,                         # 15: HCL
                1.6e15 * p_frac,                         # 16: HBR
                2.9e12 * p_frac,                         # 17: HI
                5.1e12 * p_frac,                         # 18: ClO
                1.7e10 * p_frac,                         # 19: OCS
                1.0e15 * p_frac,                         # 20: H2CO
                3.1e15 * p_frac,                         # 21: HOCl
                1.3e24 * p_frac,                         # 22: N2
            ]

            # Write in groups
            line = ""
            for j, c in enumerate(cols):
                line += f"{c:15.7E}"
                if (j + 1) % 8 == 0:
                    lines.append(line)
                    line = ""
            if line:
                lines.append(line)

        return "\n".join(lines) + "\n"

    def _parse_output(self, output_text):
        """Parse MonoRTM MONORTM.OUT to extract brightness temperatures.

        MonoRTM outputs a table with:
            FREQ(GHZ) or WAVENUMBER  TBB(K)  TRANSMISSION ...

        Returns:
            array of brightness temperatures [K], matching self.frequencies
        """
        tb = np.zeros(len(self.frequencies))

        lines = output_text.split("\n")
        data_started = False
        data_lines = []

        for line in lines:
            # Look for the data table header
            if "SPECTRAL" in line.upper() or "RADIANCE" in line.upper():
                data_started = True
                continue

            if "FREQ" in line.upper() and "TB" in line.upper():
                data_started = True
                continue

            if data_started and line.strip():
                parts = line.split()
                # Try to parse as frequency & TB
                if len(parts) >= 2:
                    try:
                        freq_val = float(parts[0])
                        # Frequency could be in GHz or cm^-1
                        if freq_val > 100:  # likely cm^-1
                            freq_ghz = freq_val * 0.0299792458
                        else:
                            freq_ghz = freq_val

                        tb_val = float(parts[1])
                        if 0 < tb_val < 400:
                            data_lines.append((freq_ghz, tb_val))
                    except (ValueError, IndexError):
                        continue

        # Match to our channels (nearest frequency)
        for i, target_freq in enumerate(self.frequencies):
            best_match = None
            best_dist = float("inf")
            for freq, val in data_lines:
                dist = abs(freq - target_freq)
                if dist < best_dist:
                    best_dist = dist
                    best_match = val
            if best_match is not None and best_dist < 1.0:  # within 1 GHz
                tb[i] = best_match
            elif best_match is not None:
                tb[i] = best_match  # use closest anyway

        return tb

    def simulate(self, profile):
        """Run MonoRTM for one atmospheric profile.

        Args:
            profile: dict with:
                T: temperature [K], shape (n_layers,)
                RH: relative humidity [%]
                CLWC: cloud liquid water [g/m^3] (optional)
                height: height [m]
                P_hPa: pressure [hPa] (optional)

        Returns:
            tb: brightness temperatures [K], shape (14,)
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write MONORTM.IN
            monortm_in = self._make_monortm_in(self.frequencies, tmpdir)
            with open(os.path.join(tmpdir, "MONORTM.IN"), "w") as f:
                f.write(monortm_in)

            # Write MONORTM_PROF.IN
            prof_in = self._make_monortm_prof_in(profile)
            with open(os.path.join(tmpdir, "MONORTM_PROF.IN"), "w") as f:
                f.write(prof_in)

            # Copy TAPE3 (symlink may not work across filesystems)
            if self.tape3_path and os.path.exists(self.tape3_path):
                shutil.copy2(self.tape3_path, os.path.join(tmpdir, "TAPE3"))

            # Run MonoRTM
            try:
                result = subprocess.run(
                    [self.monortm_path],
                    capture_output=True,
                    text=True,
                    cwd=tmpdir,
                    timeout=60,
                )
            except subprocess.TimeoutExpired:
                raise RuntimeError("MonoRTM execution timed out (60s)")

            # Check for errors
            if result.returncode != 0:
                stdout_sample = result.stdout[-500:] if result.stdout else "(empty)"
                stderr_sample = result.stderr[-500:] if result.stderr else "(empty)"
                raise RuntimeError(
                    f"MonoRTM failed with code {result.returncode}\n"
                    f"STDERR: {stderr_sample}\nSTDOUT: {stdout_sample}"
                )

            # Read output
            output_path = os.path.join(tmpdir, "MONORTM.OUT")
            if os.path.exists(output_path):
                with open(output_path, "r") as f:
                    output_text = f.read()
            else:
                output_text = result.stdout

            tb = self._parse_output(output_text)

        return tb

    def simulate_batch(self, profiles_dict):
        """Simulate BT for a batch of profiles.

        Args:
            profiles_dict: dict with arrays of shape (n_time, n_layers)
        Returns:
            tb_batch: shape (n_time, 14)
        """
        n_time = profiles_dict["T"].shape[0]
        n_chan = len(self.frequencies)
        tb_batch = np.zeros((n_time, n_chan))

        for t in range(n_time):
            profile = {
                "T": profiles_dict["T"][t],
                "RH": profiles_dict["RH"][t],
                "CLWC": profiles_dict["CLWC"][t],
                "P_hPa": profiles_dict["P"][t] if "P" in profiles_dict else
                    np.ones(len(profiles_dict["height"])) * 1013.0,
                "height": profiles_dict["height"],
            }
            tb_batch[t] = self.simulate(profile)

            if (t + 1) % 50 == 0:
                print(f"  MonoRTM: {t+1}/{n_time} profiles done")

        return tb_batch


# ============================================================
# Build helper
# ============================================================

def build_monortm_instructions():
    """Print instructions for building MonoRTM."""
    print("""
=================================================================
  MonoRTM Build Instructions
=================================================================

MonoRTM is now compiled and ready at:
  /Users/ink/test/mwr_retrieval/bin/monortm

To build from source:

  1. Clone:  git clone https://github.com/AER-RC/monoRTM.git
  2. Build:  bash build_monortm.sh

The build script compiles 16 Fortran files and links them.

=================================================================
""")


if __name__ == "__main__":
    # Quick test
    try:
        rtm = MonoRTM()
        print(f"MonoRTM found at: {rtm.monortm_path}")
        print(f"TAPE3 found at: {rtm.tape3_path}")

        # Test with standard atmosphere
        heights = np.array(config.HEIGHT_GRID)
        test_profile = {
            "T": 288.15 - 6.5 * heights / 1000.0,
            "RH": np.full_like(heights, 50.0),
            "CLWC": np.zeros_like(heights),
            "height": heights,
            "P_hPa": 1013.25 * np.exp(-heights / 8000.0),
        }

        print("\nRunning MonoRTM simulation...")
        tb = rtm.simulate(test_profile)
        print("\nBrightness Temperatures [K]:")
        for f, t in zip(config.ALL_CHANNELS, tb):
            band = "K" if f < 40 else "V"
            print(f"  {band}-band {f:.2f} GHz: {t:.2f} K")

    except FileNotFoundError as e:
        print(f"\n{e}")
        build_monortm_instructions()
