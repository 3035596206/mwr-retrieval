#!/usr/bin/env python3
"""Convert ASCII LBLRTM "fast" TAPE3 to gfortran-compatible unformatted binary.

The ASCII format (from LNFL fast output) has fixed-width fields per spectral line:
  Col 1-3:   MOL (i3)      — molecule number
  Col 4-15:  VNU (f12.6)   — wavenumber [cm⁻¹]
  Col 15-26: SP (e11.3)    — line intensity [cm⁻¹/(molec·cm⁻²)]
  Col 26-37: EPP (e11.3)   — lower-state energy [cm⁻¹]
  Col 37-44: (f7.4)        — air-broadened halfwidth [cm⁻¹/atm]
  Col 44-49: (f5.4)        — foreign width
  Col 49-56: (f7.4)        — temperature dependence exponent
  Col 56-64: (f8.5)        — pressure shift [cm⁻¹/atm]
  Col 64-70: (f6.4)        — self width
  Col 70-81: (i3,i3,i3,i2) — upper state quantum numbers (4 x int)
  Col 82-93: (i3,i3,i3,i2) — lower state quantum numbers (4 x int)
  Col 93-99: (i6)          — line coupling/I_FLG
  Col 99-100:(i1)          — speed dependence flag

Binary format (gfortran unformatted sequential):
  Each READ creates one record with 4-byte record-length markers.

  Record 1 (HEADER):
    HLINID(10)  char*8  = 80 bytes
    BMOLID(64)  char*8  = 512 bytes
    MOLCNT(64)  int32   = 256 bytes
    MCNTLC(64)  int32   = 256 bytes
    MCNTNL(64)  int32   = 256 bytes
    SUMSTR(64)  float32 = 256 bytes
    LINMOL      int32   = 4 bytes
    FLINLO      float32 = 4 bytes
    FLINHI      float32 = 4 bytes
    LINCNT      int32   = 4 bytes
    ILINLC      int32   = 4 bytes
    ILINNL      int32   = 4 bytes
    IREC        int32   = 4 bytes
    IRECTL      int32   = 4 bytes
    HID1(2)     char*8  = 16 bytes
    Total: 1664 bytes

  Record N (SPECTRAL LINE BLOCK):
    INPUT_HEADER (24 bytes):
      VMIN  real*8  = 8 bytes
      VMAX  real*8  = 8 bytes
      NREC  int32   = 4 bytes  (number of lines in this block)
      NWDS  int32   = 4 bytes  (number of 4-byte words in INPUT_BLOCK)

    INPUT_BLOCK:
      VNU(NREC)           real*8 × NREC
      SP(NREC)            real*4 × NREC
      ALFA(NREC)          real*4 × NREC
      EPP(NREC)           real*4 × NREC
      MOL(NREC)           int32  × NREC
      HWHM(NREC)          real*4 × NREC
      TMPALF(NREC)        real*4 × NREC
      PSHIFT(NREC)        real*4 × NREC
      IFLG(NREC)          int32  × NREC
      BRD_MOL_FLG_IN(7,NREC)  int32 × 7 × NREC
      BRD_MOL_DAT(21,NREC)    real*4 × 21 × NREC
      SPEED_DEP(NREC)     real*4 × NREC

    Words per line: 2 + 4 + 1 + 4 + 7 + 21 + 1 = 40
    NWDS = NREC × 40
"""

import struct, sys, os, re
import numpy as np

NLINEREC = 250  # max lines per block
MXBRDMOL = 7

def parse_ascii_tape3(path):
    """Parse ASCII LBLRTM fast TAPE3. Returns (header_info, lines_list)."""
    with open(path, 'r') as f:
        raw = f.read()

    lines = raw.strip().split('\n')
    header_line = lines[0].strip() if lines else ''

    # Collect unique molecules from spectral lines
    spectral_lines = []
    molecules = set()
    parse_errors = 0

    for line in lines[1:]:  # skip first (molecule header) line
        line = line.rstrip()
        if not line:
            continue
        rec = parse_spectral_line_regex(line)
        if rec:
            spectral_lines.append(rec)
            molecules.add(rec['mol'])
        else:
            parse_errors += 1

    molecules = sorted(molecules)
    n_good = len(spectral_lines)
    print(f"Parsed {n_good} spectral lines ({parse_errors} skipped), {len(molecules)} molecules: {molecules}")

    # Build header info — use only positive VNU values
    vnus = [r['vnu'] for r in spectral_lines if r['vnu'] > 0]

    header = {
        'hlinid': [b'LNFL    ', b'fast    ', b'0_55    ', b'        ', b'        ',
                    b'        ', b'91I     ', b'        ', b'        ', b'LNFL 91I'],
        'bmolid': [],
        'molcnt': [],
        'mcntlc': [],
        'mcntnl': [],
        'sumstr': [],
        'linmol': len(molecules),
        'flinlo': min(vnus) if vnus else 0.0,
        'flinhi': max(vnus) if vnus else 100.0,
        'lincnt': len(spectral_lines),
        'ilinlc': 0,
        'ilinnl': 0,
        'irec': 1,
        'irectl': 0,
        'hid1': [b'LBLRTM  ', b'v5.0    '],
    }

    # Also fix any negative VNU values in spectral lines
    for r in spectral_lines:
        if r['vnu'] < 0:
            r['vnu'] = abs(r['vnu'])

    # Per-molecule statistics
    for mid in molecules:
        mlines = [r for r in spectral_lines if r['mol'] == mid]
        header['bmolid'].append(f"MOL{mid:04d}  ".encode())
        header['molcnt'].append(len(mlines))
        header['mcntlc'].append(0)  # coupled line count (unknown from ASCII)
        header['mcntnl'].append(0)  # NLTE count
        header['sumstr'].append(np.float32(sum(r['sp'] for r in mlines)))

    # Pad to 64 entries
    for arr_name in ['bmolid', 'molcnt', 'mcntlc', 'mcntnl', 'sumstr']:
        arr = header[arr_name]
        while len(arr) < 64:
            if arr_name == 'bmolid':
                arr.append(b'        ')
            elif arr_name == 'sumstr':
                arr.append(np.float32(0.0))
            else:
                arr.append(0)

    return header, spectral_lines


def parse_spectral_line_regex(line):
    """Parse one spectral line from ASCII fast format using regex extraction.

    The LBLRTM LNFL fast format packs fields without spaces between them.
    We extract ALL numeric tokens (int/float/scientific) from the line
    and map them to the known field order.
    """
    # Extract tokens: integers, floats, D-format, and E-format numbers
    # Pattern matches: D/E format with optional sign, regular floats, integers
    tokens = re.findall(
        r'[+-]?\d+\.\d+[DE][+-]\d+'   # D/E format: 1.165D-32
        r'|[+-]?\d+\.\d*[DE][+-]\d+'   # D/E no digit before exp
        r'|\.\d+[DE][+-]\d+'           # .0D-35 etc
        r'|[+-]?\d+\.\d+'              # regular float
        r'|\.\d+'                       # leading-dot float: .0587
        r'|[+-]?\d+'                    # integer
        , line, re.IGNORECASE
    )

    # Convert tokens: D→E for Python float conversion
    values = []
    for t in tokens:
        try:
            v = float(t.replace('D', 'E').replace('d', 'e'))
            values.append(v)
        except ValueError:
            values.append(0.0)

    if len(values) < 4:
        return None

    # Field mapping based on LNFL fast format order:
    # mol(int), vnu, sp, epp, hwhm, fgnw, tmpalf, pshift, selfw,
    #   4 upper QNs, 4 lower QNs, coupling code(?), iflg(int)
    #
    # Note: the O2 band header lines and other special records have different format
    # We skip those (mol != 1-digit)

    mol = int(values[0])
    if mol <= 0 or mol > 100:
        return None  # skip header lines and invalid molecules

    vnu = np.float64(values[1])
    sp = np.float32(values[2])
    epp = np.float32(values[3])

    # remaining values: hwhm, fgnw, tmpalf, pshift, selfw, qn_upper(4), qn_lower(4), ...
    rest = values[4:]

    hwhm = np.float32(rest[0]) if len(rest) > 0 else np.float32(0.07)
    tmpalf = np.float32(rest[2]) if len(rest) > 2 else np.float32(0.68)

    # Pressure shift: extract from position
    pshift = np.float32(rest[3]) if len(rest) > 3 and abs(rest[3]) < 1.0 else np.float32(0.0)

    # IFLG: usually one of the last integers before the final integers
    # In fast format, IFLG is embedded among the quantum numbers
    # For most lines, IFLG=1 (no line coupling), IFLG=2 (line mixing)
    # Find the integer that looks like IFLG in the tail of the line
    iflg = 1
    for v in values:
        iv = int(v)
        if iv in (1, 2) and v == float(iv) and v < 3:
            # it's an integer, could be IFLG - use the last occurrence near the end
            pass

    # Just use 1 as default — line coupling info not critical for microwave BT
    iflg = 1

    return {
        'mol': mol,
        'vnu': vnu,
        'sp': sp,
        'alfa': hwhm,
        'epp': epp,
        'hwhm': hwhm,
        'tmpalf': tmpalf,
        'pshift': pshift,
        'iflg': iflg,
        'brd_mol_flg': [0] * MXBRDMOL,
        'brd_mol_dat': [np.float32(0.0)] * (MXBRDMOL * 3),
        'speed_dep': np.float32(0.0),
    }


def write_binary_tape3(header, spectral_lines, output_path):
    """Write gfortran-compatible unformatted binary TAPE3."""

    with open(output_path, 'wb') as f:
        # === Record 1: Header ===
        header_data = b''

        # HLINID(10): 10 × 8 chars = 80 bytes
        for i in range(10):
            s = header['hlinid'][i] if i < len(header['hlinid']) else b'        '
            header_data += s[:8].ljust(8, b' ')

        # BMOLID(64): 64 × 8 chars = 512 bytes
        for i in range(64):
            s = header['bmolid'][i]
            header_data += s[:8].ljust(8, b' ')

        # MOLCNT(64): 64 × int32 = 256 bytes
        for i in range(64):
            header_data += struct.pack('<i', int(header['molcnt'][i]))

        # MCNTLC(64): 64 × int32 = 256 bytes
        for i in range(64):
            header_data += struct.pack('<i', int(header['mcntlc'][i]))

        # MCNTNL(64): 64 × int32 = 256 bytes
        for i in range(64):
            header_data += struct.pack('<i', int(header['mcntnl'][i]))

        # SUMSTR(64): 64 × float32 = 256 bytes
        for i in range(64):
            header_data += struct.pack('<f', float(header['sumstr'][i]))

        # LINMOL: int32 = 4 bytes
        header_data += struct.pack('<i', header['linmol'])
        # FLINLO: float32 = 4 bytes
        header_data += struct.pack('<f', header['flinlo'])
        # FLINHI: float32 = 4 bytes
        header_data += struct.pack('<f', header['flinhi'])
        # LINCNT: int32 = 4 bytes
        header_data += struct.pack('<i', header['lincnt'])
        # ILINLC: int32 = 4 bytes
        header_data += struct.pack('<i', header['ilinlc'])
        # ILINNL: int32 = 4 bytes
        header_data += struct.pack('<i', header['ilinnl'])
        # IREC: int32 = 4 bytes
        header_data += struct.pack('<i', header['irec'])
        # IRECTL: int32 = 4 bytes
        header_data += struct.pack('<i', header['irectl'])
        # HID1(2): 2 × 8 chars = 16 bytes
        for i in range(2):
            s = header['hid1'][i]
            header_data += s[:8].ljust(8, b' ')

        assert len(header_data) == 1664, f"Header size mismatch: {len(header_data)} != 1664"

        # Write unformatted record (gfortran: 4-byte length prefix + data + 4-byte suffix)
        f.write(struct.pack('<i', len(header_data)))
        f.write(header_data)
        f.write(struct.pack('<i', len(header_data)))

        # === Spectral line block records ===
        # Group lines into blocks of up to NLINEREC
        for block_start in range(0, len(spectral_lines), NLINEREC):
            block_lines = spectral_lines[block_start:block_start + NLINEREC]
            nrec = len(block_lines)

            # INPUT_HEADER
            vnus = [r['vnu'] for r in block_lines]
            inp_header = struct.pack('<ddii',
                float(min(vnus)),  # VMIN
                float(max(vnus)),  # VMAX
                nrec,              # NREC
                0,                 # NWDS (will compute after building block)
            )

            # INPUT_BLOCK
            block_data = b''

            # VNU (real*8 × nrec)
            for r in block_lines:
                block_data += struct.pack('<d', float(r['vnu']))
            # Pad to NLINEREC
            for _ in range(NLINEREC - nrec):
                block_data += struct.pack('<d', 0.0)

            # SP (real*4 × NLINEREC)
            for r in block_lines:
                block_data += struct.pack('<f', float(r['sp']))
            for _ in range(NLINEREC - nrec):
                block_data += struct.pack('<f', 0.0)

            # ALFA (real*4 × NLINEREC)
            for r in block_lines:
                block_data += struct.pack('<f', float(r['alfa']))
            for _ in range(NLINEREC - nrec):
                block_data += struct.pack('<f', 0.0)

            # EPP (real*4 × NLINEREC)
            for r in block_lines:
                block_data += struct.pack('<f', float(r['epp']))
            for _ in range(NLINEREC - nrec):
                block_data += struct.pack('<f', 0.0)

            # MOL (int32 × NLINEREC)
            for r in block_lines:
                block_data += struct.pack('<i', r['mol'])
            for _ in range(NLINEREC - nrec):
                block_data += struct.pack('<i', 0)

            # HWHM (real*4 × NLINEREC)
            for r in block_lines:
                block_data += struct.pack('<f', float(r['hwhm']))
            for _ in range(NLINEREC - nrec):
                block_data += struct.pack('<f', 0.0)

            # TMPALF (real*4 × NLINEREC)
            for r in block_lines:
                block_data += struct.pack('<f', float(r['tmpalf']))
            for _ in range(NLINEREC - nrec):
                block_data += struct.pack('<f', 0.0)

            # PSHIFT (real*4 × NLINEREC)
            for r in block_lines:
                block_data += struct.pack('<f', float(r['pshift']))
            for _ in range(NLINEREC - nrec):
                block_data += struct.pack('<f', 0.0)

            # IFLG (int32 × NLINEREC)
            for r in block_lines:
                block_data += struct.pack('<i', r['iflg'])
            for _ in range(NLINEREC - nrec):
                block_data += struct.pack('<i', 0)

            # BRD_MOL_FLG_IN (int32 × 7 × NLINEREC)
            for j in range(MXBRDMOL):
                for r in block_lines:
                    block_data += struct.pack('<i', int(r['brd_mol_flg'][j]))
                for _ in range(NLINEREC - nrec):
                    block_data += struct.pack('<i', 0)

            # BRD_MOL_DAT (real*4 × 21 × NLINEREC)
            for j in range(MXBRDMOL * 3):
                for r in block_lines:
                    block_data += struct.pack('<f', float(r['brd_mol_dat'][j]))
                for _ in range(NLINEREC - nrec):
                    block_data += struct.pack('<f', 0.0)

            # SPEED_DEP (real*4 × NLINEREC)
            for r in block_lines:
                block_data += struct.pack('<f', float(r['speed_dep']))
            for _ in range(NLINEREC - nrec):
                block_data += struct.pack('<f', 0.0)

            # Compute NWDS: number of 4-byte words in INPUT_BLOCK
            # Per line: VNU(2) + SP(1)+ALFA(1)+EPP(1)+MOL(1) + HWHM(1)+TMPALF(1)+PSHIFT(1)+IFLG(1) + BRD_FLG(7) + BRD_DAT(21) + SPEED_DEP(1)
            words_per_line = 2 + 4 + 4 + 7 + 21 + 1  # = 39
            nwds = NLINEREC * words_per_line

            # INPUT_HEADER record (24 bytes, separate record)
            inp_header = struct.pack('<ddii',
                float(min(vnus)) if vnus else 0.0,
                float(max(vnus)) if vnus else 0.0,
                nrec,
                nwds,
            )

            # Write INPUT_HEADER as separate unformatted record
            f.write(struct.pack('<i', len(inp_header)))
            f.write(inp_header)
            f.write(struct.pack('<i', len(inp_header)))

            # Write INPUT_BLOCK as separate unformatted record
            f.write(struct.pack('<i', len(block_data)))
            f.write(block_data)
            f.write(struct.pack('<i', len(block_data)))

    print(f"Wrote binary TAPE3: {output_path} ({os.path.getsize(output_path)} bytes)")


def main():
    ascii_path = sys.argv[1] if len(sys.argv) > 1 else '/Users/ink/test/mwr_retrieval/data/TAPE3/TAPE3'
    output_path = sys.argv[2] if len(sys.argv) > 2 else '/Users/ink/test/mwr_retrieval/data/TAPE3/TAPE3_bin'

    print(f"Reading: {ascii_path}")
    header, lines = parse_ascii_tape3(ascii_path)

    print(f"\nHeader summary:")
    print(f"  LINMOL={header['linmol']}")
    print(f"  FLINLO={header['flinlo']:.4f} FLINHI={header['flinhi']:.4f}")
    print(f"  LINCNT={header['lincnt']}")

    write_binary_tape3(header, lines, output_path)
    print("Done!")


if __name__ == '__main__':
    main()
