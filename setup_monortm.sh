#!/bin/bash
# MonoRTM full setup script
# Run this when you have internet access to download TAPE3 spectral data
set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
echo "Setting up MonoRTM for MWR retrieval project"
echo "Project directory: $PROJECT_DIR"

# ---- Step 1: Download AER Line File Parameters from Zenodo ----
echo ""
echo "Step 1: Downloading AER Line File Parameters (v3.8.1, ~442 MB)..."
TMPDIR=$(mktemp -d)
cd "$TMPDIR"

# Try Zenodo first
if curl -L -o aer_line_file.tar.gz \
   "https://zenodo.org/records/5120012/files/aer_v3.8.1.tar.gz?download=1"; then
    echo "  Downloaded from Zenodo successfully."
else
    echo "  Zenodo failed, trying AER_Line_File Python tool from GitHub..."
    # Fallback: clone AER_Line_File tool
    git clone https://github.com/AER-RC/AER_Line_File.git
    cd AER_Line_File
    pip3 install -r requirements.txt 2>/dev/null || true
    python3 download_data.py
    cd "$TMPDIR"
fi

# ---- Step 2: Build LNFL (Line File Creation Program) ----
echo ""
echo "Step 2: Building LNFL for TAPE3 generation..."

if [ ! -d "$TMPDIR/LNFL" ]; then
    git clone https://github.com/AER-RC/LNFL.git
fi

cd "$TMPDIR/LNFL"
git checkout tags/v3.1 2>/dev/null || true

# Compile LNFL (single file)
gfortran -O2 -o lnfl lnfl.f 2>/dev/null || \
gfortran-15 -O2 -o lnfl lnfl.f 2>/dev/null || \
echo "  Note: LNFL compilation skipped (may need manual build)"

# ---- Step 3: Generate TAPE3 for microwave 20-60 GHz ----
echo ""
echo "Step 3: Generating TAPE3 for microwave frequencies..."

# Copy TAPE3 to project
TAPE3_DEST="$PROJECT_DIR/data/TAPE3"

# If we have a pre-built TAPE3 from MonoRTM repo, use it
MONORTM_TAPE3="$TMPDIR/monoRTM/run/TAPE3_spectral_lines.dat.0_55.v5.0_fast"
if [ ! -f "$TAPE3_DEST" ]; then
    # Try to clone MonoRTM to get the pre-built TAPE3
    git clone --depth 1 https://github.com/AER-RC/monoRTM.git monoRTM_tmp 2>/dev/null || true

    if [ -f "$MONORTM_TAPE3" ] && [ -s "$MONORTM_TAPE3" ]; then
        cp "$MONORTM_TAPE3" "$TAPE3_DEST"
        echo "  Copied pre-built TAPE3 from MonoRTM repository."
    elif [ -f "lnfl" ]; then
        # Generate custom TAPE3 using LNFL
        echo "  Generating custom TAPE3 with LNFL..."
        # Create LNFL input for 20-60 GHz microwave
        cat > lnfl_input.dat << 'LNFL_EOF'
 &INCONT
  LINEFIL = 'TAPE3',
  MRGFLG = 1,
  IFLG = 0,
  IBRD = 0,
  V1 = 20.0,
  V2 = 2000.0,    ! cm^-1, 20 GHz ~ 0.67 cm^-1, 60 GHz ~ 2.0 cm^-1, use wide range
  SAMPLE = 0.001,
  RELHUM = 0.0,
  XSELF = 0.0,
  XFORN = 0.0,
  XHG = 0.0,
  HWHM = 0.1,
 /
LNFL_EOF
        echo "  LNFL input written (manual intervention may be needed)"
    fi
fi

# ---- Step 4: Set up environment ----
echo ""
echo "Step 4: Setting environment..."

if [ -f "$TAPE3_DEST" ]; then
    echo "export MONORTM_TAPE3=\"$TAPE3_DEST\"" >> "$HOME/.zshrc"
    echo "  Added MONORTM_TAPE3 to ~/.zshrc"
    ls -la "$TAPE3_DEST"
else
    echo ""
    echo "  ==============================================================="
    echo "  TAPE3 file not obtained automatically."
    echo ""
    echo "  Manual options:"
    echo "  1. Download from Zenodo:"
    echo "     https://zenodo.org/records/5120012"
    echo "     (aer_v3.8.1.tar.gz, 442 MB)"
    echo ""
    echo "  2. Or clone MonoRTM when GitHub is accessible:"
    echo "     git clone https://github.com/AER-RC/monoRTM.git"
    echo "     cp monoRTM/run/TAPE3_spectral_lines.dat.0_55.v5.0_fast \\"
    echo "        $PROJECT_DIR/data/TAPE3"
    echo ""
    echo "  3. Or contact AER: aer_monortm@aer.com"
    echo "  ==============================================================="
fi

# Cleanup
rm -rf "$TMPDIR"

echo ""
echo "Setup complete!"
echo "Run: source ~/.zshrc"
echo "Then: python run.py --stage all"
