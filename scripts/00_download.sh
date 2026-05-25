#!/bin/bash
# Download CrossDocked2020 / Francoeur2020 split (PDBbind2016 + .types files)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_DIR="${PROJECT_DIR}/data"

mkdir -p "${DATA_DIR}"
cd "${PROJECT_DIR}"

if command -v aria2c &> /dev/null; then
    DOWNLOADER=aria2c
elif command -v wget &> /dev/null; then
    DOWNLOADER=wget
    echo "aria2c not found; using wget"
else
    echo "ERROR: neither aria2c nor wget found. Install one of:"
    echo "  aria2 — sudo apt-get install aria2   or   brew install aria2"
    echo "  wget  — usually preinstalled on Linux"
    exit 1
fi

download_file() {
    local url="$1"
    local out="$2"
    if [[ $DOWNLOADER == aria2c ]]; then
        aria2c -x 16 -s 16 -c "$url" -o "$out"
    else
        wget -c -O "$out" "$url"
    fi
}

echo "Downloading PDBbind2016 structures..."
download_file \
    https://bits.csb.pitt.edu/files/crossdock2020/PDBbind2016.tar.gz \
    PDBbind2016.tar.gz

echo "Downloading paper .types files..."
download_file \
    https://bits.csb.pitt.edu/files/crossdock2020/v1.0/paper_types.tar.gz \
    paper_types.tar.gz

echo "Extracting into ${DATA_DIR}..."
tar -xzf PDBbind2016.tar.gz -C data
tar -xzf paper_types.tar.gz -C data

echo "Cleaning up archives..."
rm -f PDBbind2016.tar.gz paper_types.tar.gz

echo ""
echo "Done! Data layout:"
echo "  ${DATA_DIR}/PDBbind2016/"
echo "  ${DATA_DIR}/types/ref_uff_train0.types"
echo "  ${DATA_DIR}/types/ref_uff_test0.types"
echo ""
echo "Train: bash scripts/run_training.sh"
