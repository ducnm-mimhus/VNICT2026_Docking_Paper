#!/bin/bash
# Pack 1–2 CrossDock complexes for Kaggle demo.
# Usage: bash scripts/pack_demo_inference.sh 4kqp 2ydt

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_ROOT="${PROJECT_DIR}/data"
OUT="${PROJECT_DIR}/demo_inference"
TYPES_SRC="${DATA_ROOT}/types/ref_uff_test0.types"

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/pack_demo_inference.sh PDB_ID [PDB_ID ...]"
  exit 1
fi
if [[ ! -f "${TYPES_SRC}" ]]; then
  echo "Missing ${TYPES_SRC}. Run: bash scripts/00_download.sh"
  exit 1
fi

mkdir -p "${OUT}/types"
: > "${OUT}/types/demo.types"

for pdb in "$@"; do
  pdb_lc="$(echo "${pdb}" | tr '[:upper:]' '[:lower:]')"
  line="$(grep -i "${pdb_lc}/" "${TYPES_SRC}" | head -n 1 | sed 's/#.*//' | xargs)"
  [[ -n "${line}" ]] || { echo "No .types line for ${pdb}"; exit 1; }
  rec_rel="$(echo "${line}" | awk '{print $(NF-1)}')"
  lig_rel="$(echo "${line}" | awk '{print $NF}')"
  rec_abs="${DATA_ROOT}/${rec_rel}"
  lig_abs="${DATA_ROOT}/${lig_rel}"
  [[ -f "${rec_abs}" && -f "${lig_abs}" ]] || { echo "Missing gninatypes for ${pdb}"; exit 1; }
  mkdir -p "${OUT}/$(dirname "${rec_rel}")"
  cp -f "${rec_abs}" "${OUT}/${rec_rel}"
  cp -f "${lig_abs}" "${OUT}/${lig_rel}"
  echo "${line}" >> "${OUT}/types/demo.types"
  echo "OK ${pdb}"
done
du -sh "${OUT}"
