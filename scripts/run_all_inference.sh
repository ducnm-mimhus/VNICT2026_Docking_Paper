#!/bin/bash
# Chay tools/run_inference.py cho 6 model co checkpoint (bo qua equibind — khong
# co best_model.pt), tren CUNG 1 testfile, xuat du doan tung mau vao
# results/predictions/<model>.csv.
#
# Cac file CSV nay la dau vao cho tools/exact_metrics.py (A1) va
# tools/paired_bootstrap.py (VD4).
#
# Cach dung:
#   bash scripts/run_all_inference.sh
#   DEVICE=cpu bash scripts/run_all_inference.sh          # neu khong co GPU
#   TEST_FILE=... DATA_ROOT=... bash scripts/run_all_inference.sh

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

DATA_ROOT="${DATA_ROOT:-${PROJECT_DIR}/data}"
TEST_FILE="${TEST_FILE:-${DATA_ROOT}/types/ref_uff_test0.types}"
OUT_DIR="${PROJECT_DIR}/results/predictions"
mkdir -p "${OUT_DIR}"

MODELS=(gnina_dense gnina_default2018 pafnucy potentialnet tankbind geoformerdock)

for m in "${MODELS[@]}"; do
    CKPT="${PROJECT_DIR}/results/models/${m}/best_model.pt"
    if [ ! -f "${CKPT}" ]; then
        echo "[SKIP] khong thay ${CKPT}"
        continue
    fi
    OUT_CSV="${OUT_DIR}/${m}.csv"
    if [ -f "${OUT_CSV}" ]; then
        echo "[SKIP] ${OUT_CSV} da ton tai"
        continue
    fi
    echo "=== Inference: ${m} ==="
    python3 "${PROJECT_DIR}/tools/run_inference.py" \
        --model "${m}" \
        --checkpoint "${CKPT}" \
        --testfile "${TEST_FILE}" \
        --data_root "${DATA_ROOT}" \
        --batch_size "${BATCH_SIZE:-512}" \
        -g "${DEVICE:-cuda:0}" \
        --out "${OUT_CSV}"
done

echo ""
echo "Done. CSV du doan o: ${OUT_DIR}/"
