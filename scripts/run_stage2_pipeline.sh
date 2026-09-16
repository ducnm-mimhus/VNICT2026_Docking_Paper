#!/bin/bash
# Toan bo logic Stage 2 (inference, A1, paired bootstrap) — chay duoc tren bat ky
# may Linux nao co GPU/CPU + data/ (khong rieng platform nao). B1/B2 (Stage 1) da
# train xong san — script nay CHI can
# checkpoint (results/models/geoformerdock_uncertainty/, .../geoformerdock_nobalance/)
# da co san trong repo, khong can train lai.

set -e

echo "=== 1. Tai data/ (~80GB — chua co tren may nay) ==="
bash scripts/00_download.sh
ls data/types/ref_uff_test0.types

echo "=== 2. Inference cho cac model co checkpoint (tu dong bo qua model thieu) ==="
DEVICE=cuda:0 bash scripts/run_all_inference.sh

echo "=== 3. Inference rieng cho B1/B2 neu checkpoint co mat ==="
if [ -f results/models/geoformerdock_uncertainty/best_model.pt ]; then
    python3 tools/run_inference.py --model geoformerdock --uncertainty \
        --checkpoint results/models/geoformerdock_uncertainty/best_model.pt \
        --testfile data/types/ref_uff_test0.types --data_root data -g cuda:0 \
        --out results/predictions/geoformerdock_uncertainty.csv
else
    echo "[SKIP] khong co results/models/geoformerdock_uncertainty/best_model.pt"
fi
if [ -f results/models/geoformerdock_nobalance/best_model.pt ]; then
    python3 tools/run_inference.py --model geoformerdock \
        --checkpoint results/models/geoformerdock_nobalance/best_model.pt \
        --testfile data/types/ref_uff_test0.types --data_root data -g cuda:0 \
        --out results/predictions/geoformerdock_nobalance.csv
else
    echo "[SKIP] khong co results/models/geoformerdock_nobalance/best_model.pt"
fi

echo "=== 4. A1 — chi so chinh xac cho tung model ==="
for f in results/predictions/*.csv; do
    echo "--- $f ---"
    python3 tools/exact_metrics.py --predictions "$f"
done | tee results/logs/exact_metrics_all.log

echo "=== 5. Paired bootstrap CI (VD4a) — bat buoc: gnina_dense, gnina_default2018, pafnucy ==="
mkdir -p results/logs
: > results/logs/paired_bootstrap_all.log
MANDATORY_BASELINES=(gnina_dense gnina_default2018 pafnucy)
OPTIONAL_BASELINES=(potentialnet tankbind)
MISSING_MANDATORY=0
for baseline in "${MANDATORY_BASELINES[@]}" "${OPTIONAL_BASELINES[@]}"; do
    PRED_B="results/predictions/${baseline}.csv"
    if [ ! -f "${PRED_B}" ]; then
        if [[ " ${MANDATORY_BASELINES[*]} " == *" ${baseline} "* ]]; then
            echo "[LOI] Thieu ${PRED_B} — baseline BAT BUOC theo D1." >&2
            MISSING_MANDATORY=1
        else
            echo "=== geoformerdock vs ${baseline}: [SKIP] (tuy chon) ===" | tee -a results/logs/paired_bootstrap_all.log
        fi
        continue
    fi
    echo "=== geoformerdock vs ${baseline} ===" | tee -a results/logs/paired_bootstrap_all.log
    python3 tools/paired_bootstrap.py \
        --pred_a results/predictions/geoformerdock.csv \
        --pred_b "${PRED_B}" \
        --metric all --n_boot 2000 --seed 2026 \
        | tee -a results/logs/paired_bootstrap_all.log
done
if [ "${MISSING_MANDATORY}" = "1" ]; then
    echo "CANH BAO: thieu baseline bat buoc, xem log o tren." >&2
fi

echo "=== DONE stage 2 ==="
