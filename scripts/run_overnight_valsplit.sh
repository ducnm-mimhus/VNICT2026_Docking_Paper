#!/bin/bash
# Chay qua dem: train lai geoformerdock + 3 baseline chinh voi checkpoint
# selection/early-stopping dua tren VALIDATION (khong con dua tren TEST) - sua VD9.
#
# Uu tien: geoformerdock truoc (quan trong nhat cho luan diem chinh cua bai),
# roi den gnina_dense, gnina_default2018, pafnucy neu con thoi gian/GPU.
#
# Tu cho GPU trong (khong tranh chap voi nguoi khac dang dung chung may),
# tu giam batch_size (256 -> 128 -> 64) neu OOM, tiep tuc sang model tiep theo
# neu 1 model that bai (khong dung ca chuoi qua dem chi vi 1 model loi).

set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"
source /data/miniconda/etc/profile.d/conda.sh
conda activate dockbench
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1

TYPES_DIR="data/types"
TRAIN_FILE="${TYPES_DIR}/ref_uff_train0_split.types"
VAL_FILE="${TYPES_DIR}/ref_uff_val0.types"
TEST_FILE="${TYPES_DIR}/ref_uff_test0.types"

for f in "${TRAIN_FILE}" "${VAL_FILE}" "${TEST_FILE}"; do
    if [ ! -f "${f}" ]; then
        echo "LOI: khong thay ${f} — chay tools/make_val_split.py truoc." >&2
        exit 1
    fi
done

wait_for_gpu() {
    local need_mib="$1"
    local max_wait_min="${2:-240}"
    local waited=0
    while true; do
        local free_mib
        free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
        echo "  [$(date '+%H:%M:%S')] GPU free: ${free_mib} MiB (can >= ${need_mib} MiB)"
        if [ "${free_mib}" -ge "${need_mib}" ] 2>/dev/null; then
            return 0
        fi
        if [ "${waited}" -ge "${max_wait_min}" ]; then
            echo "  Da cho ${max_wait_min} phut, GPU van khong du trong." >&2
            return 1
        fi
        sleep 120
        waited=$((waited + 2))
    done
}

train_one() {
    local MODEL="$1"
    local OUTDIR="results/models/${MODEL}_valsplit"
    local LOGFILE="results/logs/${MODEL}_valsplit.log"

    for bs in 256 128 64; do
        echo "=== ${MODEL}: thu batch_size=${bs} ($(date)) ==="
        rm -rf "${OUTDIR}"

        AMP_ARGS=()
        if [[ "${MODEL}" == "gnina_dense" || "${MODEL}" == "gnina_default2018" \
              || "${MODEL}" == "pafnucy" || "${MODEL}" == "geoformerdock" ]]; then
            AMP_ARGS+=(--use_amp)
        fi
        GEO_ARGS=()
        if [ "${MODEL}" = "geoformerdock" ]; then
            GEO_ARGS+=(--max_pseudo_atoms 12)
        fi

        if python -u -m dockbench.training \
            "${TRAIN_FILE}" \
            --testfile "${TEST_FILE}" \
            --valfile "${VAL_FILE}" \
            -d data \
            -m "${MODEL}" \
            --label_pos 0 --affinity_pos 1 \
            --base_lr 0.001 --weight_decay 0.01 \
            --batch_size "${bs}" \
            --random_translation 1.0 --clip_gradients 5.0 \
            -i 100 \
            --iteration_scheme small \
            --lr_dynamic --warmup_epochs 2 \
            --test_every 2 --checkpoint_every 100 \
            --no_roc_auc \
            --scale_affinity_loss 1.0 --delta_affinity_loss 1.0 \
            --scale_ranking 0.05 --ranking_temperature 1.0 --ranking_num_pairs 128 \
            --hard_neg_fraction 0.3 \
            --rank_warmup_epochs 10 --rank_rampup_epochs 15 \
            --scale_pose_coupling 0.00 --lambda_pose 1.2 \
            --pose_warmup_epochs 0 --pose_only_epochs 4 \
            --pose_loss_type focal --pose_focal_gamma 2.0 --pose_focal_alpha 0.75 \
            --pose_class_normalize --pose_balance_batch --pose_balance_target_per_class 32 \
            --pose_prior_logit_scale 0.25 --disable_pose_prior_init \
            --pose_loss_scale 0.5 --pose_total_weight 0.85 --aff_total_weight 0.15 \
            --metric_ema_alpha 0.3 \
            --early_stop_metric composite_cidx_balacc --early_stop_composite_w_cidx 0.5 \
            --early_stop_patience 25 --early_stop_min_delta 0.0001 \
            --scale_dist_constraint 0.02 --scale_anchor_loss 0.01 \
            --normalize_targets --seed 2026 \
            "${GEO_ARGS[@]}" "${AMP_ARGS[@]}" \
            -g cuda:0 \
            -o "${OUTDIR}" \
            2>&1 | tee "${LOGFILE}"
        then
            if [ -f "${OUTDIR}/summary.json" ]; then
                echo "=== ${MODEL}: THANH CONG voi batch_size=${bs} ($(date)) ==="
                return 0
            fi
        fi
        echo "=== ${MODEL}: THAT BAI o batch_size=${bs}, thu nho hon ==="
    done
    echo "=== ${MODEL}: THAT BAI CA 3 MUC batch_size — BO QUA, sang model tiep theo ===" >&2
    return 1
}

echo "########## BAT DAU OVERNIGHT RUN — sua VD9 (chon model bang validation) ##########"
echo "Bat dau luc: $(date)"
echo "Train (split): ${TRAIN_FILE}"
echo "Val (chon checkpoint/early-stop): ${VAL_FILE}"
echo "Test (CHUA TUNG dung de chon model, chi de bao cao): ${TEST_FILE}"

RESULTS_SUMMARY=()
for MODEL in geoformerdock gnina_dense gnina_default2018 pafnucy; do
    echo ""
    echo "############################################################"
    echo "MODEL: ${MODEL}  ($(date))"
    echo "############################################################"
    if wait_for_gpu 15000 240; then
        if train_one "${MODEL}"; then
            RESULTS_SUMMARY+=("${MODEL}: OK")
        else
            RESULTS_SUMMARY+=("${MODEL}: THAT BAI (het 3 muc batch_size)")
        fi
    else
        RESULTS_SUMMARY+=("${MODEL}: BO QUA (khong cho duoc GPU trong 4h)")
    fi
done

echo ""
echo "########## TOM TAT ##########"
for line in "${RESULTS_SUMMARY[@]}"; do
    echo "  ${line}"
done
echo "Ket thuc luc: $(date)"
echo "########## XONG TOAN BO OVERNIGHT RUN ##########"
