#!/bin/bash
# Chay qua dem: train lai geoformerdock + 3 baseline chinh voi checkpoint
# selection/early-stopping dua tren VALIDATION (khong con dua tren TEST) - sua VD9.
#
# A2 (docs/revision_plan_reviews.md Muc 3): xep theo SEED-MAJOR, khong phai
# MODEL-MAJOR — moi "luot" chay du 4 mo hinh o 1 seed, de cat ngang o bat ky dau
# van co mot bang hoan chinh (vd het GPU sau luot 1 van dung duoc, khong phai
# cho het ca 3 luot moi co so). Uu tien: geoformerdock truoc trong tung luot
# (quan trong nhat cho luan diem chinh cua bai), roi den gnina_dense,
# gnina_default2018, pafnucy.
#
# Chay lai an toan: neu OUTDIR/summary.json da co (tu lan chay truoc bi ngat),
# BO QUA cau hinh (seed, model) do, khong train lai — cho phep dut giua chung
# roi chay lai script nay ma khong mat viec da xong.
#
# Tu cho GPU trong (khong tranh chap voi nguoi khac dang dung chung may),
# tu giam batch_size (256 -> 128 -> 64) neu OOM, tiep tuc sang model tiep theo
# neu 1 model that bai (khong dung ca chuoi qua dem chi vi 1 model loi).
#
# Bien moi truong tuy chinh:
#   SEEDS="2026 2027 2028"   danh sach seed, cach nhau boi dau cach (mac dinh 3 seed)
#   MODELS="geoformerdock gnina_dense gnina_default2018 pafnucy"  (mac dinh)

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
    local SEED="$2"
    local OUTDIR="results/models/${MODEL}_valsplit_s${SEED}"
    local LOGFILE="results/logs/${MODEL}_valsplit_s${SEED}.log"

    if [ -f "${OUTDIR}/summary.json" ]; then
        echo "=== ${MODEL} seed=${SEED}: DA CO ${OUTDIR}/summary.json — BO QUA (xoa thu muc neu muon chay lai) ==="
        return 0
    fi

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
            --normalize_targets --seed "${SEED}" \
            "${GEO_ARGS[@]}" "${AMP_ARGS[@]}" \
            -g cuda:0 \
            -o "${OUTDIR}" \
            2>&1 | tee "${LOGFILE}"
        then
            if [ -f "${OUTDIR}/summary.json" ]; then
                echo "=== ${MODEL} seed=${SEED}: THANH CONG voi batch_size=${bs} ($(date)) ==="
                return 0
            fi
        fi
        echo "=== ${MODEL} seed=${SEED}: THAT BAI o batch_size=${bs}, thu nho hon ==="
    done
    echo "=== ${MODEL} seed=${SEED}: THAT BAI CA 3 MUC batch_size — BO QUA, sang cau hinh tiep theo ===" >&2
    return 1
}

# SEEDS/MODELS co the ghi de qua bien moi truong (xem chu thich dau file).
read -ra SEED_LIST <<< "${SEEDS:-2026 2027 2028}"
read -ra MODEL_LIST <<< "${MODELS:-geoformerdock gnina_dense gnina_default2018 pafnucy}"

echo "########## BAT DAU OVERNIGHT RUN — sua VD9 (chon model bang validation) ##########"
echo "Bat dau luc: $(date)"
echo "Train (split): ${TRAIN_FILE}"
echo "Val (chon checkpoint/early-stop): ${VAL_FILE}"
echo "Test (CHUA TUNG dung de chon model, chi de bao cao): ${TEST_FILE}"
echo "Seeds (thu tu luot): ${SEED_LIST[*]}"
echo "Models (thu tu trong tung luot): ${MODEL_LIST[*]}"

RESULTS_SUMMARY=()
for SEED in "${SEED_LIST[@]}"; do
    echo ""
    echo "=========================================================="
    echo "LUOT seed=${SEED}  ($(date))"
    echo "=========================================================="
    for MODEL in "${MODEL_LIST[@]}"; do
        echo ""
        echo "############################################################"
        echo "MODEL: ${MODEL}  seed=${SEED}  ($(date))"
        echo "############################################################"
        if wait_for_gpu 15000 240; then
            if train_one "${MODEL}" "${SEED}"; then
                RESULTS_SUMMARY+=("seed=${SEED} ${MODEL}: OK")
            else
                RESULTS_SUMMARY+=("seed=${SEED} ${MODEL}: THAT BAI (het 3 muc batch_size)")
            fi
        else
            RESULTS_SUMMARY+=("seed=${SEED} ${MODEL}: BO QUA (khong cho duoc GPU trong 4h)")
        fi
    done
done

echo ""
echo "########## TOM TAT ##########"
for line in "${RESULTS_SUMMARY[@]}"; do
    echo "  ${line}"
done
echo "Ket thuc luc: $(date)"
echo "########## XONG TOAN BO OVERNIGHT RUN ##########"
