#!/bin/bash
# Track B (docs/revision_plan_reviews.md Muc 4, docs/phan_cong_2_phan.md B-5):
# chay 5 ablation kien truc cua GeoFormerDock (B-1..B-5, Bang IV) DUOI VAL-SPLIT
# — khac voi scripts/run_geoformerdock_ablations.sh (ban cu, ablation
# uncertainty/nobalance, KHONG dung --valfile, chon checkpoint tren test).
#
# LY DO CO FILE RIENG thay vi sua run_overnight_valsplit.sh: file do dung cho
# 4 mo hinh CHINH (Track A, so sanh kien truc), con file nay dung CHO 1 minh
# geoformerdock voi cac gia tri --geo_ablation khac nhau (Track B, do dong gop
# tung mo-dun). Gop chung se lam roi muc dich cua tung script.
#
# Moi sieu tham so (trong TRAIN_FILE/VAL_FILE/TEST_FILE, seed, batch size,
# tat ca cac trong so loss...) duoc SAO CHEP NGUYEN VAN tu train_one() trong
# run_overnight_valsplit.sh cho dong geoformerdock — CHI them dung 1 co
# --geo_ablation, de phep so sanh hop le (chi thay doi 1 bien so tai 1 thoi
# diem). Dung seed=2026 CO DINH (khop lam Luot 1 cua Track A) — theo dung
# yeu cau "B-1..B-5 chay tren val-split, seed 2026, cung batch_size voi luot A".
#
# Cach dung:
#   bash scripts/run_geo_ablations_valsplit.sh no_geometry
#   bash scripts/run_geo_ablations_valsplit.sh no_geometry,no_key_bias   # nhieu ablation, phan cach dau phay
#   bash scripts/run_geo_ablations_valsplit.sh all                       # ca 5, theo dung thu tu uu tien B-1..B-5
#
# Thu tu uu tien khi GPU han che (docs/phan_cong_2_phan.md B-5..B-7):
#   B-1 no_geometry, B-2 <cân bằng lớp — xem scripts/run_geoformerdock_ablations.sh nobalance>,
#   B-3 concat_fusion, B-4 simple_affinity, B-5 no_key_bias.
#   B-2 (bo pose_balance_batch) KHONG chay o day — no la mot co huan luyen rieng
#   (--pose_balance_batch), khong phai --geo_ablation; dung
#   scripts/run_geoformerdock_ablations.sh nobalance nhung PHAI sua thanh co
#   --valfile truoc (hien ban do van chua co val-split — xem ghi chu cuoi file).

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
SEED=2026

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
        # Kiem tra numeric TRUOC khi so sanh: neu nvidia-smi loi, dung de chuoi
        # loi lot vao phep so sanh "-ge" roi treo trong im lang toi 4h (xem
        # ghi chu tuong tu trong run_overnight_valsplit.sh).
        free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
        if [[ "${free_mib}" =~ ^[0-9]+$ ]]; then
            echo "  [$(date '+%H:%M:%S')] GPU free: ${free_mib} MiB (can >= ${need_mib} MiB)"
            if [ "${free_mib}" -ge "${need_mib}" ]; then
                return 0
            fi
        else
            echo "  [$(date '+%H:%M:%S')] KHONG doc duoc so MiB trong tu nvidia-smi (driver GPU co the chua san sang) — thu lai sau"
        fi
        if [ "${waited}" -ge "${max_wait_min}" ]; then
            echo "  Da cho ${max_wait_min} phut, GPU van khong san sang." >&2
            return 1
        fi
        sleep 120
        waited=$((waited + 2))
    done
}

train_ablation() {
    local ABLATION="$1"     # gia tri hop le cua --geo_ablation
    local OUTDIR="results/models/geoformerdock_ablation_${ABLATION}_valsplit_s${SEED}"
    local LOGFILE="results/logs/geoformerdock_ablation_${ABLATION}_valsplit_s${SEED}.log"

    if [ -f "${OUTDIR}/summary.json" ]; then
        echo "=== ablation=${ABLATION}: DA CO ${OUTDIR}/summary.json — BO QUA (xoa thu muc neu muon chay lai) ==="
        return 0
    fi

    for bs in 256 128 64; do
        echo "=== ablation=${ABLATION}: thu batch_size=${bs} ($(date)) ==="
        rm -rf "${OUTDIR}"

        if python -u -m dockbench.training \
            "${TRAIN_FILE}" \
            --testfile "${TEST_FILE}" \
            --valfile "${VAL_FILE}" \
            -d data \
            -m geoformerdock \
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
            --max_pseudo_atoms 12 \
            --geo_ablation "${ABLATION}" \
            --use_amp \
            -g cuda:0 \
            -o "${OUTDIR}" \
            2>&1 | tee "${LOGFILE}"
        then
            if [ -f "${OUTDIR}/summary.json" ]; then
                echo "=== ablation=${ABLATION}: THANH CONG voi batch_size=${bs} ($(date)) ==="
                return 0
            fi
        fi
        echo "=== ablation=${ABLATION}: THAT BAI o batch_size=${bs}, thu nho hon ==="
    done
    echo "=== ablation=${ABLATION}: THAT BAI CA 3 MUC batch_size — BO QUA, sang ablation tiep theo ===" >&2
    return 1
}

ALL_ABLATIONS_IN_PRIORITY_ORDER="no_geometry concat_fusion simple_affinity no_key_bias"

MODE="${1:-}"
if [ -z "${MODE}" ]; then
    echo "Cach dung: bash scripts/run_geo_ablations_valsplit.sh {no_geometry|concat_fusion|no_key_bias|simple_affinity|all|<a,b,c>}"
    exit 1
fi

if [ "${MODE}" = "all" ]; then
    read -ra ABLATION_LIST <<< "${ALL_ABLATIONS_IN_PRIORITY_ORDER}"
else
    IFS=',' read -ra ABLATION_LIST <<< "${MODE}"
fi

# "none" (khong ablation) khong hop le O DAY: do la dong GeoFormerDock day du,
# thuoc Track A (scripts/run_overnight_valsplit.sh), khong phai mot ablation.
VALID="no_geometry concat_fusion no_key_bias simple_affinity"
for a in "${ABLATION_LIST[@]}"; do
    if [[ ! " ${VALID} " =~ " ${a} " ]]; then
        echo "ERROR: '${a}' khong phai gia tri ablation hop le cho script nay (${VALID})." >&2
        echo "  ('none' khong dung o day — dung scripts/run_overnight_valsplit.sh cho dong geoformerdock day du.)" >&2
        exit 1
    fi
done

echo "########## BAT DAU TRACK B — ablation kien truc (val-split, seed=${SEED}) ##########"
echo "Bat dau luc: $(date)"
echo "Ablation se chay: ${ABLATION_LIST[*]}"

RESULTS_SUMMARY=()
for ABLATION in "${ABLATION_LIST[@]}"; do
    echo ""
    echo "############################################################"
    echo "ABLATION: ${ABLATION}  ($(date))"
    echo "############################################################"
    if wait_for_gpu 15000 240; then
        if train_ablation "${ABLATION}"; then
            RESULTS_SUMMARY+=("${ABLATION}: OK")
        else
            RESULTS_SUMMARY+=("${ABLATION}: THAT BAI (het 3 muc batch_size)")
        fi
    else
        RESULTS_SUMMARY+=("${ABLATION}: BO QUA (khong cho duoc GPU trong 4h)")
    fi
done

echo ""
echo "########## TOM TAT ##########"
for line in "${RESULTS_SUMMARY[@]}"; do
    echo "  ${line}"
done
echo "Ket thuc luc: $(date)"
echo "########## XONG TRACK B ##########"
