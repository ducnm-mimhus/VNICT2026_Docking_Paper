#!/bin/bash
# Chay B1 (uncertainty head) va/hoac B2 (bo pose_balance_batch) cho GeoFormerDock.
#
# LY DO CO FILE RIENG THAY VI SUA run_training.sh:
#   run_training.sh dat OUTDIR="${MODELS_DIR}/${MODEL_NAME}" (vd: results/models/geoformerdock).
#   Neu goi lai train_model geoformerdock voi co khac, no se GHI DE thang len
#   best_model.pt / summary.json GOC dang dung cho VD11 va lam bang ket qua chinh.
#   Script nay dung thu muc output RIENG cho tung ablation, khong dung lai
#   run_training.sh, nen khong co rui ro ghi de.
#
# Moi sieu tham so khac (lr, batch, epoch, seed, cac trong so loss...) duoc
# sao chep NGUYEN VAN tu train_model() trong run_training.sh cho dong
# geoformerdock, chi doi DUNG MOT co dang ablation — de phep so sanh hop le
# (chi thay doi 1 bien so).
#
# Cach dung:
#   bash scripts/run_geoformerdock_ablations.sh uncertainty
#   bash scripts/run_geoformerdock_ablations.sh nobalance
#   bash scripts/run_geoformerdock_ablations.sh both      # chay tuan tu ca hai
#
# Truoc khi chay B1/B2 tren data that, nen smoke-test truoc bang demo_inference:
#   bash scripts/run_geoformerdock_ablations.sh smoketest
# (xem huong dan chi tiet o cuoi file nay)

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${PROJECT_DIR}/results"
LOGS_DIR="${RESULTS_DIR}/logs"
MODELS_DIR="${RESULTS_DIR}/models"

# cd vao PROJECT_DIR (khong chi export PYTHONPATH): `python -m dockbench.training`
# tu dua CWD vao dau sys.path, nen day la lop phong ve thu 2 doc lap voi
# PYTHONPATH — tranh loi "ModuleNotFoundError: No module named 'dockbench'"
# neu PYTHONPATH bi mot co che nao khac (vd cau hinh may lab dung chung) ghi de.
cd "${PROJECT_DIR}"

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export GIT_PYTHON_REFRESH=quiet
export PYTHONWARNINGS="ignore::DeprecationWarning"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1

mkdir -p "${LOGS_DIR}" "${MODELS_DIR}"

DATA_ROOT="${DATA_ROOT:-${PROJECT_DIR}/data}"
TRAIN_FILE="${TRAIN_FILE:-${DATA_ROOT}/types/ref_uff_train0.types}"
TEST_FILE="${TEST_FILE:-${DATA_ROOT}/types/ref_uff_test0.types}"

# ---- Cac gia tri MAC DINH giong het run_training.sh cho dong geoformerdock ----
EPOCHS="${EPOCHS:-100}"
LR="${LR:-0.001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
TEST_EVERY="${TEST_EVERY:-2}"
RANKING_NUM_PAIRS="${RANKING_NUM_PAIRS:-128}"
HARD_NEG_FRACTION="${HARD_NEG_FRACTION:-0.3}"
# 1024 la mac dinh cua run_training.sh, nhung DA XAC NHAN OOM tren GPU 44.39GB vRAM
# thuc (l40s) — grid voxel 48^3 qua nang. Mac dinh o day giam xuong 256; neu chay
# tren GPU lon hon (>=80GB thuc) co the nang len 1024 de khop chinh xac baseline.
BATCH_SIZE="${BATCH_SIZE:-256}"
GEOFORMER_MAX_PSEUDO_ATOMS="${GEOFORMER_MAX_PSEUDO_ATOMS:-12}"
SCALE_RANKING="${SCALE_RANKING:-0.05}"

SCALE_POSE_COUPLING="0.00"
SCALE_POSE_LOSS="1.2"
POSE_LOSS_SCALE="0.5"
POSE_TOTAL_WEIGHT="0.85"
AFF_TOTAL_WEIGHT="0.15"
POSE_WARMUP_EPOCHS="0"
POSE_ONLY_EPOCHS="4"

POSE_FOCAL_GAMMA="${POSE_FOCAL_GAMMA:-2.0}"
POSE_FOCAL_ALPHA="${POSE_FOCAL_ALPHA:-0.75}"
POSE_BALANCE_TARGET_PER_CLASS="${POSE_BALANCE_TARGET_PER_CLASS:-32}"
POSE_PRIOR_LOGIT_SCALE="${POSE_PRIOR_LOGIT_SCALE:-0.25}"
DISABLE_POSE_PRIOR_INIT="${DISABLE_POSE_PRIOR_INIT:-1}"
LR_WARMUP_EPOCHS="${LR_WARMUP_EPOCHS:-2}"
EARLY_STOP_METRIC="${EARLY_STOP_METRIC:-composite_cidx_balacc}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-25}"
EARLY_STOP_MIN_DELTA="${EARLY_STOP_MIN_DELTA:-0.0001}"
EARLY_STOP_COMPOSITE_W_CIDX="${EARLY_STOP_COMPOSITE_W_CIDX:-0.5}"
SEED="${SEED:-2026}"
ITERATION_SCHEME="${ITERATION_SCHEME:-small}"

run_ablation() {
    local ABLATION_NAME="$1"     # uncertainty | nobalance
    local OUTDIR="${MODELS_DIR}/geoformerdock_${ABLATION_NAME}"
    local LOGFILE="${LOGS_DIR}/geoformerdock_${ABLATION_NAME}.log"

    if [ -d "${OUTDIR}" ] && [ -f "${OUTDIR}/summary.json" ]; then
        echo "  [SKIP] ${OUTDIR}/summary.json da ton tai. Xoa thu muc neu muon chay lai."
        return 0
    fi

    local EXTRA_ARGS=()
    local BALANCE_ARGS=()

    case "${ABLATION_NAME}" in
        uncertainty)
            EXTRA_ARGS+=(--geoformer_uncertainty)
            BALANCE_ARGS+=(--pose_balance_batch --pose_balance_target_per_class "${POSE_BALANCE_TARGET_PER_CLASS}")
            ;;
        nobalance)
            # KHONG them --pose_balance_batch: day chinh la ablation B2.
            BALANCE_ARGS=()
            ;;
        *)
            echo "ERROR: ablation khong hop le: ${ABLATION_NAME} (dung: uncertainty|nobalance)"
            exit 1
            ;;
    esac

    echo ""
    echo "=========================================="
    echo "Ablation: geoformerdock_${ABLATION_NAME}"
    echo "  out: ${OUTDIR}"
    echo "  log: ${LOGFILE}"
    echo "=========================================="

    python -u -m dockbench.training \
        "${TRAIN_FILE}" \
        --testfile "${TEST_FILE}" \
        -d "${DATA_ROOT}" \
        -m geoformerdock \
        --label_pos 0 \
        --affinity_pos 1 \
        --base_lr "${LR}" \
        --weight_decay "${WEIGHT_DECAY}" \
        --batch_size "${BATCH_SIZE}" \
        --random_translation 1.0 \
        --clip_gradients 5.0 \
        -i "${EPOCHS}" \
        --iteration_scheme "${ITERATION_SCHEME}" \
        --lr_dynamic \
        --warmup_epochs "${LR_WARMUP_EPOCHS}" \
        --test_every "${TEST_EVERY}" \
        --checkpoint_every "${EPOCHS}" \
        --no_roc_auc \
        --scale_affinity_loss 1.0 \
        --delta_affinity_loss 1.0 \
        --scale_ranking "${SCALE_RANKING}" \
        --ranking_temperature 1.0 \
        --ranking_num_pairs "${RANKING_NUM_PAIRS}" \
        --hard_neg_fraction "${HARD_NEG_FRACTION}" \
        --rank_warmup_epochs 10 \
        --rank_rampup_epochs 15 \
        --scale_pose_coupling "${SCALE_POSE_COUPLING}" \
        --lambda_pose "${SCALE_POSE_LOSS}" \
        --pose_warmup_epochs "${POSE_WARMUP_EPOCHS}" \
        --pose_only_epochs "${POSE_ONLY_EPOCHS}" \
        --pose_loss_type focal \
        --pose_focal_gamma "${POSE_FOCAL_GAMMA}" \
        --pose_focal_alpha "${POSE_FOCAL_ALPHA}" \
        --pose_class_normalize \
        "${BALANCE_ARGS[@]}" \
        --pose_prior_logit_scale "${POSE_PRIOR_LOGIT_SCALE}" \
        $( [ "${DISABLE_POSE_PRIOR_INIT}" = "1" ] && echo "--disable_pose_prior_init" ) \
        --pose_loss_scale "${POSE_LOSS_SCALE}" \
        --pose_total_weight "${POSE_TOTAL_WEIGHT}" \
        --aff_total_weight "${AFF_TOTAL_WEIGHT}" \
        --metric_ema_alpha 0.3 \
        --early_stop_metric "${EARLY_STOP_METRIC}" \
        --early_stop_composite_w_cidx "${EARLY_STOP_COMPOSITE_W_CIDX}" \
        --early_stop_patience "${EARLY_STOP_PATIENCE}" \
        --early_stop_min_delta "${EARLY_STOP_MIN_DELTA}" \
        --scale_dist_constraint 0.02 \
        --scale_anchor_loss 0.01 \
        --normalize_targets \
        --seed "${SEED}" \
        --max_pseudo_atoms "${GEOFORMER_MAX_PSEUDO_ATOMS}" \
        --use_amp \
        "${EXTRA_ARGS[@]}" \
        -o "${OUTDIR}" \
        2>&1 | tee "${LOGFILE}"

    if [ -f "${OUTDIR}/summary.json" ]; then
        echo "  Done: ${OUTDIR}/summary.json"
    else
        echo "  WARNING: khong thay summary.json — 20 dong log cuoi:"
        tail -20 "${LOGFILE}" 2>/dev/null || true
    fi
}

run_smoketest() {
    local DEMO_ROOT="${PROJECT_DIR}/demo_inference"
    local DEMO_TYPES="${DEMO_ROOT}/types/demo.types"
    local OUTDIR="${MODELS_DIR}/geoformerdock_smoketest"
    local LOGFILE="${LOGS_DIR}/geoformerdock_smoketest.log"

    if [ ! -f "${DEMO_TYPES}" ]; then
        echo "ERROR: khong thay ${DEMO_TYPES}"
        exit 1
    fi

    echo "=========================================="
    echo "SMOKE TEST — demo_inference (2 mau, CHI kiem tra pipeline co chay khong)"
    echo "  KHONG dung ket qua nay de danh gia mo hinh: ca 2 mau deu label=1,"
    echo "  cac chi so phan loai (BalAcc, PR-AUC, Recall Neg...) se suy bien."
    echo "=========================================="

    python -u -m dockbench.training \
        "${DEMO_TYPES}" \
        --testfile "${DEMO_TYPES}" \
        -d "${DEMO_ROOT}" \
        -m geoformerdock \
        --label_pos 0 \
        --affinity_pos 1 \
        --rmsd_pos 2 \
        --batch_size 2 \
        --no_random_rotation \
        --random_translation 0.0 \
        -i 3 \
        --test_every 1 \
        --checkpoint_every 3 \
        --no_roc_auc \
        --max_pseudo_atoms 12 \
        --disable_pose_prior_init \
        --seed 2026 \
        -g "${SMOKETEST_DEVICE:-cpu}" \
        -o "${OUTDIR}" \
        2>&1 | tee "${LOGFILE}"

    if [ -f "${OUTDIR}/summary.json" ]; then
        echo ""
        echo "  Smoke test PASS: pipeline chay het 3 epoch, checkpoint + summary.json duoc tao."
        echo "  Kiem tra thu:"
        echo "    python3 -c \"import json; d=json.load(open('${OUTDIR}/summary.json')); print(d.get('best_epoch'))\""
        echo "  Neu best_epoch khong phai None va khong nhat thiet =3 (epoch cuoi), fix nonlocal hoat dong dung."
    else
        echo "  Smoke test FAIL — xem log:"
        tail -40 "${LOGFILE}" 2>/dev/null || true
        exit 1
    fi
}

MODE="${1:-}"
case "${MODE}" in
    uncertainty)
        run_ablation uncertainty
        ;;
    nobalance)
        run_ablation nobalance
        ;;
    both)
        run_ablation uncertainty
        run_ablation nobalance
        ;;
    smoketest)
        run_smoketest
        ;;
    *)
        echo "Cach dung: bash scripts/run_geoformerdock_ablations.sh {uncertainty|nobalance|both|smoketest}"
        echo ""
        echo "  smoketest    — chay 3 epoch tren demo_inference (2 mau), KIEM TRA moi truong"
        echo "                 (molgrid, pipeline) truoc khi tai du lieu day du. Khong danh gia model."
        echo "  uncertainty  — B1: bat --geoformer_uncertainty, output rieng, KHONG dung data/ that."
        echo "  nobalance    — B2: bo --pose_balance_batch, output rieng."
        echo "  both         — chay uncertainty roi nobalance, tuan tu."
        echo ""
        echo "Yeu cau: data/ (hoac DATA_ROOT tro toi noi khac) va GPU cho uncertainty/nobalance."
        echo "smoketest chi can demo_inference/ co san trong repo + molgrid cai duoc."
        exit 1
        ;;
esac
