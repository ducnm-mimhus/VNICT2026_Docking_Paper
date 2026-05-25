#!/bin/bash
# code_docking — 7-model benchmark (dockbench / GeoFormerDock + baselines)
# Prerequisites: bash scripts/00_download.sh

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLS_DIR="${PROJECT_DIR}/tools"
RESULTS_DIR="${PROJECT_DIR}/results"
LOGS_DIR="${RESULTS_DIR}/logs"
MODELS_DIR="${RESULTS_DIR}/models"
PLOTS_DIR="${RESULTS_DIR}/plots"
DATA_REPORT_FILE="${LOGS_DIR}/dataset_report.log"

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"
export GIT_PYTHON_REFRESH=quiet
export PYTHONWARNINGS="ignore::DeprecationWarning"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1

SHOW_PROGRESS_BAR="${SHOW_PROGRESS_BAR:-0}"
mkdir -p "${LOGS_DIR}" "${MODELS_DIR}" "${PLOTS_DIR}"

ONLY_MODEL="${ONLY_MODEL:-all}"
export ONLY_MODEL

DATA_ROOT="${PROJECT_DIR}/data"
TYPES_DIR="${PROJECT_DIR}/data/types"
TRAIN_FILE="${TYPES_DIR}/ref_uff_train0.types"
TEST_FILE="${TYPES_DIR}/ref_uff_test0.types"

if [ ! -f "${TRAIN_FILE}" ]; then
    echo "ERROR: Training data not found: ${TRAIN_FILE}"
    echo "  bash scripts/00_download.sh"
    exit 1
fi

# ---- Shared training parameters ----
EPOCHS="${EPOCHS:-100}"
LR="${LR:-0.001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
TEST_EVERY="${TEST_EVERY:-2}"
RANKING_NUM_PAIRS="${RANKING_NUM_PAIRS:-128}"
HARD_NEG_FRACTION="${HARD_NEG_FRACTION:-0.3}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
GEOFORMER_MAX_PSEUDO_ATOMS="${GEOFORMER_MAX_PSEUDO_ATOMS:-12}"
SCALE_RANKING="${SCALE_RANKING:-0.05}"

ALLOW_POSE_OVERRIDES="${ALLOW_POSE_OVERRIDES:-0}"
if [ "${ALLOW_POSE_OVERRIDES}" = "1" ]; then
    SCALE_POSE_COUPLING="${SCALE_POSE_COUPLING:-0.00}"
    SCALE_POSE_LOSS="${SCALE_POSE_LOSS:-1.2}"
    POSE_LOSS_SCALE="${POSE_LOSS_SCALE:-0.5}"
    POSE_TOTAL_WEIGHT="${POSE_TOTAL_WEIGHT:-0.85}"
    AFF_TOTAL_WEIGHT="${AFF_TOTAL_WEIGHT:-0.15}"
    POSE_WARMUP_EPOCHS="${POSE_WARMUP_EPOCHS:-0}"
    POSE_ONLY_EPOCHS="${POSE_ONLY_EPOCHS:-4}"
else
    SCALE_POSE_COUPLING="0.00"
    SCALE_POSE_LOSS="1.2"
    POSE_LOSS_SCALE="0.5"
    POSE_TOTAL_WEIGHT="0.85"
    AFF_TOTAL_WEIGHT="0.15"
    POSE_WARMUP_EPOCHS="0"
    POSE_ONLY_EPOCHS="4"
fi

USE_SAM_FOR_GEO="${USE_SAM_FOR_GEO:-0}"
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
FULL_DATASET_EPOCH="${FULL_DATASET_EPOCH:-1}"
ITERATION_SCHEME="${ITERATION_SCHEME:-small}"

echo "=========================================="
echo "code_docking — DockBench 7-model benchmark"
echo "=========================================="
if [ -n "${ONLY_MODEL}" ] && [ "${ONLY_MODEL}" != "all" ] && [ "${ONLY_MODEL}" != "benchmark" ] && [ "${ONLY_MODEL}" != "full" ]; then
    echo "Single model: ONLY_MODEL=${ONLY_MODEL}"
else
    echo "Training all 7 benchmark models"
fi
echo "Train: ${TRAIN_FILE}"
echo "Test:  ${TEST_FILE}"
echo "Epochs: ${EPOCHS} | Batch: ${BATCH_SIZE} | LR: ${LR} | Seed: ${SEED}"
echo "Outputs:"
echo "  models/  ${MODELS_DIR}/<model>/"
echo "  logs/    ${LOGS_DIR}/<model>.log"
echo "  plots/   ${PLOTS_DIR}/"
echo "=========================================="

echo ""
echo "[0] Dataset report..."
python3 "${TOOLS_DIR}/dataset_report.py" \
    --project-dir "${PROJECT_DIR}" \
    --data-root "${DATA_ROOT}" \
    --train-file "${TRAIN_FILE}" \
    --test-file "${TEST_FILE}" \
    --batch-size "${BATCH_SIZE}" \
    2>&1 | tee "${DATA_REPORT_FILE}"
echo "  Saved: ${DATA_REPORT_FILE}"

train_model() {
    local MODEL_NAME=$1
    local DISPLAY_NAME=$2
    local IDX=$3
    local TOTAL=$4
    shift 4
    local EXTRA_ARGS=("$@")

    local OUTDIR="${MODELS_DIR}/${MODEL_NAME}"
    local LOGFILE="${LOGS_DIR}/${MODEL_NAME}.log"
    local MODEL_BATCH_SIZE="${BATCH_SIZE}"
    local MODEL_LR="${LR}"
    local SAM_ARGS=()
    local GEO_EXTRA_ARGS=()
    local AMP_ARGS=()

    if [ "${MODEL_NAME}" = "gnina_dense" ] || [ "${MODEL_NAME}" = "gnina_default2018" ] \
       || [ "${MODEL_NAME}" = "pafnucy" ] || [ "${MODEL_NAME}" = "geoformerdock" ]; then
        AMP_ARGS+=(--use_amp)
    fi
    if [ "${MODEL_NAME}" = "geoformerdock" ]; then
        GEO_EXTRA_ARGS+=(--max_pseudo_atoms "${GEOFORMER_MAX_PSEUDO_ATOMS}")
        if [ "${USE_SAM_FOR_GEO}" = "1" ]; then
            SAM_ARGS+=(--use_sam)
        fi
    fi

    echo ""
    echo "[${IDX}/${TOTAL}] ${DISPLAY_NAME}"
    echo "  out: ${OUTDIR}"
    echo "  log: ${LOGFILE}"
    echo "  tail -f \"${LOGFILE}\""

    STRATIFY_ARGS=()
    if [ "${FULL_DATASET_EPOCH}" = "0" ]; then
        STRATIFY_ARGS+=(--stratify_receptor)
    fi

    python -u -m dockbench.training \
        "${TRAIN_FILE}" \
        --testfile "${TEST_FILE}" \
        -d "${DATA_ROOT}" \
        -m "${MODEL_NAME}" \
        --label_pos 0 \
        --affinity_pos 1 \
        --base_lr "${MODEL_LR}" \
        --weight_decay "${WEIGHT_DECAY}" \
        --batch_size "${MODEL_BATCH_SIZE}" \
        --random_translation 1.0 \
        --clip_gradients 5.0 \
        -i "${EPOCHS}" \
        "${STRATIFY_ARGS[@]}" \
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
        --pose_balance_batch \
        --pose_balance_target_per_class "${POSE_BALANCE_TARGET_PER_CLASS}" \
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
        "${GEO_EXTRA_ARGS[@]}" \
        "${SAM_ARGS[@]}" \
        "${AMP_ARGS[@]}" \
        "${EXTRA_ARGS[@]}" \
        -o "${OUTDIR}" \
        2>&1 | tee "${LOGFILE}"

    if [ -f "${OUTDIR}/summary.json" ]; then
        echo "  Done."
    else
        echo "  WARNING: missing summary.json — last 20 log lines:"
        tail -20 "${LOGFILE}" 2>/dev/null || true
    fi
}

run_single() {
    local key=$1
    case "${key}" in
        geoformerdock)
            train_model geoformerdock "GeoFormerDock (Ours)" 1 1 ;;
        gnina_dense|default2017)
            train_model gnina_dense "GNINA Dense" 1 1 ;;
        gnina_default2018|default2018)
            train_model gnina_default2018 "GNINA Default2018" 1 1 ;;
        pafnucy)
            train_model pafnucy "Pafnucy" 1 1 ;;
        potentialnet)
            train_model potentialnet "PotentialNet" 1 1 ;;
        equibind)
            train_model equibind "EquiBind" 1 1 ;;
        tankbind)
            train_model tankbind "TankBind" 1 1 ;;
        *)
            echo "ERROR: ONLY_MODEL='${key}' is not supported."
            echo "Use: gnina_dense | gnina_default2018 | pafnucy | potentialnet | equibind | tankbind | geoformerdock"
            echo "Aliases: default2017, default2018. Unset ONLY_MODEL for all 7 models."
            exit 1
            ;;
    esac
}

if [ -n "${ONLY_MODEL}" ] && [ "${ONLY_MODEL}" != "all" ] && [ "${ONLY_MODEL}" != "benchmark" ] && [ "${ONLY_MODEL}" != "full" ]; then
    run_single "${ONLY_MODEL}"
else
    train_model gnina_dense       "GNINA Dense"           1 7
    train_model gnina_default2018 "GNINA Default2018"     2 7
    train_model pafnucy           "Pafnucy"               3 7
    train_model potentialnet      "PotentialNet"          4 7
    train_model equibind          "EquiBind"              5 7
    train_model tankbind          "TankBind"              6 7
    train_model geoformerdock     "GeoFormerDock (Ours)"  7 7
fi

echo ""
echo "=========================================="
echo "Benchmark summary"
echo "=========================================="
PLOT_ONLY_MODEL="${ONLY_MODEL}"
if [ "${PLOT_ONLY_MODEL}" = "all" ] || [ "${PLOT_ONLY_MODEL}" = "benchmark" ] || [ "${PLOT_ONLY_MODEL}" = "full" ]; then
    PLOT_ONLY_MODEL=""
fi

python3 "${TOOLS_DIR}/summarize_benchmark.py" \
    --models_dir "${MODELS_DIR}" \
    --logs_dir "${LOGS_DIR}" \
    --only_model "${PLOT_ONLY_MODEL}"

echo ""
echo "Plotting training curves..."
python3 "${TOOLS_DIR}/plot_training_curves.py" \
    --models_dir "${MODELS_DIR}" \
    --plots_dir "${PLOTS_DIR}" \
    --only_model "${PLOT_ONLY_MODEL}" \
    --also_geo_only

echo ""
echo "Done."
echo "  models: ${MODELS_DIR}/"
echo "  logs:   ${LOGS_DIR}/"
echo "  plots:  ${PLOTS_DIR}/"
echo "=========================================="
