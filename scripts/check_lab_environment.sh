#!/bin/bash
# "Kham suc khoe" may lab GPU truoc khi cham vao training that — dung sau moi
# lan server bi restart/mat ket noi de biet CHINH XAC phai lam lai tu dau hay
# chi can chay tiep, thay vi doan mo.
#
# Chay HET moi kiem tra (khong dung o loi dau tien), in ro [OK]/[CANH BAO]/[LOI]
# cho tung muc, roi tong ket cuoi cung.
#
# Cach dung — chay TRUC TIEP tren may lab (khong chay tren may soan code):
#   cd VNICT2026_Docking_Paper
#   bash scripts/check_lab_environment.sh

set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

PASS=0
WARN=0
FAIL=0

ok()   { echo "  [OK]     $1"; PASS=$((PASS + 1)); }
warn() { echo "  [CANH BAO] $1"; WARN=$((WARN + 1)); }
fail() { echo "  [LOI]    $1"; FAIL=$((FAIL + 1)); }

echo "=================================================="
echo "1. Repo — commit hien tai, dong bo voi remote?"
echo "=================================================="
if [ -d .git ]; then
    LOCAL_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "?")
    LOCAL_MSG=$(git log -1 --format=%s 2>/dev/null || echo "?")
    echo "  Commit hien tai: ${LOCAL_COMMIT} (${LOCAL_MSG})"
    git fetch origin master --quiet 2>/dev/null
    REMOTE_COMMIT=$(git rev-parse --short origin/master 2>/dev/null || echo "?")
    if [ "${LOCAL_COMMIT}" = "${REMOTE_COMMIT}" ]; then
        ok "Dung ban moi nhat tren origin/master (${REMOTE_COMMIT})"
    else
        warn "Khac origin/master (local=${LOCAL_COMMIT}, remote=${REMOTE_COMMIT}) — chay 'git pull'"
    fi
    UNCOMMITTED=$(git status --porcelain | wc -l | tr -d ' ')
    if [ "${UNCOMMITTED}" -gt 0 ]; then
        warn "Co ${UNCOMMITTED} file thay doi chua commit — kiem tra 'git status'"
    else
        ok "Working tree sach"
    fi
else
    fail "Khong tim thay .git — co dang dung thu muc repo khong?"
fi

echo ""
echo "=================================================="
echo "2. Dia trong"
echo "=================================================="
df -h . 2>/dev/null | tail -1
AVAIL_GB=$(df --output=avail -BG . 2>/dev/null | tail -1 | tr -dc '0-9')
if [ -n "${AVAIL_GB}" ] && [ "${AVAIL_GB}" -lt 30 ] 2>/dev/null; then
    warn "Chi con ${AVAIL_GB}GB trong — neu phai tai lai data/ (~11GB file nen, ~80GB+ sau giai nen theo tai lieu du an) co the khong du"
elif [ -n "${AVAIL_GB}" ]; then
    ok "Con ${AVAIL_GB}GB trong"
else
    warn "Khong doc duoc dung luong dia (lenh df khac ban tren he thong nay)"
fi

echo ""
echo "=================================================="
echo "3. Du lieu — data/ con nguyen khong?"
echo "=================================================="
check_types_file() {
    local path="$1" expected_lines="$2"
    if [ -f "${path}" ]; then
        local n
        n=$(wc -l < "${path}" 2>/dev/null | tr -d ' ')
        if [ "${n}" = "${expected_lines}" ]; then
            ok "${path}: ${n} dong (dung nhu ky vong)"
        else
            warn "${path}: ${n} dong (ky vong ${expected_lines} — co the bi cat cut giua chung hoac khac ban)"
        fi
    else
        fail "${path}: KHONG TIM THAY"
    fi
}
check_types_file "data/types/ref_uff_train0.types" 62335
check_types_file "data/types/ref_uff_test0.types" 4618

if [ -d "data/PDBbind2016" ]; then
    N_RECEPTORS=$(find data/PDBbind2016 -maxdepth 1 -mindepth 1 -type d 2>/dev/null | wc -l | tr -d ' ')
    if [ "${N_RECEPTORS}" -gt 100 ] 2>/dev/null; then
        ok "data/PDBbind2016/ ton tai, ${N_RECEPTORS} thu muc con (receptor)"
    else
        warn "data/PDBbind2016/ ton tai nhung chi co ${N_RECEPTORS} thu muc con — co the giai nen chua xong/bi ngat giua chung"
    fi
else
    fail "data/PDBbind2016/ KHONG TIM THAY — can chay lai scripts/00_download.sh"
fi

if [ -f "data/types/ref_uff_train0_split.types" ] && [ -f "data/types/ref_uff_val0.types" ]; then
    ok "Da co san val split (ref_uff_train0_split.types + ref_uff_val0.types) tu lan chay truoc — KHONG can chay lai tools/make_val_split.py"
else
    warn "Chua co val split — se can chay tools/make_val_split.py truoc khi train (xem docs/revision_plan_reviews.md Muc 3, A0)"
fi

echo ""
echo "=================================================="
echo "4. Moi truong Python (conda env 'dockbench')"
echo "=================================================="
# `conda` co the chua co tren PATH trong mot phien SSH khong tuong tac (khong
# nap .bashrc), du conda DA duoc cai — thu nap thang conda.sh o duong dan cac
# script san xuat (run_overnight_valsplit.sh...) da hardcode, truoc khi ket
# luan la thieu.
if ! command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source /data/miniconda/etc/profile.d/conda.sh 2>/dev/null
fi
if command -v conda >/dev/null 2>&1; then
    if conda env list 2>/dev/null | grep -q "^dockbench "; then
        ok "conda env 'dockbench' ton tai"
        # shellcheck disable=SC1091
        source "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" 2>/dev/null
        conda activate dockbench 2>/dev/null
        for pkg in torch molgrid ignite; do
            if python3 -c "import ${pkg}" 2>/dev/null; then
                ok "import ${pkg}: OK"
            else
                fail "import ${pkg}: LOI — moi truong co the da bi hong sau restart, xem lai docs/handoff_runbook.md Muc 1"
            fi
        done
        CUDA_STATUS=$(python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)
        if [ "${CUDA_STATUS}" = "True" ]; then
            ok "torch.cuda.is_available() = True"
        else
            warn "torch.cuda.is_available() = ${CUDA_STATUS:-khong doc duoc} — torch khong thay GPU"
        fi
    else
        fail "conda env 'dockbench' KHONG TON TAI — can cai lai (xem docs/handoff_runbook.md Muc 1)"
    fi
else
    fail "Khong tim thay lenh 'conda'"
fi

echo ""
echo "=================================================="
echo "5. GPU (nvidia-smi)"
echo "=================================================="
if command -v nvidia-smi >/dev/null 2>&1; then
    if nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader
        FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
        ok "nvidia-smi hoat dong, GPU free: ${FREE_MIB} MiB"
    else
        fail "nvidia-smi co nhung chay loi (vd 'khong giao tiep duoc voi driver') — driver GPU chua san sang sau restart"
    fi
else
    fail "Khong tim thay nvidia-smi — GPU driver chua duoc cai/nhan dien tren may nay"
fi

echo ""
echo "=================================================="
echo "6. Checkpoint da co san (tu lan chay truoc / khoa luan goc)"
echo "=================================================="
for m in geoformerdock geoformerdock_nobalance geoformerdock_uncertainty \
         gnina_dense gnina_default2018 pafnucy potentialnet tankbind; do
    if [ -f "results/models/${m}/best_model.pt" ]; then
        ok "results/models/${m}/best_model.pt ton tai"
    else
        warn "results/models/${m}/best_model.pt KHONG co (co the chua tung train, hoac bi mat sau restart)"
    fi
done

echo ""
echo "=================================================="
echo "7. Cac run val-split da lam do (neu co, tu lan chay truoc bi ngat)"
echo "=================================================="
FOUND_ANY=0
for d in results/models/*_valsplit_s*/; do
    [ -d "${d}" ] || continue
    FOUND_ANY=1
    if [ -f "${d}summary.json" ]; then
        echo "  [XONG]     ${d} (co summary.json — se duoc BO QUA neu chay lai run_overnight_valsplit.sh)"
    else
        echo "  [DO DANG]  ${d} (CHUA co summary.json — se bi XOA va chay lai neu goi lai script)"
    fi
done
if [ "${FOUND_ANY}" -eq 0 ]; then
    echo "  (chua co run val-split nao — se bat dau tu lam dau, dung nhu ke hoach)"
fi

echo ""
echo "=================================================="
echo "TOM TAT: ${PASS} OK, ${WARN} CANH BAO, ${FAIL} LOI"
echo "=================================================="
if [ "${FAIL}" -gt 0 ]; then
    echo "=> CO LOI NGHIEM TRONG — xu ly cac dong [LOI] o tren TRUOC KHI chay training that."
    exit 1
elif [ "${WARN}" -gt 0 ]; then
    echo "=> Chay duoc nhung co diem can luu y — doc cac dong [CANH BAO] o tren truoc khi tiep tuc."
    exit 0
else
    echo "=> MOI THU BINH THUONG, an toan de chay training that ngay."
    exit 0
fi
