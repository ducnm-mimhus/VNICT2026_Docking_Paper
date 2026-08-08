# Runbook Bàn giao — Chạy trên Data Đầy đủ (GPU + `data/`)

**Mục tiêu:** tài liệu này là **tất cả những gì người được giao việc cần**, để chạy trọn phần
còn lại của kế hoạch (B1, B2, inference, A1, paired bootstrap) mà **không cần hỏi lại** người
đã chuẩn bị. Toàn bộ script trong runbook này đã được viết và kiểm thử end-to-end (bằng
checkpoint thật + `demo_inference`) trước khi bàn giao — chỉ chưa chạy được trên dữ liệu đầy
đủ vì máy chuẩn bị không đủ dung lượng (~80GB).

**Không cần đọc lại toàn bộ lịch sử dự án để làm theo runbook này** — chỉ cần làm đúng theo
thứ tự các bước dưới.

---

## 0. Trước khi bắt đầu — kiểm tra tài nguyên

```
[ ] Máy có GPU (Kaggle Notebook / JetBrains Cadence / máy khác)?
[ ] Máy có ít nhất ~90–100GB dung lượng trống (80GB data + buffer cho checkpoint mới)?
[ ] Có internet ổn định để tải PDBbind2016.tar.gz (~80GB)?
```

Chạy lệnh sau để kiểm tra dung lượng đĩa TRƯỚC KHI tải bất cứ gì:
```bash
df -h .
```
Nếu dưới ~90GB trống, **dừng lại và báo về** trước khi tải — đừng để tải nửa chừng rồi hết chỗ.

---

## 1. Cài môi trường

Đã kiểm chứng cách cài sau chạy được (nhanh hơn hướng dẫn conda-forge trong README —
`pip install molgrid` tải thẳng được từ PyPI, không cần conda-forge):

```bash
# Tạo env riêng, đúng Python 3.10 (khớp requirements.txt)
conda create -n dockbench python=3.10 -y
conda activate dockbench

# numpy PHẢI <2 — molgrid build cho ABI numpy 1.x, numpy 2.x sẽ crash lúc import
pip install "numpy<2" scipy scikit-learn pandas PyYAML tqdm

# Torch — nếu có GPU CUDA, đổi index-url cho đúng bản CUDA của máy
# (xem https://pytorch.org/get-started/locally/); ví dụ CUDA 12.1:
pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121
# Nếu chỉ có CPU (không khuyến khích cho B1/B2 vì rất chậm):
#   pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install pytorch-ignite mlflow molgrid
```

Kiểm tra cài đúng:
```bash
python3 -c "
import torch, molgrid, ignite
print('torch', torch.__version__, 'cuda:', torch.cuda.is_available())
print('molgrid OK')
"
```
Nếu `torch.cuda.is_available()` ra `False` trong khi máy có GPU, kiểm tra lại bản CUDA đã chọn
ở bước cài torch — đây là lỗi hay gặp nhất.

---

## 2. Tải dữ liệu (~80GB, tốn thời gian nhất trong toàn bộ runbook)

```bash
cd VNICT2026_Docking_Paper   # thư mục gốc repo
bash scripts/00_download.sh
```

Sau khi xong, xác nhận:
```bash
ls data/types/ref_uff_train0.types data/types/ref_uff_test0.types
ls data/PDBbind2016/ | head
```

---

## 3. Smoke test — BẮT BUỘC chạy trước, trước khi đụng đến bước 4/5

Việc này xác nhận môi trường đúng (đã kiểm thử pass trên máy chuẩn bị, chỉ ~3 giây, KHÔNG
cần `data/` đầy đủ — dùng `demo_inference/` có sẵn trong repo):

```bash
bash scripts/run_geoformerdock_ablations.sh smoketest
```

Kỳ vọng: in ra `Smoke test PASS`, và:
```bash
python3 -c "import json; d=json.load(open('results/models/geoformerdock_smoketest/summary.json')); print(d['best_epoch'])"
```
phải in ra một số (không phải `None`, không nhất thiết là `3`). Nếu lỗi ở bước này — **dừng
lại, đừng chạy tiếp bước 4/5** — báo lại lỗi cụ thể trước khi tốn GPU cho việc khác.

Sau khi xác nhận pass, dọn thư mục rác này:
```bash
rm -rf results/models/geoformerdock_smoketest results/logs/geoformerdock_smoketest.log
```

---

## 3.5. Tải checkpoint cũ từ Drive + kiểm tra sanity (THAY CHO việc train lại 6 mô hình gốc)

**D0 đã chốt:** dùng lại checkpoint cũ (đã lưu trên Drive), không train lại từ đầu 6 mô hình
gốc. **D1 đã chốt: bỏ 3 baseline đồ thị** (`equibind`, `tankbind`, `potentialnet`) khỏi bảng
kết quả chính (VĐ1) — vì cả ba cho kết quả ≈ ngẫu nhiên do lỗi kỹ thuật không sửa được khi port
sang voxel (tọa độ pseudo-atom không khả vi). Hệ quả: **không cần train `equibind`**.

**Bước 1 — Tải 3 file `best_model.pt` cần cho bảng chính** từ Drive, đặt đúng đường dẫn:
```
results/models/gnina_dense/best_model.pt
results/models/gnina_default2018/best_model.pt
results/models/pafnucy/best_model.pt
```
(Tùy chọn, không bắt buộc: `tankbind/best_model.pt`, `potentialnet/best_model.pt` — chỉ nếu
muốn giữ làm minh chứng phụ lục cho VĐ1, KHÔNG đưa vào bảng chính. `geoformerdock/best_model.pt`
cũng tải nếu muốn có dòng đối chứng trước khi B1/B2 chạy xong.)

⚠️ **KHÔNG tải/dùng `summary.json` đi kèm trên Drive** (nếu có) — số liệu đó được ghi bằng code
cũ có lỗi tracking `best_epoch` (VĐ11), không đáng tin. Chỉ dùng file `.pt`; mọi chỉ số phải
tính lại từ đầu ở bước 5–6 dưới đây.

**Bước 2 — Sanity-check ngay sau khi tải, TRƯỚC khi tin bất kỳ số nào:**
```bash
python3 tools/inspect_checkpoints.py
```
Kỳ vọng: `status: ok` cho **đúng những model đã tải** (3 bắt buộc + bao nhiêu tùy chọn đã tải),
và **cùng một giá trị `pi_implied`** (≈0.135) ở mọi dòng `ok` — đây chính là phép kiểm
`Acc = π·Rpos + (1−π)·Rneg` đã dùng để phát hiện VĐ11 lần đầu. `[MISSING]` cho `equibind` và
cho bất kỳ model tùy chọn nào bạn chưa tải là **bình thường, không phải lỗi** — script quét cả
7 model theo mặc định. Chỉ coi là lỗi nếu `[MISSING]` xuất hiện ở 1 trong 3 model **bắt buộc**
(`gnina_dense, gnina_default2018, pafnucy`), hoặc nếu `pi_implied` lệch nhau giữa các dòng `ok`
— khi đó **dừng lại**, checkpoint tải về có thể bị hỏng hoặc không khớp `ref_uff_test0.types`
hiện tại (ví dụ tải nhầm từ một lần chạy với split khác). Báo lại trước khi chạy tiếp bước 4.

---

## 4. Chạy B1, B2 (ablation — mỗi cái ~1 ngày GPU với epoch=100 mặc định)

```bash
bash scripts/run_geoformerdock_ablations.sh both
```
(hoặc chạy riêng `uncertainty` / `nobalance` nếu muốn theo dõi từng cái).

Script này **không đụng** đến `results/models/geoformerdock/` gốc — ghi ra
`results/models/geoformerdock_uncertainty/` và `results/models/geoformerdock_nobalance/`.

Nếu cần rút ngắn thời gian thử nghiệm trước khi chạy full 100 epoch, có thể test nhanh với:
```bash
EPOCHS=10 bash scripts/run_geoformerdock_ablations.sh uncertainty
```
nhưng **kết quả dùng cho bài báo phải chạy đủ epoch mặc định (100)** để so sánh hợp lệ với các
dòng khác trong bảng.

---

## 5. Chạy inference lấy dự đoán từng mẫu

```bash
bash scripts/run_all_inference.sh
```

Mặc định dùng `data/types/ref_uff_test0.types` + GPU. Nếu máy không có GPU:
```bash
DEVICE=cpu bash scripts/run_all_inference.sh
```

Kết quả: `results/predictions/<model>.csv`. Theo **D1 = bỏ 3 baseline đồ thị**, chỉ 4 model sau
là bắt buộc cho bảng chính: `gnina_dense, gnina_default2018, pafnucy, geoformerdock`.
`potentialnet`, `tankbind` là **tùy chọn** (chỉ chạy nếu muốn giữ làm minh chứng phụ lục cho
VĐ1 — không đưa số của chúng vào Bảng kết quả chính Mục IV). `equibind` bị bỏ hẳn theo D1,
không cần checkpoint, không cần chạy.

**Nếu muốn có dự đoán cho B1/B2** (sau khi bước 4 xong), chạy thêm thủ công:
```bash
python3 tools/run_inference.py --model geoformerdock --uncertainty \
    --checkpoint results/models/geoformerdock_uncertainty/best_model.pt \
    --testfile data/types/ref_uff_test0.types --data_root data \
    --out results/predictions/geoformerdock_uncertainty.csv

python3 tools/run_inference.py --model geoformerdock \
    --checkpoint results/models/geoformerdock_nobalance/best_model.pt \
    --testfile data/types/ref_uff_test0.types --data_root data \
    --out results/predictions/geoformerdock_nobalance.csv
```
⚠️ **`--uncertainty` là bắt buộc** khi nạp checkpoint của B1 — kiến trúc đầu ra ái lực khác
nhau (`UncertaintyHead` vs. `Sequential` thường), thiếu cờ này sẽ lỗi khi nạp trọng số.

---

## 6. A1 — chỉ số chính xác (không lấy mẫu ngẫu nhiên) cho từng model

```bash
for f in results/predictions/*.csv; do
    echo "=== $f ==="
    python3 tools/exact_metrics.py --predictions "$f"
    echo
done | tee results/logs/exact_metrics_all.log
```

Gửi lại file `results/logs/exact_metrics_all.log` — đây là số liệu **thay thế cho C-index lấy
mẫu ngẫu nhiên** hiện có trong `training_metrics_test.csv` (VĐ4b).

---

## 7. Paired bootstrap CI (VĐ4a) — so sánh mô hình đề xuất với từng baseline

`gnina_dense, gnina_default2018, pafnucy` là **bắt buộc** (D1). `potentialnet, tankbind` là
**tùy chọn** — script dưới đây tự bỏ qua nếu thiếu file dự đoán, không dừng cả vòng lặp:

```bash
mkdir -p results/logs
: > results/logs/paired_bootstrap_all.log
MANDATORY_BASELINES=(gnina_dense gnina_default2018 pafnucy)
OPTIONAL_BASELINES=(potentialnet tankbind)
MISSING_MANDATORY=0

for baseline in "${MANDATORY_BASELINES[@]}" "${OPTIONAL_BASELINES[@]}"; do
    PRED_B="results/predictions/${baseline}.csv"
    if [ ! -f "${PRED_B}" ]; then
        if [[ " ${MANDATORY_BASELINES[*]} " == *" ${baseline} "* ]]; then
            echo "[LOI] Thieu ${PRED_B} — day la baseline BAT BUOC theo D1. Kiem tra lai buoc 3.5/5." >&2
            MISSING_MANDATORY=1
        else
            echo "=== geoformerdock vs ${baseline}: [SKIP] khong co ${PRED_B} (baseline tuy chon) ===" \
                | tee -a results/logs/paired_bootstrap_all.log
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
    echo "DUNG LAI: thieu it nhat 1 baseline bat buoc — xem [LOI] o tren." >&2
    exit 1
fi
```

Nếu script in cảnh báo `"hai file co ve KHONG cung hang/cung testfile"` — **dừng lại**, đây là
dấu hiệu bước 5 chạy sai `--testfile` giữa các model (phải cùng một file `ref_uff_test0.types`
cho tất cả). Đối chiếu lại trước khi tin bất kỳ con số nào từ bước này.

---

## 8. A2 — đã chạy sẵn, không cần làm lại trừ khi có B1/B2

Việc này **đã chạy xong trên máy chuẩn bị** (không cần `data/`), kết quả nằm ở
`results/logs/epoch_selection_audit.tsv` (đã có trong repo qua commit). Chỉ cần chạy lại nếu
muốn có thêm 2 dòng B1/B2 sau khi có `training_metrics_train/test.csv` của chúng (bước 4
tự động tạo ra):

```bash
python3 tools/select_epoch_by_train.py
```

---

## 9. Gửi lại cho Đức sau khi xong

```
results/models/geoformerdock_uncertainty/     (toàn bộ thư mục — B1)
results/models/geoformerdock_nobalance/       (toàn bộ thư mục — B2)
results/predictions/*.csv                     (4-8 file tuy da tai baseline tuy chon
                                                chua va B1/B2 xong chua — buoc 5)
results/logs/exact_metrics_all.log            (bước 6)
results/logs/paired_bootstrap_all.log         (bước 7)
results/logs/epoch_selection_audit.tsv        (cập nhật nếu chạy lại ở bước 8)
```

Không cần gửi lại `data/` (quá nặng, Đức không cần bản sao) hay `results/models/<6 model gốc>`
(đã có sẵn).

---

## 10. Nếu gặp lỗi không có trong runbook này

Dán nguyên thông báo lỗi + lệnh đã chạy, gửi lại cho Đức — **không tự sửa code training/model**
(vi phạm ràng buộc "chỉ viết lại, không phát triển thêm" của cả dự án). Các script hỗ trợ
(`tools/*.py`, `scripts/run_geoformerdock_ablations.sh`, `scripts/run_all_inference.sh`) có thể
sửa nếu lỗi rõ ràng là lỗi script (đường dẫn sai, thiếu thư viện) — nhưng không sửa
`dockbench/training.py`, `dockbench/models/`, `dockbench/losses.py` mà không hỏi trước.
