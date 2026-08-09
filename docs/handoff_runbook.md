# Runbook Bàn giao — Chạy trên Data Đầy đủ (GPU + `data/`)

**Mục tiêu:** tài liệu này là **tất cả những gì người được giao việc cần**, để chạy trọn phần
còn lại của kế hoạch (inference, A1, paired bootstrap) mà **không cần hỏi lại** người đã
chuẩn bị. Toàn bộ script trong runbook này đã được viết và kiểm thử end-to-end.

**Không cần đọc lại toàn bộ lịch sử dự án để làm theo runbook này** — chỉ cần làm đúng theo
thứ tự các bước dưới.

## ✅ TÌNH TRẠNG HIỆN TẠI — chỉ còn Bước 5, 6, 7 phải chạy

**Train (Bước 4) đã xong hết cho TẤT CẢ model** — không chỉ B1/B2, mà cả 3 baseline bắt buộc
(`gnina_dense`, `gnina_default2018`, `pafnucy`) và 2 baseline tùy chọn (`potentialnet`,
`tankbind`) đều đã có checkpoint sẵn từ trước (huấn luyện hồi làm khóa luận). B1/B2 mới train
thêm gần đây trên JetBrains Cadence (l40s, batch_size=256), log sạch (không NaN/lỗi/OOM),
`best_epoch` đúng (62 và 42) sau khi đã sửa lỗi `nonlocal` trong `dockbench/training.py`.

**→ Bỏ hẳn Bước 3.5 và Bước 4 trong runbook này — KHÔNG cần tải gì từ Drive, KHÔNG cần train
gì cả.** Chỉ cần:
1. Giải nén file `handoff_all_checkpoints.zip` (gửi kèm, ~153MB) vào đúng vị trí — file zip đã
   giữ nguyên cấu trúc thư mục, chỉ cần giải nén ngay tại thư mục gốc repo:
   ```
   results/models/gnina_dense/best_model.pt (+ summary.json)
   results/models/gnina_default2018/best_model.pt (+ summary.json)
   results/models/pafnucy/best_model.pt (+ summary.json)
   results/models/geoformerdock/best_model.pt (+ summary.json)
   results/models/geoformerdock_uncertainty/best_model.pt (+ summary.json)   ← B1
   results/models/geoformerdock_nobalance/best_model.pt (+ summary.json)    ← B2
   results/models/potentialnet/best_model.pt (+ summary.json)   (tùy chọn, phụ lục)
   results/models/tankbind/best_model.pt (+ summary.json)       (tùy chọn, phụ lục)
   ```
2. Làm **Bước 1 → 2 → 3** như bình thường (cài env, tải ~80GB data, smoke test) — máy nhận
   việc là máy khác, chưa có gì, vẫn cần làm đủ 3 bước này để chắc môi trường đúng.
3. **Bỏ qua Bước 3.5 và Bước 4** — đi thẳng từ Bước 3 → 5.
4. Bước 5 (inference) sẽ tự chạy đủ cho mọi model vì checkpoint đã có sẵn.
5. Bước 8 (A2) đã được sửa để tự quét luôn `geoformerdock_uncertainty`/`geoformerdock_nobalance`
   (trước đó có lỗi nhỏ: script chỉ quét 7 model chuẩn, bỏ sót 2 ablation — đã fix trong
   `tools/select_epoch_by_train.py`). Kết quả đã có sẵn ở `results/logs/epoch_selection_audit.tsv`
   (đã cập nhật, có đủ 9 dòng kể cả 2 ablation, đã gửi kèm trong repo) — không cần chạy lại trừ
   khi muốn double-check.

**Việc CHƯA làm** (đây là toàn bộ phần còn lại, ~1-1.5h không tính thời gian tải data):
Bước 5 (inference — 6-8 model), Bước 6 (A1), Bước 7 (paired bootstrap).

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

## 3.5. Giải nén checkpoint (đã gửi kèm) + kiểm tra sanity — KHÔNG cần Drive

**D0 đã chốt:** dùng lại checkpoint cũ, không train lại từ đầu 6 mô hình gốc. **D1 đã chốt: bỏ
3 baseline đồ thị** (`equibind`, `tankbind`, `potentialnet`) khỏi bảng kết quả chính (VĐ1) — vì
cả ba cho kết quả ≈ ngẫu nhiên do lỗi kỹ thuật không sửa được khi port sang voxel (tọa độ
pseudo-atom không khả vi). Hệ quả: **không cần checkpoint `equibind`** (và thực tế cũng không
train được — xem `results/logs/best_checkpoint_audit.tsv`, dòng `equibind` = `missing_checkpoint`).

**Bước 1 — Giải nén `handoff_all_checkpoints.zip`** (đã gửi kèm cùng repo, ~153MB) ngay tại thư
mục gốc repo — file zip giữ đúng cấu trúc `results/models/<model>/...`, tự đặt đúng chỗ:
```bash
unzip -o handoff_all_checkpoints.zip
```
Đủ cả 8 model rồi (3 baseline bắt buộc + `geoformerdock` gốc + B1 + B2 + 2 baseline tùy chọn) —
**không cần tải gì thêm từ Drive**.

⚠️ **KHÔNG dùng `summary.json` đi kèm để trích số liệu cuối cùng** — những file này (trừ 2 file
của B1/B2, đã được tính lại đúng) được ghi bằng code cũ có lỗi tracking `best_epoch` (VĐ11),
không đáng tin. Chỉ dùng để tham khảo nhanh; mọi chỉ số dùng cho bài báo phải tính lại từ đầu ở
Bước 5–6 dưới đây.

**Bước 2 — Sanity-check ngay sau khi giải nén, TRƯỚC khi tin bất kỳ số nào:**
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

## 4. Chạy B1, B2 (ablation) — ✅ ĐÃ XONG, BỎ QUA BƯỚC NÀY

**Đã chạy xong trên Cadence — xem banner "TÌNH TRẠNG HIỆN TẠI" ở đầu file.** Checkpoint B1/B2
đã nằm trong `handoff_all_checkpoints.zip` giải nén ở Bước 3.5 — đi thẳng xuống Bước 5.

Giữ lại phần dưới đây **chỉ để tham khảo** — nếu vì lý do nào đó cần train lại (ví dụ file zip
bị hỏng), đây là lệnh gốc (thực tế đo được ~2.9h/B1 + ~2.6h/B2 ở `batch_size=256` trên l40s,
KHÔNG phải "~1 ngày" như ước tính ban đầu dưới đây — ước tính cũ quá cao):

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

## 8. A2 — đã chạy sẵn kể cả B1/B2, không cần làm lại

Việc này **đã chạy xong**, kết quả nằm ở `results/logs/epoch_selection_audit.tsv` (đã có trong
repo qua bản gửi kèm) — đủ cả 9 dòng, kể cả `geoformerdock_uncertainty`/`geoformerdock_nobalance`
(script `tools/select_epoch_by_train.py` đã fix để tự quét thêm 2 model này). Chỉ cần chạy lại
nếu bạn nghi ngờ kết quả hoặc muốn double-check:

```bash
python3 tools/select_epoch_by_train.py
```

---

## 9. Gửi lại cho Đức sau khi xong

Đức đã có sẵn mọi checkpoint (gửi kèm ở Bước 3.5) — **không cần gửi lại `results/models/`**.
Chỉ cần gửi 3 thứ MỚI mà runbook này tạo ra:
```
results/predictions/*.csv                     (6-8 file tùy đã chạy baseline tùy chọn
                                                chưa — bước 5)
results/logs/exact_metrics_all.log            (bước 6)
results/logs/paired_bootstrap_all.log         (bước 7)
```
(Bỏ luôn nếu bước 8 không chạy lại — file đó Đức đã có, không đổi.)

Không cần gửi lại `data/` (quá nặng, Đức không cần bản sao).

---

## 10. Nếu gặp lỗi không có trong runbook này

Dán nguyên thông báo lỗi + lệnh đã chạy, gửi lại cho Đức — **không tự sửa code training/model**
(vi phạm ràng buộc "chỉ viết lại, không phát triển thêm" của cả dự án). Các script hỗ trợ
(`tools/*.py`, `scripts/run_geoformerdock_ablations.sh`, `scripts/run_all_inference.sh`) có thể
sửa nếu lỗi rõ ràng là lỗi script (đường dẫn sai, thiếu thư viện) — nhưng không sửa
`dockbench/training.py`, `dockbench/models/`, `dockbench/losses.py` mà không hỏi trước.
