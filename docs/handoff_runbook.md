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

## 5. Chạy inference lấy dự đoán từng mẫu (cho 6 model đã có checkpoint)

```bash
bash scripts/run_all_inference.sh
```

Mặc định dùng `data/types/ref_uff_test0.types` + GPU. Nếu máy không có GPU:
```bash
DEVICE=cpu bash scripts/run_all_inference.sh
```

Kết quả: `results/predictions/<model>.csv` cho 6 model (`gnina_dense, gnina_default2018,
pafnucy, potentialnet, tankbind, geoformerdock`). **`equibind` bị bỏ qua** — không có
`best_model.pt` (đã xác nhận từ trước, không phải lỗi của script).

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

```bash
mkdir -p results/logs
for baseline in gnina_dense gnina_default2018 pafnucy potentialnet tankbind; do
    echo "=== geoformerdock vs ${baseline} ===" | tee -a results/logs/paired_bootstrap_all.log
    python3 tools/paired_bootstrap.py \
        --pred_a results/predictions/geoformerdock.csv \
        --pred_b "results/predictions/${baseline}.csv" \
        --metric all --n_boot 2000 --seed 2026 \
        | tee -a results/logs/paired_bootstrap_all.log
done
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
results/predictions/*.csv                     (7-8 file, bước 5)
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
