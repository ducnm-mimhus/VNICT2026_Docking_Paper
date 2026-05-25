# code_docking

Mã nguồn huấn luyện và đánh giá **benchmark 7 mô hình** cho bài toán **dự đoán tư thế docking và affinity protein–ligand** trên lưới voxel 3D, phục vụ khóa luận **GeoFormerDock**.

Package Python: **`dockbench`** — chứa kiến trúc mạng, loss, metrics và script train.

---

## Mục lục

1. [Bài toán](#1-bài-toán)
2. [Dữ liệu](#2-dữ-liệu)
3. [Bảng benchmark 7 mô hình](#3-bảng-benchmark-7-mô-hình)
4. [GeoFormerDock (mô hình đề xuất)](#4-geoformerdock-mô-hình-đề-xuất)
5. [Pipeline huấn luyện](#5-pipeline-huấn-luyện)
6. [Cấu trúc thư mục](#6-cấu-trúc-thư-mục)
7. [Cài đặt môi trường](#7-cài-đặt-môi-trường)
8. [Tải dữ liệu](#8-tải-dữ-liệu)
9. [Chạy huấn luyện](#9-chạy-huấn-luyện)
10. [Tham số cấu hình](#10-tham-số-cấu-hình)
11. [Train trực tiếp bằng Python](#11-train-trực-tiếp-bằng-python)
12. [Kết quả đầu ra](#12-kết-quả-đầu-ra)
13. [Screening UI](#13-screening-ui-load-mô-hình--chọn-compound)
14. [Công cụ phân tích sau train](#14-công-cụ-phân-tích-sau-train)
15. [Xử lý lỗi thường gặp](#15-xử-lý-lỗi-thường-gặp)

---

## 1. Bài toán

Trong **structure-based drug design (SBDD)**, cần đánh giá mức độ phù hợp giữa một phân tử thuốc (ligand) và pocket protein khi ligand được đặt trong pocket (docking pose). Repo này giải quyết **hai nhiệm vụ đồng thời** trên cùng một đầu vào voxel 3D (từ `molgrid`):

| Nhiệm vụ | Mô tả | Nhãn trong file `.types` |
|----------|--------|---------------------------|
| **Pose classification** | Phân loại pose **tốt** (gần native) vs **xấu** (decoy) | Cột 0: `1` = good, `0` = bad |
| **Affinity regression** | Dự đoán **pK / affinity** (năng lượng tương tác) | Cột 1: giá trị liên tục |

Ngoài ra, trên tập test còn báo cáo các chỉ số **screening / ranking** (C-index, enrichment factor EF@k%, success@k) — mô phỏng bài toán sàng lọc compound theo affinity dự đoán.

**Đầu vào mô hình:** tensor voxel `(C, D, H, W)` — lưới 3D quanh complex protein–ligand, kích thước mặc định **23.5 Å**, resolution **0.5 Å** (theo chuẩn GNINA / CrossDocked).

**Mục tiêu benchmark:** so sánh công bằng **7 kiến trúc** (3D CNN, GNN, mô hình đề xuất) trên **cùng dữ liệu, cùng loss, cùng siêu tham số train** — chỉ khác backbone.

---

## 2. Dữ liệu

- **Nguồn:** [CrossDocked2020](https://bits.csb.pitt.edu/files/crossdock2020/) (cấu trúc PDBbind2016 + file `.types` từ paper).
- **Split:** Francoeur et al. 2020 (*J. Chem. Inf. Model.*) — file train/test UFF:
  - `data/types/ref_uff_train0.types`
  - `data/types/ref_uff_test0.types`
- **Cấu trúc tham chiếu:** `data/PDBbind2016/` — file `.gninatypes` được tham chiếu trong từng dòng `.types`.

Mỗi dòng file `.types` (rút gọn):

```text
<label_pose>  <affinity>  ...  <đường_dẫn.gninatypes>  ...
```

- `label_pose`: 0/1 (bad/good pose).
- `affinity`: pK chuẩn hóa theo convention GNINA (affinity > 0 thường được coi là “positive” khi tính C-index / EF).

---

## 3. Bảng benchmark 7 mô hình

| # | ID (`ONLY_MODEL` / `-m`) | Họ mô hình | Vai trò trong benchmark |
|---|--------------------------|------------|-------------------------|
| 1 | `gnina_dense` | 3D DenseNet (GNINA dense) | CNN mạnh, baseline GNINA hiện đại |
| 2 | `gnina_default2018` | 3D CNN (GNINA default) | CNN chuẩn GNINA 2018 |
| 3 | `pafnucy` | 3D CNN (Pafnucy-style) | CNN affinity kinh điển |
| 4 | `potentialnet` | Gated GNN | Message passing hai giai đoạn |
| 5 | `equibind` | Equivariant GNN | EGNN trên pseudo-atom |
| 6 | `tankbind` | Triangle-aware GNN | Cơ chế tam giác / pair |
| 7 | `geoformerdock` | **GeoFormerDock (Ours)** | CNN + geometry tokens + Pocket Transformer |

**Alias legacy (CLI):** `default2017` → `gnina_dense`, `default2018` → `gnina_default2018`.

Tất cả mô hình dùng chung pipeline multi-task trong `scripts/run_training.sh` (Focal pose loss, 5-loss affinity, early stopping, v.v.).

---

## 4. GeoFormerDock (mô hình đề xuất)

**GeoFormerDock** (`dockbench/models/geoformerdock.py`) là kiến trúc lai **voxel + graph**:

- **Backbone 3D CNN** dùng chung, sau đó **tách sớm** nhánh pose và affinity.
- **Nhánh pose:** CNN cục bộ + **pseudo-atom geometry** (top-K voxel, mã hóa RBF) + gated fusion → logits 2 lớp.
- **Nhánh affinity:** CNN + **Voxel Tokenizer** + **Pocket-Aware Transformer** → hồi quy pK (có calibration scale/bias).
- Siêu tham số kiến trúc khi train: `--max_pseudo_atoms` (mặc định 12), `--num_transformer_layers` (mặc định 2).

---

## 5. Pipeline huấn luyện

### 5.1. Loss affinity (5 thành phần)

Tất cả 7 mô hình dùng **cùng** hàm mất mát affinity:

1. **Uncertainty-aware regression** (scale = 1.0)
2. **Soft ranking loss** (scale = 0.05, warmup epoch 10–25)
3. **Distribution alignment** (scale = 0.02)
4. **Anchor loss** (scale = 0.01)
5. **Pose auxiliary:** Focal Loss trên nhánh pose, trọng số `w_pose=0.85`, `w_aff=0.15`

Tổng quát mỗi bước (sau giai đoạn pose-only):

```text
L = L_aff + 0.3 × (λ_pose × pose_loss_scale × L_pose)
```

### 5.2. Lịch train

| Giai đoạn epoch | Nội dung |
|-----------------|----------|
| 0 → `POSE_ONLY_EPOCHS` (4) | Chỉ backprop **pose** (+ backbone chung) |
| 0 → 10 | Regression + pose |
| 10 → 25 | Thêm ranking (ramp 0→1) |
| 25+ | Full ranking + distribution alignment |

### 5.3. Tối ưu & ổn định

- Optimizer: **AdamW** + warmup LR + cosine decay
- **Gradient clipping** (max norm 5.0)
- **Chuẩn hóa target pK** (`--normalize_targets`)
- **Early stopping** trên test: `composite_cidx_balacc` (mặc định 0.5×C-index + 0.5×Balanced Accuracy), patience 25
- **Focal loss** cho pose + cân bằng minibatch theo lớp
- **AMP** bật tự động cho các mô hình CNN nặng (`gnina_dense`, `gnina_default2018`, `pafnucy`, `geoformerdock`)

### 5.4. Chỉ số đánh giá

**Pose:** Recall pos/neg, Balanced Accuracy, PR-AUC, MCC.

**Affinity:** MAE, RMSE, Pearson, Spearman, **C-index**.

**Screening:** EF@1%, EF@5%, Success@1/5/10.

---

## 6. Cấu trúc thư mục

```text
code_docking/
├── README.md
├── requirements.txt
├── scripts/
│   ├── 00_download.sh       # tải & giải nén dữ liệu → data/
│   └── run_training.sh      # benchmark 7 model + báo cáo + vẽ đồ thị
├── tools/
│   ├── dataset_report.py    # thống kê dataset trước khi train
│   ├── summarize_benchmark.py
│   └── plot_training_curves.py
├── dockbench/               # package Python
│   ├── training.py          # entry point: python -m dockbench.training
│   ├── losses.py, metrics.py, dataloaders.py, ...
│   └── models/              # 7 kiến trúc + registry.py
├── screening_ui/            # UI nhỏ: load weights, 3D screening, chọn ligand
│   ├── app.py               # Streamlit
│   └── README.md
├── data/                    # (tạo sau khi download)
│   ├── PDBbind2016/
│   └── types/
└── results/                 # (tạo khi train)
    ├── models/<tên_model>/
    ├── logs/
    └── plots/
```

---

## 7. Cài đặt môi trường

**Khuyến nghị:** Linux hoặc **WSL2**, GPU NVIDIA, **bash**.

### 7.1. Conda + PyTorch

```bash
cd code_docking

conda create -n dockbench python=3.10 -y
conda activate dockbench

# Chọn bản CUDA phù hợp GPU — ví dụ CUDA 12.1:
pip install torch==2.2.2 torchvision --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

### 7.2. molgrid (bắt buộc)

`molgrid` đọc file `.types` và tạo lưới voxel. Trên nhiều máy cần cài qua conda-forge:

```bash
conda install -c conda-forge molgrid
```

Nếu `pip install molgrid` thất bại, dùng lệnh conda ở trên.

### 7.3. Công cụ tải dữ liệu

Script download cần **một trong hai:** `aria2c` (nhanh hơn) hoặc `wget`.

```bash
# Ubuntu/Debian
sudo apt-get install -y aria2

# macOS
brew install aria2
```

### 7.4. Kiểm tra cài đặt

```bash
conda activate dockbench
python -c "import torch, molgrid; from dockbench.models import BENCHMARK_MODELS; print(torch.cuda.is_available(), BENCHMARK_MODELS)"
```

---

## 8. Tải dữ liệu

Chạy **một lần** từ thư mục gốc `code_docking/`:

```bash
bash scripts/00_download.sh
```

Script sẽ:

1. Tải `PDBbind2016.tar.gz` và `paper_types.tar.gz` (~vài GB).
2. Giải nén vào `data/`.
3. Xóa file nén tạm.

Sau khi hoàn tất, kiểm tra:

```bash
ls data/types/ref_uff_train0.types
ls data/types/ref_uff_test0.types
ls data/PDBbind2016/ | head
```

Xem báo cáo thống kê (không bắt buộc):

```bash
python tools/dataset_report.py \
  --project-dir . \
  --data-root data \
  --train-file data/types/ref_uff_train0.types \
  --test-file data/types/ref_uff_test0.types
```

---

## 9. Chạy huấn luyện

Mọi lệnh dưới đây chạy từ **`code_docking/`**.

### 9.1. Benchmark đủ 7 mô hình

```bash
bash scripts/run_training.sh
```

Thứ tự train: GNINA Dense → GNINA Default2018 → Pafnucy → PotentialNet → EquiBind → TankBind → GeoFormerDock.

Cuối pipeline tự động:

- In bảng so sánh metric (`summarize_benchmark.py`)
- Vẽ learning curves (`plot_training_curves.py`)

**Thời gian:** phụ thuộc GPU; với 100 epoch và batch 1024 có thể kéo dài nhiều ngày trên một GPU. Nên chạy từng model hoặc giảm `EPOCHS` khi thử.

### 9.2. Train một mô hình

```bash
ONLY_MODEL=geoformerdock bash scripts/run_training.sh
```

Các giá trị `ONLY_MODEL` hợp lệ:

`gnina_dense` | `gnina_default2018` | `pafnucy` | `potentialnet` | `equibind` | `tankbind` | `geoformerdock`

### 9.3. Smoke test (2 epoch)

```bash
EPOCHS=2 BATCH_SIZE=256 ONLY_MODEL=geoformerdock bash scripts/run_training.sh
```

Dùng để kiểm tra môi trường, đường dẫn `data/`, và GPU trước khi chạy full.

### 9.4. Theo dõi log trực tiếp

```bash
tail -f results/logs/geoformerdock.log
```

Mỗi epoch in loss và metric train/test (Recall pose, MAE, C-index, EF, …).

---

## 10. Tham số cấu hình

Ghi đè bằng **biến môi trường** trước `bash scripts/run_training.sh`:

| Biến | Mặc định | Ý nghĩa |
|------|----------|---------|
| `ONLY_MODEL` | `all` | Một model hoặc `all` / `benchmark` / `full` cho cả 7 |
| `EPOCHS` | `100` | Số epoch |
| `BATCH_SIZE` | `1024` | Batch size (giảm nếu OOM) |
| `LR` | `0.001` | Learning rate ban đầu |
| `SEED` | `2026` | Random seed |
| `TEST_EVERY` | `2` | Đánh giá test mỗi N epoch |
| `GEOFORMER_MAX_PSEUDO_ATOMS` | `12` | Số pseudo-atom (chỉ GeoFormerDock) |
| `USE_SAM_FOR_GEO` | `0` | Bật SAM optimizer cho GeoFormerDock (`1`) |
| `POSE_FOCAL_ALPHA` | `0.75` | Focal loss — trọng số lớp good pose |
| `POSE_FOCAL_GAMMA` | `2.0` | Focal loss — focusing parameter |
| `EARLY_STOP_PATIENCE` | `25` | Early stopping patience |
| `FULL_DATASET_EPOCH` | `1` | `1` = full epoch; `0` = stratify receptor (legacy) |

**Ví dụ:**

```bash
EPOCHS=80 BATCH_SIZE=512 SEED=2026 ONLY_MODEL=gnina_dense bash scripts/run_training.sh
```

```bash
GEOFORMER_MAX_PSEUDO_ATOMS=8 EPOCHS=100 ONLY_MODEL=geoformerdock bash scripts/run_training.sh
```

---

## 11. Train trực tiếp bằng Python

Khi cần tinh chỉnh tham số không có trong `run_training.sh`:

```bash
export PYTHONPATH="$(pwd):${PYTHONPATH}"

python -m dockbench.training \
  data/types/ref_uff_train0.types \
  --testfile data/types/ref_uff_test0.types \
  -d data \
  -m geoformerdock \
  --label_pos 0 \
  --affinity_pos 1 \
  --base_lr 0.001 \
  --batch_size 1024 \
  -i 100 \
  --test_every 2 \
  --lr_dynamic \
  --warmup_epochs 2 \
  --normalize_targets \
  --max_pseudo_atoms 12 \
  --num_transformer_layers 2 \
  --use_amp \
  --seed 2026 \
  -o results/models/geoformerdock
```

Xem đầy đủ tham số:

```bash
python -m dockbench.training --help
```

---

## 12. Kết quả đầu ra

### 12.1. Mỗi mô hình — `results/models/<tên_model>/`

| File | Mô tả |
|------|--------|
| `summary.json` | Metric tốt nhất / cuối cùng trên test |
| `best_model.pt` | Checkpoint theo early stopping |
| `final_model.pt` | Checkpoint epoch cuối |
| `training_metrics_train.csv` | Metric theo epoch (train) |
| `training_metrics_test.csv` | Metric theo epoch (test) |
| `training.log` | Log chi tiết trong thư mục model |

### 12.2. Log terminal — `results/logs/<tên_model>.log`

Bản sao stdout/stderr khi chạy qua `tee` trong `run_training.sh`.

### 12.3. Bảng tổng hợp — `results/logs/benchmark_summary.tsv`

Tạo tự động sau `run_training.sh` (hoặc chạy lại tool §13).

### 12.4. Đồ thị — `results/plots/`

- `*.png` — đường cong metric (train `--`, test `-`) cho từng chỉ số.
- `geoformerdock_only/` — chỉ curves của GeoFormerDock (khi chạy full benchmark).

---

## 13. Screening UI (load mô hình & chọn compound)

Thư mục [`screening_ui/`](screening_ui/README.md): giao diện Streamlit nhỏ — load `best_model.pt`, suy luận trên `.types`, **không gian screening 3D** (Plotly), xếp hạng top-k ligand theo target.

```bash
pip install -r screening_ui/requirements-ui.txt
streamlit run screening_ui/app.py
```

Chế độ **Demo** (CSV mẫu) chạy ngay không cần molgrid; chế độ inference đầy đủ cần `data/` + molgrid như khi train.

---

## 14. Công cụ phân tích sau train

Chạy từ `code_docking/` (không cần train lại):

**Bảng so sánh 7 model:**

```bash
python tools/summarize_benchmark.py --models_dir results/models --logs_dir results/logs
```

**Vẽ lại learning curves:**

```bash
python tools/plot_training_curves.py \
  --models_dir results/models \
  --plots_dir results/plots \
  --also_geo_only
```

Chỉ một model:

```bash
python tools/plot_training_curves.py \
  --models_dir results/models \
  --plots_dir results/plots \
  --only_model geoformerdock
```

---

## 15. Xử lý lỗi thường gặp

### Thiếu file `.types` hoặc PDBbind

```text
ERROR: Training data not found: .../data/types/ref_uff_train0.types
```

→ Chạy lại `bash scripts/00_download.sh` và kiểm tra `data/`.

### `ModuleNotFoundError: molgrid`

→ `conda install -c conda-forge molgrid`, kích hoạt đúng env `dockbench`.

### CUDA out of memory

```bash
BATCH_SIZE=256 bash scripts/run_training.sh
# hoặc
BATCH_SIZE=128 ONLY_MODEL=geoformerdock bash scripts/run_training.sh
```
### `No module named dockbench`

→ Đảm bảo `PYTHONPATH` trỏ tới thư mục gốc `code_docking/` (script `run_training.sh` đã set sẵn).

### Train trên Windows PowerShell

Script bash cần **Git Bash** hoặc **WSL**. Không dùng `&&` trong PowerShell thuần; mở WSL:

```bash
cd /mnt/d/HUS/KLTN/KhoaLuan/code_docking
bash scripts/run_training.sh
```
Ư
