# KhoaLuanTotNghiep_HVMD — DockBench / GeoFormerDock

Mã nguồn huấn luyện và benchmark **7 mô hình** dự đoán **tư thế docking (pose)** và **affinity protein–ligand (pK)** trên lưới voxel 3D. Package Python: **`dockbench`**.

**GitHub:** https://github.com/nWoWolfpac/KhoaLuanTotNghiep — thư mục `code_docking/`

---

## 1. Giới thiệu bài toán

Trong **thiết kế thuốc dựa trên cấu trúc (SBDD)**, cần đánh giá ligand đặt trong pocket protein: pose có hợp lý không và năng lượng tương tác (affinity) ra sao.

Repo giải quyết **hai nhiệm vụ** trên cùng đầu vào — tensor voxel 3D từ **molgrid** (hộp **23.5 Å**, bước **0.5 Å**):

| Nhiệm vụ | Mô tả | Trong file `.types` |
|----------|--------|---------------------|
| **Pose classification** | Pose **tốt** (gần native) vs **xấu** (decoy) | Cột 0: `1` = good, `0` = bad |
| **Affinity regression** | Dự đoán **pK** | Cột 1: giá trị liên tục |

Trên tập test còn báo cáo metric **sàng lọc**: C-index, EF@1/5%, Success@k.

**Benchmark 7 kiến trúc** — cùng dữ liệu, cùng loss, cùng siêu tham số train; chỉ khác backbone:

| ID (`ONLY_MODEL`) | Mô hình |
|-------------------|---------|
| `gnina_dense` | GNINA DenseNet |
| `gnina_default2018` | GNINA Default 2018 |
| `pafnucy` | Pafnucy (3D CNN) |
| `potentialnet` | PotentialNet (GNN) |
| `equibind` | EquiBind (GNN) |
| `tankbind` | TankBind (GNN) |
| `geoformerdock` | **GeoFormerDock (đề xuất)** |

---

## 2. Clone repo & cài môi trường

**Yêu cầu:** Linux hoặc **WSL2**, GPU NVIDIA, **bash**.

```bash
git clone https://github.com/nWoWolfpac/KhoaLuanTotNghiep.git
cd KhoaLuanTotNghiep/code_docking

conda create -n dockbench python=3.10 -y
conda activate dockbench

pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
conda install -c conda-forge molgrid -y
```

Kiểm tra:

```bash
python -c "import torch, molgrid; print('CUDA:', torch.cuda.is_available())"
```

> Trên Windows: chạy các lệnh trên trong **WSL**, ví dụ `cd /mnt/d/HUS/KLTN/KhoaLuan/KhoaLuanTotNghiep/code_docking`.

---

## 3. Tải dữ liệu

**Nguồn:** [CrossDocked2020](https://bits.csb.pitt.edu/files/crossdock2020/) (PDBbind2016 + file `.types` theo Francoeur et al. 2020).

Từ thư mục `code_docking/`:

```bash
bash scripts/00_download.sh
```

Script tải và giải nén vào `data/` (cần `aria2c` hoặc `wget`).

Sau khi xong, kiểm tra:

```bash
ls data/types/ref_uff_train0.types
ls data/types/ref_uff_test0.types
ls data/PDBbind2016/ | head
```

**Cấu trúc dùng khi train:**

```text
data/
├── types/
│   ├── ref_uff_train0.types    # train
│   └── ref_uff_test0.types     # test
└── PDBbind2016/                # file .gninatypes (đường dẫn trong .types)
```

Mỗi dòng `.types` (rút gọn): `label_pose  affinity  ...  rec.gninatypes  lig.gninatypes`

Báo cáo thống kê (tuỳ chọn):

```bash
python tools/dataset_report.py --project-dir . --data-root data \
  --train-file data/types/ref_uff_train0.types \
  --test-file data/types/ref_uff_test0.types
```

---

## 4. Huấn luyện (train)

Mọi lệnh chạy trong `code_docking/`.

### 4.1. Benchmark đủ 7 model

```bash
bash scripts/run_training.sh
```

Thứ tự: GNINA Dense → GNINA Default2018 → Pafnucy → PotentialNet → EquiBind → TankBind → GeoFormerDock. Cuối script tự tạo bảng so sánh và đồ thị.

### 4.2. Chỉ một model

```bash
ONLY_MODEL=geoformerdock bash scripts/run_training.sh
```

### 4.3. Chạy thử (smoke test)

```bash
EPOCHS=2 BATCH_SIZE=256 ONLY_MODEL=geoformerdock bash scripts/run_training.sh
```

### 4.4. Theo dõi tiến trình

```bash
tail -f results/logs/geoformerdock.log
```

### 4.5. Tham số hay chỉnh (biến môi trường)

| Biến | Mặc định | Ghi chú |
|------|----------|---------|
| `ONLY_MODEL` | `all` | Một ID ở bảng trên |
| `EPOCHS` | `100` | Số epoch |
| `BATCH_SIZE` | `1024` | Giảm nếu hết VRAM |
| `GEOFORMER_MAX_PSEUDO_ATOMS` | `12` | Chỉ GeoFormerDock |

Ví dụ:

```bash
EPOCHS=80 BATCH_SIZE=512 ONLY_MODEL=gnina_dense bash scripts/run_training.sh
```

### 4.6. Train trực tiếp bằng Python

```bash
export PYTHONPATH="$(pwd):${PYTHONPATH}"

python -m dockbench.training data/types/ref_uff_train0.types \
  --testfile data/types/ref_uff_test0.types \
  -d data -m geoformerdock \
  --batch_size 1024 -i 100 --test_every 2 \
  --normalize_targets --use_amp --seed 2026 \
  -o results/models/geoformerdock
```

---

## 5. Thư mục trọng số & kết quả (`results/`)

Sau train, toàn bộ **checkpoint**, log và metric nằm trong:

```text
code_docking/results/
├── models/
│   ├── geoformerdock/
│   │   ├── summary.json          # metric tốt nhất trên test
│   │   ├── best_model.pt         # ← trọng số dùng inference
│   │   ├── final_model.pt
│   │   ├── training.log
│   │   └── training_metrics_*.csv
│   ├── gnina_dense/
│   ├── gnina_default2018/
│   └── ...                       # các model còn lại
├── logs/                         # *.log, benchmark_summary.tsv
└── plots/                        # learning curves (.png)
```

**File quan trọng nhất:** `results/models/<tên_model>/best_model.pt` (đường dẫn đầy đủ trong `summary.json` → `best_weights`, ví dụ `results/models/geoformerdock_20260511_063122/best_model.pt`).

**Không đẩy `results/` lên GitHub** (dung lượng lớn). Tải bản đã train:

| Nguồn | Link |
|-------|------|
| **Google Drive** | https://drive.google.com/drive/folders/XXXXXXXX ← **dán link folder hoặc file zip `results` tại đây** |
| OneDrive (tuỳ chọn) | |

Sau khi tải/giải nén, đặt vào `code_docking/results/` (giữ cấu trúc `models/`, `logs/`, `plots/` như trên).

**Phân tích sau train (local):**

```bash
python tools/summarize_benchmark.py --models_dir results/models --logs_dir results/logs
python tools/plot_training_curves.py --models_dir results/models --plots_dir results/plots
```

**Inference không cần train lại:** dùng [`DockBench_Inference.ipynb`](DockBench_Inference.ipynb) trên Colab/Kaggle — clone repo + trỏ tới thư mục `results/` đã tải (ô cấu hình `RESULTS_*` trong notebook).

---

## Phụ lục

### Cấu trúc repo

```text
code_docking/
├── README.md
├── requirements.txt
├── DockBench_Inference.ipynb
├── dockbench/
├── scripts/00_download.sh, run_training.sh
├── tools/
├── data/          ← sau bước 3
└── results/       ← sau bước 4 (link ngoài GitHub)
```
