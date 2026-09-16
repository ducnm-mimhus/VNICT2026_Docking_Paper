# Kế hoạch sửa bài theo phản biện — VNICT 2026, submission 3531

**Bài:** GeoFormerDock: Kiến trúc học sâu đa nhiệm khai thác thông tin hình học cho bài toán
liên kết protein – phối tử
**Điểm phản biện:** R1 = 0 (borderline), R2 = +1 (weak accept)
**Deadline:** 10 ngày (tính từ ngày 0 = ngày bắt đầu thực hiện kế hoạch này)
**Tài nguyên:** máy lab dùng chung, GPU + `data/` (80GB) đã có sẵn; thời lượng GPU rảnh
chưa xác định → kế hoạch được xếp theo **thứ tự ưu tiên chạy được cắt ngang ở bất kỳ đâu**.
**Nguồn bản thảo:** `docs/VNICT2026_GeoFormerDock/` (IEEEtran, 10pt, hai cột, `\usepackage[utf8]{vietnam}`).
Mọi vị trí sửa được neo theo `file:dòng` ở §11.

> Tài liệu này **chỉ nói làm gì và làm thế nào**, không đánh giá lại phản biện. Mọi phản biện
> được coi là phải đáp ứng.

---

## 0. Phát hiện then chốt (đã xác minh, không tốn GPU) — nền của toàn bộ chiến lược

Phản biện R1 điểm 1 là điểm chặn nặng nhất: *"chọn checkpoint trên tập test + dao động 0.163
Balanced Accuracy giữa epoch 98 và 100, lớn hơn cả khoảng cách 0.040 giữa các kiến trúc"*.

Chạy lại trên `results/models/*/training_metrics_test.csv` (số liệu đã có sẵn trong repo, không
cần train lại) cho thấy **ba sự thật định lượng** sau:

### Z0.1 — Dao động đó KHÔNG phải đặc tính riêng của GeoFormerDock

Biên độ Balanced Accuracy trong 21 lần đánh giá cuối (epoch 60→100):

| Mô hình | BalAcc [min, max] | Biên độ | Recall Neg [min, max] | Biên độ |
|---|---|---|---|---|
| GeoFormerDock | [0.6258, 0.8295] | **0.204** | [0.256, 0.947] | 0.691 |
| GNINA Dense | [0.5860, 0.7871] | **0.201** | [0.172, 0.761] | 0.589 |
| GNINA Default2018 | [0.6562, 0.7614] | 0.105 | [0.336, 0.585] | 0.249 |
| Pafnucy | [0.6949, 0.7897] | 0.095 | [0.425, 0.652] | 0.227 |

→ GNINA Dense dao động **ngang hệt** GeoFormerDock (0.201 vs 0.204). Đây là tính chất của
**ngưỡng quyết định cố định 0.5 trên dữ liệu mất cân bằng 12.4%**, không phải của kiến trúc.

### Z0.2 — Chỉ số không phụ thuộc ngưỡng thì ổn định hơn 2.5 lần

Cùng khoảng epoch 60→100, GeoFormerDock: BalAcc biên độ 0.204 nhưng **PR-AUC biên độ chỉ
0.083**, C-index biên độ 0.052. Nói cách khác, trọng số mô hình ổn định; cái trôi là **điểm
vận hành (operating point)** của bộ phân loại.

### Z0.3 — Ưu thế PR-AUC là ổn định với MỌI cách chọn checkpoint; ưu thế BalAcc thì KHÔNG

Xét toàn bộ các epoch ≥ 70 (16 lần đánh giá cuối), **không chọn epoch nào cả**:

| Chỉ số | GeoFormerDock | Pafnucy | GNINA-Ds | GNINA-D18 | Kết luận |
|---|---|---|---|---|---|
| **PR-AUC** | [0.5858, 0.6459] | [0.4954, 0.5622] | [0.4430, 0.5192] | [0.4520, 0.4915] | **Rời nhau hoàn toàn** — min của GeoFormerDock > max của mọi baseline |
| BalAcc | [0.6428, 0.8295] | [0.6949, 0.7897] | [0.5860, 0.7871] | [0.6562, 0.7614] | **Chồng lấn** — không kết luận được nếu không chọn epoch |

**→ Đây là câu trả lời mạnh nhất cho R1-W1/W4 và nó không tốn một giờ GPU nào:**
> Trên 16 checkpoint cuối cùng, PR-AUC **thấp nhất** của GeoFormerDock (0.5858) vẫn cao hơn
> PR-AUC **cao nhất** của mọi baseline (0.5622). Kết luận về PR-AUC do đó không phụ thuộc vào
> cách chọn checkpoint.

**Hệ quả chiến lược — 3 quyết định:**

- **QĐ-1.** Đổi chỉ số chủ đạo của nhiệm vụ phân loại tư thế: **PR-AUC là chỉ số chính**
  (không phụ thuộc ngưỡng, phù hợp dữ liệu mất cân bằng), Balanced Accuracy là chỉ số phụ
  **kèm khai báo rõ ngưỡng được chọn ở đâu**. Điều này cũng đáp ứng luôn R2 ("tập trung hơn
  vào phân tích các chỉ số đánh giá chính và ý nghĩa thực tế").
- **QĐ-2.** Ngưỡng quyết định phải được chọn **trên tập validation**, không để cứng 0.5 và
  không dò trên test. Đây là nguồn bất ổn định thật sự, và sửa nó tốn 0 giờ GPU.
- **QĐ-3.** Vẫn phải train lại với validation-based selection + đa seed (Track A) vì phản biện
  yêu cầu trực tiếp; nhưng Z0 là **lưới an toàn**: nếu GPU không đủ, bài vẫn có một luận điểm
  vững về tính ổn định.

**Lệnh tái lập Z0** (viết thành `tools/checkpoint_robustness.py` ở việc **C1**):
```bash
python3 tools/checkpoint_robustness.py --models_dir results/models \
    --models geoformerdock,pafnucy,gnina_dense,gnina_default2018 \
    --min_epoch 70 --out results/logs/checkpoint_robustness.tsv
```

---

## 1. Ánh xạ phản biện → việc phải làm

| # | Phản biện | Mức | Việc | Track | Cần GPU? |
|---|---|---|---|---|---|
| R1-W7 | R+ ghi 0.578 trong văn bản nhưng 0.851 trong Bảng II | 🔴 lỗi rõ ràng | Sửa 1 câu ở IV.C | C | Không |
| R1-W2 | Cân bằng lớp: abstract nói 4 mô hình, IV.B nói riêng GeoFormerDock | 🔴 | **Đã xác minh bằng log: cả 4 mô hình đều `pose_balance_batch=True`** → sửa câu chữ + thêm 1 câu bằng chứng | C | Không |
| R1-W6 | Không có tuyên bố công bố mã nguồn | 🔴 rẻ | Thêm mục "Công bố mã nguồn" + dọn repo public | C | Không |
| R1-W1, W4 | Chọn checkpoint trên test; 1 run/mô hình; bất ổn định | 🔴 chặn | Z0 (ngay) + validation split + đa seed | A | **Có** |
| R2-4 | "Việc lựa chọn mô hình sử dụng dữ liệu kiểm thử là chưa chuẩn tắc" | 🔴 | Trùng R1-W1 | A | Có |
| R1-W3 | Không có ablation | 🔴 | Ablation: luồng hình học, cổng hợp nhất, cân bằng lớp | B | **Có** |
| R1-W5 | Nhánh ái lực phức tạp nhất nhưng yếu nhất → thử head hồi quy đơn giản | 🟡 | Ablation head ái lực đơn giản | B | Có |
| R2-1, R2-5 | Tiếng Việt cần trong sáng hơn; thuật ngữ cần kèm nguyên gốc | 🔴 | Rà thuật ngữ + rà văn phong toàn bài | C | Không |
| R2-2 | Dữ liệu gốc và bài toán cần mô tả tường minh | 🔴 | Viết lại IV.A + thêm 3–4 câu ở III.A | C | Không |
| R2-3 | Làm rõ vì sao mô hình đề xuất thắng; phân tích đặc điểm các mô hình so sánh | 🟡 | Viết mới 1 đoạn ở IV.D | C | Không |
| R2-4b | Tập trung phân tích chỉ số chính + ý nghĩa thực tế | 🟡 | QĐ-1 + viết lại IV.C | C | Không |

**Nguyên tắc xếp lịch:** Track C (văn bản, 0 GPU) làm **trước và độc lập**, xong hết trong
ngày 1–3, để nếu Track A/B trượt thì bài vẫn được cải thiện đáng kể. Track A/B chạy nền.

---

## 2. Ràng buộc đã xác minh (đừng lập kế hoạch trái với các mục này)

| Sự kiện | Bằng chứng |
|---|---|
| Cả 4 mô hình đều dùng cân bằng minibatch | `grep "pose_balance_batch" results/models/{geoformerdock,gnina_dense,gnina_default2018,pafnucy}/training.log` → đều `= True` |
| Bảng II hiện tại lấy từ checkpoint chọn trên test, GeoFormerDock = epoch 94 | `results/logs/best_checkpoint_audit.tsv` cột `true_epoch` |
| "Dao động 0.163" = BalAcc epoch 98 (0.8055) vs epoch 100 (0.6428) | `results/logs/epoch_selection_audit.tsv` |
| 1 run 100 epoch ≈ 1.5–2.1h GPU thật | commit `9d184d6`; `Elapsed Time` trong `results/models/*/training.log` |
| `--valfile` đã được cài vào `dockbench/training.py` (chưa commit) | `git diff dockbench/training.py` |
| `tools/make_val_split.py` tách theo **receptor**, không tách ngẫu nhiên từng dòng | đã đọc, đúng |
| Chưa có ablation nào cho luồng hình học / cổng hợp nhất / key-bias | `dockbench/models/geoformerdock.py` không có cờ nào cho ba thứ này |
| Đã có sẵn 2 ablation (`_nobalance`, `_uncertainty`) nhưng **chọn checkpoint trên test** → phải chạy lại dưới val-split | `results/logs/epoch_selection_audit.tsv` |
| `geoformerdock_nobalance` hiện có BalAcc **cao hơn** bản có cân bằng (0.8380 vs 0.8295) | `results/logs/exact_metrics_all.log` |
| Repo public: `github.com/ducnm-mimhus/VNICT2026_Docking_Paper` | `git remote -v` |

> ⚠️ Lưu ý về dòng áp chót: kết quả `nobalance` hiện tại **gợi ý** cân bằng lớp không phải
> nguồn gốc của ưu thế BalAcc (ngược với lo ngại của R1-W2), nhưng cả hai con số đều lấy ở
> epoch chọn-trên-test nên **chưa dùng được**. Phải chạy lại ở B-4.

---

## 3. TRACK A — Thực nghiệm chính: validation-based selection + đa seed

Mục tiêu: mọi con số trong Bảng II được sinh ra mà **test set chưa từng được dùng để chọn bất
cứ thứ gì** (epoch, ngưỡng, siêu tham số), và có **độ lệch chuẩn qua nhiều seed**.

### A0. Tạo validation split (≈ 15 phút, không cần GPU)

```bash
python3 tools/make_val_split.py \
    --train data/types/ref_uff_train0.types \
    --out_train data/types/ref_uff_train0_split.types \
    --out_val   data/types/ref_uff_val0.types \
    --val_frac 0.15 --seed 2026
```

**Kiểm tra bắt buộc sau khi chạy** (bổ sung vào `make_val_split.py`, việc **A0b**):
- [ ] Giao tập receptor train/val = 0 (script đã assert).
- [ ] Số mẫu `y_aff > 0` trong val ≥ 500 — nếu ít hơn, C-index trên val quá nhiễu để chọn
      checkpoint. Ước tính: 7.758 mẫu tốt × 15% ≈ 1.160 → an toàn, nhưng **phải in ra và ghi
      vào log**, vì phản biện sẽ hỏi.
- [ ] Tỉ lệ lớp dương trong val ≈ 12.4% (giống train) — nếu lệch nhiều, đổi seed split.
- [ ] Giao receptor giữa **val và test** — báo cáo con số này. Split `ref_uff` gốc đã tách
      train/test theo pocket, nên val (con của train) cũng phải rời test; xác nhận lại bằng số.

### A1. Sửa code trước khi chạy (≈ 2–3 giờ, không cần GPU)

| Việc | File | Nội dung | Vì sao bắt buộc |
|---|---|---|---|
| **A1a** | `dockbench/training.py` | Ghi `training_metrics_val.csv` song song với `_test.csv` | Không có file này thì không chứng minh được việc chọn dựa trên val |
| **A1b** | `dockbench/training.py` | Ghi vào `summary.json`: `best_epoch`, `selection_split` (`"val"`/`"test"`), `best_val_score`, `val_metrics_at_best` | Bằng chứng kiểm chứng được cho phản biện |
| **A1c** | `dockbench/training.py` | **Chọn ngưỡng phân loại trên val** (quét ngưỡng tối đa hóa BalAcc trên val tại epoch tốt nhất), ghi `pose_threshold` vào checkpoint | QĐ-2 — đây là nguồn bất ổn định thật (Z0.2) |
| **A1d** | `tools/run_inference.py` + `tools/exact_metrics.py` | Đọc `pose_threshold` từ checkpoint thay vì mặc định 0.5 | Nếu không, ngưỡng chọn trên val không có tác dụng lúc báo cáo |
| **A1e** | `dockbench/training.py` | In cảnh báo to nếu `--valfile` không được truyền | Chống tái diễn lỗi cũ |

> ⚠️ **A1c/A1d là thay đổi có ảnh hưởng tới con số cũ.** Phải giữ đường mặc định (không có
> `--valfile`, không có `pose_threshold` trong checkpoint) hoạt động **y hệt trước**, để các
> kết quả cũ vẫn tái lập được. Kiểm tra bằng `scripts/run_geoformerdock_ablations.sh smoketest`.

### A2. Hàng đợi chạy — xếp theo SEED-MAJOR, không phải MODEL-MAJOR

Vì GPU rảnh không xác định, phải xếp sao cho **cắt ngang ở bất kỳ đâu vẫn có một bảng hoàn
chỉnh**. Chạy theo lượt seed, mỗi lượt đủ 4 mô hình:

| Lượt | Runs | Tích lũy | GPU-h | Sau lượt này bài có gì |
|---|---|---|---|---|
| Lượt 1 (seed 2026) | geoformerdock, pafnucy, gnina_dense, gnina_default2018 | 4 | ~7–8h | Bảng II val-selected, 1 seed — **đã đáp ứng R1-W1 phần chọn checkpoint và R2-4** |
| Lượt 2 (seed 2027) | 4 mô hình | 8 | ~15h | mean ± half-range 2 seed |
| Lượt 3 (seed 2028) | 4 mô hình | 12 | ~22h | **mean ± std 3 seed — đáp ứng đủ R1-W4** |

Sửa `scripts/run_overnight_valsplit.sh` (việc **A2a**):
- Thêm vòng lặp `for SEED in 2026 2027 2028` **bọc ngoài** vòng lặp model.
- `OUTDIR="results/models/${MODEL}_valsplit_s${SEED}"`.
- Bỏ qua nếu `${OUTDIR}/summary.json` đã tồn tại (chạy lại được sau khi đứt).
- Giữ nguyên `wait_for_gpu` và cơ chế giảm batch_size 256→128→64.

> ⚠️ **Cảnh báo về batch_size.** Bảng II hiện tại train ở `batch_size=1024`; script overnight
> mặc định 256 vì GPU lab OOM ở 1024. Nếu lượt mới chạy ở 256 thì **không so được với số cũ**
> — nhưng điều đó không sao, vì toàn bộ Bảng II sẽ được thay bằng số mới, và cả 4 mô hình đều
> dùng cùng batch_size. **Điều bắt buộc: 4 mô hình trong cùng một lượt phải cùng batch_size.**
> Ghi batch_size thật vào chú thích Bảng II.

### A3. Sinh lại kết quả sau mỗi lượt (≈ 45 phút/lượt, cần GPU nhẹ)

```bash
# 1. inference trên test, cho từng model × từng seed
python3 tools/run_inference.py --model $M --checkpoint results/models/${M}_valsplit_s${S}/best_model.pt \
    --testfile data/types/ref_uff_test0.types -d data \
    --out results/predictions/${M}_valsplit_s${S}.csv
# 2. chỉ số chính xác
python3 tools/exact_metrics.py --predictions results/predictions/${M}_valsplit_s${S}.csv
# 3. tổng hợp đa seed  (công cụ MỚI, việc A3a)
python3 tools/aggregate_seeds.py --pred_glob 'results/predictions/*_valsplit_s*.csv' \
    --out results/logs/table2_multiseed.tsv --latex results/logs/table2_multiseed.tex
```

**A3a — `tools/aggregate_seeds.py` (mới):** gom các CSV theo `(model, seed)`, xuất
`mean ± std` cho từng chỉ số, và sinh thẳng phần thân bảng LaTeX để dán vào bản thảo (tránh
lỗi chép tay — chính là lỗi R1-W7).

**A3b — Paired bootstrap có tính cả phương sai seed (mới).** Với đa seed, cách bootstrap cũ
(`tools/paired_bootstrap_classification.py`) chỉ tính phương sai lấy mẫu test. Mở rộng thành
**bootstrap hai tầng**: mỗi vòng lặp (i) lấy mẫu lại 4.618 mẫu test có hoàn lại, **và** (ii)
bốc ngẫu nhiên 1 seed cho mỗi mô hình. Khoảng tin cậy thu được bao trùm cả nhiễu dữ liệu lẫn
nhiễu khởi tạo — đây chính là thứ R1 đòi.
Cài đặt: thêm cờ `--pred_a a_s2026.csv,a_s2027.csv,a_s2028.csv` (tương tự `--pred_b`).
*Phương án lùi nếu không kịp:* chạy bootstrap riêng cho từng seed, báo cáo "có ý nghĩa ở k/3
seed" + biên độ CI.

### A4. Cổng cắt (ngày 6)

Đến hết ngày 6, chốt theo số lượt đã xong:

| Đã xong | Bảng II báo cáo gì | Câu phải viết ở Mục V |
|---|---|---|
| 3 lượt | mean ± std qua 3 seed, chọn trên val | — (đã đủ) |
| 2 lượt | mean ± nửa biên độ qua 2 seed | "do giới hạn tài nguyên tính toán, mỗi cấu hình chạy 2 seed" |
| 1 lượt | 1 run/mô hình, chọn trên val | "mỗi cấu hình chạy một lần; tính ổn định theo checkpoint được đánh giá gián tiếp qua phân tích Z0 (Bảng III)" |
| 0 lượt | Giữ số cũ + **bắt buộc** có Bảng Z0 | "checkpoint được chọn trên tập kiểm tra; để bù, chúng tôi báo cáo dải giá trị trên 16 checkpoint cuối..." |

---

## 4. TRACK B — Ablation (R1-W3, R1-W5)

Tất cả chạy trên GeoFormerDock, **dùng val-split, seed 2026, cùng batch_size với lượt A**, để
so được trực tiếp với dòng GeoFormerDock đầy đủ của lượt 1.

### B0. Thêm cờ ablation vào code (≈ 3 giờ, không cần GPU) — việc chặn Track B

Sửa `dockbench/models/geoformerdock.py`, thêm tham số khởi tạo (mặc định = hành vi hiện tại,
để checkpoint cũ vẫn load được):

| Cờ mới | Mặc định | Khi bật ablation | Chạm vào |
|---|---|---|---|
| `use_geometry: bool` | `True` | `False` → nhánh tư thế chỉ còn `hlocal`, bỏ `SimpleGeometryEncoder` và bỏ luôn cổng hợp nhất | `pose_geometry`, `pose_fusion` |
| `pose_fusion_mode: str` | `"gated"` | `"concat"` → nối 2 vector rồi chiếu tuyến tính, **giữ nguyên luồng hình học** | `GatedFeatureFusion` |
| `use_key_bias: bool` | `True` | `False` → `B_pocket = 0`, attention chuẩn | `MultiHeadAttention.pocket_gate` |
| `simple_affinity: bool` | `False` | `True` → bỏ 2 khối Transformer + tokenizer, nhánh ái lực = GAP + MLP | `tokenizer`, `transformer_layers`, `aff_fusion` |

Và trong `dockbench/training.py`: thêm `--geo_ablation {none,no_geometry,concat_fusion,no_key_bias,simple_affinity}`
→ đổ vào `geoformer_kwargs` (chỗ dòng ~1108). In cấu hình ablation vào log và `summary.json`.

**Kiểm tra bắt buộc sau B0:**
- [ ] `--geo_ablation none` cho **đúng 1.594.573 tham số** (bằng số hiện tại trong bài).
- [ ] Load được `results/models/geoformerdock/best_model.pt` không lỗi `unexpected keys`.
- [ ] Mỗi biến thể chạy được `smoketest` 3 epoch trên `demo_inference/`.
- [ ] **Ghi lại số tham số của từng biến thể** — bảng ablation phải có cột Params, vì luận
      điểm của bài là "nhỏ mà tốt".

### B1–B5. Danh sách run, theo thứ tự ưu tiên cắt

| # | Ablation | Trả lời phản biện nào | GPU-h | Ưu tiên |
|---|---|---|---|---|
| **B-1** | `--geo_ablation no_geometry` | R1-W3 "geometric stream" — **đóng góp #1 của bài** | ~2h | 🔴 không được cắt |
| **B-2** | bỏ `--pose_balance_batch` (val-split) | R1-W2 + R1-W3 "class balancing" — tách công thức huấn luyện khỏi kiến trúc | ~2h | 🔴 không được cắt |
| **B-3** | `--geo_ablation concat_fusion` | R1-W3 "gated fusion" | ~2h | 🟠 cắt thứ 3 |
| **B-4** | `--geo_ablation simple_affinity` | R1-W5 "head hồi quy đơn giản có tốt bằng không" | ~2h | 🟠 cắt thứ 2 |
| **B-5** | `--geo_ablation no_key_bias` | R1-W3 "key-saliency bias" | ~2h | 🟡 cắt đầu tiên |

Tổng Track B: ~10 GPU-h. **Tổng A + B ≈ 32 GPU-h ≈ 3.2h/ngày trong 10 ngày** — vừa sức một
máy lab chạy đêm, nhưng không có dư địa cho sự cố → phải bắt đầu chạy từ ngày 2.

### B6. Đã có sẵn, dùng lại được có điều kiện

`results/models/geoformerdock_uncertainty/` (B1 cũ) trả lời một phần R1-W5: head ái lực có
ước lượng bất định cho **MAE 1.0817 vs 1.221** của bản gốc — tốt hơn rõ rệt. Nhưng con số này
chọn-trên-test → **chỉ được nhắc trong Mục V như hướng phát triển, không đưa vào bảng ablation**,
trừ khi chạy lại dưới val-split (ưu tiên sau B-5).

---

## 5. TRACK C — Sửa bản thảo (0 GPU, làm ngay ngày 1–3)

### C0. Chuẩn bị nguồn — ✅ ĐÃ CÓ

Nguồn LaTeX nằm ở **`docs/VNICT2026_GeoFormerDock/`** (không phải `paper/`):

```
docs/VNICT2026_GeoFormerDock/
├── VNICT2026_Template_LaTeX.tex   (495 dòng — preamble + abstract + \input)
├── IEEEtran.cls
├── ref.bib                        (15 mục, CẢ 15 đều được trích dẫn)
├── tex/introduction.tex   (14 dòng)
├── tex/relatedwork.tex    (9 dòng)
├── tex/method.tex         (157 dòng)   ← Mục III
├── tex/result.tex         (53 dòng)    ← Mục IV + Bảng II
├── tex/conclusion.tex     (4 dòng)     ← Mục V
└── figures/  (7 file .png; `shared_backbone.png` HIỆN KHÔNG ĐƯỢC DÙNG)
```

- [ ] `git add docs/VNICT2026_GeoFormerDock/` và commit **nguyên trạng bản đã nộp** trước khi
      sửa một ký tự nào — để diff được mọi thay đổi về sau.
- [ ] Giữ `docs/VNICT2026_GeoFormerDock.pdf` làm mốc so sánh.
- [ ] Dựng được bản PDF hiện tại trên máy làm việc (`pdflatex` + `bibtex`) **trước** khi sửa,
      để biết chắc lỗi build về sau là do mình.

### C1. Công cụ sinh bằng chứng Z0

Viết `tools/checkpoint_robustness.py`: đọc `training_metrics_test.csv` của N mô hình, xuất
[min, max] của từng chỉ số trên các epoch ≥ `--min_epoch`, kèm cột "rời nhau/chồng lấn" so với
mô hình tốt nhất. Xuất TSV + thân bảng LaTeX.

### C2. Các sửa chữa văn bản — chi tiết từng chỗ

#### C2-1 🔴 Lỗi số R+ (Mục IV.C) — R1-W7

- **Câu cũ:** "…cao hơn so với GNINA-D18 (0.578) và Pafnucy (0.652), đổi lại R+=0.578 giảm nhẹ
  so với hai mô hình này (0.945 và 0.928 tương ứng)."
- **Câu mới:** "…cao hơn so với GNINA-D18 (0.578) và Pafnucy (0.652), đổi lại R+=0.851 thấp hơn
  hai mô hình này (0.945 và 0.928 tương ứng)."
- **Kiểm chứng:** 0.135·0.851 + 0.865·0.808 = 0.814 = Acc ✓; (0.851+0.808)/2 = 0.830 = BalAcc ✓.
- **Việc kèm theo:** mọi con số trong văn bản phải được sinh từ `tools/aggregate_seeds.py`,
  không chép tay. Thêm bước **P8'** ở cuối: script tự đối chiếu mọi số trong `.tex` với TSV.

#### C2-2 🔴 Cân bằng lớp (Tóm tắt + Mục III.B.2 + Mục IV.B) — R1-W2

Đây **không phải** điểm yếu thật mà là lỗi diễn đạt: log của cả 4 mô hình đều có
`pose_balance_batch = True`. Cách sửa biến nó thành điểm mạnh:

- **Mục III.B.2, câu cũ:** "Đồng thời, trước khi tính $L_{pose}$, một tập con 64 mẫu (32 tốt/32
  xấu) được lấy mẫu lại có hoàn lại từ minibatch 1024 mẫu… Đây là một yếu tố thuộc về thiết lập
  huấn luyện…"
  → **Chuyển toàn bộ đoạn này sang Mục IV.B** (nó là thiết lập huấn luyện, không phải kiến
  trúc), chỉ để lại ở III.B.2 một câu: "Chi tiết về chiến lược lấy mẫu cân bằng lớp dùng chung
  cho mọi mô hình được trình bày ở Mục IV.B."
- **Mục IV.B, câu cũ:** "Đối với nhánh phân loại tư thế của GeoFormerDock được huấn luyện với
  cân bằng lớp ở mức minibatch (32 mẫu tốt/32 mẫu xấu mỗi batch)."
- **Câu mới (3 câu):** "Cân bằng lớp ở mức minibatch được áp dụng **đồng nhất cho nhánh phân
  loại tư thế của cả bốn mô hình**: trước khi tính $L_{pose}$, một tập con 64 mẫu (32 tốt/32
  xấu) được lấy mẫu lại có hoàn lại từ mỗi minibatch. Tương tự, hàm mất mát tiêu điểm
  ($\gamma=2.0$, $\alpha=0.75$) và trung bình cân bằng theo lớp cũng dùng chung cho bốn mô
  hình. Do đó chênh lệch Balanced Accuracy và PR-AUC giữa các kiến trúc trong Bảng II **không
  đến từ khác biệt về chiến lược lấy mẫu hay hàm mất mát phân loại**; ảnh hưởng của chính chiến
  lược này được đo trực tiếp ở dòng ablation (−cân bằng lớp) trong Bảng III."
- **Tóm tắt:** giữ nguyên "khung đánh giá chung cho bốn mô hình… tích hợp chiến lược lấy mẫu
  cân bằng lớp ở mức minibatch" — nay đã nhất quán với IV.B.

#### C2-3 🔴 Validation split + tính ổn định (Mục III.C, IV.B, Bảng III mới) — R1-W1/W4, R2-4

- **Mục III.C, câu cũ:** "Mô hình được đánh giá sau mỗi 2 epoch; checkpoint tốt nhất được lựa
  chọn theo chỉ số tổng hợp 0.5·C-index + 0.5·Balanced Accuracy…"
- **Câu mới:** "…checkpoint tốt nhất, điểm dừng sớm **và ngưỡng quyết định của bộ phân loại tư
  thế** đều được lựa chọn trên một **tập validation độc lập**, tách từ tập huấn luyện theo
  **receptor** (15% số protein, không có receptor nào xuất hiện ở cả hai tập). Tập kiểm tra
  không tham gia vào bất kỳ quyết định nào của quá trình huấn luyện."
- **Mục IV.B, thêm:** số mẫu train-split / val / test sau khi tách; số mẫu `y_aff>0` trong val;
  seed của phép tách.
- **Mục IV.C, thêm Bảng III (Z0) + 2–3 câu:**
  > "Balanced Accuracy phụ thuộc ngưỡng và do đó nhạy với checkpoint: trong 16 lần đánh giá
  > cuối, biên độ dao động của chỉ số này là 0.204 với GeoFormerDock và 0.201 với GNINA Dense
  > — nghĩa là đặc tính của ngưỡng cố định trên dữ liệu mất cân bằng, không của riêng kiến trúc
  > nào. Ngược lại, PR-AUC — chỉ số không phụ thuộc ngưỡng — chỉ dao động 0.083, và **giá trị
  > thấp nhất của GeoFormerDock trên toàn bộ 16 checkpoint cuối (0.5858) vẫn cao hơn giá trị
  > cao nhất của mọi baseline (0.5622)**. Kết luận về PR-AUC do đó độc lập với cách chọn
  > checkpoint."
- **Thay số Bảng II** bằng `mean ± std` từ `tools/aggregate_seeds.py`; chú thích rõ: số seed,
  batch_size, "checkpoint và ngưỡng chọn trên validation".

#### C2-4 🔴 Bảng ablation mới (Bảng IV) — R1-W3/W5

Bảng nhỏ, 1 cột, chỉ 4 chỉ số chính (PR-AUC, BalAcc, MAE, C-index) + Params:

| Cấu hình | Params | PR-AUC | BalAcc | MAE | C-idx |
|---|---|---|---|---|---|
| GeoFormerDock (đầy đủ) | 1.59M | — | — | — | — |
| − luồng hình học (chỉ CNN cục bộ) | — | — | — | — | — |
| − cổng hợp nhất (thay bằng nối) | — | — | — | — | — |
| − cân bằng lớp minibatch | 1.59M | — | — | — | — |
| nhánh ái lực đơn giản (bỏ Transformer) | — | — | — | — | — |
| − thiên vị chú ý theo khóa | — | — | — | — | — |

Kèm 3–4 câu diễn giải, trong đó **bắt buộc** có một câu trả lời thẳng R1-W5: hoặc "head đơn
giản đạt MAE tương đương với ít tham số hơn, cho thấy độ phức tạp của nhánh ái lực chưa được
đền đáp" (nếu đúng vậy), hoặc ngược lại. Không được tránh né câu hỏi.

#### C2-5 🔴 Công bố mã nguồn — R1-W6

Thêm mục **"Công bố mã nguồn và dữ liệu"** (2–3 câu, đặt cuối Mục IV hoặc trước Kết luận):
> "Mã nguồn, cấu hình huấn luyện đầy đủ, script tách validation và toàn bộ log huấn luyện được
> công bố tại `https://github.com/ducnm-mimhus/VNICT2026_Docking_Paper`. Dữ liệu
> CrossDocked2020 và split `ref_uff` là dữ liệu công khai của nhóm tác giả bộ dữ liệu [4];
> `scripts/00_download.sh` tải đúng phiên bản đã dùng."

**Việc kèm theo (quan trọng, dễ quên):** trước ngày nộp phải dọn repo public:
- [ ] `results/logs/`, `results/predictions/`, `training_metrics_*.csv` — giữ (nhỏ, là bằng chứng).
- [ ] Checkpoint `.pt` — đang bị `.gitignore`; cân nhắc đưa lên GitHub Release (~153MB).
- [ ] README nêu đúng cách tái lập Bảng II/III/IV bằng 3 lệnh.
- [ ] Xóa/soát thông tin cá nhân, đường dẫn tuyệt đối máy lab trong script.

#### C2-6 🔴 Mô tả dữ liệu gốc và bài toán (Mục III.A + IV.A) — R2-2

Hiện bài nhảy thẳng vào "tệp types" và "lưới voxel 48×48×48". Thêm 4–5 câu **trước** đoạn đó:
- Dữ liệu gốc là gì: cấu trúc 3D thực nghiệm của phức hợp protein–phối tử (PDBbind2016), được
  **tái ghép nối chéo (cross-docking)** để sinh nhiều tư thế giả định cho mỗi phức hợp →
  CrossDocked2020.
- Mỗi mẫu = (1 cấu trúc protein, 1 tư thế phối tử), lưu ở định dạng `.gninatypes`.
- Bài toán 1 (phân loại tư thế): cho một tư thế do docking sinh ra, quyết định nó có **tái lập
  đúng** tư thế thực nghiệm hay không (RMSD ≤ 2 Å) — tức là "kết quả docking này có đáng tin
  không".
- Bài toán 2 (dự đoán ái lực): ước lượng ái lực liên kết (thang pK) cho các mẫu có nhãn thực nghiệm.
- Ý nghĩa thực tế (R2-4b): trong sàng lọc ảo, PR-AUC cao nghĩa là khi lấy top-N tư thế theo
  điểm số, tỉ lệ tư thế đúng trong đó cao hơn — trực tiếp giảm số thí nghiệm phải làm.
- Nói rõ vì sao 4.618 mẫu test nhưng chỉ 623 mẫu tính chỉ số ái lực.

#### C2-7 🔴 Thuật ngữ kèm nguyên gốc tiếng Anh — R2-1

Ở **lần xuất hiện đầu tiên** của mỗi thuật ngữ, thêm nguyên gốc trong ngoặc. Danh sách tối
thiểu (R2 nêu đích danh 4 mục đầu):

| Tiếng Việt | Nguyên gốc cần thêm |
|---|---|
| tư thế gắn kết | binding pose |
| dự đoán ái lực | binding affinity prediction |
| đánh giá tư thế | pose scoring |
| dự đoán nhị phân (tốt/xấu) | binary classification (good/bad pose) |
| nhãn tư thế nhị phân | binary pose label |
| liên kết protein–phối tử | protein–ligand docking |
| học đa nhiệm | multi-task learning |
| lưới voxel | voxel grid |
| cơ chế thiên vị chú ý theo khóa | key-saliency attention bias |
| cổng thích nghi / hợp nhất có cổng | gated fusion |
| điểm trọng yếu | pseudo-atom |
| hàm cơ sở xuyên tâm | radial basis function (RBF) |
| hàm mất mát tiêu điểm | focal loss |
| lấy mẫu cân bằng lớp ở mức minibatch | minibatch class-balanced resampling |
| tập kiểm định / validation | validation set |
| điểm dừng sớm | early stopping |
| đường biên Pareto | Pareto frontier |
| tái ghép nối chéo | cross-docking |

Áp dụng **đồng thời cho phần Từ khóa** (R2 nêu đích danh).

#### C2-8 🟡 Vì sao GeoFormerDock thắng + đặc điểm các mô hình so sánh (Mục IV.D) — R2-3

Viết mới 1 đoạn (~0.5 cột), bám vào số liệu chứ không suy diễn:
- **Đặc điểm kiến trúc đối chiếu:** GNINA-D18 = CNN 3D thuần, 2.16M, không có mô-đun hình học
  tường minh; GNINA-Dense = kết nối dày đặc, chỉ 0.46M, mạnh nhất ở ái lực; Pafnucy = CNN sâu
  7.99M, thiết kế gốc cho hồi quy ái lực đơn nhiệm; GeoFormerDock = 1.59M, có luồng hình học
  tường minh cho nhánh tư thế.
- **Cơ chế được cho là tạo ra ưu thế:** nhánh tư thế kết hợp đặc trưng cục bộ với quan hệ
  khoảng cách giữa các điểm trọng yếu, tức là mô hình hóa được "hình dạng" của tư thế chứ không
  chỉ mật độ nguyên tử cục bộ. **Bảng IV (ablation) là bằng chứng trực tiếp cho câu này** — nếu
  bỏ luồng hình học mà PR-AUC không giảm thì phải sửa lại câu này, không được giữ.
- **Trả lời thẳng câu "đây có phải kết quả tốt nhất trên tập test chưa":** không. Nói rõ:
  đây là kết quả tốt nhất **trong khung so sánh thống nhất của nghiên cứu này** (cùng biểu
  diễn voxel, cùng siêu tham số, train từ đầu), không phải kết quả tốt nhất từng công bố trên
  CrossDocked2020 — các kết quả công bố gốc dùng thiết lập huấn luyện và tiền xử lý khác nên
  không so trực tiếp được.

#### C2-9 🔴 Viết lại Mục V (Kết luận + Hạn chế)

- **Bỏ** hạn chế (i) cũ ("checkpoint chọn trên chính tập kiểm tra… dao động 0.163") — đã được
  sửa thật ở Track A, không còn là hạn chế.
- **Giữ và làm rõ** (iii) mô hình đồ thị, (iv) 623/4.618 mẫu.
- **Thêm/sửa** các hạn chế mới, trung thực với những gì thật sự còn thiếu:
  - Tách validation theo receptor, **chưa phải theo cụm tương đồng chuỗi/túi liên kết** —
    vẫn có thể còn rò rỉ họ protein giữa train và val.
  - Số seed thực tế chạy được (điền theo cổng cắt A4).
  - Ablation chạy 1 seed/cấu hình.
  - Chưa quét trọng số $w_{pose}/w_{aff}$ nên "đường biên Pareto" vẫn là quan sát trên các điểm
    rời rạc (câu này đã có, giữ).

#### C2-10 🔴 Rà văn phong tiếng Việt toàn bài — R2-1, R2-5

Checklist cụ thể (giao cho Hiếu, đọc phản biện bởi Phát):
- [ ] Bỏ câu bị động dịch máy: "được thực hiện bằng", "được tiến hành", "mang lại hiệu của
      chính" (→ "hiệu quả chính").
- [ ] Thống nhất: "tập kiểm tra" (test) vs "tập kiểm định" (validation) — **không dùng lẫn**.
- [ ] Thống nhất "ái lực liên kết" xuyên suốt (không lúc "ái lực", lúc "độ gắn kết").
- [ ] Mỗi đoạn ≤ 5 câu; câu ≤ 40 chữ.
- [ ] Bỏ dấu gạch nối "-" dùng thay dấu ngoặc đơn (bài hiện dùng nhiều, ví dụ "…decoy; hai nhãn
      này không được dùng lẫn cho nhau ở bất kỳ bước huấn luyện/đánh giá nào - nhánh tư thế…").
- [ ] Tóm tắt viết lại cho khớp kết luận mới (PR-AUC là chỉ số chủ đạo, có validation, có ablation).

---

## 6. Ngân sách trang (6 trang, hai cột) — bắt buộc cân đối

Nội dung **thêm** ≈ 1.35 cột. Phải cắt tương đương.

| Thêm | Ước lượng |
|---|---|
| Mô tả dữ liệu gốc + bài toán (C2-6) | +0.35 cột |
| Thuật ngữ nguyên gốc (C2-7) | +0.10 cột |
| Validation split + Bảng III (Z0) (C2-3) | +0.45 cột |
| Bảng IV ablation + diễn giải (C2-4) | +0.55 cột |
| Công bố mã nguồn (C2-5) | +0.08 cột |
| Đoạn "vì sao thắng" (C2-8) | +0.45 cột |
| Cân bằng lớp viết lại (C2-2) | +0.05 cột |
| **Tổng thêm** | **≈ +2.0 cột** |

| Cắt | Tiết kiệm |
|---|---|
| Bỏ hạn chế (i) cũ ở Mục V | −0.15 cột |
| Bảng II: bỏ cột **Acc** (suy được từ R+/R−/π) và **RMSE** (giữ MAE) → còn chỗ cho ± std | −0.25 cột |
| Mục II rút còn 4–5 câu (bỏ câu về mạng đồ thị, vì Mục V đã nêu lý do loại nhóm này) | −0.40 cột |
| Gộp Hình 3 và Hình 4 thành một hình 2 panel | −0.35 cột |
| Rút đoạn Pareto ở IV.D (đã dài, nay có Bảng IV thay thế phần lập luận) | −0.30 cột |
| Rút mô tả $L_{dist}$, $L_{anchor}$ còn 1 câu + 1 công thức gộp | −0.30 cột |
| Thu 3 hình đơn cột từ `width=0.5\textwidth` xuống `0.44` (hiện đang **tràn cột**, xem §11) | −0.25 cột |
| Rút đoạn quy ước dấu $y_{aff}$ ở `method.tex:23` (hiện ~15 dòng một đoạn) còn ~8 dòng | −0.25 cột |
| **Tổng cắt** | **≈ −2.0 cột** |

> ⚠️ **Không cắt tài liệu tham khảo.** Đã kiểm tra: `ref.bib` có đúng 15 mục và **cả 15 đều
> được trích dẫn** trong bài. Cắt mục nào cũng phải bỏ một trích dẫn thật — đây là khoản
> tiết kiệm giả.

> Đo **số trang thật sau khi format**, không ước lượng. Nếu vẫn vượt: cắt tiếp theo thứ tự
> Bảng III (Z0) → dòng ablation B-5 → dòng ablation B-3. **Không bao giờ cắt:** Bảng IV dòng
> "− luồng hình học" và "− cân bằng lớp", mục Công bố mã nguồn, và câu về validation split.

---

## 7. Lịch 10 ngày

Ký hiệu: **Đ** = Đức (code/GPU/số liệu), **H** = Hiếu (văn phong/định dạng), **P** = Phát
(bảng biểu/rà soát).

| Ngày | Việc | Ai | Chặn cái gì |
|---|---|---|---|
| **1** | C0 (copy `.tex` vào `paper/`, commit nguyên trạng) | Đ | mọi việc Track C |
| 1 | **C2-1** (sửa R+), **C2-2** (cân bằng lớp), **C2-5** (công bố mã nguồn) — 3 việc rẻ, xong dứt điểm | P | — |
| 1 | **C1** viết `tools/checkpoint_robustness.py`, sinh Bảng III | Đ | C2-3 |
| 1 | **A0** tạo val split + kiểm tra A0b | Đ | A2 |
| 1 | **A1a–A1e** sửa `training.py` (val CSV, summary, ngưỡng trên val) | Đ | A2 |
| **2** | **B0** thêm cờ ablation + kiểm tra 1.594.573 tham số + smoketest | Đ | Track B |
| 2 | **A2a** sửa `run_overnight_valsplit.sh` (vòng seed, skip-if-done) | Đ | A2 |
| 2 tối | ▶ **Khởi động lượt 1 (seed 2026, 4 mô hình)** | Đ | Bảng II |
| 2 | **C2-6** (mô tả dữ liệu/bài toán), **C2-7** (thuật ngữ) | H | — |
| **3** | **A3** inference + exact metrics lượt 1; **A3a** viết `aggregate_seeds.py` | Đ | Bảng II |
| 3 tối | ▶ **Khởi động B-1, B-2** (2 ablation không được cắt) | Đ | Bảng IV |
| 3 | **C2-3** viết lại III.C + IV.B (validation) — điền số sau | H | — |
| 3 | **C2-10** rà văn phong vòng 1 trên các mục chưa đụng số liệu (I, II, III.A/B) | H | — |
| **4** | **A3b** bootstrap hai tầng; chạy trên lượt 1 | Đ | IV.C |
| 4 tối | ▶ **Khởi động lượt 2 (seed 2027)** | Đ | ± std |
| 4 | **C2-9** viết lại Mục V | P | — |
| **5** | Dựng **Bảng II** (từ `aggregate_seeds.py`, không chép tay) + **Bảng III** | P | Mục IV |
| 5 tối | ▶ **Khởi động B-3, B-4** | Đ | Bảng IV |
| 5 | **C2-8** viết đoạn "vì sao thắng + đặc điểm mô hình so sánh" | Đ | — |
| **6** | 🚩 **CỔNG CẮT (mục 4 A4).** Chốt: bao nhiêu seed, bao nhiêu ablation. Chốt luôn câu chữ ở Mục V tương ứng | cả 3 | mọi việc còn lại |
| 6 tối | ▶ **Khởi động lượt 3 (seed 2028)** nếu GPU cho phép; nếu không → chạy B-5 | Đ | — |
| **7** | Dựng **Bảng IV** (ablation) + viết diễn giải C2-4 | Đ + P | Mục IV |
| 7 | Ghép bản thảo v2 hoàn chỉnh | Đ | phản biện nội bộ |
| **8** | Phản biện nội bộ chéo: H đọc Mục III–IV, P đọc toàn bài, đối chiếu **từng phản biện trong mục 1** | H + P | sửa ngày 9 |
| 8 | **Đo số trang thật**; nếu vượt 6 trang → áp mục 6 | H | — |
| **9** | Sửa theo phản biện nội bộ; **P8'** script đối chiếu mọi số trong `.tex` với TSV | cả 3 | — |
| 9 | Dọn repo public theo checklist C2-5; tạo GitHub Release cho checkpoint | Đ | — |
| **10** | Rà chính tả cuối; xuất PDF; nộp | P → Đ | — |

**Đường găng:** `C0 → A1 → A2 lượt 1 → A3 → Bảng II → Mục IV → ghép v2 → phản biện → nộp`.
Nếu lượt 1 không xong trước hết ngày 4, kích hoạt ngay phương án A4 dòng "0 lượt".

---

## 8. Kịch bản dự phòng (phải quyết trước, không quyết lúc thấy kết quả)

### K1. Nếu val-split + đa seed làm mất ý nghĩa thống kê của ưu thế BalAcc

Rất có thể xảy ra: theo `epoch_selection_audit.tsv`, ở epoch chọn theo train (98), BalAcc của
GeoFormerDock là 0.8055 so với Pafnucy 0.7897 — khoảng cách co từ +0.040 xuống +0.016.

**Quyết trước:** giữ nguyên kết quả, **đổi luận điểm chứ không đổi số**:
- Luận điểm chính chuyển hẳn sang **PR-AUC** (theo QĐ-1), nơi ưu thế rộng và ổn định (Z0.3).
- BalAcc trình bày như chỉ số phụ, phụ thuộc ngưỡng, có ± std.
- Sửa Tóm tắt, đóng góp #1, và Kết luận cho khớp: "dẫn đầu về PR-AUC với khác biệt có ý nghĩa
  thống kê; ở Balanced Accuracy, ưu thế nằm trong khoảng dao động giữa các seed".
- **Không** thử nhiều seed rồi chọn seed đẹp. Không thử nhiều tỉ lệ val rồi chọn tỉ lệ đẹp.

### K2. Nếu ablation cho thấy luồng hình học KHÔNG đóng góp

**Quyết trước:** báo cáo trung thực và sửa đóng góp #1 xuống thành "kiến trúc gọn nhẹ đạt
PR-AUC cao nhất", đồng thời ghi ở Mục V rằng nguồn gốc ưu thế chưa được quy về mô-đun cụ thể.
Phản biện R1 đánh giá cao chính sự trung thực này ("Honest reporting of a negative result").
Che giấu kết quả ablation âm là rủi ro lớn hơn nhiều so với việc hạ một câu tuyên bố.

### K3. Nếu GPU không rảnh đủ

Thứ tự hy sinh (cắt từ trên xuống, dừng khi vừa tài nguyên):
1. Lượt 3 (seed 2028) → còn 2 seed.
2. B-5 (key-bias), rồi B-4 (simple affinity), rồi B-3 (gating).
3. Lượt 2 (seed 2027) → còn 1 seed, **nhưng vẫn là val-selected**.
4. Cuối cùng mới đến Track A hoàn toàn — lúc đó Bảng III (Z0) trở thành bắt buộc và Mục V phải
   giữ lại nguyên hạn chế (i) cũ.

**Không hy sinh:** B-1 (luồng hình học) và B-2 (cân bằng lớp) — hai ablation mà cả R1-W2 lẫn
R1-W3 đều đòi đích danh.

---

## 9. Checklist nghiệm thu trước khi nộp

Mỗi dòng phải trỏ được tới bằng chứng cụ thể trong repo:

**Đáp ứng phản biện**
- [ ] R1-W1/W4, R2-4: Bảng II sinh từ checkpoint chọn trên **val**; `summary.json` có
      `selection_split="val"`; có ± std qua ≥ 2 seed **hoặc** Bảng III (Z0) nếu chỉ 1 seed.
- [ ] R1-W2: Mục IV.B nói rõ cân bằng lớp dùng chung 4 mô hình; có dòng ablation "− cân bằng lớp".
- [ ] R1-W3: Bảng IV có tối thiểu 3 dòng (− hình học, − cân bằng lớp, và ≥1 dòng nữa).
- [ ] R1-W5: có câu trả lời thẳng cho "head hồi quy đơn giản có tốt bằng không", dựa trên số.
- [ ] R1-W6: có mục Công bố mã nguồn; link GitHub mở được; README tái lập được Bảng II bằng 3 lệnh.
- [ ] R1-W7: R+ = 0.851 (hoặc số mới tương ứng); **script đối chiếu số tự động đã chạy sạch**.
- [ ] R2-1/R2-5: mọi thuật ngữ trong bảng C2-7 có nguyên gốc ở lần đầu; đã rà văn phong 2 vòng.
- [ ] R2-2: có mô tả dữ liệu gốc (PDBbind2016 → cross-docking → CrossDocked2020) và cả hai bài toán.
- [ ] R2-3: có đoạn phân tích đặc điểm 4 kiến trúc + câu nói rõ "không phải kết quả tốt nhất
      từng công bố, mà là tốt nhất trong khung so sánh thống nhất này".
- [ ] R2-4b: PR-AUC được nêu là chỉ số chính, có giải thích ý nghĩa thực tế trong sàng lọc ảo.

**Tính toàn vẹn số liệu**
- [ ] Mọi con số trong `.tex` khớp TSV do script sinh ra (P8').
- [ ] Bất biến $Acc = \pi R^+ + (1-\pi) R^-$ đúng cho mọi dòng Bảng II ($\pi = 0.1349$).
- [ ] Bảng IV: số tham số của cấu hình đầy đủ = 1.594.573, khớp con số trong Tóm tắt.
- [ ] Batch size / số seed / split của **mọi** dòng bảng được ghi trong chú thích.
- [ ] Không con số nào trong bài đến từ một epoch khác với các con số cùng dòng.

**Kỹ thuật**
- [ ] `--geo_ablation none` load được checkpoint cũ (không lỗi `unexpected keys`).
- [ ] Chạy không có `--valfile` cho kết quả y hệt trước khi sửa (hồi quy không đổi hành vi cũ).
- [ ] `results/logs/` có log của mọi run mới; không log nào có NaN/OOM.
- [ ] Giao receptor train/val = 0 và val/test = 0, có in ra trong log.

---

## 10. Việc KHÔNG làm (chốt trước để khỏi tranh luận giữa chừng)

- ❌ Không thêm baseline mới (PotentialNet/EquiBind/TankBind vẫn ở ngoài, lý do đã nêu ở Mục V).
- ❌ Không đổi kiến trúc để chạy theo phản biện — chỉ đo, không sửa mô hình.
- ❌ Không quét siêu tham số để "cứu" một chỉ số.
- ❌ Không chạy nhiều seed rồi chọn seed cho kết quả đẹp (K1).
- ❌ Không tự dò ngưỡng phân loại trên test, kể cả "chỉ để xem".
- ❌ Không viết lại Mục III theo hướng thêm mô-đun mới.

---

## 11. Phụ lục — Neo chính xác trong mã nguồn LaTeX

Đã đọc toàn bộ `docs/VNICT2026_GeoFormerDock/`. Mỗi việc ở §5 được neo vào đúng file và dòng
của **bản đã nộp** (dòng sẽ trôi sau lần sửa đầu tiên — dùng đoạn trích để tìm lại).

### 11.1. Bảng neo: việc → vị trí

| Việc §5 | File:dòng | Đoạn nhận dạng |
|---|---|---|
| **C2-1** sửa R+ | `tex/result.tex:33` | `đổi lại R+=0.578 giảm nhẹ so với hai mô hình này` |
| **C2-2** cân bằng lớp (IV.B) | `tex/result.tex:28` | `Đối với nhánh phân loại tư thế của GeoFormerDock được huấn luyện với cân bằng lớp` |
| **C2-2** chuyển đoạn cân bằng khỏi Mục III | `tex/method.tex:65` | `Đồng thời, trước khi tính $\mathcal{L}_{pose}$, một tập con 64 mẫu` |
| **C2-2** Tóm tắt | `VNICT2026_Template_LaTeX.tex:442` | `tích hợp chiến lược lấy mẫu cân bằng lớp ở mức minibatch` |
| **C2-3** chọn checkpoint (III.C) | `tex/method.tex:139` | `checkpoint tốt nhất được lựa chọn theo chỉ số tổng hợp` |
| **C2-3** chọn trên test (IV.B) | `tex/result.tex:28` (cuối câu) | `được chọn theo chỉ số tổng hợp trên chính tập kiểm tra` |
| **C2-3** Bảng III (Z0) chèn mới | sau `tex/result.tex:42` | sau đoạn paired bootstrap phân loại |
| **C2-4** Bảng IV ablation chèn mới | sau `tex/result.tex:43` | trước `\textit{D. Phân tích đánh đổi}` |
| **C2-5** công bố mã nguồn | cuối `tex/result.tex` | chèn `\textit{E. Công bố mã nguồn và dữ liệu}` |
| **C2-6** mô tả dữ liệu gốc | `tex/method.tex:4` + `tex/result.tex:24` | `Dữ liệu ban đầu được tổ chức dưới dạng tệp types` |
| **C2-7** thuật ngữ | toàn bộ + `…LaTeX.tex:446` | khối `\begin{IEEEkeywords}` |
| **C2-8** vì sao thắng | `tex/result.tex:53` | đoạn cuối `D. Phân tích đánh đổi` |
| **C2-9** viết lại Mục V | `tex/conclusion.tex:2` và `:4` | `(i) checkpoint tốt nhất và điểm dừng sớm hiện được chọn trên chính tập kiểm tra` |
| **Bảng II** thay số | `tex/result.tex:5–22` | `\label{tab:combined_results}` |

### 11.2. Bẫy định dạng — phải xử lý TRƯỚC khi thêm nội dung

Bản hiện tại vừa đúng 6 trang nhờ một số thủ thuật bố cục sẽ **vỡ ngay khi thêm chữ**:

1. 🔴 **`\newpage` thủ công ở `tex/method.tex:46`** — nằm giữa Mục III.B.2, ngay sau phương
   trình hợp nhất. Thêm bất kỳ câu nào phía trước sẽ đẩy nó sai chỗ và tạo một cột trắng.
   **Xóa `\newpage`, để LaTeX tự ngắt.**
2. 🔴 **Sáu hình đều dùng `\begin{figure}[H]`** (`method.tex:7,49,82`; `result.tex:35,46`) —
   `[H]` của gói `float` **cấm** LaTeX dời hình, gây khoảng trắng lớn và tràn cột khi nội dung
   đổi. **Đổi hết sang `[!t]` hoặc `[!htb]`**, trừ `figure*` ở `method.tex:16` (giữ `[t]`).
3. 🔴 **Ba hình đơn cột đặt `width=0.5\textwidth`** (`method.tex:9,51,84`). Trong IEEEtran hai
   cột, một cột rộng ≈ `0.47\textwidth` → **ba hình này đang tràn cột**. Hạ xuống `0.44` vừa
   sửa lỗi vừa tiết kiệm ~0.25 cột (đã tính vào §6).
4. 🟡 **`\usepackage{url}` đang bị comment** (`VNICT2026_Template_LaTeX.tex:331`). Phải bỏ
   comment trước khi dùng `\url{}` cho link GitHub ở **C2-5**.
5. 🟡 **Mục IV dùng `\textit{A. Dữ liệu}` viết tay** (`result.tex:3,26,31,44`) trong khi Mục III
   dùng `\subsection{}` (`method.tex:3,25,131`). Không đồng nhất; các mục con của IV **không vào
   mục lục và không đánh số tự động**. Giữ nguyên cách viết tay để khỏi xô số trang, nhưng khi
   thêm mục E (công bố mã nguồn) phải theo đúng kiểu `\textit{E. …}`.
6. 🟡 **Phương trình focal loss ở `method.tex:57–59` không có `\label`** — thêm label nếu Mục IV
   cần tham chiếu tới nó khi diễn giải ablation.
7. 🟢 **`figures/shared_backbone.png` chưa được dùng ở đâu cả** — dùng được làm panel khi gộp
   Hình 3 + Hình 4 (khoản cắt −0.35 cột ở §6), hoặc xóa khỏi repo cho gọn.

### 11.3. Bảng II — vì sao bắt buộc phải bỏ bớt cột

`tex/result.tex:10` hiện là `\begin{tabular}{lc ccccc ccccc}` = **12 cột trong một
`table*` toàn trang**, cỡ `\footnotesize`. Thêm `± std` vào 11 cột số là không thể vừa.

Cột bỏ và lý do đưa vào chú thích bảng:
- **Acc** — suy được từ $R^+, R^-$ và $\pi = 0.1349$: $Acc = \pi R^+ + (1-\pi)R^-$.
- **RMSE** — cùng chiều thông tin với MAE trên tập 623 mẫu; giữ MAE là chỉ số chính.

Còn lại 10 cột: `Mô hình | Params | BalAcc | R+ | R- | PR-AUC | MAE | r | ρ | C-idx`, đủ chỗ
cho `0.830 ± 0.012`.

### 11.4. Số liệu trong bài cần kiểm chứng lại (không do phản biện nêu, nhưng sẽ lộ khi công bố mã nguồn)

| Chỗ | Vấn đề | Cách xử lý |
|---|---|---|
| `method.tex:153` | Bảng I ghi `$w_{pose}, w_{aff}$ = 0.51, 0.15`. Trong code, `pose_total_weight=0.85`, `lambda_pose=1.2`, `pose_loss_scale=0.5` → **0.85 × 1.2 × 0.5 = 0.51** (khớp), còn `aff_total_weight=0.15`. Nhưng 0.51 là **tích hiệu dụng**, không phải một siêu tham số đơn | Thêm một dòng chú thích dưới Bảng I nói rõ 0.51 là giá trị hiệu dụng, và liệt kê ba cờ gốc — nếu không, người đọc mã nguồn sẽ không tìm thấy số 0.51 ở đâu cả |
| `method.tex:154` | Bảng I ghi `Batch size (chuẩn) 1024`, nhưng các run mới trên GPU lab sẽ chạy ở 256 (OOM ở 1024) | Sau khi chốt lượt chạy, sửa thành batch size **thật** của Bảng II, hoặc ghi cả hai kèm giải thích |
| `result.tex:24` | "62.335 mẫu huấn luyện" | Sau khi tách val, phải đổi thành `train_split / val / test` với số thật |
| `method.tex:23` + `result.tex:24` | Tỉ lệ dương ghi 12.4% (train) và 13.5% (test) | Sau tách val phải bổ sung tỉ lệ của val; kiểm tra lại 13.5% vs $\pi = 0.1349$ trong audit (khớp) |

### 11.5. Trạng thái `ref.bib`

15 mục, **cả 15 đều được trích dẫn** — không có mục chết. Không có khoản tiết kiệm nào ở đây
(xem cảnh báo ở §6). Nếu C2-6/C2-8 cần thêm trích dẫn mới (ví dụ cho cross-docking hoặc cho
việc chọn ngưỡng trên validation), phải **thêm** mục, tức là **tốn** thêm chỗ chứ không tiết
kiệm — đã tính dự phòng trong §6.
