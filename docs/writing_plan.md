# Kế hoạch Viết Bài báo – VNICT 2026
**Nguồn:** Khóa luận tốt nghiệp "Phát triển Mô hình Học sâu Đa nhiệm..."  
**Nguyên tắc:** Viết lại (rewrite), không phát triển thêm nội dung khoa học mới, trừ ablation nhỏ nếu còn thời gian.  
**Quỹ thời gian:** **14 ngày** cho toàn bộ công việc (từ ngày 1 đến ngày nộp).  
**Độ dài mục tiêu:** **~6 trang, khổ hai cột kiểu hội nghị quốc tế** (IEEE/ACM two-column). ⚠️ **Đây là ràng buộc chặt hơn "6–8 trang" ở bản trước**, không phải nới lỏng hơn — một trang hai cột chứa lượng nội dung tương đương ~1.5–2 trang một cột thông thường. **Vẫn cần Đ1 xác nhận CFP chính thức**: khổ cột, cỡ chữ, và tài liệu tham khảo có tính vào 6 trang hay được cộng thêm.

> **Phiên bản 4** — đính chính khung độ dài. Bản 3 viết "6–8 trang" không phân biệt khổ cột; đây
> là lỗi giả định, không phải thay đổi yêu cầu. Hệ quả của việc siết khổ:
> - Ngân sách nội dung được tính lại theo **cột** (6 trang × 2 cột = 12 cột), không theo trang.
> - **Ablation (B2) không còn là mục con IV.4 riêng** — gộp thành 1–2 dòng phụ ngay trong bảng
>   kết quả chính (IV.3) để tiết kiệm diện tích. Không đủ chỗ cho một bảng riêng.
> - Mục II (Công trình liên quan) nén còn ~5–6 câu tổng, không phải 2–3 câu mỗi nhóm như bản 3.
> - Danh sách "quyết định D1/D2/D3", "VĐ1–VĐ12", "A1/A2/B1/B2", ma trận phụ thuộc, và lịch 14
>   ngày ở bản 3 **không đổi về nội dung** — chỉ đổi phần khung độ dài và cách phân bổ chữ.

---

## 0. Điều kiện tiên quyết — kiểm tra trước khi lập lịch

```
[ ] results/models/<model>/best_model.pt          còn cho cả 7 mô hình?
[ ] results/models/<model>/training_metrics_*.csv còn cho cả 7 mô hình?
[ ] results/models/<model>/summary.json           còn cho cả 7 mô hình?
[ ] data/ (PDBbind2016 + .types)                   còn nguyên, molgrid còn chạy được?
[ ] GPU còn truy cập được (Kaggle / JetBrains Cadence)?
```

**Đã xác minh:** `results/` có checkpoint đầy đủ cho 6/7 mô hình (thiếu `equibind`); `demo_inference/`
chỉ có dữ liệu mẫu 2 phức hợp, không đủ cho inference toàn tập test; `data/` đầy đủ **chưa có**,
đang xin theo hướng dẫn `scripts/00_download.sh`.

**Phát hiện quan trọng khi kiểm tra `results/`** (không phải giả định — đã chạy
`tools/inspect_checkpoints.py` và xác nhận trên dữ liệu thật):
- `dockbench/training.py` thiếu `nonlocal best_epoch, best_metrics` trong `log_test_results()`,
  khiến `summary.json` luôn phản ánh **epoch cuối**, không phải epoch tốt nhất.
- Bảng 6/7 trong khóa luận (dòng GeoFormerDock) được ghép từ **ít nhất 5 epoch khác nhau**
  (48, 58, 68, 92, 94) — không phải một checkpoint duy nhất.
- Với số liệu thật, tự nhất quán (`best_model.pt`, epoch 94): Balanced Accuracy (0.8295) và
  PR-AUC (0.6306) vẫn cao nhất so với 3 baseline CNN, nhưng **MAE (1.221) và C-index (0.771) tệ
  hơn cả Pafnucy** — khác với mô tả "gần biên Pareto, không hy sinh gì" trong khóa luận.
- **Đây là quyết định còn treo, chưa chốt**: dùng số liệu thật hay số khóa luận, và có viết lại
  câu chuyện Pareto hay không. Xem §2 — phần này **chưa được cập nhật** trong bản 4, vì đây là
  quyết định của nhóm, không phải của việc đính chính độ dài. Cần chốt ở Họp 1 hoặc sớm hơn.

---

## 1. Đường găng (critical path)

```
Ngày 1  ── VĐ11: đối chiếu lại Bảng 6 (ĐÃ XONG — xem §0)              ┐
        ── Q2:  lấy N của tập con y_aff > 0                          │
                                                                      ▼
Ngày 2-3 ─ Q4: chạy inference lấy dự đoán từng mẫu (cần data/ + GPU) ──► Ngày 3-4: paired bootstrap
                                                                                 │
                                                                                 ▼
                                              Ngày 5-6: Phát dựng bảng kết quả cuối
                                                                                 │
                                                                                 ▼
                                              Ngày 5-6: Đức viết Mục IV
```

**Nhánh song song, không chặn đường găng:** A2 (script chọn epoch, đã viết được ngay, không cần
`data/`), kick off B1/B2 (chạy nền khi có `data/` + GPU).

---

## 2. Định vị câu chuyện của bài báo

> ⚠️ **Phần này CHƯA cập nhật theo phát hiện ở §0** (GeoFormerDock thua Pafnucy ở MAE/C-index
> với số liệu thật). Giữ nguyên nội dung bản 3 dưới đây làm điểm khởi đầu thảo luận — nhóm cần
> quyết định trước khi Hiếu viết Mục I, nếu không sẽ phải viết lại.

### ❌ Không viết theo hướng

- *"Chúng tôi đề xuất mô hình tốt hơn tất cả"* — GNINA vẫn thắng MAE/C-index.
- *"Ba thành phần kiến trúc của chúng tôi đều đóng góp"* — Geometry Encoder chiếm 0.7% tham số
  với tọa độ không khả vi, `B_pocket` chiếm 0.06% và không dùng thông tin không gian.
- *"Benchmark hoàn toàn công bằng: cùng dữ liệu, cùng loss, cùng siêu tham số"* — sai cho 6/7
  dòng (VĐ10: GNINA dùng Gaussian NLL, còn lại dùng Huber).

### ✅ Viết theo hướng (bản nháp — chờ chốt sau khi có quyết định về số liệu thật)

> *"Trong bài toán scoring đa nhiệm cho docking, tối ưu ái lực liên kết và lọc cấu hình là hai
> mục tiêu có tính đánh đổi. Các mô hình hiện có thường nằm lệch về một cực. Chúng tôi xây dựng
> một khung so sánh thống nhất trên cùng biểu diễn voxel để định lượng sự đánh đổi này, và đề
> xuất một kiến trúc lai gọn nhẹ (1.59M tham số) đạt vị trí lệch về phía lọc cấu hình trên biên
> Pareto — dẫn đầu về PR-AUC và Balanced Accuracy trên dữ liệu mất cân bằng lớp, đổi lại độ chính
> xác ái lực thấp hơn nhóm CNN chuyên biệt."*

Câu này **thận trọng hơn** bản 3 — không còn khẳng định "không hy sinh gì", vì số liệu thật chưa
xác nhận điều đó. Cần nhóm chốt lại đúng mức độ mạnh/yếu của tuyên bố trước khi đưa vào Mục I.

### Ba đóng góp (bản nháp, chờ chốt cùng §2)

1. Kiến trúc đa nhiệm gọn nhẹ (~1.59M tham số): backbone 3D CNN chung, tách sớm 2 nhánh, kết hợp
   Transformer token voxel (ái lực) và đặc trưng hình học RBF trên pseudo-atom (cấu hình).
2. Khung so sánh thống nhất 7 kiến trúc trên cùng biểu diễn voxel, định lượng đánh đổi
   pose–affinity bằng biểu đồ Pareto.
3. Phân tách định lượng các yếu tố ngoài kiến trúc (dạng hàm hồi quy, cân bằng minibatch) ảnh
   hưởng tới vị trí trên biên Pareto — *chỉ tuyên bố được nếu B1/B2 hoàn thành*.

> ⚠️ Không viết "chưa có mô hình nào giải quyết đồng thời" (VĐ5). Cụm **"theo hiểu biết của
> chúng tôi"** bắt buộc nếu giữ một tuyên bố dạng khoảng trống.

---

## 3. Quyết định phải chốt trong Họp 1

| # | Quyết định | Khuyến nghị | Hệ quả |
|---|---|---|---|
| **D0 (mới)** | Dùng số liệu thật (checkpoint, tự nhất quán) hay số khóa luận (ghép nhiều epoch)? | **Số liệu thật** — số khóa luận không tái lập được từ bất kỳ checkpoint nào | Viết lại phần định vị câu chuyện ở §2; MAE/C-index của mô hình đề xuất sẽ **kém hơn** Pafnucy |
| **D1** | VĐ1: giữ hay bỏ 3 baseline đồ thị? | **Bỏ** (Phương án B) | Càng cần thiết ở khổ 6 trang hai cột — không đủ chỗ giải thích 3 dòng bằng ngẫu nhiên |
| **D2** | Có chạy B1 (uncertainty) + B2 (no-balance) không? | **Có, cả hai** — chạy nền, không chặn viết | B2 gộp vào bảng chính (không có mục ablation riêng ở khổ 6 trang) |
| **D3** | Trình bày VĐ10 ở đâu? | Mục IV.2 (1 câu) + Mục V (1 câu) | Không đủ chỗ cho phân tích dài |

---

## 4. Cấu trúc bài báo — ngân sách theo CỘT (6 trang × 2 cột = 12 cột)

⚠️ Con số dưới đây là **ước lượng** dựa trên mật độ chữ điển hình của template IEEE hai cột
(~450–550 từ/cột với cỡ chữ 10pt). Đức cần đối chiếu với template CFP thật (Đ1) để hiệu chỉnh.

| Mục | Ngân sách (cột) | ~Số từ | Nội dung chính |
|---|---|---|---|
| Tiêu đề + Tóm tắt | 0.3 | 150–200 | Trên đầu trang 1 |
| I. Giới thiệu | 1.0 | 450–500 | Bối cảnh, đánh đổi, 3 đóng góp |
| II. Công trình liên quan | 0.6 | 280–320 | **Nén còn 5–6 câu tổng**, không phải 2–3 câu/nhóm |
| III. Phương pháp đề xuất | 3.0 | 1300–1400 | Kiến trúc, backbone, 2 nhánh, loss + 1 hình |
| IV. Thực nghiệm | 4.2 | 1800–1900 | Dữ liệu, thiết lập, 1 bảng kết quả (kèm dòng ablation), Pareto |
| V. Thảo luận & Hạn chế | 0.9 | 380–420 | 6 điểm bắt buộc, **1 câu/điểm** |
| VI. Kết luận | 0.3 | 120–150 | 3–4 câu |
| Tài liệu tham khảo | 1.4 | ~15–18 mục | IEEE compact |
| **Tổng** | **~11.7 / 12** | | Còn ~0.3 cột dự phòng cho hình/bảng tràn |

**Cắt so với bản 3 để vừa khổ:**
- Ablation (B2) **không còn là Mục IV.4 riêng** — chỉ thêm 1–2 dòng vào bảng kết quả chính (IV.3),
  không có đoạn văn giải thích dài.
- Mục II không còn chia 4 đoạn theo nhóm — viết liền thành 5–6 câu.
- Mục V: mỗi điểm đúng 1 câu, không phải 1–2 câu.
- Hình 3 (kiến trúc) và Hình 17 (Pareto): nên vẽ vừa **1 cột** nếu được; nếu buộc phải tràn 2 cột
  (`\begin{figure*}`), tính thêm ~0.3–0.4 cột phát sinh cho mỗi hình tràn cột.

---

## 5. Chi tiết từng mục

### I. Giới thiệu (~1.0 cột)

Giữ đoạn mở đầu về ý nghĩa docking (rút gọn còn 2–3 câu), đặt quan sát về đánh đổi (Hình 17)
ngay trong đoạn 2. Ba đóng góp theo §2 — dạng gạch đầu dòng ngắn, không viết thành câu dài.

### II. Công trình liên quan (~0.6 cột — nén mạnh)

Viết liền mạch 5–6 câu, không tách đoạn theo nhóm:
1. Phương pháp truyền thống (Vina, GOLD) — 1 câu.
2. CNN 3D voxel (GNINA, Pafnucy) — 2 câu, nhóm baseline chính.
3. Mạng đồ thị/hình học — 1 câu; nếu D1 = Bỏ, thêm nửa câu giải thích không đưa vào so sánh.
4. Học đa nhiệm — 1–2 câu, dẫn GNINA 1.0 (VĐ5, có "theo hiểu biết của chúng tôi").

### III. Phương pháp đề xuất (~3.0 cột)

Nguồn duy nhất: `model_architecture.md` v2. 1 hình kiến trúc (ưu tiên vừa 1 cột). Backbone
(2–3 câu + phương trình), Pose Branch (Gated Fusion + Focal, RBF gộp 16 số), Affinity Branch
(Pocket-aware Attention **kèm định nghĩa `B_pocket`** — VĐ3, 27 token), hàm mất mát (1 phương
trình tổng + bảng trọng số nhỏ + phương trình $S = \{i : y_{\text{aff},i} > 0\}$ — VĐ2), 1 câu
tiền xử lý (z-score).

**Cắt:** lan truyền ngược/AdamW, ví dụ số minh họa gradient — như bản 3.

> ⚠️ Đối chiếu phụ lục 14 điểm cuối `model_architecture.md` trước khi nộp bản thảo Mục III.

### IV. Thực nghiệm (~4.2 cột)

**IV.1 Dữ liệu (~0.5 cột).** 1 câu CrossDocked2020 + số liệu (C=28, 48³, 0.5 Å, 23.5 Å), 1 câu
tăng cường (xoay + tịnh tiến ±1.0 Å, test không tăng cường). Không có bảng thống kê riêng —
gộp vào văn xuôi.

**IV.2 Thiết lập thực nghiệm (~0.6 cột).** Bốn khai báo, mỗi khai báo **1 câu**:
1. Siêu tham số dùng chung (VĐ6).
2. Cân bằng lớp ở mức minibatch.
3. Chọn mô hình bằng tập test (VĐ9) + 1 nửa câu về A2 nếu có.
4. Khác biệt `L_reg` (VĐ10) + 1 nửa câu về B1 nếu có.

**IV.3 Kết quả chính (~2.2 cột).** **Một bảng duy nhất** (gộp Bảng 6+7), có:
- Cột chính: 4–7 mô hình (tùy D1) × các chỉ số chính (BalAcc, PR-AUC, MAE, C-index — bỏ bớt cột
  phụ nếu bảng quá rộng cho 1 cột; cân nhắc xoay bảng ngang `\begin{table*}` nếu cần tràn 2 cột).
- **1–2 dòng phụ ở cuối bảng cho B1/B2** (không tách bảng riêng) — đây là cách xử lý ablation ở
  khổ 6 trang.
- Chú thích dưới bảng: N của tập con `y_aff > 0` (VĐ2), paired bootstrap CI cho Δ (VĐ4) — viết
  gọn trong 1 dòng chú thích, không phải đoạn văn.

**IV.4 Phân tích đánh đổi (~0.9 cột).** Hình 17 (Pareto) — hình trung tâm — + 2–3 câu phân tích.
*(Không còn Mục IV.4 riêng cho ablation — đã gộp vào IV.3 ở trên; đây là IV.4 mới = phân tích
Pareto, trước là IV.5 ở bản 3.)*

### V. Thảo luận & Hạn chế (~0.9 cột — 6 điểm, mỗi điểm 1 câu)

1. GNINA thắng MAE/C-index; nếu D0 = số thật, chênh lệch **lớn hơn** khóa luận mô tả — nêu đúng
   mức độ. Kèm 1 nửa câu về VĐ10 nếu có B1.
2. Chọn mô hình bằng tập test (VĐ9).
3. Chỉ 1 fold train/test.
4. Baseline đồ thị theo D1.
5. Chỉ số ái lực trên tập con `y_aff > 0` (VĐ2).
6. Nếu B2 xong: 1 câu về nguồn gốc Balanced Accuracy; nếu không: 1 câu thừa nhận chưa tách được
   đóng góp riêng từng thành phần.

### VI. Kết luận (~0.3 cột)

3–4 câu, không lặp số liệu.

---

## 6. Lịch 14 ngày

| Ngày | Việc | Người | Ghi chú |
|---|---|---|---|
| **1** | ⚠️ Kiểm tra CFP — **xác nhận khổ cột, cỡ chữ, ref có tính vào 6 trang không** | Đức | Quyết định lại ngân sách §4 nếu khác giả định |
| **1** | Kiểm tra điều kiện tiên quyết §0 | Đức | Đã có kết quả sơ bộ — xác nhận lại `data/` |
| **1** | VĐ11 — đã xong (xem §0); chuẩn bị trình bày phát hiện cho D0 | Đức | Mang ra họp sớm hơn dự kiến nếu cần |
| **1** | Q2: lấy N của tập con `y_aff > 0` | Đức | Dùng `check_label_affinity_sign.py` đã có sẵn |
| **1–2** | A2: script chọn epoch bằng train composite | Đức | Không cần GPU/`data/` |
| **1** | Đọc `model_architecture.md` v2 + `issues_and_fixes.md` v2 + phát hiện D0 | Cả 3 | Bắt buộc trước Họp 1 |
| **2** | **Họp 1** — chốt **D0**, D1, D2, D3, outline theo khổ 6 trang | Cả 3 | ~1 giờ, quan trọng hơn dự kiến do D0 |
| **2** | Kick off B1 (nếu `data/` đã có) | Đức | ~1 ngày GPU |
| **2–3** | Q4: chạy inference 7 mô hình → dự đoán từng mẫu | Đức | Cần `data/` |
| **2–3** | Mục I + Mục II (bản nháp, theo khung §2 đã chốt ở Họp 1) | Hiếu | Không viết trước khi D0 chốt |
| **2–3** | Khung Mục V + xử lý D1 | Phát | |
| **3** | Kick off B2 | Đức | ~1 ngày GPU |
| **3–4** | A1 (C-index chính xác, từ dự đoán Đ4) + paired bootstrap | Đức | |
| **3–5** | Mục III (Phương pháp) | Đức | Nguồn: `model_architecture.md` v2 |
| **4** | Check-in nhanh | Cả 3 | |
| **4–5** | Kết quả B1, B2 | Đức | |
| **4–5** | Mục V | Phát | |
| **5–6** | Dựng bảng kết quả cuối (1 bảng, có dòng B1/B2) | Phát | |
| **5–6** | Mục IV | Đức | |
| **6** | **Họp 2** — ghép 3 mục, rà số liệu, **kiểm tra tổng số cột đã dùng so với ngân sách §4** | Cả 3 | |
| **6** | Mục VI | Hiếu | |
| **7** | Vẽ lại Hình 3 (ưu tiên vừa 1 cột), xuất Hình 17 | Đức | |
| **8** | Ghép bản thảo đầy đủ lần 1 | Đức | |
| **9** | Đọc phản biện + **đo số trang thật sau khi format** | Phát, Hiếu | Nếu vượt 6 trang → kích hoạt cắt ở §7 |
| **9** | Họp 3 | Cả 3 | |
| **10** | Sửa theo góp ý chéo | Từng người | |
| **11** | Định dạng + tài liệu tham khảo | Hiếu | |
| **11** | Khớp số liệu | Phát | |
| **12** | Đọc lại lần cuối | Cả 3 | |
| **13** | Dự phòng | Cả 3 | |
| **14** | Nộp bài | Đức | |

---

## 7. Phương án rút gọn

### 7a. Nếu deadline thực tế < 14 ngày

| Còn | Cắt gì | Giữ bằng mọi giá |
|---|---|---|
| 10 ngày | Bỏ B1+B2 → đóng góp #3 hạ về suy đoán | VĐ11, VĐ9, VĐ10 (khai báo), VĐ2, VĐ3, A1, A2 |
| 7 ngày | Bỏ paired bootstrap | A1, A2, Mục V đầy đủ |
| 5 ngày | Bỏ 3 baseline đồ thị (D1 = Bỏ, bắt buộc) | VĐ11, A1, A2, Mục V |

### 7b. Nếu sau khi format thật, bài vượt quá 6 trang (mới — do khổ 2 cột dễ ước lượng sai)

Cắt theo thứ tự, dừng ngay khi vừa khổ:

1. Rút Mục II xuống 3–4 câu (bỏ câu về mạng đồ thị nếu D1 = Bỏ).
2. Bỏ dòng B1 khỏi bảng chính nếu chưa mang lại kết luận rõ ràng (giữ lại B2 vì trả lời trực
   tiếp câu hỏi nguồn gốc BalAcc).
3. Gộp Hình 3 và Hình 17 thành 1 hình 2 panel thay vì 2 hình riêng.
4. Rút Mục V xuống 4 điểm quan trọng nhất (bỏ điểm 3 và 6 nếu cần), **không bao giờ bỏ điểm 1
   và điểm 2** (VĐ10, VĐ9 — đây là hai điểm liêm chính khoa học then chốt nhất).
5. Rút danh mục tài liệu tham khảo còn ~12–14 mục thực sự được trích dẫn trực tiếp.

**Không bao giờ cắt:** phương trình $B_{\text{pocket}}$ (VĐ3), phương trình $S = \{y_{\text{aff}}>0\}$
(VĐ2), và ít nhất 1 câu về VĐ9 + VĐ10 ở Mục V.

---

## 8. Danh sách việc KHÔNG làm

- ❌ Không huấn luyện lại mô hình với kiến trúc mới, không đổi dữ liệu, không thêm baseline mới.
- ❌ Không mở rộng dataset hoặc thêm k-fold cross-validation.
- ❌ Không dịch sang tiếng Anh (trừ khi CFP bắt buộc).
- ❌ Không thêm nhiệm vụ thứ 3 (screening, lead optimization).
- ❌ Không sửa code training/loss để "vá" VĐ9 — chỉ báo cáo thêm phép chọn epoch thay thế (A2).
- ❌ **Không cố nhồi nội dung của bản kế hoạch "10–13 trang" đã bị hủy vào khổ 6 trang** — nếu
  thấy bản thảo dài hơn ngân sách §4, đó là tín hiệu cần cắt (§7b), không phải tín hiệu đổi khổ.

**✅ Ngoại lệ có kiểm soát — B1 và B2:** chỉ bật/tắt cờ dòng lệnh có sẵn, áp dụng cho đúng 1 mô
hình, không đổi kiến trúc/dữ liệu — là "ablation nhỏ" đúng như ngoại lệ đầu tài liệu này.

---

## 9. Checklist trước khi nộp

```
[ ] Đã xác nhận CFP: khổ cột, số trang thật, cỡ chữ, ngôn ngữ, ref có tính vào không
[ ] Đã kiểm tra điều kiện tiên quyết §0

🔴 CRITICAL
[ ] D0 — Đã chốt dùng số liệu thật hay số khóa luận; Mục I/IV/V nhất quán với lựa chọn này
[ ] VĐ11 — Bảng kết quả đã dùng số liệu một epoch nhất quán (không ghép nhiều epoch)
[ ] VĐ9  — Đã khai báo chọn mô hình bằng tập test ở Mục IV.2 VÀ Mục V
[ ] VĐ10 — Đã mô tả đúng khác biệt L_reg ở Mục IV.2 VÀ Mục V
[ ] VĐ1  — Đã xử lý 3 baseline đồ thị theo D1
[ ] VĐ2  — Có phương trình S = {i : y_aff > 0}; có N của tập con
[ ] VĐ3  — Có công thức B_pocket đúng; không còn diễn giải "spatial/khoảng cách"

🟡 HIGH
[ ] Đã đo số trang thật sau khi format — nếu > 6 trang, áp dụng §7b
[ ] A1 — C-index là giá trị đủ cặp, không xấp xỉ
[ ] A2 — Có đối chứng "epoch chọn bằng train" (dù chỉ nhắc 1 câu do giới hạn không gian)
[ ] VĐ12 — Mục III đã đối chiếu phụ lục 14 điểm
[ ] VĐ4  — Có paired bootstrap CI cho Δ (nếu đủ chỗ) hoặc ít nhất nêu trong văn bản
[ ] VĐ5  — Câu Related Work đã sửa, có "theo hiểu biết của chúng tôi"
[ ] VĐ6  — Có câu minh bạch siêu tham số
[ ] Ngôn ngữ đã hạ tông

🟢 MEDIUM
[ ] B1/B2 — dòng phụ trong bảng chính (không phải mục riêng)
[ ] Mục V có đủ 6 điểm, mỗi điểm 1 câu
[ ] Hình 3, Hình 17 vừa khổ cột đã chọn
[ ] Định dạng đúng template; tài liệu tham khảo đầy đủ, đúng số lượng cho phép
[ ] Mọi con số trong văn bản khớp chính xác với bảng
```
