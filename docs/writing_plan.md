# Kế hoạch Viết Bài báo – VNICT 2026
**Nguồn:** Khóa luận tốt nghiệp "Phát triển Mô hình Học sâu Đa nhiệm..."  
**Nguyên tắc:** Viết lại (rewrite), không phát triển thêm nội dung khoa học mới, trừ ablation nhỏ nếu còn thời gian.  
**Quỹ thời gian:** **14 ngày** cho toàn bộ công việc (từ ngày 1 đến ngày nộp).  
**Độ dài mục tiêu:** 6–8 trang (⚠️ **xác nhận CFP trong ngày 1** — nếu giới hạn là 4–6 trang, xem §6 để cắt).

> **Phiên bản 3** — tích hợp 4 nâng cấp rẻ (A1, A2, B1, B2) được đánh giá độ khó thực tế:
> - **A1 — C-index chính xác** (bỏ lấy mẫu ngẫu nhiên): chỉ là một script hậu xử lý ~15 dòng
>   chạy trên file dự đoán đã có, **không cần sửa code training, không cần chạy lại eval**.
>   → Đóng hẳn VĐ4b, xóa việc "đo 20 lần" khỏi lịch.
> - **A2 — chọn epoch không nhìn tập test**: ~1 giờ pandas trên `training_metrics_*.csv` đã có
>   sẵn, không cần GPU. → Biến VĐ9 từ "chỉ thú nhận" thành "thú nhận + có cột đối chứng".
> - **B1 — bật `--geoformer_uncertainty`**: đổi 1 dòng, train lại **1 mô hình** (~1 ngày GPU).
>   → Đóng VĐ10 bằng một dòng so sánh cùng dạng hàm mất mát với GNINA.
> - **B2 — ablation bỏ `--pose_balance_batch`**: đổi 0 dòng (bỏ 2 cờ), train lại **1 mô hình**
>   (~1 ngày GPU). → Trả lời câu hỏi quan trọng nhất: Balanced Accuracy 0.830 đến từ kiến trúc
>   hay từ thủ thuật huấn luyện.
>
> Tổng chi phí bốn nâng cấp: **~2 giờ code + 1–2 ngày GPU chạy nền** (không chặn việc viết).
> Đổi lại: đóng góp #3 (§1) chuyển từ "chỉ tuyên bố được nếu có ablation" sang **có bằng chứng
> thực nghiệm cụ thể**, và VĐ4b + VĐ10 đóng hẳn thay vì chỉ khai báo.

---

## 0. Điều kiện tiên quyết — kiểm tra trước khi lập lịch

**Trước khi Họp 1 diễn ra, Đức phải xác nhận:**

```
[ ] results/models/<model>/best_model.pt          còn cho cả 7 mô hình?
[ ] results/models/<model>/training_metrics_*.csv còn cho cả 7 mô hình?
[ ] results/models/<model>/summary.json           còn cho cả 7 mô hình?
[ ] data/ (PDBbind2016 + .types)                   còn nguyên, molgrid còn chạy được?
[ ] GPU còn truy cập được (để chạy B1, B2, và inference Đ4)?
```

- Nếu **thiếu checkpoint/CSV** → A1, A2, VĐ11 không thực hiện được như thiết kế; phải kích hoạt
  phương án rút gọn ở §6 ngay từ đầu.
- Nếu **thiếu `data/` hoặc GPU** → B1, B2, và Đ4 (inference) đều không chạy được — quay lại
  khai báo thuần văn bản cho VĐ9, VĐ10, và bỏ paired bootstrap (cần dự đoán từng mẫu).
- A2 là nâng cấp **duy nhất không phụ thuộc `data/`/GPU** — chỉ cần CSV đã có, luôn làm được.

---

## 1. Đường găng (critical path) — đọc trước tiên

Chỉ có **ba** việc thực sự chặn tiến độ toàn nhóm. A1/A2/B1/B2 chạy song song, không nằm trên
đường găng — nếu B1 hoặc B2 trễ, Mục IV vẫn ra được bằng phương án khai báo.

```
Ngày 1  ── VĐ11: đối chiếu lại Bảng 6 với summary.json ────┐
        ── Q2:  lấy N của tập con y_aff > 0                │
                                                            ▼
Ngày 2-3 ─ Q4: chạy inference lấy dự đoán từng mẫu ──► Ngày 3-4: paired bootstrap
                (7 mô hình, chỉ forward pass)          + A1 (C-index chính xác,
                                                          hậu xử lý từ dự đoán)
                                                                   │
                                                                   ▼
                                              Ngày 5-6: Phát dựng bảng kết quả cuối
                                                                   │
                                                                   ▼
                                              Ngày 5-6: Đức viết Mục IV
```

**Nhánh song song, không chặn đường găng:**

```
Ngày 1-2 ─ A2: script chọn epoch bằng train composite (thuần pandas, không GPU)
Ngày 2   ─ Kick off B1 (bật uncertainty) + B2 (bỏ pose_balance_batch) — chạy nền
Ngày 3-4 ─ Kết quả B1, B2 sẵn sàng (1 GPU: tuần tự ~2 ngày; ≥2 GPU: song song ~1 ngày)
                                                                   │
                                                                   ▼
                                    Nếu xong trước ngày 6: gộp vào bảng của Phát (P4)
                                    Nếu trễ: dùng khai báo văn bản, không chặn gì cả
```

**Nếu VĐ11 phát hiện Bảng 6 sai thật** (4/7 dòng đang mâu thuẫn số học), mọi con số trong bài
phải lấy lại từ `summary.json` → **đây là việc phải xong trong ngày 1, không được lùi.**

---

## 2. Định vị câu chuyện của bài báo

### ❌ Không viết theo hướng

- *"Chúng tôi đề xuất mô hình tốt hơn tất cả"* — GNINA vẫn thắng MAE/C-index.
- *"Ba thành phần kiến trúc của chúng tôi đều đóng góp"* — kiểm tra code cho thấy Geometry
  Encoder chiếm **0.7%** tham số với tọa độ **không khả vi**, và `B_pocket` chiếm **0.06%** tham
  số và **không dùng thông tin không gian**. Ablation B2 (nếu có) là bằng chứng thay thế tốt
  hơn nhiều so với suy luận từ tỉ trọng tham số.
- *"Benchmark hoàn toàn công bằng: cùng dữ liệu, cùng loss, cùng siêu tham số"* — **sai cho 6/7
  dòng**: GNINA Dense và GNINA Default2018 dùng biến thể Gaussian NLL của `L_reg`, các mô hình
  còn lại dùng Huber (VĐ10). **Ngay cả sau khi B1 chạy xong**, chỉ có 1 dòng GeoFormerDock
  (uncertainty) dùng cùng dạng loss với GNINA — 5 dòng còn lại vẫn khác. Không được khái quát
  thành "đã công bằng cho tất cả".

### ✅ Viết theo hướng

> *"Trong bài toán scoring đa nhiệm cho docking, tối ưu ái lực liên kết và lọc cấu hình là hai
> mục tiêu có tính đánh đổi. Các mô hình hiện có thường nằm lệch về một cực. Chúng tôi xây dựng
> một khung so sánh thống nhất trên cùng biểu diễn voxel để định lượng sự đánh đổi này, và đề
> xuất một kiến trúc lai gọn nhẹ (1.59M tham số) đạt vị trí gần biên Pareto hơn — đặc biệt ở khả
> năng lọc cấu hình trên dữ liệu mất cân bằng lớp (PR-AUC 0.646, Balanced Accuracy 0.830). Thực
> nghiệm bổ sung cho thấy vị trí trên biên Pareto chịu ảnh hưởng đáng kể từ cấu hình huấn luyện
> — không chỉ từ kiến trúc."*

Vế cuối là trục chính của bản 3. Với B1+B2, nó **không còn là suy đoán** mà có số liệu cụ thể:
B1 tách được phần chênh lệch MAE/C-index do khác hàm loss; B2 tách được phần Balanced Accuracy
đến từ cân bằng minibatch.

### Ba đóng góp (diễn đạt thận trọng, cập nhật theo B1/B2)

1. **Kiến trúc đa nhiệm gọn nhẹ (~1.59M tham số)** tách nhánh sớm sau backbone 3D CNN dùng
   chung, kết hợp Transformer trên token voxel cho nhánh ái lực và đặc trưng hình học tường
   minh (RBF trên pseudo-atom) cho nhánh cấu hình.
2. **Khung so sánh thống nhất 7 kiến trúc** trên cùng biểu diễn voxel, định lượng sự đánh đổi
   pose–affinity; mô hình đề xuất đạt vị trí cân bằng gần biên Pareto.
3. **Phân tách định lượng các yếu tố ngoài kiến trúc ảnh hưởng tới vị trí Pareto**: dạng hàm hồi
   quy (Huber vs Gaussian NLL, qua B1) và chiến lược cân bằng lớp ở mức minibatch (qua B2).
   *→ Nếu B1/B2 xong trước ngày 6: tuyên bố đầy đủ với số liệu. Nếu một trong hai trễ: hạ xuống
   còn phần đã có bằng chứng, phần còn lại chuyển thành "hiệu năng không tỉ lệ thuận với số tham
   số" kèm cảnh báo về `log_sigma_head`.*

> ⚠️ Không viết "chưa có mô hình nào giải quyết đồng thời" — xem VĐ5. Cụm
> **"theo hiểu biết của chúng tôi"** là bắt buộc nếu vẫn muốn giữ một tuyên bố dạng khoảng trống.

---

## 3. Quyết định phải chốt trong Họp 1 (ngày 2)

| # | Quyết định | Khuyến nghị | Hệ quả |
|---|---|---|---|
| **D1** | VĐ1: giữ hay bỏ 3 baseline đồ thị? | **Bỏ** (Phương án B) | Bảng còn 4 dòng gốc + tối đa 2 dòng phụ (B1/B2), gọn hơn ~0.3 trang |
| **D2** | Có chạy B1 (uncertainty) + B2 (no-balance) không? | **Có, cả hai** — chi phí thấp (~1–2 ngày GPU chạy nền, không chặn viết), giá trị cao (đóng VĐ10 + trả lời câu hỏi nguồn gốc BalAcc) | Kick off ngay đầu ngày 2. Nếu GPU/data không sẵn sàng (xem §0) → tự động rơi về khai báo văn bản, không mất gì thêm |
| **D3** | Trình bày VĐ10 ở đâu? | **Mục IV.2** (mô tả khác biệt loss) + **Mục IV.3** (nếu B1 xong, thêm dòng so sánh) + **Mục V** (thừa nhận confound còn lại ở 5/6 dòng khác) | ~4–6 câu |

**Lưu ý:** D2 = Có **không** đổi ràng buộc "chỉ viết lại, không phát triển thêm" — B1 chỉ bật
một cờ đã tồn tại trong code (`--geoformer_uncertainty`), B2 chỉ tắt hai cờ đã tồn tại
(`--pose_balance_batch`, `--pose_balance_target_per_class`). Không viết dòng code kiến trúc mới
nào, không đổi dữ liệu.

---

## 4. Cấu trúc bài báo (6–8 trang)

| Mục | Tiêu đề | Trang | Nội dung chính |
|---|---|---|---|
| I | Giới thiệu | 0.75 | Bối cảnh, 2 nhiệm vụ, quan sát đánh đổi, 3 đóng góp |
| II | Công trình liên quan | 0.6 | CNN voxel / đồ thị / hình học / đa nhiệm, mỗi hướng 2–3 câu |
| III | Phương pháp đề xuất | 2.0 | Kiến trúc, backbone, 2 nhánh, hàm mất mát |
| IV | Thực nghiệm | 2.6 | Dữ liệu, thiết lập, bảng có CI + cột A2 + dòng B1, ablation B2, Pareto |
| V | Thảo luận & Hạn chế | 0.9 | Thừa nhận đầy đủ, cập nhật theo B1/B2 |
| VI | Kết luận | 0.25 | 3–4 câu |

Mục IV tăng nhẹ (2.5 → 2.6 trang) do thêm cột A2 và dòng B1 trong bảng kết quả; bù bằng bỏ 3
baseline đồ thị (D1) và rút gọn Mục II.

---

## 5. Chi tiết từng mục

### I. Giới thiệu

**Giữ từ khóa luận:** đoạn mở đầu về ý nghĩa docking, hai nhiệm vụ, thách thức mất cân bằng lớp.

**Viết mới:** đặt quan sát về sự đánh đổi (Hình 17) **ngay trong Giới thiệu**. Ba đóng góp theo §2.

### II. Công trình liên quan

Mỗi ý 2–3 câu:
- Phương pháp truyền thống (AutoDock Vina, GOLD) — 1 câu
- **CNN 3D voxel (GNINA, Pafnucy)** — 3 câu, nhóm baseline chính
- Mạng đồ thị / hình học (PotentialNet, EquiBind, TankBind) — 2 câu.
  *Nếu D1 = Bỏ:* vẫn nhắc trong Related Work + 1 câu giải thích vì sao không đưa vào so sánh.
- Học đa nhiệm — 2 câu, dẫn GNINA 1.0 như tiền lệ gần nhất (VĐ5)

### III. Phương pháp đề xuất

**Nguồn duy nhất: `model_architecture.md` phiên bản 2.**

- 1 hình kiến trúc tổng quan (Hình 3 vẽ lại, dạng ngang)
- **Backbone:** 3–4 câu + `F_shared = Φ(X)`.
- **Pose Branch:** Gated Fusion + Focal Loss, RBF gộp thành 16 số vô hướng.
- **Affinity Branch:** công thức Pocket-aware Attention kèm định nghĩa `B_pocket` (VĐ3), 27 token.
- **Hàm mất mát:** 1 phương trình tổng + bảng trọng số + phương trình tập mẫu
  $S = \{i : y_{\text{aff},i} > 0\}$ (VĐ2).
- **Tiền xử lý:** 1 câu về chuẩn hóa z-score target.
- *(Tùy chọn, 1 câu cuối mục nếu B1 xong)*: "Một biến thể có nhánh ước lượng phương sai
  (Mục IV.3) được huấn luyện riêng để đối chiếu công bằng với hai baseline GNINA."

**Cắt bỏ hoàn toàn:** lan truyền ngược / AdamW, ví dụ số minh họa gradient.

> ⚠️ Đối chiếu với **phụ lục 14 điểm** ở cuối `model_architecture.md` trước khi nộp bản thảo Mục III.

### IV. Thực nghiệm

**IV.1 Dữ liệu (0.4 trang).** CrossDocked2020, bảng thống kê ngắn, 1 câu về 28 kênh. Giữ: C=28,
48³, 0.5 Å, 23.5 Å. Tăng cường: xoay ngẫu nhiên + tịnh tiến ±1.0 Å, test không tăng cường.

**IV.2 Thiết lập thực nghiệm (0.6 trang).** Bốn khai báo bắt buộc:

1. **Siêu tham số dùng chung** (VĐ6) + thừa nhận cấu hình tinh chỉnh cho mô hình đề xuất.
2. **Cân bằng lớp ở mức minibatch** (32 good / 32 bad).
3. **Lựa chọn mô hình bằng tập test** (VĐ9) — **kèm câu về A2**: *"Để kiểm tra ảnh hưởng của quy
   tắc này, chúng tôi bổ sung một phép chọn epoch thay thế dựa hoàn toàn trên tập huấn luyện
   (Mục IV.3)."*
4. **Khác biệt về `L_reg`** giữa GNINA và các mô hình còn lại (VĐ10) — **kèm câu về B1** nếu xong:
   *"Để đối chiếu công bằng hơn, chúng tôi huấn luyện thêm một biến thể của mô hình đề xuất với
   cùng dạng hàm hồi quy (Mục IV.3)."*

**IV.3 Kết quả chính (1.0 trang).** 1 bảng gộp (Bảng 6+7) kèm:
- **paired bootstrap CI** cho hiệu số Δ so với baseline mạnh nhất (VĐ4)
- **N của tập con `y_aff > 0`** ghi dưới bảng (VĐ2)
- **C-index tính đủ cặp** (A1) — không còn là giá trị xấp xỉ; bỏ mọi chú thích về nhiễu lấy mẫu
- **cột phụ "test @ epoch chọn bằng train"** cạnh cột "best-on-test" (A2), cho mọi mô hình; nếu
  thứ hạng giữ nguyên ở cả hai cột → câu kết luận về so sánh tương đối được củng cố
- **dòng phụ "GeoFormerDock (uncertainty)"** (B1, nếu xong) đặt ngay dưới dòng GeoFormerDock gốc,
  kèm chú thích so sánh trực tiếp với 2 dòng GNINA
- văn phong đã hạ tông

**IV.4 Ablation (0.4 trang).**
```
Mô hình A (full)       : cấu hình đầy đủ                → PR-AUC / BalAcc / MAE / C-index
Mô hình B (no-balance) : bỏ --pose_balance_batch  (B2)  → ...
```
B2 là dòng **cam kết chạy** (D2 = Có), không còn "chỉ nếu kịp" như bản trước. Nếu còn thời gian
GPU sau B1+B2 (không bắt buộc): thêm `--max_pseudo_atoms 0` (no-geo) và bỏ `B_pocket` (no-Bpocket)
— cùng số epoch và cùng quy tắc chọn mô hình với dòng full. Nếu B2 tự nó cũng trễ: bỏ mục này,
hạ đóng góp #3 xuống dạng phỏng đoán có cảnh báo (như bản trước khi có B1/B2).

**IV.5 Phân tích đánh đổi (0.4 trang).** Hình 17 (Pareto) là hình trung tâm + 3–4 câu phân tích.
Nếu có B1/B2: 1 câu chỉ ra hai điểm dữ liệu bổ sung này nằm ở đâu trên biểu đồ Pareto so với
dòng gốc.

### V. Thảo luận & Hạn chế (0.9 trang)

**Sáu điểm bắt buộc**, mỗi điểm 1–2 câu:

1. GNINA Dense/Default2018 vượt trội về MAE và C-index. **Nếu B1 xong:** *"Khi kiểm soát cùng
   dạng hàm hồi quy (biến thể uncertainty), khoảng cách với GNINA là ___; phần chênh lệch còn
   lại với 5 mô hình khác không kiểm soát được yếu tố này."* **Nếu B1 chưa xong:** giữ câu thừa
   nhận confound thuần văn bản như bản 2.
2. **Chọn mô hình bằng early stopping trên tập test** (VĐ9) → giá trị tuyệt đối là ước lượng lạc
   quan. **Nếu A2 xong:** thêm 1 câu về mức chênh lệch giữa hai cách chọn epoch và việc thứ hạng
   có đổi hay không.
3. Chỉ đánh giá trên **1 fold** train/test, chưa có k-fold.
4. *(Theo D1)* baseline đồ thị hoặc lý do không đưa vào so sánh.
5. Chỉ số ái lực tính trên **tập con** `y_aff > 0`, không phải toàn bộ tập test.
6. **Nếu B2 xong:** *"Phần chênh lệch Balanced Accuracy quy về cân bằng lớp ở mức minibatch là
   ___; phần còn lại (nếu có) quy về kiến trúc."* **Nếu chưa:** giữ câu thừa nhận chưa tách được
   đóng góp riêng của từng thành phần.

### VI. Kết luận

3–4 câu, không lặp số liệu, nêu đóng góp cốt lõi và một hướng phát triển ngắn.

---

## 6. Lịch 14 ngày

| Ngày | Việc | Người | Ghi chú |
|---|---|---|---|
| **1** | ⚠️ Kiểm tra CFP (template, deadline thật, số trang, ngôn ngữ) | Đức | Nếu deadline < 14 ngày → §7 |
| **1** | ⚠️ **Kiểm tra điều kiện tiên quyết §0** (checkpoint, data/, GPU) | Đức | Gate cho toàn bộ lịch dưới đây |
| **1** | ⚠️ **VĐ11: đối chiếu Bảng 6 với `summary.json`** | Đức | **Đường găng — không được lùi** |
| **1** | Q2: lấy N của tập con `y_aff > 0` | Đức | 15 phút |
| **1–2** | **A2: script chọn epoch bằng train composite** | Đức | Thuần pandas, không GPU, không phụ thuộc gì khác |
| **1** | Đọc `model_architecture.md` v2 + `issues_and_fixes.md` v2 | Cả 3 | Bắt buộc trước Họp 1 |
| **2** | **Họp 1** — chốt D1/D2/D3, outline, phân công | Cả 3 | ~1 giờ |
| **2** | **Kick off B1** (sửa 1 dòng, bật `--geoformer_uncertainty`, chạy nền) | Đức | ~1 ngày GPU |
| **2–3** | Q4: chạy inference 7 mô hình → dự đoán từng mẫu | Đức | GPU, chỉ forward pass |
| **2–3** | Mục I + Mục II (bản nháp) | Hiếu | Không phụ thuộc số liệu |
| **2–3** | Khung Mục V + soạn bảng baseline theo D1 | Phát | |
| **3** | **Kick off B2** (bỏ 2 cờ, chạy nền) — sau khi B1 xong nếu 1 GPU, song song nếu ≥2 GPU | Đức | ~1 ngày GPU |
| **3–4** | **A1: tính C-index chính xác** từ file dự đoán của Đ4 (script offline, không cần rerun eval) | Đức | ~15 phút, phụ thuộc Đ4 |
| **3–4** | Paired bootstrap CI cho Δ | Đức | Phụ thuộc Đ4 |
| **3–5** | **Mục III (Phương pháp)** | Đức | Nguồn: `model_architecture.md` v2 |
| **4** | **Check-in nhanh** — Đức báo VĐ11/A1/A2; Hiếu báo Mục I–II | Cả 3 | 20 phút |
| **4** | Kết quả B1 sẵn sàng (nếu kịp) | Đức | Thu thập `summary.json` mới |
| **4–5** | Kết quả B2 sẵn sàng (nếu kịp) | Đức | Thu thập `summary.json` mới |
| **4–5** | Mục V (Thảo luận & Hạn chế) — 6 điểm bắt buộc | Phát | Phụ thuộc VĐ11 + B1/B2 (nếu có) |
| **5–6** | Dựng bảng kết quả cuối (gộp 6+7, tích hợp CI + cột A2 + dòng B1 + hàng B2) | Phát | Phụ thuộc Đức ngày 3–5 |
| **5–6** | **Mục IV.1–IV.3, IV.5** | Đức | Phụ thuộc bảng của Phát |
| **6** | **Họp 2** — ghép 3 mục chính, rà nhất quán số liệu | Cả 3 | ~1 giờ |
| **6** | Mục VI (Kết luận) | Hiếu | |
| **7** | Vẽ lại Hình 3 (ngang), xuất Hình 17 độ phân giải cao | Đức | |
| **7** | Bảng ablation IV.4 (B2 + tùy chọn no-geo/no-Bpocket nếu còn GPU) | Phát | |
| **8** | **Ghép bản thảo đầy đủ lần 1** | Đức (chủ biên) | |
| **9** | Đọc phản biện: đối chiếu từng dòng với checklist CRITICAL/HIGH | Phát | Gồm phụ lục 14 điểm |
| **9** | Đọc phản biện Mục III về tính dễ hiểu | Hiếu | |
| **9** | **Họp 3** — thống nhất sửa đổi | Cả 3 | |
| **10** | Sửa theo góp ý chéo | Từng người | |
| **11** | Định dạng theo template + danh mục tài liệu tham khảo | Hiếu | |
| **11** | Kiểm tra khớp số liệu văn bản ↔ bảng | Phát | |
| **12** | Đọc lại lần cuối, rà chính tả/ngữ pháp | Cả 3 → Phát chốt | |
| **13** | **Dự phòng** | Cả 3 | Giữ trống có chủ đích |
| **14** | Nộp bài | Đức | |

---

## 7. Phương án rút gọn nếu deadline thực tế < 14 ngày

| Còn | Cắt gì | Giữ bằng mọi giá |
|---|---|---|
| **10 ngày** | **Cắt B1 + B2 trước tiên** (GPU-dependent, đắt nhất) → đóng góp #3 hạ về suy đoán có cảnh báo; bỏ Hình 12 | VĐ11, VĐ9, VĐ10 (khai báo), VĐ2, VĐ3, **A1, A2** (rẻ, không GPU), paired bootstrap |
| **7 ngày** | Thêm: bỏ paired bootstrap, chỉ giữ A2 làm thước đo dao động duy nhất | VĐ11, VĐ9, VĐ10, VĐ2, VĐ3, A1, A2, Mục V đầy đủ |
| **5 ngày** | Thêm: bỏ 3 baseline đồ thị (D1 = Bỏ, bắt buộc), rút Mục II còn 0.4 trang | **VĐ11, A1, A2, và Mục V** — không bao giờ cắt (A1/A2 gần như miễn phí) |

**Nguyên tắc cắt:** B1/B2 là hai nâng cấp duy nhất cần GPU + thời gian chờ, nên là ứng viên cắt
đầu tiên khi thời gian eo hẹp. A1 và A2 gần như miễn phí (không GPU, dưới 2 giờ tổng cộng) nên
**giữ ở mọi kịch bản**, kể cả 5 ngày. Ưu tiên giữ **tính trung thực** (VĐ9, VĐ10, Mục V) hơn
**tính đầy đủ** (ablation mở rộng, B1/B2).

---

## 8. Danh sách việc KHÔNG làm

- ❌ Không huấn luyện lại mô hình với kiến trúc mới, không đổi dữ liệu, không thêm baseline mới.
- ❌ Không mở rộng dataset hoặc thêm k-fold cross-validation.
- ❌ Không dịch sang tiếng Anh (trừ khi CFP bắt buộc).
- ❌ Không thêm nhiệm vụ thứ 3 (screening, lead optimization).
- ❌ Không sửa code training/loss để "vá" VĐ9 — quy tắc chọn mô hình giữ nguyên, chỉ **báo cáo
  thêm** một phép chọn thay thế (A2) mà không đổi checkpoint đã có.

**✅ Ngoại lệ có kiểm soát — B1 và B2 KHÔNG vi phạm nguyên tắc trên:**
Cả hai chỉ bật/tắt cờ dòng lệnh đã tồn tại sẵn trong `training.py`/`run_training.sh`
(`--geoformer_uncertainty`, `--pose_balance_batch`), áp dụng cho **đúng 1 mô hình**
(GeoFormerDock), không đổi kiến trúc, không đổi dữ liệu, không đổi mô hình khác. Đây là "ablation
nhỏ" đúng như ngoại lệ được nêu ở đầu tài liệu này.

---

## 9. Checklist trước khi nộp

```
[ ] Đã xác nhận CFP: khuôn mẫu, số trang, deadline, ngôn ngữ
[ ] Đã kiểm tra điều kiện tiên quyết §0 (checkpoint, data/, GPU)

🔴 CRITICAL
[ ] VĐ11 — Bảng 6 đã đối chiếu với summary.json; kiểm tra Acc = π·Rpos + (1−π)·Rneg đúng cho cả 7 dòng
[ ] VĐ9  — Đã khai báo chọn mô hình bằng tập test ở Mục IV.2 VÀ Mục V
[ ] VĐ10 — Đã mô tả đúng khác biệt L_reg (Huber vs Gaussian NLL) ở Mục IV.2 VÀ Mục V
[ ] VĐ1  — Đã xử lý 3 baseline đồ thị theo quyết định D1
[ ] VĐ2  — Có phương trình S = {i : y_aff > 0}; có N của tập con dưới Bảng 7
[ ] VĐ3  — Có công thức B_pocket đúng; KHÔNG còn bất kỳ diễn giải "spatial/khoảng cách" nào

🟡 HIGH
[ ] A1 — C-index trong bảng là giá trị đủ cặp, không phải xấp xỉ ngẫu nhiên
[ ] A2 — Có cột "test @ epoch chọn bằng train" cạnh cột "best-on-test" cho cả 7 mô hình
[ ] VĐ12 — Mục III đã đối chiếu với phụ lục 14 điểm cuối model_architecture.md
[ ] VĐ4  — Có paired bootstrap CI cho Δ
[ ] VĐ5  — Câu Related Work đã sửa, có cụm "theo hiểu biết của chúng tôi"
[ ] VĐ6  — Có câu minh bạch siêu tham số + thừa nhận (a) lr chung, (b) AMP không đồng nhất
[ ] Ngôn ngữ đã hạ tông: không còn "áp đảo", "bứt phá", "không thể quy về nhiễu"

🟢 MEDIUM
[ ] B1 — (nếu hoàn thành) dòng GeoFormerDock (uncertainty) trong Bảng kết quả + Mục V mục 1;
        (nếu không) đã khai báo confound thuần văn bản
[ ] B2 — (nếu hoàn thành) bảng ablation IV.4 + Mục V mục 6; (nếu không) đã khai báo giới hạn
[ ] Mục V có đủ 6 điểm bắt buộc
[ ] VĐ8 — Đã cắt nội dung thừa; Hình 3 vẽ lại dạng ngang; Hình 12 có chú thích về log_sigma_head
[ ] Định dạng đúng template; tài liệu tham khảo đầy đủ
[ ] Tên tác giả, đơn vị, người hướng dẫn chính xác
[ ] Mọi con số trong văn bản khớp chính xác với bảng
```
