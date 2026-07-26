# Kế hoạch Viết Bài báo – VNICT 2026
**Nguồn:** Khóa luận tốt nghiệp "Phát triển Mô hình Học sâu Đa nhiệm..."  
**Nguyên tắc:** Viết lại (rewrite), không phát triển thêm nội dung khoa học mới, trừ ablation nhỏ nếu còn thời gian.  
**Quỹ thời gian:** **14 ngày** cho toàn bộ công việc (từ ngày 1 đến ngày nộp).  
**Độ dài mục tiêu:** 6–8 trang (⚠️ **xác nhận CFP trong ngày 1** — nếu giới hạn là 4–6 trang, xem §6 để cắt).

> **Phiên bản 2** — cập nhật sau khi đối chiếu toàn bộ mã nguồn. Ba thay đổi lớn so với bản 1:
> 1. **VĐ2 và VĐ3 đã có lời giải** → tiết kiệm ~2 ngày của Đức ở đầu dự án.
> 2. **Ba vấn đề CRITICAL mới (VĐ9, VĐ10, VĐ11)** → cần thêm dung lượng cho Mục IV.2 và Mục V,
>    và VĐ11 nằm trên đường găng (chặn toàn bộ Mục IV).
> 3. **Danh sách đóng góp được định vị lại** — hai trong ba đóng góp của bản 1 không đứng vững
>    trước kiểm tra code (xem §1).

---

## 0. Đường găng (critical path) — đọc trước tiên

Chỉ có **ba** việc thực sự chặn tiến độ. Mọi thứ khác chạy song song được.

```
Ngày 1  ── VĐ11: đối chiếu lại Bảng 6 với summary.json ────┐
        ── Q2:  lấy N của tập con y_aff > 0                │
                                                            ▼
Ngày 2-3 ─ Q4: chạy inference lấy dự đoán từng mẫu ──► Ngày 3-4: paired bootstrap
                (7 mô hình, chỉ forward pass)                      │
                                                                   ▼
                                              Ngày 5-6: Phát dựng bảng kết quả cuối
                                                                   │
                                                                   ▼
                                              Ngày 5-6: Đức viết Mục IV
```

**Nếu VĐ11 phát hiện Bảng 6 sai thật** (4/7 dòng đang mâu thuẫn số học), mọi con số trong bài
phải lấy lại từ `summary.json` → **đây là việc phải xong trong ngày 1, không được lùi.**

Ngoài đường găng, chỉ có một việc cần GPU: chạy inference (ngày 2–3, rẻ) và ablation
(tùy chọn, chạy nền — xem §4).

---

## 1. Định vị câu chuyện của bài báo

### ❌ Không viết theo hướng

- *"Chúng tôi đề xuất mô hình tốt hơn tất cả"* — GNINA vẫn thắng MAE/C-index.
- *"Ba thành phần kiến trúc của chúng tôi đều đóng góp"* — kiểm tra code cho thấy Geometry
  Encoder chiếm **0.7%** tham số với tọa độ **không khả vi**, và `B_pocket` chiếm **0.06%** tham
  số và **không dùng thông tin không gian**. Không có bằng chứng thực nghiệm cho từng thành phần.
- *"Benchmark công bằng: cùng dữ liệu, cùng loss, cùng siêu tham số"* — **sai**: GNINA Dense và
  GNINA Default2018 dùng biến thể Gaussian NLL của `L_reg`, các mô hình còn lại dùng Huber
  (VĐ10).

### ✅ Viết theo hướng

> *"Trong bài toán scoring đa nhiệm cho docking, tối ưu ái lực liên kết và lọc cấu hình là hai
> mục tiêu có tính đánh đổi. Các mô hình hiện có thường nằm lệch về một cực. Chúng tôi xây dựng
> một khung so sánh thống nhất trên cùng biểu diễn voxel để định lượng sự đánh đổi này, và đề
> xuất một kiến trúc lai gọn nhẹ (1.59M tham số) đạt vị trí gần biên Pareto hơn — đặc biệt ở khả
> năng lọc cấu hình trên dữ liệu mất cân bằng lớp (PR-AUC 0.646, Balanced Accuracy 0.830) — đồng
> thời chỉ ra rằng vị trí của một mô hình trên biên Pareto chịu ảnh hưởng đáng kể từ cấu hình
> huấn luyện, không chỉ từ kiến trúc."*

Vế cuối là điểm mới của bản 2. Nó biến hai "điểm yếu" (VĐ10 khác hàm loss, và cân bằng lớp
minibatch) thành **một quan sát thực nghiệm có giá trị**, thay vì thứ phải giấu.

### Ba đóng góp (đã định vị lại — diễn đạt thận trọng)

1. **Kiến trúc đa nhiệm gọn nhẹ (~1.59M tham số)** tách nhánh sớm sau backbone 3D CNN dùng
   chung, kết hợp Transformer trên token voxel cho nhánh ái lực và đặc trưng hình học tường
   minh (RBF trên pseudo-atom) cho nhánh cấu hình.
2. **Khung so sánh thống nhất 7 kiến trúc** trên cùng biểu diễn voxel, định lượng sự đánh đổi
   pose–affinity; mô hình đề xuất đạt vị trí cân bằng gần biên Pareto.
3. **Phân tích các yếu tố ngoài kiến trúc ảnh hưởng tới vị trí Pareto**: dạng hàm hồi quy
   (Huber vs Gaussian NLL) và chiến lược cân bằng lớp ở mức minibatch.
   *→ Đóng góp #3 chỉ tuyên bố được nếu có ablation (§4). Nếu không kịp, thay bằng: "hiệu năng
   không tỉ lệ thuận với số tham số", kèm cảnh báo về `log_sigma_head` của GNINA Default2018.*

> ⚠️ Không viết "chưa có mô hình nào giải quyết đồng thời" — xem VĐ5. Cụm
> **"theo hiểu biết của chúng tôi"** là bắt buộc nếu vẫn muốn giữ một tuyên bố dạng khoảng trống.

---

## 2. Quyết định phải chốt trong Họp 1 (ngày 2)

Ba quyết định này ảnh hưởng tới cấu trúc bảng và dung lượng. Không chốt sớm sẽ phải viết lại.

| # | Quyết định | Khuyến nghị | Hệ quả |
|---|---|---|---|
| **D1** | VĐ1: giữ hay bỏ 3 baseline đồ thị? | **Bỏ** (Phương án B) | Bảng còn 4 dòng, gọn hơn ~0.3 trang, không phải giải thích BalAcc ≈ 0.5 |
| **D2** | Có chạy ablation không? | **Có, chạy nền** — nhưng viết bài với giả định KHÔNG có | Nếu kịp trước ngày 8 thì thêm vào; không thì bỏ đóng góp #3 |
| **D3** | Trình bày VĐ10 (khác hàm loss) ở đâu? | **Mục IV.2** (mô tả) + **Mục V** (thừa nhận confound) | Cần ~4 câu; đổi lại có được luận điểm cho đóng góp #3 |

---

## 3. Cấu trúc bài báo (6–8 trang)

| Mục | Tiêu đề | Trang | Nội dung chính | Thay đổi so với bản 1 |
|---|---|---|---|---|
| I | Giới thiệu | 0.75 | Bối cảnh, 2 nhiệm vụ, quan sát đánh đổi, 3 đóng góp | Đóng góp định vị lại (§1) |
| II | Công trình liên quan | 0.6 | CNN voxel / đồ thị / hình học / đa nhiệm, mỗi hướng 2–3 câu | Rút bớt 0.15 trang |
| III | Phương pháp đề xuất | 2.0 | Kiến trúc, backbone, 2 nhánh, hàm mất mát | **Dùng `model_architecture.md` v2** |
| IV | Thực nghiệm | 2.5 | Dữ liệu, thiết lập, bảng có CI, ablation, Pareto | **IV.2 dài thêm** (VĐ9, VĐ10) |
| V | Thảo luận & Hạn chế | **0.9** | Thừa nhận đầy đủ | **Tăng từ 0.5 → 0.9** |
| VI | Kết luận | 0.25 | 3–4 câu | không đổi |

Dung lượng tăng thêm ở IV.2 và V được bù bằng: bỏ 3 baseline (D1), rút gọn Mục II, và bỏ
Bảng 3 (28 kênh) xuống 1 câu.

---

## 4. Chi tiết từng mục

### I. Giới thiệu

**Giữ từ khóa luận:** đoạn mở đầu về ý nghĩa docking, hai nhiệm vụ, thách thức mất cân bằng lớp.

**Viết mới:** đặt quan sát về sự đánh đổi (Hình 17) **ngay trong Giới thiệu**, không chờ đến
Thảo luận. Đây là trục của toàn bài.

**Đóng góp:** theo §1, ba gạch đầu dòng.

### II. Công trình liên quan

Mỗi ý 2–3 câu, không khai triển như khóa luận:
- Phương pháp truyền thống (AutoDock Vina, GOLD) — 1 câu
- **CNN 3D voxel (GNINA, Pafnucy)** — 3 câu, đây là nhóm baseline chính của bài
- Mạng đồ thị / hình học (PotentialNet, EquiBind, TankBind) — 2 câu.
  *Nếu chọn D1 = Bỏ:* vẫn nhắc trong Related Work như một hướng nghiên cứu, kèm 1 câu giải thích
  vì sao không đưa vào so sánh (yêu cầu tọa độ nguyên tử, không tương thích thiết lập voxel
  thống nhất). Điều này **tốt hơn** là im lặng.
- Học đa nhiệm — 2 câu, dẫn GNINA 1.0 như tiền lệ gần nhất (VĐ5)

### III. Phương pháp đề xuất

**Nguồn duy nhất: `model_architecture.md` phiên bản 2.** Không dùng bản 1, không viết từ trí nhớ.

- 1 hình kiến trúc tổng quan (Hình 3 vẽ lại, dạng ngang)
- **Backbone:** 3–4 câu + `F_shared = Φ(X)`. Nhắc residual có nhánh chiếu tắt (§3.2 tài liệu KT).
- **Pose Branch:** Gated Fusion + Focal Loss. Nêu rõ Geometry Encoder gộp ma trận RBF thành
  16 số vô hướng (không phải message passing), và `d_ij` ở đơn vị lưới chuẩn hóa, không phải Å.
- **Affinity Branch:** công thức Pocket-aware Attention **kèm định nghĩa `B_pocket`** (VĐ3):
  $B^{(h)}[i,j] = \sigma(\mathbf{w}_h^\top \mathbf{x}_j + b_h)$. Nêu 3 tính chất: không dùng tọa
  độ, chỉ phụ thuộc key, miền (0,1). Nêu **27 token**.
- **Hàm mất mát:** 1 phương trình tổng + bảng trọng số + **phương trình tập mẫu
  $S = \{i : y_{\text{aff},i} > 0\}$** (VĐ2).
- **Tiền xử lý:** 1 câu về chuẩn hóa z-score target.

**Cắt bỏ hoàn toàn:** lan truyền ngược / AdamW (mục 2.3.4 khóa luận), ví dụ số minh họa gradient.

> ⚠️ **Không viết:** lịch ramp-up `L_rank` (không tồn tại trong code), `B_pocket` dạng khoảng
> cách 3D, Affine Calibrator như thành phần hoạt động, `L_dist` dạng bình phương.
> Đối chiếu với **phụ lục 14 điểm** ở cuối `model_architecture.md` trước khi nộp bản thảo Mục III.

### IV. Thực nghiệm

**IV.1 Dữ liệu (0.4 trang).** CrossDocked2020, bảng thống kê ngắn, 1 câu về 28 kênh (bỏ Bảng 3).
Giữ: C=28, 48³, 0.5 Å, 23.5 Å. Nêu tăng cường: xoay ngẫu nhiên + tịnh tiến ±1.0 Å, test không
tăng cường.

**IV.2 Thiết lập thực nghiệm (0.6 trang — mục dài nhất và quan trọng nhất về mặt liêm chính).**
Bốn khai báo bắt buộc:

1. **Siêu tham số dùng chung** cho cả 7 mô hình + câu minh bạch (VĐ6), kèm thừa nhận rằng cấu
   hình được tinh chỉnh cho mô hình đề xuất.
2. **Cân bằng lớp ở mức minibatch** (32 good / 32 bad) — không được bỏ qua.
3. **⚠️ Lựa chọn mô hình bằng tập test** (VĐ9), kèm câu giảm nhẹ: quy tắc áp dụng đồng nhất
   cho cả 7 mô hình nên so sánh tương đối vẫn có ý nghĩa.
4. **⚠️ Khác biệt về `L_reg`** giữa GNINA và các mô hình còn lại (VĐ10).

**IV.3 Kết quả chính (0.9 trang).** 1 bảng gộp (Bảng 6+7) kèm:
- **paired bootstrap CI** cho hiệu số Δ so với baseline mạnh nhất (VĐ4)
- **N của tập con `y_aff > 0`** ghi ngay dưới bảng (VĐ2)
- văn phong đã hạ tông

**IV.4 Ablation (0.3 trang — chỉ nếu kịp).** Bảng 3–4 dòng. Thứ tự ưu tiên chạy:
1. bỏ `--pose_balance_batch` ← **quan trọng nhất**
2. `--max_pseudo_atoms 0`
3. bỏ `B_pocket`

Mọi dòng phải **cùng số epoch và cùng quy tắc chọn mô hình**. Nếu không kịp → bỏ mục này và bỏ
đóng góp #3.

**IV.5 Phân tích đánh đổi (0.4 trang).** Hình 17 (Pareto) là hình trung tâm + 3–4 câu phân tích.

### V. Thảo luận & Hạn chế (0.9 trang)

Đây là nơi bài báo qua được vòng phản biện. **Sáu điểm bắt buộc**, mỗi điểm 1–2 câu:

1. GNINA Dense/Default2018 vượt trội về MAE và C-index. **Kèm giải thích VĐ10** — khoảng cách
   chịu ảnh hưởng của khác biệt dạng hàm hồi quy, không thuần túy là khác biệt kiến trúc.
2. **Chọn mô hình bằng early stopping trên tập test** (VĐ9) → giá trị tuyệt đối là ước lượng
   lạc quan.
3. Chỉ đánh giá trên **1 fold** train/test, chưa có k-fold.
4. *(Nếu D1 = Giữ)* 3 baseline đồ thị là bản cài đặt lại, tọa độ pseudo-atom không khả vi →
   không rút ra kết luận về họ mô hình đồ thị.
   *(Nếu D1 = Bỏ)* 1 câu giải thích lý do không đưa nhóm đồ thị vào so sánh.
5. Chỉ số ái lực tính trên **tập con** `y_aff > 0`, không phải toàn bộ tập test.
6. *(Nếu không có ablation)* Chưa tách được đóng góp riêng của từng thành phần kiến trúc.

### VI. Kết luận

3–4 câu, không lặp số liệu, nêu đóng góp cốt lõi và một hướng phát triển ngắn.

---

## 5. Lịch 14 ngày

| Ngày | Việc | Người | Ghi chú |
|---|---|---|---|
| **1** | ⚠️ Kiểm tra CFP (template, deadline thật, số trang, ngôn ngữ) | Đức | Nếu deadline < 14 ngày → §6 |
| **1** | ⚠️ **VĐ11: đối chiếu Bảng 6 với `summary.json`** | Đức | **Đường găng — không được lùi** |
| **1** | Q2: lấy N của tập con `y_aff > 0` | Đức | 15 phút |
| **1** | Đọc `model_architecture.md` v2 + `issues_and_fixes.md` v2 | Cả 3 | Bắt buộc trước Họp 1 |
| **2** | **Họp 1** — chốt D1/D2/D3, outline, phân công | Cả 3 | ~1 giờ |
| **2–3** | Q4: chạy inference 7 mô hình → dự đoán từng mẫu | Đức | GPU, chỉ forward pass |
| **2–3** | Mục I + Mục II (bản nháp) | Hiếu | Không phụ thuộc số liệu |
| **2–3** | Khung Mục V + soạn bảng baseline theo D1 | Phát | |
| **2–8** | *(Nếu D2 = Có)* Ablation chạy nền | Đức | Ưu tiên: no-balance → no-geo → no-Bpocket |
| **3** | Đo độ lệch chuẩn C-index (20 lần eval) | Đức | 1 giờ, VĐ4b |
| **3–4** | Paired bootstrap CI cho Δ | Đức | VĐ4a |
| **3–5** | **Mục III (Phương pháp)** | Đức | Nguồn: `model_architecture.md` v2 |
| **4** | **Check-in nhanh** — Đức báo VĐ11/VĐ4; Hiếu báo Mục I–II | Cả 3 | 20 phút |
| **4–5** | Mục V (Thảo luận & Hạn chế) — 6 điểm bắt buộc | Phát | Phụ thuộc kết quả VĐ11 |
| **5–6** | Dựng bảng kết quả cuối (gộp 6+7, tích hợp CI) | Phát | Phụ thuộc Đức ngày 3–4 |
| **5–6** | **Mục IV.1–IV.3, IV.5** | Đức | Phụ thuộc bảng của Phát |
| **6** | **Họp 2** — ghép 3 mục chính, rà nhất quán số liệu | Cả 3 | ~1 giờ |
| **6** | Mục VI (Kết luận) | Hiếu | |
| **7** | Vẽ lại Hình 3 (ngang), xuất Hình 17 độ phân giải cao | Đức | |
| **7** | *(Nếu kịp)* Bảng ablation IV.4 | Phát | Phụ thuộc ablation |
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

## 6. Phương án rút gọn nếu deadline thực tế < 14 ngày

Kiểm tra CFP là việc **đầu tiên** của ngày 1. Nếu quỹ thời gian thật ngắn hơn, cắt theo thứ tự này:

| Còn | Cắt gì | Giữ bằng mọi giá |
|---|---|---|
| **10 ngày** | Bỏ ablation (D2 = Không) → bỏ đóng góp #3; rút Hình 12 | VĐ11, VĐ9, VĐ10, VĐ2, VĐ3, paired bootstrap |
| **7 ngày** | Thêm: bỏ paired bootstrap, chỉ báo cáo giá trị 5 epoch cuối làm thước đo dao động (VĐ9 mục 3) | VĐ11, VĐ9, VĐ10, VĐ2, VĐ3, Mục V đầy đủ |
| **5 ngày** | Thêm: bỏ luôn 3 baseline đồ thị (D1 = Bỏ, bắt buộc), rút Mục II còn 0.4 trang | **VĐ11 và Mục V** — hai thứ không bao giờ được cắt |

**Nguyên tắc cắt:** ưu tiên giữ **tính trung thực** (VĐ9, VĐ10, Mục V) hơn **tính đầy đủ**
(ablation, CI). Một bài ngắn nhưng thành thật sẽ qua phản biện; một bài đầy đủ nhưng có bảng số
tự mâu thuẫn (VĐ11) hoặc giấu việc chọn mô hình trên test (VĐ9) thì không.

---

## 7. Danh sách việc KHÔNG làm

- ❌ Không huấn luyện lại mô hình với kiến trúc mới
- ❌ Không thêm baseline mới ngoài 7 mô hình đã có
- ❌ Không mở rộng dataset hoặc thêm k-fold cross-validation
- ❌ Không train lại chỉ để sửa VĐ10 (bật `--geoformer_uncertainty`) — chỉ khai báo bằng văn bản
- ❌ Không dịch sang tiếng Anh (trừ khi CFP bắt buộc)
- ❌ Không thêm nhiệm vụ thứ 3 (screening, lead optimization)
- ❌ **Không sửa code để "vá" VĐ9 hay VĐ10** — sửa xong sẽ phải train lại toàn bộ, vượt xa quỹ
  thời gian. Cách xử lý duy nhất trong 14 ngày là **khai báo trung thực**.

---

## 8. Checklist trước khi nộp

```
[ ] Đã xác nhận CFP: khuôn mẫu, số trang, deadline, ngôn ngữ

🔴 CRITICAL
[ ] VĐ11 — Bảng 6 đã đối chiếu với summary.json; kiểm tra Acc = π·Rpos + (1−π)·Rneg đúng cho cả 7 dòng
[ ] VĐ9  — Đã khai báo chọn mô hình bằng tập test ở Mục IV.2 VÀ Mục V
[ ] VĐ10 — Đã mô tả đúng khác biệt L_reg (Huber vs Gaussian NLL) ở Mục IV.2 VÀ Mục V
[ ] VĐ1  — Đã xử lý 3 baseline đồ thị theo quyết định D1
[ ] VĐ2  — Có phương trình S = {i : y_aff > 0}; có N của tập con dưới Bảng 7
[ ] VĐ3  — Có công thức B_pocket đúng; KHÔNG còn bất kỳ diễn giải "spatial/khoảng cách" nào

🟡 HIGH
[ ] VĐ12 — Mục III đã đối chiếu với phụ lục 14 điểm cuối model_architecture.md
[ ] VĐ4  — Có paired bootstrap CI cho Δ; đã kiểm tra độ lệch chuẩn C-index
[ ] VĐ5  — Câu Related Work đã sửa, có cụm "theo hiểu biết của chúng tôi"
[ ] VĐ6  — Có câu minh bạch siêu tham số + thừa nhận (a) lr chung, (b) AMP không đồng nhất
[ ] Ngôn ngữ đã hạ tông: không còn "áp đảo", "bứt phá", "không thể quy về nhiễu"

🟢 MEDIUM
[ ] Mục V có đủ 6 điểm bắt buộc
[ ] VĐ8 — Đã cắt nội dung thừa; Hình 3 vẽ lại dạng ngang; Hình 12 có chú thích về log_sigma_head
[ ] Định dạng đúng template; tài liệu tham khảo đầy đủ
[ ] Tên tác giả, đơn vị, người hướng dẫn chính xác
[ ] Mọi con số trong văn bản khớp chính xác với bảng
```
