# Phân công Chi tiết – Nhóm Viết Bài báo VNICT 2026
**Thành viên:** Đức, Hiếu, Phát  
**Quỹ thời gian:** **14 ngày** cho toàn bộ công việc  
**Độ dài mục tiêu:** **~6 trang, khổ hai cột kiểu hội nghị quốc tế** (đính chính — chặt hơn "6–8
trang" ở bản trước, xem `writing_plan.md` v4 §4 để biết ngân sách theo cột).  
**Vai trò tổng quát:** Đức là tác giả khóa luận gốc, nắm toàn bộ code và số liệu → đầu mối kỹ thuật và tác giả liên hệ. Hiếu và Phát hỗ trợ viết, phản biện chéo, và xử lý các phần không đòi hỏi truy cập code/model.

> **Phiên bản 4** — đính chính độ dài (6 trang hai cột) + cập nhật theo phát hiện thực tế từ
> `tools/inspect_checkpoints.py`. Thay đổi so với v3:
> - **D0 (mới, ưu tiên cao nhất):** dùng số liệu thật (checkpoint, tự nhất quán) hay số khóa luận
>   (ghép nhiều epoch)? Đã xác nhận Bảng 6/7 khóa luận không tái lập được từ bất kỳ checkpoint
>   nào — xem `writing_plan.md` §0. Cần chốt **trước** khi Hiếu viết Mục I.
> - **P5 (bảng ablation riêng) bị loại bỏ** — khổ 6 trang hai cột không đủ chỗ cho một mục con
>   ablation riêng; B2 gộp thành 1–2 dòng phụ trong bảng chính (P4).
> - Các mô tả "~0.75 trang", "~0.6 trang" trong v3 được thay bằng ngân sách theo cột.
> - Tài liệu tham chiếu bắt buộc: **`model_architecture.md` v2**, **`issues_and_fixes.md` v2**,
>   **`writing_plan.md` v4**.

---

## Nguyên tắc phân công

1. Việc nào cần chạy code / truy cập model / dữ liệu gốc → **bắt buộc Đức**.
2. Việc viết văn bản, tổng hợp, trình bày, định dạng → chia đều cho cả 3.
3. Mỗi phần có **1 người viết chính** và **1 người đọc phản biện** (không phải người viết).
4. **Ngày 1, cả 3 người đọc `model_architecture.md` v2, `issues_and_fixes.md` v2, VÀ phần §0
   của `writing_plan.md` v4 (phát hiện checkpoint) trước Họp 1.**

---

## Điều kiện tiên quyết — trạng thái đã xác minh

```
[x] results/models/<model>/best_model.pt          — có cho 6/7 mô hình (thiếu equibind)
[x] results/models/<model>/training_metrics_*.csv — có cho 6/7 mô hình
[x] results/models/<model>/summary.json           — có, nhưng KHÔNG đáng tin (xem D0)
[ ] data/ (PDBbind2016 + .types)                   — CHƯA có, đang xin theo scripts/00_download.sh
[ ] GPU (Kaggle / JetBrains Cadence)                — chưa test molgrid trên môi trường này
```

---

## Phát hiện cần mang ra Họp 1 (quan trọng hơn cả D1/D2/D3 cũ)

`tools/inspect_checkpoints.py` đã chạy trên `results/` thật (không phải giả định) và cho thấy:

1. **Lỗi code thật** trong `training.py::log_test_results` (thiếu `nonlocal best_epoch,
   best_metrics`) khiến `summary.json` của cả 6 mô hình luôn là **epoch cuối**, không phải epoch
   tốt nhất theo early stopping.
2. **Bảng 6/7 khóa luận (dòng GeoFormerDock) ghép từ 5 epoch khác nhau** (48, 58, 68, 92, 94) —
   đã xác định chính xác từng epoch cho từng cột, không còn là nghi vấn.
3. **Số liệu thật, tự nhất quán** (checkpoint epoch 94, kiểm tra `Acc = π·Rpos + (1−π)·Rneg` khớp
   đúng): Balanced Accuracy 0.8295 và PR-AUC 0.6306 **vẫn cao nhất**; nhưng MAE 1.221 và C-index
   0.771 **kém hơn cả Pafnucy** (MAE 1.149, C-index 0.785) — khác với câu chuyện "gần biên
   Pareto, không hy sinh gì" hiện có trong khóa luận và trong bản kế hoạch v3.

**→ Quyết định D0 phải chốt trước khi bất kỳ ai viết Mục I, IV, hoặc V.**

---

## Ba việc chặn tiến độ (đường găng)

| Việc | Ai | Hạn | Chặn cái gì |
|---|---|---|---|
| **D0 — chốt số liệu thật vs. số khóa luận** | Cả 3 (Họp 1) | Ngày 2 | Mục I, IV, V — viết trước khi chốt sẽ phải viết lại |
| **Q4 — chạy inference lấy dự đoán từng mẫu** | Đức | Hết ngày 3 | A1, paired bootstrap → bảng kết quả → Mục IV |
| **Bảng kết quả cuối** | Phát | Hết ngày 6 | Đức viết Mục IV |

---

## PHÂN CÔNG ĐỨC

**Vai trò:** Đầu mối kỹ thuật, tác giả liên hệ, chủ biên nội dung Phương pháp + Thực nghiệm.

| # | Việc | Input | Output | Ưu tiên | Hạn |
|---|---|---|---|---|---|
| **Đ1** | ⚠️ Kiểm tra CFP — **đặc biệt xác nhận: khổ 1 hay 2 cột, cỡ chữ, tài liệu tham khảo có tính vào 6 trang không** | Trang web hội thảo | Thông báo nhóm; điều chỉnh ngân sách §4 `writing_plan.md` nếu khác giả định | 🔴 | Ngày 1 |
| **Đ2** | Trình bày lại phát hiện từ `tools/inspect_checkpoints.py` cho Hiếu/Phát trước Họp 1 | `results/logs/best_checkpoint_audit.tsv` | Cả 3 hiểu rõ vấn đề D0 trước khi họp | 🔴 | Ngày 1 |
| **Đ3** | **Q2 — Lấy N của tập con `y_aff > 0`** trên tập test | `results/data_plots/check_label_affinity_sign.py` + CSV đã có | 1 con số | 🔴 | Ngày 1 |
| **Đ4** | **A2 — Chọn epoch bằng train composite.** Từ `training_metrics_train.csv`, lấy epoch có `0.5·C-index + 0.5·BalAcc` lớn nhất trên **train**; tra giá trị **test** tại epoch đó | `training_metrics_train/test.csv` × 6 | Cột đối chứng cho bảng kết quả | 🟡 | Ngày 1–2 |
| **Đ5** | **Kick off B1** (bật `--geoformer_uncertainty`, model riêng, output riêng — KHÔNG ghi đè `results/models/geoformerdock`) | `data/` (khi có) + GPU | `results/models/geoformerdock_uncertainty/` | 🟡 | Sau khi có `data/`, ~1 ngày GPU |
| **Đ6** | **Kick off B2** (bỏ `--pose_balance_batch`, output riêng) | `data/` (khi có) + GPU | `results/models/geoformerdock_nobalance/` | 🟡 | Sau khi có `data/`, ~1 ngày GPU |
| **Đ7** | **Q4 — Chạy inference 6 mô hình** từ `best_model.pt` trên tập test, xuất dự đoán từng mẫu ra CSV | `data/`, `best_model.pt` × 6 | 6 file `predictions_<model>.csv` | 🔴 | Sau khi có `data/` |
| **Đ8** | **A1 — C-index chính xác** (đủ cặp, không lấy mẫu) từ file dự đoán của Đ7 | CSV từ Đ7 | C-index chính xác cho 6 mô hình | 🟡 | Ngay sau Đ7 |
| **Đ9** | **Paired bootstrap CI** cho hiệu số Δ | CSV từ Đ7 | Bảng CI (nếu đủ chỗ trong khổ 6 trang — xem §7b `writing_plan.md`) | 🟡 | Sau Đ7 |
| **Đ10** | **Viết Mục III – Phương pháp đề xuất** (~3.0 cột). Nguồn duy nhất `model_architecture.md` v2 | `model_architecture.md` v2 | Bản thảo Mục III | 🔴 | Ngày 5 |
| **Đ11** | **Viết Mục IV** (~4.2 cột). IV.2: 4 khai báo, mỗi khai báo 1 câu. IV.3: 1 bảng duy nhất kèm dòng B1/B2. IV.4: phân tích Pareto | Bảng của Phát (P4) | Bản thảo Mục IV | 🔴 | Ngày 6 |
| **Đ12** | Vẽ lại Hình 3 (ưu tiên vừa 1 cột), xuất Hình 17 (Pareto) độ phân giải in ấn | Hình gốc khóa luận | File .png/.pdf | 🟡 | Ngày 7 |
| **Đ13** | **Ghép bản thảo đầy đủ lần 1** (chủ biên) | 6 mục từ 3 người | Bản thảo v1 | 🔴 | Ngày 8 |
| **Đ14** | Tác giả liên hệ: nộp bài, theo dõi trạng thái | — | — | 🔴 | Ngày 14+ |

### ✅ Việc đã HOÀN THÀNH

- ~~*`L_reg` tính trên tập nào*~~ → **Đáp án: (B)**, chỉ mẫu `y_aff > 0` (VĐ2).
- ~~*Công thức `B_pocket`*~~ → `B^(h)[i,j] = σ(w_hᵀ x_j + b_h)` (VĐ3).
- ~~*Siêu tham số dùng chung*~~ → Có, dùng chung (VĐ6).
- ~~*VĐ11 — đối chiếu Bảng 6*~~ → **Xong, và phát hiện sâu hơn dự kiến** (xem phần "Phát hiện cần mang ra Họp 1" ở trên). Không cần làm lại — chỉ cần D0 quyết định cách xử lý.

### Deadline nội bộ cho Đức

| Hạn | Việc |
|---|---|
| **Hết ngày 1** | Đ1, Đ2, Đ3 |
| Hết ngày 2 | Đ4, kick off Đ5 (nếu có `data/`) |
| Hết ngày 3 | Kick off Đ6, Đ7 (nếu có `data/`) |
| Hết ngày 4 | Đ8, Đ9 |
| Hết ngày 5 | Đ10 (bản nháp Mục III) |
| Hết ngày 6 | Đ11 (bản nháp Mục IV) |
| Hết ngày 7 | Đ12 |
| Hết ngày 8 | Đ13 |

---

## PHÂN CÔNG HIẾU

**Vai trò:** Viết Giới thiệu / Liên quan / Kết luận, kiểm soát văn phong, định dạng.

| # | Việc | Input | Output | Ưu tiên | Hạn |
|---|---|---|---|---|---|
| **H1** | **Viết Mục I – Giới thiệu** (~1.0 cột, ~450–500 từ). ⚠️ **Không viết trước khi D0 chốt ở Họp 1** — câu chuyện đánh đổi phụ thuộc trực tiếp vào việc dùng số liệu nào | `writing_plan.md` §2 (bản đã chốt D0) | Bản thảo Mục I | 🔴 | Ngày 3 |
| **H2** | **VĐ5 — Sửa câu overclaim** Related Work, có "theo hiểu biết của chúng tôi" | `issues_and_fixes.md` VĐ5 | Đoạn đã sửa | 🔴 | Ngày 3 |
| **H3** | **Viết Mục II** (~0.6 cột — **nén còn 5–6 câu liền mạch**, không tách đoạn theo nhóm như bản cũ) | Khóa luận mục 1.3 | Bản thảo Mục II | 🔴 | Ngày 3 |
| **H4** | **Viết Mục VI – Kết luận** (~0.3 cột, 3–4 câu) | Khóa luận phần Kết luận | Bản thảo Mục VI | 🟡 | Ngày 6 |
| **H5** | **Đọc phản biện Mục III** — đối chiếu phụ lục 14 điểm cuối `model_architecture.md` v2 | Bản thảo Đ10 | Danh sách lỗi | 🔴 | Ngày 9 |
| **H6** | Kiểm soát văn phong toàn bài | Bản thảo ghép v1 | Bản thảo đã hiệu đính | 🟡 | Ngày 10 |
| **H7** | **Định dạng theo template hội thảo — đo số trang thật.** Nếu > 6 trang, áp dụng `writing_plan.md` §7b | Bản thảo + template CFP | File đúng chuẩn, đúng số trang | 🔴 | Ngày 11 |
| **H8** | Danh mục tài liệu tham khảo (~1.4 cột, ~15–18 mục — **ít hơn v3** do khổ hẹp) | Danh mục khóa luận | File .bib | 🟡 | Ngày 11 |

### Deadline nội bộ cho Hiếu

| Hạn | Việc |
|---|---|
| Hết ngày 3 | H1 (chỉ sau khi D0 chốt), H2, H3 |
| Hết ngày 6 | H4 |
| Hết ngày 9 | H5 |
| Hết ngày 10 | H6 |
| Hết ngày 11 | H7, H8 |

---

## PHÂN CÔNG PHÁT

**Vai trò:** Viết Thảo luận / Hạn chế, xử lý bảng biểu, đọc phản biện tổng thể, rà soát cuối.

| # | Việc | Input | Output | Ưu tiên | Hạn |
|---|---|---|---|---|---|
| **P1** | **Soạn khung Mục V** (~0.9 cột — 6 điểm, **mỗi điểm đúng 1 câu**). Chờ D0 để biết mức độ mạnh/yếu của điểm 1 | `issues_and_fixes.md` v2 + D0 | Khung Mục V | 🔴 | Ngày 3 |
| **P2** | Xử lý VĐ1 theo D1 | `issues_and_fixes.md` VĐ1 | Đoạn đã xử lý | 🔴 | Ngày 4 |
| **P3** | Soạn đoạn IV.2 (4 khai báo, mỗi khai báo 1 câu — VĐ6, cân bằng minibatch, VĐ9, VĐ10) | `issues_and_fixes.md` VĐ6/9/10 | 1 đoạn ngắn (~0.6 cột) | 🔴 | Ngày 4 |
| **P4** | **Dựng bảng kết quả chính — MỘT bảng duy nhất.** Gộp Bảng 6+7, thêm **1–2 dòng phụ cho B1/B2 ở cuối bảng** (không tách bảng ablation riêng — không đủ chỗ ở khổ 6 trang), chú thích N tập con + CI dưới bảng | Đ3, Đ4, Đ8, Đ9, D0 | Bảng hoàn chỉnh cho Mục IV | 🔴 | Ngày 6 |
| **P5** | ~~Bảng ablation riêng~~ **— ĐÃ LOẠI BỎ khỏi phân công.** B2 nay là một phần của P4, không phải việc riêng | — | — | — | — |
| **P6** | **Viết hoàn chỉnh Mục V** từ khung P1 | Khung P1 + bảng P4 | Bản thảo Mục V | 🔴 | Ngày 6 |
| **P7** | Đọc phản biện bản thảo ghép lần 1 | Bản thảo v1 (Đ13) | Danh sách lỗi | 🔴 | Ngày 9 |
| **P8** | Kiểm tra khớp số liệu + chạy lại `Acc = π·Rpos + (1−π)·Rneg` trên bảng cuối | Bản thảo hoàn chỉnh | Danh sách sai lệch | 🔴 | Ngày 11 |
| **P9** | Rà soát chính tả, ngữ pháp | Bản thảo cuối | Bản thảo sạch lỗi | 🔴 | Ngày 12 |

### Deadline nội bộ cho Phát

| Hạn | Việc |
|---|---|
| Hết ngày 3 | P1 |
| Hết ngày 4 | P2, P3 |
| Hết ngày 6 | **P4 (chặn Đ11)**, P6 |
| Hết ngày 9 | P7 |
| Hết ngày 11 | P8 |
| Hết ngày 12 | P9 |

---

## Ma trận Phụ thuộc Công việc

```
NGÀY 1 ─────────────────────────────────────────────────────────────
  Đ1 (CFP + khổ cột) ──► điều chỉnh ngân sách §4
  Đ2 (trình bày phát hiện D0) ──┐
  Đ3 (N tập con), Đ4 (A2)       │
                                 ▼
NGÀY 2  Họp 1: CHỐT D0 (quan trọng nhất) + D1/D2/D3, outline theo khổ 6 trang
                                 │
NGÀY 2-4 ────────────────────────┼─────────────────────────────────────
  Đ5,Đ6 kick off B1/B2 (khi có data/) │
  Đ7 (inference) ──► Đ8 (A1) + Đ9 (bootstrap)
  H2,H3 (Mục II)  ← không phụ thuộc D0
  H1 (Mục I) CHỈ SAU KHI D0 chốt
  P1,P2,P3
                                 │
NGÀY 5-6 ────────────────────────┴─────────────────────────────────────
  Đ3+Đ4+Đ8+Đ9(+B1/B2 nếu xong) ──► P4 (MỘT bảng, có dòng B1/B2) ──► Đ11 (Mục IV)
  Đ10 (Mục III)                                                  ──► P6 (Mục V)
  H4 (Mục VI)
  NGÀY 6  Họp 2: ghép 3 mục, rà số liệu, KIỂM TRA TỔNG SỐ CỘT vs ngân sách §4

NGÀY 7-8 ────────────────────────────────────────────────────────────
  Đ12 (hình, ưu tiên vừa 1 cột) ──► Đ13 (ghép bản thảo v1)

NGÀY 9-12 ───────────────────────────────────────────────────────────
  P7 + H5 (phản biện) + ĐO SỐ TRANG THẬT (H7) — nếu >6 trang, kích hoạt §7b
       └──► NGÀY 9 Họp 3 ──► sửa (ngày 10) ──► H6 ──► H7,H8 ──► P8 ──► P9

NGÀY 13  Dự phòng
NGÀY 14  Đức nộp bài
```

**Chuỗi dài nhất:** `Đ2/Họp1(D0) → Đ7 → Đ9 → P4 → Đ11 → Đ13 → P7 → sửa → H7 → P9 → nộp` = 14 ngày.
**D0 giờ là điểm khởi đầu của chuỗi găng**, không chỉ VĐ11 như bản v3 — nếu Họp 1 không chốt được
D0, mọi việc viết (H1, P1, Đ11) đều có nguy cơ phải làm lại.

---

## Họp & Check-in

| Sự kiện | Ngày | Thời lượng | Nội dung |
|---|---|---|---|
| Đọc tài liệu | 1 | tự làm | `model_architecture.md` v2, `issues_and_fixes.md` v2, phát hiện D0 |
| **Họp 1** | 2 | 75' (dài hơn — có thêm D0) | **Chốt D0** trước tiên, rồi D1, D2, D3, outline theo khổ 6 trang |
| Check-in | 4 | 20' | Tiến độ Đ7/A1/A2/B1/B2; Mục I–II |
| **Họp 2** | 6 | 60' | Ghép 3 mục, rà số liệu, **kiểm tra tổng cột vs ngân sách §4** |
| Họp 3 | 9 | 60' | Sau phản biện + đo trang thật — thống nhất sửa đổi |
| Họp cuối | 12 | 45' | Rà soát lần cuối |

---

## Lưu ý chung cho cả 3 người

1. **Mọi thay đổi về số liệu phải được Đức xác nhận.**
2. **Không tự ý thêm nội dung khoa học mới.** B1/B2 là ngoại lệ đã kiểm soát.
3. **Không sửa code training/loss để "vá" VĐ9.** Chỉ khai báo + A2 đối chứng.
4. **`issues_and_fixes.md` v2 là checklist bắt buộc.**
5. ⚠️ **Không dùng `model_architecture.md` bản 1, không tin `README.md` mục 5.1–5.2** (VĐ12).
6. **Khi phải cắt vì thời gian:** dùng `writing_plan.md` §7a. **Khi phải cắt vì vượt khổ 6
   trang:** dùng §7b — hai bảng cắt khác nhau, đừng nhầm lẫn nguyên nhân.
7. **D0 ưu tiên cao hơn mọi việc khác ở Họp 1** — nếu hết giờ họp mà chưa chốt D0, hoãn D1/D2/D3
   sang check-in ngày 4, nhưng D0 phải xong trước khi bất kỳ ai chạm vào Mục I/IV/V.
