# Phân công Chi tiết – Nhóm Viết Bài báo VNICT 2026
**Thành viên:** Đức, Hiếu, Phát  
**Quỹ thời gian:** **14 ngày** cho toàn bộ công việc  
**Vai trò tổng quát:** Đức là tác giả khóa luận gốc, nắm toàn bộ code và số liệu → đầu mối kỹ thuật và tác giả liên hệ. Hiếu và Phát hỗ trợ viết, phản biện chéo, và xử lý các phần không đòi hỏi truy cập code/model.

> **Phiên bản 2** — cập nhật sau khi đối chiếu mã nguồn.
> - **VĐ2 và VĐ3 đã có lời giải** → hai việc cũ của Đức (#2, #3) bị xóa, giải phóng ~2 ngày.
> - **Ba việc CRITICAL mới** phát sinh từ VĐ9, VĐ10, VĐ11.
> - **Việc kiểm tra số liệu của Phát được nâng lên 🔴 và chuyển lên ngày 1–2** vì nó chặn toàn
>   bộ Mục IV.
> - Tài liệu tham chiếu bắt buộc: **`model_architecture.md` v2**, **`issues_and_fixes.md` v2**,
>   `writing_plan.md` v2.

---

## Nguyên tắc phân công

1. Việc nào cần chạy code / truy cập model / dữ liệu gốc → **bắt buộc Đức**.
2. Việc viết văn bản, tổng hợp, trình bày, định dạng → chia đều cho cả 3.
3. Mỗi phần có **1 người viết chính** và **1 người đọc phản biện** (không phải người viết).
4. **Ngày 1, cả 3 người đọc `model_architecture.md` v2 và `issues_and_fixes.md` v2 trước Họp 1.**
   Không họp khi chưa đọc — sẽ mất cả buổi để giải thích lại.

---

## Ba việc chặn tiến độ (đường găng)

Nếu chỉ nhớ được ba dòng trong tài liệu này thì nhớ ba dòng dưới đây:

| Việc | Ai | Hạn | Chặn cái gì |
|---|---|---|---|
| **VĐ11 — đối chiếu Bảng 6 với `summary.json`** | Đức | **Hết ngày 1** | Toàn bộ Mục IV, Mục V, mọi con số trong bài |
| **Q4 — chạy inference lấy dự đoán từng mẫu** | Đức | Hết ngày 3 | Paired bootstrap → bảng kết quả → Mục IV.3 |
| **Bảng kết quả cuối** | Phát | Hết ngày 6 | Đức viết Mục IV.3 |

---

## PHÂN CÔNG ĐỨC

**Vai trò:** Đầu mối kỹ thuật, tác giả liên hệ, chủ biên nội dung Phương pháp + Thực nghiệm.

| # | Việc | Input | Output | Ưu tiên | Hạn |
|---|---|---|---|---|---|
| **Đ1** | ⚠️ Kiểm tra CFP chính thức VNICT 2026 (template, **deadline thật**, số trang, ngôn ngữ) | Trang web hội thảo | Thông báo nhóm trong ngày 1. **Nếu deadline < 14 ngày → kích hoạt §6 của `writing_plan.md`** | 🔴 | Ngày 1 |
| **Đ2** | ⚠️ **VĐ11 — Đối chiếu lại Bảng 6.** Lấy `final_acc`, `final_bal_acc`, `final_pose_recall_pos/neg`, `final_pr_auc` từ `results/models/<model>/summary.json` của cả 7 mô hình. Kiểm tra đẳng thức `Acc = π·Rpos + (1−π)·Rneg` với **cùng một π** cho mọi dòng. Hiện 4/7 dòng đang mâu thuẫn (π suy ra: 0.13 / 0.20 / 0.69 / 0.77 / 0.84) | `summary.json`, `training_metrics_test.csv` | Bảng 6 đã xác minh + báo cáo ngắn: dòng nào sai, sai do đâu | 🔴 | **Ngày 1** |
| **Đ3** | **Q2 — Lấy N của tập con `y_aff > 0`** trên tập test (từ dòng `[DEBUG] Affinity target stats (N=...)` trong log, hoặc đếm trực tiếp trên `ref_uff_test0.types`) | Log train hoặc file `.types` | 1 con số, đặt dưới Bảng 7 | 🔴 | Ngày 1 |
| **Đ4** | **Q4 — Chạy inference 7 mô hình** từ `best_model.pt` trên tập test, xuất dự đoán **từng mẫu** ra CSV (`y_pose`, `y_aff`, `p_good`, `ŷ_aff`). Chỉ forward pass, không train | `best_model.pt` × 7, `ref_uff_test0.types` | 7 file `predictions_<model>.csv` | 🔴 | Ngày 3 |
| **Đ5** | **VĐ4b — Đo nhiễu C-index.** Chạy `test_evaluator` 20 lần trên cùng một `best_model.pt`, ghi C-index mỗi lần, tính độ lệch chuẩn. `concordance_index` lấy mẫu cặp ngẫu nhiên **không seed** (`metrics.py:200-217`) | `best_model.pt` | σ của C-index. **Nếu σ ≳ 0.005 → không được diễn giải khoảng cách 0.011 giữa GeoFormerDock và GNINA là khác biệt thực** | 🟡 | Ngày 3 |
| **Đ6** | **VĐ4a — Paired bootstrap CI.** Resample **cùng bộ chỉ số** cho 2 mô hình, lấy CI của hiệu số Δ. Snippet có sẵn trong `issues_and_fixes.md` VĐ4 | 7 file CSV từ Đ4 | Bảng CI cho Δ ở: PR-AUC, Bal Acc, MAE, RMSE, r, ρ, C-index | 🟡 | Ngày 4 |
| **Đ7** | **VĐ9 — Trích số liệu dao động.** Từ `training_metrics_test.csv`, lấy giá trị **5 lần đánh giá cuối** của mỗi mô hình (mean ± std), để báo cáo cạnh giá trị best | `training_metrics_test.csv` × 7 | Bảng phụ hoặc cột thêm | 🟡 | Ngày 4 |
| **Đ8** | **Viết Mục III – Phương pháp đề xuất.** Nguồn **duy nhất** là `model_architecture.md` **v2**. Bắt buộc gồm: công thức `B_pocket` (VĐ3), phương trình `S = {i : y_aff > 0}` (VĐ2), chuẩn hóa z-score, 27 token, cân bằng minibatch | `model_architecture.md` v2 | Bản thảo Mục III | 🔴 | Ngày 5 |
| **Đ9** | **Viết Mục IV.1–IV.3 + IV.5.** IV.2 phải có đủ **4 khai báo bắt buộc**: siêu tham số chung (VĐ6), cân bằng minibatch, **chọn mô hình trên test (VĐ9)**, **khác biệt `L_reg` (VĐ10)** | Bảng của Phát (P4) + CI từ Đ6 | Bản thảo Mục IV | 🔴 | Ngày 6 |
| **Đ10** | *(Nếu D2 = Có)* **Chạy ablation, ưu tiên theo thứ tự:** (1) bỏ `--pose_balance_batch`, (2) `GEOFORMER_MAX_PSEUDO_ATOMS=0`, (3) comment `geoformerdock.py:65-67`. **Cùng số epoch và cùng quy tắc chọn mô hình cho mọi dòng, kể cả dòng full** | Code hiện có | Kết quả 2–4 dòng ablation | 🟢 | Chạy nền ngày 2–8 |
| **Đ11** | Chuẩn bị hình: vẽ lại Hình 3 (kiến trúc, **dạng ngang gọn**), xuất Hình 17 (Pareto) độ phân giải in ấn | Hình gốc khóa luận | File .png/.pdf | 🟡 | Ngày 7 |
| **Đ12** | **Ghép bản thảo đầy đủ lần 1** (chủ biên) | 6 mục từ 3 người | Bản thảo v1 | 🔴 | Ngày 8 |
| **Đ13** | Tác giả liên hệ: nộp bài, theo dõi trạng thái, phản hồi phản biện | — | — | 🔴 | Ngày 14+ |

### ✅ Hai việc ở bản 1 đã HOÀN THÀNH — không cần làm nữa

- ~~*Việc #2: kiểm tra `L_reg` tính trên tập nào*~~ → **Đáp án: (B)**, chỉ mẫu có `y_aff > 0`, cả
  khi train (`training.py:496`) lẫn khi eval (`transforms.py:131-147`, `metrics.py:368-373`).
  ⚠️ Điều kiện là `y_aff > 0`, **không phải** `y_pose = 1`.
- ~~*Việc #3: xác định công thức `B_pocket`*~~ → **`B^(h)[i,j] = σ(w_hᵀ x_j + b_h)`**
  (`geoformerdock.py:54-57, 65-67`). Không dùng tọa độ 3D; chỉ phụ thuộc key `j`; miền (0,1).
- ~~*Việc #4: xác nhận siêu tham số dùng chung*~~ → **Có, dùng chung.** `run_training.sh:111-214`
  dùng cùng một hàm `train_model` cho cả 7 mô hình.

### Deadline nội bộ cho Đức

| Hạn | Việc |
|---|---|
| **Hết ngày 1** | Đ1, **Đ2 (đường găng)**, Đ3 |
| Hết ngày 3 | Đ4, Đ5 |
| Hết ngày 4 | Đ6, Đ7 |
| Hết ngày 5 | Đ8 (bản nháp Mục III) |
| Hết ngày 6 | Đ9 (bản nháp Mục IV) |
| Hết ngày 7 | Đ11 |
| Hết ngày 8 | Đ12 |

---

## PHÂN CÔNG HIẾU

**Vai trò:** Viết Giới thiệu / Liên quan / Kết luận, kiểm soát văn phong tổng thể, định dạng.

| # | Việc | Input | Output | Ưu tiên | Hạn |
|---|---|---|---|---|---|
| **H1** | **Viết Mục I – Giới thiệu**, theo định vị câu chuyện mới ở `writing_plan.md` §1. ⚠️ **Ba đóng góp đã được định vị lại** — không dùng danh sách đóng góp của bản 1 | Khóa luận Ch.1 + `writing_plan.md` §1 | Bản thảo Mục I (~0.75 trang) | 🔴 | Ngày 3 |
| **H2** | **VĐ5 — Sửa câu overclaim** trong Related Work. Câu thay thế **bắt buộc** có cụm "theo hiểu biết của chúng tôi", vì bản thân câu thay thế ở bản 1 cũng là một overclaim | Mục 1.3.2 khóa luận + `issues_and_fixes.md` VĐ5 | Đoạn đã sửa | 🔴 | Ngày 3 |
| **H3** | **Viết Mục II – Công trình liên quan** (~0.6 trang). 4 nhóm, mỗi nhóm 2–3 câu. *Nếu D1 = Bỏ baseline đồ thị:* vẫn nhắc nhóm đồ thị trong Related Work + 1 câu giải thích vì sao không đưa vào so sánh | Khóa luận mục 1.3 | Bản thảo Mục II | 🔴 | Ngày 3 |
| **H4** | **Viết Mục VI – Kết luận** (3–4 câu, không lặp số liệu) | Khóa luận phần Kết luận | Bản thảo Mục VI | 🟡 | Ngày 6 |
| **H5** | **Đọc phản biện Mục III.** ⚠️ Không chỉ đọc về tính dễ hiểu — **đối chiếu từng công thức với phụ lục 14 điểm ở cuối `model_architecture.md` v2**. Bất kỳ câu nào trùng cột "Bản 1 viết" đều là lỗi | Bản thảo Đ8 + phụ lục | Danh sách lỗi + góp ý | 🔴 | Ngày 9 |
| **H6** | **Kiểm soát văn phong toàn bài (VĐ4c).** Xóa "áp đảo", "bứt phá", "vượt trội" không có CI hậu thuẫn. ⚠️ Cũng xóa câu *"không thể quy về nhiễu với dataset 4.618 mẫu"* nếu nó lọt vào bản thảo | Bản thảo ghép v1 | Bản thảo đã hiệu đính | 🟡 | Ngày 10 |
| **H7** | **Định dạng theo template hội thảo** (font, margin, số trang, style trích dẫn) | Bản thảo + template CFP | File đúng chuẩn | 🔴 | Ngày 11 |
| **H8** | **Danh mục tài liệu tham khảo** — lọc từ 27 mục của khóa luận còn phần thực sự trích dẫn | Danh mục khóa luận | File .bib hoặc danh mục đúng format | 🟡 | Ngày 11 |

### Deadline nội bộ cho Hiếu

| Hạn | Việc |
|---|---|
| Hết ngày 3 | H1, H2, H3 (bản nháp) |
| Hết ngày 6 | H4 |
| Hết ngày 9 | H5 (đọc phản biện Mục III) |
| Hết ngày 10 | H6 |
| Hết ngày 11 | H7, H8 |

---

## PHÂN CÔNG PHÁT

**Vai trò:** Viết Thảo luận / Hạn chế, xử lý bảng biểu, đọc phản biện tổng thể, rà soát cuối.

| # | Việc | Input | Output | Ưu tiên | Hạn |
|---|---|---|---|---|---|
| **P1** | **Soạn khung Mục V – Thảo luận & Hạn chế.** ⚠️ Nay có **6 điểm bắt buộc** (bản 1 chỉ có 4): (1) GNINA thắng MAE/C-index **+ giải thích VĐ10**, (2) **chọn mô hình trên test — VĐ9**, (3) chỉ 1 fold, (4) baseline đồ thị theo D1, (5) **chỉ số ái lực trên tập con — VĐ2**, (6) chưa có ablation (nếu không kịp) | `issues_and_fixes.md` v2 | Khung Mục V ~0.9 trang | 🔴 | Ngày 3 |
| **P2** | **Xử lý VĐ1 theo quyết định D1.** *Nếu Bỏ:* soạn 1 câu giải thích cho Related Work (phối hợp H3). *Nếu Giữ:* soạn bảng đã đổi tên + ghi chú nêu đúng nguyên nhân (tọa độ pseudo-atom **không khả vi**), không nói chung chung | `issues_and_fixes.md` VĐ1 | Bảng/đoạn đã xử lý | 🔴 | Ngày 4 |
| **P3** | **Soạn đoạn IV.2 về minh bạch thiết lập** — gộp cả 4 khai báo: siêu tham số chung (VĐ6, kèm thừa nhận lr chung + AMP không đồng nhất), cân bằng minibatch, **VĐ9**, **VĐ10**. Chuyển cho Đức ráp vào Mục IV | Xác nhận từ Đ2/Đ7 + `issues_and_fixes.md` VĐ6/9/10 | 2 đoạn văn (~0.4 trang) | 🔴 | Ngày 4 |
| **P4** | **Dựng bảng kết quả chính** — gộp Bảng 6+7 thành 1 bảng, tích hợp paired bootstrap CI từ Đ6, ghi **N của tập con** dưới bảng (từ Đ3), thêm cột/hàng giá trị 5 epoch cuối từ Đ7 | Bảng đã xác minh (Đ2) + CI (Đ6) + N (Đ3) | Bảng hoàn chỉnh cho Mục IV | 🔴 | Ngày 6 |
| **P5** | *(Nếu Đ10 hoàn thành)* Dựng bảng ablation 3–4 dòng | Kết quả Đ10 | Bảng ablation | 🟢 | Ngày 7 |
| **P6** | **Viết hoàn chỉnh Mục V** từ khung P1, cập nhật theo số liệu thật | Khung P1 + bảng P4 | Bản thảo Mục V | 🔴 | Ngày 6 |
| **P7** | **Đọc phản biện bản thảo ghép lần 1** — đối chiếu từng dòng với checklist CRITICAL/HIGH trong `issues_and_fixes.md` v2 và §8 của `writing_plan.md` | Bản thảo v1 (Đ12) | Danh sách lỗi/thiếu sót | 🔴 | Ngày 9 |
| **P8** | **Kiểm tra khớp số liệu:** mọi con số trong văn bản khớp chính xác với bảng; làm tròn nhất quán; **chạy lại phép kiểm tra `Acc = π·Rpos + (1−π)·Rneg` trên bảng cuối** | Bản thảo hoàn chỉnh | Danh sách sai lệch | 🔴 | Ngày 11 |
| **P9** | Rà soát chính tả, ngữ pháp tiếng Việt lần cuối | Bản thảo cuối | Bản thảo sạch lỗi | 🔴 | Ngày 12 |

### Deadline nội bộ cho Phát

| Hạn | Việc |
|---|---|
| Hết ngày 3 | P1 (khung Mục V) |
| Hết ngày 4 | P2, P3 |
| Hết ngày 6 | **P4 (bảng kết quả — chặn Đ9)**, P6 |
| Hết ngày 7 | P5 (nếu có) |
| Hết ngày 9 | P7 |
| Hết ngày 11 | P8 |
| Hết ngày 12 | P9 |

---

## Ma trận Phụ thuộc Công việc

```
NGÀY 1 ─────────────────────────────────────────────────────────────
  Đ1 (CFP) ──► quyết định có kích hoạt phương án rút gọn không
  Đ2 (VĐ11: xác minh Bảng 6) ─┐   ← ĐƯỜNG GĂNG
  Đ3 (N tập con)              │
                              │
NGÀY 2  Họp 1: chốt D1 / D2 / D3
                              │
NGÀY 2-4 ─────────────────────┼──────────────────────────────────────
  Đ4 (inference) ──► Đ5, Đ6 (bootstrap + nhiễu C-index) ──┐
  Đ7 (5 epoch cuối) ──────────────────────────────────────┤
  H1,H2,H3 (Mục I, II)  ← không phụ thuộc số liệu         │
  P1 (khung Mục V), P2 (VĐ1), P3 (đoạn IV.2)              │
                              │                            │
NGÀY 5-6 ─────────────────────┴────────────────────────────┴─────────
  Đ2 + Đ3 + Đ6 + Đ7 ──► P4 (bảng kết quả) ──► Đ9 (Mục IV)
  Đ8 (Mục III)                             ──► P6 (Mục V)
  H4 (Mục VI)
  NGÀY 6  Họp 2: ghép 3 mục chính, rà nhất quán số liệu

NGÀY 7-8 ────────────────────────────────────────────────────────────
  Đ11 (hình)  +  P5 (ablation, nếu có)  ──► Đ12 (ghép bản thảo v1)

NGÀY 9-12 ───────────────────────────────────────────────────────────
  P7 (phản biện tổng thể) + H5 (phản biện Mục III)
       └──► NGÀY 9 Họp 3 ──► sửa (ngày 10)
                            ──► H6 (văn phong) ──► H7,H8 (format, refs)
                            ──► P8 (khớp số liệu) ──► P9 (chính tả)

NGÀY 13  Dự phòng
NGÀY 14  Đức nộp bài
```

**Chuỗi dài nhất:** `Đ2 → Đ4 → Đ6 → P4 → Đ9 → Đ12 → P7 → sửa → H7 → P9 → nộp` = 14 ngày,
**không có slack**. Mọi trễ hạn ở Đ2 hoặc Đ4 đều đẩy lùi ngày nộp.

---

## Họp & Check-in

| Sự kiện | Ngày | Thời lượng | Nội dung |
|---|---|---|---|
| **Đọc tài liệu** | 1 | tự làm | Cả 3 đọc `model_architecture.md` v2 + `issues_and_fixes.md` v2 |
| **Họp 1** | 2 | 60' | Đức báo CFP + kết quả VĐ11. **Chốt D1 (bỏ/giữ baseline đồ thị), D2 (có/không ablation), D3 (trình bày VĐ10 ở đâu).** Chốt outline |
| **Check-in** | 4 | 20' | Đức báo VĐ4/VĐ9; Hiếu báo Mục I–II; Phát báo khung Mục V |
| **Họp 2** | 6 | 60' | Ghép 3 mục chính, rà nhất quán số liệu giữa các mục |
| **Họp 3** | 9 | 60' | Sau khi Phát + Hiếu đọc phản biện — thống nhất sửa đổi |
| **Họp cuối** | 12 | 45' | Rà soát lần cuối trước nộp |

---

## Lưu ý chung cho cả 3 người

1. **Mọi thay đổi về số liệu phải được Đức xác nhận** — chỉ Đức có quyền truy cập code/kết quả gốc.
2. **Không tự ý thêm nội dung khoa học mới** ngoài phạm vi đã thống nhất (ràng buộc "chỉ viết
   lại, không phát triển thêm").
3. **Không sửa code để "vá" VĐ9 hay VĐ10.** Sửa xong sẽ phải train lại toàn bộ 7 mô hình — vượt
   xa 14 ngày. Cách xử lý duy nhất là **khai báo trung thực bằng văn bản**.
4. **`issues_and_fixes.md` v2 là checklist bắt buộc.** Mọi mục 🔴 phải được đánh dấu hoàn thành
   trước khi ghép bản thảo cuối (ngày 8).
5. ⚠️ **Không dùng `model_architecture.md` bản 1** (đã bị thay thế) và **không tin `README.md`**
   ở các mục 5.1–5.2 — README vẫn ghi "5 thành phần loss", lịch ramp-up `L_rank`, và "cùng loss
   cho cả 7 mô hình", cả ba đều sai so với code (VĐ12).
6. **Nguyên tắc khi phải cắt việc:** ưu tiên giữ **tính trung thực** (VĐ9, VĐ10, Mục V đầy đủ)
   hơn **tính đầy đủ** (ablation, bootstrap CI). Xem `writing_plan.md` §6.
