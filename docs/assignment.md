# Phân công Chi tiết – Nhóm Viết Bài báo VNICT 2026
**Thành viên:** Đức, Hiếu, Phát  
**Quỹ thời gian:** **14 ngày** cho toàn bộ công việc  
**Vai trò tổng quát:** Đức là tác giả khóa luận gốc, nắm toàn bộ code và số liệu → đầu mối kỹ thuật và tác giả liên hệ. Hiếu và Phát hỗ trợ viết, phản biện chéo, và xử lý các phần không đòi hỏi truy cập code/model.

> **Phiên bản 3** — tích hợp 4 nâng cấp rẻ đã đánh giá độ khó (A1, A2, B1, B2). Ba việc cũ của
> Đức (Đ5 — đo 20 lần eval; Đ7 — trung bình 5 epoch cuối) **bị xóa và thay thế** bằng phiên bản
> rẻ hơn và đúng hơn (A1, A2). Hai việc mới cần GPU (B1, B2) được thêm, chạy nền, không chặn
> việc viết. Tài liệu tham chiếu bắt buộc: **`model_architecture.md` v2**,
> **`issues_and_fixes.md` v2**, **`writing_plan.md` v3**.

---

## Nguyên tắc phân công

1. Việc nào cần chạy code / truy cập model / dữ liệu gốc → **bắt buộc Đức**.
2. Việc viết văn bản, tổng hợp, trình bày, định dạng → chia đều cho cả 3.
3. Mỗi phần có **1 người viết chính** và **1 người đọc phản biện** (không phải người viết).
4. **Ngày 1, cả 3 người đọc `model_architecture.md` v2 và `issues_and_fixes.md` v2 trước Họp 1.**

---

## Điều kiện tiên quyết — kiểm tra TRƯỚC ngày 1 kết thúc

```
[ ] results/models/<model>/best_model.pt          còn cho cả 7 mô hình?
[ ] results/models/<model>/training_metrics_*.csv còn cho cả 7 mô hình?
[ ] results/models/<model>/summary.json           còn cho cả 7 mô hình?
[ ] data/ (PDBbind2016 + .types)                   còn nguyên, molgrid còn chạy được?
[ ] GPU còn truy cập được?
```

Nếu thiếu bất kỳ mục nào, báo ngay trong Họp 1 — lịch dưới đây phải điều chỉnh theo
`writing_plan.md` §7 (phương án rút gọn).

---

## Bốn nâng cấp mới — độ khó thực tế (đọc trước khi phân việc)

| # | Nội dung | Chi phí thật | Đóng vấn đề gì |
|---|---|---|---|
| **A1** | C-index tính đủ cặp thay vì lấy mẫu ngẫu nhiên | **~15 phút**, chỉ là script offline chạy trên file dự đoán đã có (Đ4) — **không sửa code, không train lại, không chạy lại eval** | VĐ4b (đóng hẳn) |
| **A2** | Chọn epoch bằng chỉ số trên tập train, không nhìn test | **~1 giờ pandas**, dùng `training_metrics_*.csv` có sẵn — **không cần GPU** | VĐ9 (từ "thú nhận" → "thú nhận + đối chứng") |
| **B1** | Bật `--geoformer_uncertainty`, train lại GeoFormerDock | Đổi 1 dòng lệnh + **~1 ngày GPU chạy nền** | VĐ10 (đóng bằng 1 dòng so sánh cùng dạng loss) |
| **B2** | Bỏ `--pose_balance_batch`, train lại GeoFormerDock | Đổi 0 dòng code (bỏ 2 cờ) + **~1 ngày GPU chạy nền** | Ablation quan trọng nhất — nguồn gốc Balanced Accuracy 0.830 |

A1 và A2 gần như miễn phí — luôn làm, ở mọi kịch bản thời gian. B1/B2 cần GPU nên chạy nền song
song với việc viết, không nằm trên đường găng.

---

## Ba việc chặn tiến độ (đường găng)

| Việc | Ai | Hạn | Chặn cái gì |
|---|---|---|---|
| **VĐ11 — đối chiếu Bảng 6 với `summary.json`** | Đức | **Hết ngày 1** | Toàn bộ Mục IV, Mục V, mọi con số trong bài |
| **Q4 — chạy inference lấy dự đoán từng mẫu** | Đức | Hết ngày 3 | A1, paired bootstrap → bảng kết quả → Mục IV.3 |
| **Bảng kết quả cuối** | Phát | Hết ngày 6 | Đức viết Mục IV.3 |

B1 và B2 **không** nằm trong bảng trên: nếu trễ, Mục IV vẫn ra được bằng phương án khai báo văn
bản (xem `writing_plan.md` §5 mục 1 và 6).

---

## PHÂN CÔNG ĐỨC

**Vai trò:** Đầu mối kỹ thuật, tác giả liên hệ, chủ biên nội dung Phương pháp + Thực nghiệm.

| # | Việc | Input | Output | Ưu tiên | Hạn |
|---|---|---|---|---|---|
| **Đ1** | ⚠️ Kiểm tra CFP chính thức VNICT 2026 (template, **deadline thật**, số trang, ngôn ngữ) | Trang web hội thảo | Thông báo nhóm trong ngày 1. **Nếu deadline < 14 ngày → kích hoạt `writing_plan.md` §7** | 🔴 | Ngày 1 |
| **Đ2** | ⚠️ **Kiểm tra điều kiện tiên quyết** (bảng trên) | `results/`, `data/`, GPU | Xác nhận/cảnh báo cho cả nhóm | 🔴 | Ngày 1 |
| **Đ3** | ⚠️ **VĐ11 — Đối chiếu lại Bảng 6.** Lấy `final_acc`, `final_bal_acc`, `final_pose_recall_pos/neg`, `final_pr_auc` từ `results/models/<model>/summary.json` của cả 7 mô hình. Kiểm tra đẳng thức `Acc = π·Rpos + (1−π)·Rneg` với **cùng một π** cho mọi dòng. Hiện 4/7 dòng đang mâu thuẫn (π suy ra: 0.13 / 0.20 / 0.69 / 0.77 / 0.84) | `summary.json`, `training_metrics_test.csv` | Bảng 6 đã xác minh + báo cáo ngắn: dòng nào sai, sai do đâu | 🔴 | **Ngày 1** |
| **Đ4** | **Q2 — Lấy N của tập con `y_aff > 0`** trên tập test | Log train hoặc `ref_uff_test0.types` | 1 con số, đặt dưới Bảng 7 | 🔴 | Ngày 1 |
| **Đ5** | **A2 — Chọn epoch bằng train composite.** Từ `training_metrics_train.csv`, lấy epoch có `0.5·C-index + 0.5·BalAcc` lớn nhất trên **train**; tra giá trị **test** tại đúng epoch đó từ `training_metrics_test.csv`. Làm cho cả 7 mô hình | `training_metrics_train/test.csv` × 7 | Cột "test @ epoch chọn bằng train" cạnh cột "best-on-test" | 🟡 | Ngày 1–2 |
| **Đ6** | **Kick off B1** — thêm `--geoformer_uncertainty` vào `GEO_EXTRA_ARGS` trong `run_training.sh` cho dòng `geoformerdock`, chạy lại **chỉ mô hình này**, chạy nền | `run_training.sh`, `data/` | `results/models/geoformerdock_uncertainty/summary.json` | 🟡 | Kick off ngày 2, ~1 ngày GPU |
| **Đ7** | **Kick off B2** — bỏ `--pose_balance_batch --pose_balance_target_per_class ...` khỏi dòng lệnh train `geoformerdock`, chạy lại, chạy nền. Song song với B1 nếu ≥2 GPU, tuần tự sau B1 nếu 1 GPU | `run_training.sh`, `data/` | `results/models/geoformerdock_nobalance/summary.json` | 🟡 | Kick off ngày 2–3, ~1 ngày GPU |
| **Đ8** | **Q4 — Chạy inference 7 mô hình** từ `best_model.pt` trên tập test, xuất dự đoán **từng mẫu** ra CSV (`y_pose`, `y_aff`, `p_good`, `ŷ_aff`). Chỉ forward pass, không train | `best_model.pt` × 7, `ref_uff_test0.types` | 7 file `predictions_<model>.csv` | 🔴 | Ngày 3 |
| **Đ9** | **A1 — C-index chính xác.** Script offline (numpy, đủ cặp, không lấy mẫu) chạy trên `predictions_<model>.csv` của Đ8. Không sửa `metrics.py`, không chạy lại eval | 7 file CSV từ Đ8 | C-index chính xác cho 7 mô hình | 🟡 | Ngày 3–4, ngay sau Đ8 |
| **Đ10** | **Paired bootstrap CI.** Resample **cùng bộ chỉ số** cho 2 mô hình, lấy CI của hiệu số Δ. Snippet có sẵn trong `issues_and_fixes.md` VĐ4 | 7 file CSV từ Đ8 | Bảng CI cho Δ ở: PR-AUC, Bal Acc, MAE, RMSE, r, ρ, C-index | 🟡 | Ngày 4 |
| **Đ11** | **Thu thập kết quả B1, B2** khi chạy xong; so sánh với dòng GeoFormerDock gốc | `summary.json` mới của B1/B2 | 1–2 dòng/cột phụ cho bảng của Phát (P4) | 🟡 | Ngày 4–5 |
| **Đ12** | **Viết Mục III – Phương pháp đề xuất.** Nguồn **duy nhất** là `model_architecture.md` **v2**. Bắt buộc gồm: công thức `B_pocket` (VĐ3), phương trình `S = {i : y_aff > 0}` (VĐ2), chuẩn hóa z-score, 27 token, cân bằng minibatch | `model_architecture.md` v2 | Bản thảo Mục III | 🔴 | Ngày 5 |
| **Đ13** | **Viết Mục IV.1–IV.3 + IV.5.** IV.2 phải có đủ **4 khai báo bắt buộc** (VĐ6, cân bằng minibatch, VĐ9 + câu về A2, VĐ10 + câu về B1). IV.3 dùng bảng của Phát (đã tích hợp A1/A2/B1) | Bảng của Phát (P4) + CI từ Đ10 | Bản thảo Mục IV | 🔴 | Ngày 6 |
| **Đ14** | *(Tùy chọn, chỉ nếu còn GPU sau B1+B2)* Ablation mở rộng: `GEOFORMER_MAX_PSEUDO_ATOMS=0` (no-geo), comment `geoformerdock.py:65-67` (no-Bpocket). **Cùng số epoch và cùng quy tắc chọn mô hình cho mọi dòng** | Code hiện có | Kết quả bổ sung cho bảng ablation | 🟢 | Chạy nền, không có deadline cứng |
| **Đ15** | Chuẩn bị hình: vẽ lại Hình 3 (kiến trúc, **dạng ngang gọn**), xuất Hình 17 (Pareto) độ phân giải in ấn | Hình gốc khóa luận | File .png/.pdf | 🟡 | Ngày 7 |
| **Đ16** | **Ghép bản thảo đầy đủ lần 1** (chủ biên) | 6 mục từ 3 người | Bản thảo v1 | 🔴 | Ngày 8 |
| **Đ17** | Tác giả liên hệ: nộp bài, theo dõi trạng thái, phản hồi phản biện | — | — | 🔴 | Ngày 14+ |

### ✅ Việc đã HOÀN THÀNH — không cần làm nữa (giải quyết bằng đọc code, không cần chạy gì)

- ~~*Kiểm tra `L_reg` tính trên tập nào*~~ → **Đáp án: (B)**, chỉ mẫu có `y_aff > 0`, cả khi
  train (`training.py:496`) lẫn khi eval (`transforms.py:131-147`, `metrics.py:368-373`).
- ~~*Xác định công thức `B_pocket`*~~ → **`B^(h)[i,j] = σ(w_hᵀ x_j + b_h)`**
  (`geoformerdock.py:54-57, 65-67`). Không dùng tọa độ 3D; chỉ phụ thuộc key `j`; miền (0,1).
- ~~*Xác nhận siêu tham số dùng chung*~~ → **Có, dùng chung.** `run_training.sh:111-214` dùng
  cùng một hàm `train_model` cho cả 7 mô hình.

### ❌ Việc bị XÓA khỏi bản 2 — thay bằng A1/A2

- ~~*Đo độ lệch chuẩn C-index qua 20 lần eval (VĐ4b)*~~ → thay bằng **A1** (Đ9): tính C-index
  đủ cặp một lần, không còn là đại lượng ngẫu nhiên nên không cần đo độ lệch chuẩn nữa.
- ~~*Trích giá trị 5 epoch đánh giá cuối làm thước đo dao động (VĐ9)*~~ → thay bằng **A2** (Đ5):
  một phép chọn epoch độc lập với test, thông tin hơn hẳn so với trung bình 5 epoch cuối.

### Deadline nội bộ cho Đức

| Hạn | Việc |
|---|---|
| **Hết ngày 1** | Đ1, Đ2, **Đ3 (đường găng)**, Đ4 |
| Hết ngày 2 | Đ5 (A2), kick off Đ6 (B1) |
| Hết ngày 3 | Đ7 (kick off B2), Đ8 (inference) |
| Hết ngày 4 | Đ9 (A1), Đ10 (bootstrap), Đ11 (thu thập B1/B2 nếu xong) |
| Hết ngày 5 | Đ12 (bản nháp Mục III) |
| Hết ngày 6 | Đ13 (bản nháp Mục IV) |
| Hết ngày 7 | Đ15 |
| Hết ngày 8 | Đ16 |

---

## PHÂN CÔNG HIẾU

**Vai trò:** Viết Giới thiệu / Liên quan / Kết luận, kiểm soát văn phong tổng thể, định dạng.

| # | Việc | Input | Output | Ưu tiên | Hạn |
|---|---|---|---|---|---|
| **H1** | **Viết Mục I – Giới thiệu**, theo định vị câu chuyện mới ở `writing_plan.md` §2. ⚠️ Đóng góp #3 nay có thể có bằng chứng thực nghiệm (B1/B2) — kiểm tra với Đức trước khi chốt câu chữ | Khóa luận Ch.1 + `writing_plan.md` §2 | Bản thảo Mục I (~0.75 trang) | 🔴 | Ngày 3 |
| **H2** | **VĐ5 — Sửa câu overclaim** trong Related Work, có cụm "theo hiểu biết của chúng tôi" | Mục 1.3.2 khóa luận + `issues_and_fixes.md` VĐ5 | Đoạn đã sửa | 🔴 | Ngày 3 |
| **H3** | **Viết Mục II – Công trình liên quan** (~0.6 trang), 4 nhóm mỗi nhóm 2–3 câu. *Nếu D1 = Bỏ:* nhắc nhóm đồ thị + 1 câu giải thích vì sao không so sánh | Khóa luận mục 1.3 | Bản thảo Mục II | 🔴 | Ngày 3 |
| **H4** | **Viết Mục VI – Kết luận** (3–4 câu, không lặp số liệu) | Khóa luận phần Kết luận | Bản thảo Mục VI | 🟡 | Ngày 6 |
| **H5** | **Đọc phản biện Mục III** — đối chiếu từng công thức với phụ lục 14 điểm ở cuối `model_architecture.md` v2 | Bản thảo Đ12 + phụ lục | Danh sách lỗi + góp ý | 🔴 | Ngày 9 |
| **H6** | **Kiểm soát văn phong toàn bài.** Xóa "áp đảo", "bứt phá", "vượt trội" không có CI hậu thuẫn | Bản thảo ghép v1 | Bản thảo đã hiệu đính | 🟡 | Ngày 10 |
| **H7** | **Định dạng theo template hội thảo** | Bản thảo + template CFP | File đúng chuẩn | 🔴 | Ngày 11 |
| **H8** | **Danh mục tài liệu tham khảo** | Danh mục khóa luận | File .bib hoặc danh mục đúng format | 🟡 | Ngày 11 |

### Deadline nội bộ cho Hiếu

| Hạn | Việc |
|---|---|
| Hết ngày 3 | H1, H2, H3 (bản nháp) |
| Hết ngày 6 | H4 |
| Hết ngày 9 | H5 |
| Hết ngày 10 | H6 |
| Hết ngày 11 | H7, H8 |

---

## PHÂN CÔNG PHÁT

**Vai trò:** Viết Thảo luận / Hạn chế, xử lý bảng biểu, đọc phản biện tổng thể, rà soát cuối.

| # | Việc | Input | Output | Ưu tiên | Hạn |
|---|---|---|---|---|---|
| **P1** | **Soạn khung Mục V – Thảo luận & Hạn chế.** 6 điểm bắt buộc theo `writing_plan.md` §5 — mỗi điểm có 2 phiên bản câu chữ (có B1/B2 hoặc không), chốt bản nào dùng ở ngày 5 sau khi biết B1/B2 có xong không | `issues_and_fixes.md` v2 | Khung Mục V ~0.9 trang | 🔴 | Ngày 3 |
| **P2** | **Xử lý VĐ1 theo quyết định D1** | `issues_and_fixes.md` VĐ1 | Bảng/đoạn đã xử lý | 🔴 | Ngày 4 |
| **P3** | **Soạn đoạn IV.2 về minh bạch thiết lập** — gộp VĐ6, cân bằng minibatch, VĐ9 (+ câu về A2), VĐ10 (+ câu về B1) | Xác nhận từ Đ3/Đ5 + `issues_and_fixes.md` VĐ6/9/10 | 2 đoạn văn (~0.4 trang) | 🔴 | Ngày 4 |
| **P4** | **Dựng bảng kết quả chính** — gộp Bảng 6+7, tích hợp paired bootstrap CI (Đ10), N tập con (Đ4), **cột A2** (Đ5), **C-index chính xác** (Đ9), **dòng B1** (Đ11, nếu xong) | Bảng đã xác minh (Đ3) + CI (Đ10) + A1/A2 (Đ9/Đ5) + B1 (Đ11) | Bảng hoàn chỉnh cho Mục IV | 🔴 | Ngày 6 |
| **P5** | **Dựng bảng ablation IV.4** — dòng B2 là **cam kết** (không còn điều kiện "nếu kịp"); thêm no-geo/no-Bpocket nếu Đ14 hoàn thành | Kết quả Đ11 (B2) + Đ14 (nếu có) | Bảng ablation | 🔴 (B2) / 🟢 (phần mở rộng) | Ngày 7 |
| **P6** | **Viết hoàn chỉnh Mục V** từ khung P1, chọn đúng phiên bản câu chữ theo kết quả B1/B2 thật | Khung P1 + bảng P4 | Bản thảo Mục V | 🔴 | Ngày 6 |
| **P7** | **Đọc phản biện bản thảo ghép lần 1** — đối chiếu với checklist trong `issues_and_fixes.md` v2 và `writing_plan.md` §9 | Bản thảo v1 (Đ16) | Danh sách lỗi/thiếu sót | 🔴 | Ngày 9 |
| **P8** | **Kiểm tra khớp số liệu** — mọi con số khớp bảng; **chạy lại phép kiểm tra `Acc = π·Rpos + (1−π)·Rneg`** trên bảng cuối, kể cả dòng B1/B2 nếu có | Bản thảo hoàn chỉnh | Danh sách sai lệch | 🔴 | Ngày 11 |
| **P9** | Rà soát chính tả, ngữ pháp tiếng Việt lần cuối | Bản thảo cuối | Bản thảo sạch lỗi | 🔴 | Ngày 12 |

### Deadline nội bộ cho Phát

| Hạn | Việc |
|---|---|
| Hết ngày 3 | P1 (khung Mục V) |
| Hết ngày 4 | P2, P3 |
| Hết ngày 6 | **P4 (bảng kết quả — chặn Đ13)**, P6 |
| Hết ngày 7 | P5 |
| Hết ngày 9 | P7 |
| Hết ngày 11 | P8 |
| Hết ngày 12 | P9 |

---

## Ma trận Phụ thuộc Công việc

```
NGÀY 1 ─────────────────────────────────────────────────────────────
  Đ1 (CFP) ──► quyết định có kích hoạt phương án rút gọn không
  Đ2 (điều kiện tiên quyết) ──► gate cho B1/B2/Đ8
  Đ3 (VĐ11: xác minh Bảng 6) ─┐   ← ĐƯỜNG GĂNG
  Đ4 (N tập con)              │
  Đ5 bắt đầu (A2, độc lập, không cần gì)

NGÀY 2  Họp 1: chốt D1 / D2 / D3
  Đ6 kick off B1 (chạy nền)
                              │
NGÀY 2-4 ─────────────────────┼──────────────────────────────────────
  Đ7 kick off B2 (chạy nền)  │
  Đ8 (inference) ──► Đ9 (A1) + Đ10 (bootstrap) ──┐
  H1,H2,H3 (Mục I, II)  ← không phụ thuộc số liệu │
  P1 (khung Mục V), P2 (VĐ1), P3 (đoạn IV.2)      │
                              │                    │
NGÀY 4-5 ─────────────────────┴────────────────────┤
  B1, B2 hoàn thành (nếu đúng lịch) ──► Đ11 (thu thập kết quả)
                                                    │
NGÀY 5-6 ───────────────────────────────────────────┴─────────────────
  Đ3+Đ4+Đ5+Đ9+Đ10+Đ11 ──► P4 (bảng kết quả) ──► Đ13 (Mục IV)
  Đ12 (Mục III)                              ──► P6 (Mục V)
  H4 (Mục VI)
  NGÀY 6  Họp 2: ghép 3 mục chính, rà nhất quán số liệu

NGÀY 7-8 ────────────────────────────────────────────────────────────
  Đ15 (hình)  +  P5 (bảng ablation, B2 + tùy chọn Đ14)  ──► Đ16 (ghép bản thảo v1)

NGÀY 9-12 ───────────────────────────────────────────────────────────
  P7 (phản biện tổng thể) + H5 (phản biện Mục III)
       └──► NGÀY 9 Họp 3 ──► sửa (ngày 10)
                            ──► H6 (văn phong) ──► H7,H8 (format, refs)
                            ──► P8 (khớp số liệu) ──► P9 (chính tả)

NGÀY 13  Dự phòng
NGÀY 14  Đức nộp bài
```

**Chuỗi dài nhất:** `Đ3 → Đ8 → Đ10 → P4 → Đ13 → Đ16 → P7 → sửa → H7 → P9 → nộp` = 14 ngày,
**không có slack**. B1/B2 chạy song song với chuỗi này, không kéo dài nó — nếu B1/B2 trễ qua
ngày 6, Phát vẫn dựng được bảng kết quả (P4) mà không có 2 dòng phụ, và Mục V dùng bản khai báo
văn bản thay vì bản có số liệu.

---

## Họp & Check-in

| Sự kiện | Ngày | Thời lượng | Nội dung |
|---|---|---|---|
| **Đọc tài liệu** | 1 | tự làm | Cả 3 đọc `model_architecture.md` v2 + `issues_and_fixes.md` v2 |
| **Họp 1** | 2 | 60' | Đức báo CFP + điều kiện tiên quyết + kết quả VĐ11. **Chốt D1, D2, D3.** Chốt outline |
| **Check-in** | 4 | 20' | Đức báo VĐ11/A1/A2/tiến độ B1/B2; Hiếu báo Mục I–II; Phát báo khung Mục V |
| **Họp 2** | 6 | 60' | Ghép 3 mục chính, rà nhất quán số liệu, **chốt B1/B2 có vào bài hay không** |
| **Họp 3** | 9 | 60' | Sau khi Phát + Hiếu đọc phản biện — thống nhất sửa đổi |
| **Họp cuối** | 12 | 45' | Rà soát lần cuối trước nộp |

---

## Lưu ý chung cho cả 3 người

1. **Mọi thay đổi về số liệu phải được Đức xác nhận** — chỉ Đức có quyền truy cập code/kết quả gốc.
2. **Không tự ý thêm nội dung khoa học mới** ngoài phạm vi đã thống nhất. B1 và B2 là ngoại lệ
   đã được kiểm soát (chỉ bật/tắt cờ có sẵn, 1 mô hình, không đổi kiến trúc/dữ liệu) — xem
   `writing_plan.md` §8.
3. **Không sửa code training/loss để "vá" VĐ9.** Quy tắc chọn mô hình giữ nguyên; A2 chỉ bổ sung
   một phép chọn epoch thay thế để đối chứng, không đổi checkpoint đã lưu.
4. **`issues_and_fixes.md` v2 là checklist bắt buộc.** Mọi mục 🔴 phải hoàn thành trước khi ghép
   bản thảo cuối (ngày 8).
5. ⚠️ **Không dùng `model_architecture.md` bản 1** và **không tin `README.md`** ở các mục
   5.1–5.2 — README vẫn ghi "5 thành phần loss", lịch ramp-up `L_rank`, và "cùng loss cho cả 7
   mô hình", cả ba đều sai so với code (VĐ12).
6. **Nguyên tắc khi phải cắt việc:** giữ A1, A2 ở mọi kịch bản (gần như miễn phí); B1, B2 là ứng
   viên cắt đầu tiên nếu GPU không sẵn sàng hoặc thời gian eo hẹp — xem `writing_plan.md` §7.
7. **B1/B2 không có deadline cứng gắn với ngày cụ thể** — chúng chạy nền. Chốt tại Họp 2 (ngày 6)
   xem có kịp đưa vào bài không; nếu không kịp, dùng bản khai báo văn bản đã chuẩn bị sẵn ở P1/P6.
