# Phân công 2 phần — sửa bài theo phản biện VNICT 2026

**Deadline:** 10 ngày.
**Nguồn bản thảo: Overleaf** — đó là bản duy nhất được sửa. `docs/VNICT2026_GeoFormerDock/`
trong repo chỉ là **ảnh chụp bản đã nộp**, giữ nguyên làm mốc so sánh, **không đồng bộ liên tục**.
**Kế hoạch đầy đủ:** `docs/revision_plan_reviews.md` (tài liệu này chỉ chia việc, không lặp lại
lý do; mã việc `C2-x`, `A-x`, `B-x`, `§11` đều trỏ về kế hoạch đó).

| | Ai | Nội dung |
|---|---|---|
| **PHẦN A** | 2 người (tự phân chia nội bộ) | Mọi thứ trong bản thảo **sửa được ngay**, không phụ thuộc kết quả chạy lại |
| **PHẦN B** | Đức | Toàn bộ code + chạy GPU + mọi chỗ trong bản thảo **mang số liệu mới** |

---

## 0. Làm việc trên Overleaf — 3 quy ước, chỉ 3 thôi

Overleaf soạn thảo cộng tác thời gian thực nên **không có xung đột file** — ba người sửa cùng
lúc vẫn được. Không cần luật sở hữu file, không cần tách file bảng. Chỉ cần 3 thứ:

**1. Đánh dấu chỗ chờ số bằng comment LaTeX.** Phần A gặp chỗ cần số mới thì viết:

```latex
% TODO-DUC: dien PR-AUC cua GeoFormerDock sau luot 1
```

Tuyệt đối không gõ số cũ vào rồi định bụng sửa sau — đó đúng là cách đã sinh ra lỗi R1-W7 (văn
bản ghi `R+ = 0.578` trong khi bảng ghi `0.851`). Trước khi nộp, tìm `TODO-DUC` trong Overleaf
phải ra **0 kết quả**.

**2. Overleaf là bản gốc duy nhất trong suốt 10 ngày.** Không ai sửa `.tex` trong repo. Đến ngày
nộp mới tải project về và cập nhật `docs/VNICT2026_GeoFormerDock/` một lần, phục vụ A-4 (công bố
mã nguồn). Sửa song song ở hai nơi là chắc chắn mất bài.

**3. Đức không sửa văn xuôi của phần A; phần A không viết 3 chỗ của Đức** (liệt kê ở **B-9**).
Ngoài ranh giới đó ra thì ai sửa gì cũng được, không cần xin phép nhau.

---

## PHẦN A — 2 người, sửa được ngay

### A-0. LÀM TRƯỚC TIÊN, trước khi thêm bất kỳ chữ nào 🔴

Bài hiện vừa đúng 6 trang nhờ vài thủ thuật bố cục sẽ vỡ ngay khi thêm nội dung. Không sửa
trước thì mọi việc viết sau đều phải làm lại.

- [ ] Xóa `\newpage` thủ công ở `tex/method.tex:46`
- [ ] Đổi `\begin{figure}[H]` → `[!t]` ở 5 chỗ: `method.tex:7,49,82` và `result.tex:35,46`
      (giữ nguyên `figure*[t]` ở `method.tex:16`)
- [ ] 3 hình đơn cột `width=0.5\textwidth` → `0.44` ở `method.tex:9,51,84`
      (một cột IEEEtran chỉ rộng ≈ `0.47\textwidth` → 3 hình này **đang tràn cột sẵn**)
- [ ] Bỏ comment `\usepackage{url}` ở `VNICT2026_Template_LaTeX.tex:331` (cần cho A-4)

### A-1. Dựng khung 3 bảng

Dựng sẵn khung để khi Đức có số thì chỉ việc dán vào, không phải vừa dựng bảng vừa điền số lúc
gấp:

- [ ] **Bảng III** (tính ổn định theo checkpoint) — khung theo mẫu ở `revision_plan_reviews.md` §0
- [ ] **Bảng IV** (ablation) — 6 dòng, các cột `Cấu hình | Params | PR-AUC | BalAcc | MAE | C-idx`
- [ ] **Bảng II** — đổi cấu trúc theo A-10
- [ ] Mọi ô số để trống kèm `% TODO-DUC:`

### A-2. Sửa lỗi số R+ — R1-W7 🔴

`tex/result.tex:33`

- Cũ: "…đổi lại **R+=0.578** giảm nhẹ so với hai mô hình này (0.945 và 0.928 tương ứng)."
- Mới: "…đổi lại **R+=0.851** thấp hơn hai mô hình này (0.945 và 0.928 tương ứng)."
- Kiểm chứng: $0.135 \times 0.851 + 0.865 \times 0.808 = 0.814 = $ Acc ✓ và $(0.851+0.808)/2 = 0.830 = $ BalAcc ✓

### A-3. Cân bằng lớp dùng chung cho cả 4 mô hình — R1-W2 🔴

Đây là lỗi diễn đạt, không phải lỗi thí nghiệm. Bằng chứng đã có sẵn, không cần chạy lại:
`results/models/{geoformerdock,gnina_dense,gnina_default2018,pafnucy}/training.log` đều ghi
`pose_balance_batch = True`.

- [ ] `tex/method.tex:65` — chuyển cả đoạn "Đồng thời, trước khi tính $\mathcal{L}_{pose}$, một
      tập con 64 mẫu…" sang Mục IV.B (đây là thiết lập huấn luyện, không phải kiến trúc). Ở
      `method.tex` chỉ để lại một câu trỏ sang IV.B.
- [ ] `tex/result.tex:28` — thay câu "Đối với nhánh phân loại tư thế của GeoFormerDock được huấn
      luyện với cân bằng lớp…" (câu này còn **thiếu vị ngữ**) bằng 3 câu:
      (1) cân bằng lớp minibatch áp dụng **đồng nhất cho cả bốn mô hình**;
      (2) hàm mất mát tiêu điểm $\gamma{=}2.0$, $\alpha{=}0.75$ và trung bình cân bằng theo lớp
      cũng dùng chung;
      (3) do đó chênh lệch BalAcc/PR-AUC giữa các kiến trúc **không đến từ khác biệt về chiến
      lược lấy mẫu**, và ảnh hưởng của chính chiến lược này được đo ở dòng ablation trong Bảng IV.

### A-4. Công bố mã nguồn — R1-W6 🔴

- [ ] Thêm mục `\textit{E. Công bố mã nguồn và dữ liệu}` ở cuối `tex/result.tex`, 2–3 câu, có
      `\url{https://github.com/ducnm-mimhus/VNICT2026_Docking_Paper}` và nhắc
      `scripts/00_download.sh` tải đúng phiên bản dữ liệu đã dùng
- [ ] Rà repo trước khi bài chỉ tới nó: xóa đường dẫn tuyệt đối của máy lab trong script, kiểm
      tra `docs/issues_and_fixes.md` và `model_architecture.md` (tài liệu nội bộ, đã public),
      cập nhật README nêu 3 lệnh tái lập Bảng II/III/IV

### A-5. Mô tả dữ liệu gốc và bài toán — R2-2 🔴

Thêm 4–5 câu vào `tex/method.tex:4` và/hoặc `tex/result.tex:24`, trước phần nói về tệp types:

- [ ] Dữ liệu gốc: cấu trúc 3D thực nghiệm của phức hợp protein–phối tử (PDBbind2016), được
      **tái ghép nối chéo (cross-docking)** để sinh nhiều tư thế giả định → CrossDocked2020
- [ ] Mỗi mẫu = (1 cấu trúc protein, 1 tư thế phối tử), lưu ở định dạng `.gninatypes`
- [ ] Bài toán 1: cho một tư thế do docking sinh ra, quyết định nó có **tái lập đúng** tư thế
      thực nghiệm không (RMSD ≤ 2 Å) — tức "kết quả docking này có đáng tin không"
- [ ] Bài toán 2: ước lượng ái lực liên kết (thang pK) cho mẫu có nhãn thực nghiệm
- [ ] Ý nghĩa thực tế (R2-4b): PR-AUC cao ⇒ khi lấy top-N tư thế theo điểm số, tỉ lệ tư thế đúng
      trong đó cao hơn ⇒ giảm trực tiếp số thí nghiệm phải làm
- [ ] Nói rõ vì sao 4.618 mẫu test nhưng chỉ 623 mẫu dùng tính chỉ số ái lực

### A-6. Thuật ngữ kèm nguyên gốc tiếng Anh — R2-1 🔴

Thêm nguyên gốc trong ngoặc ở **lần xuất hiện đầu tiên**, và **cả trong phần Từ khóa**
(`VNICT2026_Template_LaTeX.tex:446` — R2 nêu đích danh mục này).

| Tiếng Việt | Nguyên gốc |
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
| tập kiểm định | validation set |
| điểm dừng sớm | early stopping |
| đường biên Pareto | Pareto frontier |
| tái ghép nối chéo | cross-docking |

### A-7. Đặc điểm các kiến trúc so sánh — R2-3 🟡

`tex/result.tex:53`. Phần này **dùng số cũ là đủ**, viết được ngay:

- [ ] Đối chiếu đặc điểm: GNINA-D18 = CNN 3D thuần, 2.16M, không có mô-đun hình học tường minh ·
      GNINA-Ds = kết nối dày đặc, 0.46M, mạnh nhất ở ái lực · Pafnucy = CNN sâu 7.99M, thiết kế
      gốc cho hồi quy ái lực đơn nhiệm · GeoFormerDock = 1.59M, có luồng hình học tường minh
- [ ] Trả lời thẳng câu hỏi của R2 "đây có phải kết quả tốt nhất trên tập test chưa": **không** —
      đây là tốt nhất **trong khung so sánh thống nhất của nghiên cứu này** (cùng biểu diễn
      voxel, cùng siêu tham số, train từ đầu), không phải kết quả tốt nhất từng công bố trên
      CrossDocked2020

> ⚠️ Phần "cơ chế nào tạo ra ưu thế" thuộc **B-9** — chờ ablation, đừng viết trước.

### A-8. Văn phong tiếng Việt — R2-1, R2-5 🔴

Áp dụng cho **Mục I, II, III** (ba mục này không đụng số mới nên sửa thoải mái):

- [ ] Bỏ câu bị động dịch máy: "được thực hiện bằng", "được tiến hành", "mang lại hiệu của chính"
      (→ "hiệu quả chính")
- [ ] Phân biệt dứt khoát **"tập kiểm tra"** (test) và **"tập kiểm định"** (validation), không dùng lẫn
- [ ] Thống nhất "ái lực liên kết" xuyên suốt
- [ ] Mỗi đoạn ≤ 5 câu, mỗi câu ≤ 40 chữ. Đoạn ở `method.tex:23` hiện ~15 dòng liền — tách ra
- [ ] Bỏ dấu gạch nối `-` dùng thay dấu ngoặc đơn (bài hiện dùng rất nhiều)

### A-9. Cắt trang

Nội dung thêm ≈ **+2.0 cột**, phải cắt tương đương:

- [ ] Rút Mục II còn 4–5 câu · **−0.40 cột**
- [ ] Gộp Hình 3 + Hình 4 thành một hình 2 panel (có sẵn `figures/shared_backbone.png` chưa dùng) · **−0.35 cột**
- [ ] Rút đoạn quy ước dấu $y_{aff}$ ở `method.tex:23` từ ~15 dòng còn ~8 · **−0.25 cột**
- [ ] Hình đơn cột về `0.44` (đã làm ở A-0) · **−0.25 cột**
- [ ] Rút mô tả $\mathcal{L}_{dist}$, $\mathcal{L}_{anchor}$ còn 1 câu + 1 công thức gộp · **−0.30 cột**
- [ ] Rút đoạn Pareto ở IV.D · **−0.30 cột**

> ⚠️ **Không cắt tài liệu tham khảo.** `ref.bib` có 15 mục và **cả 15 đều đang được trích dẫn** —
> cắt mục nào cũng phải bỏ một trích dẫn thật.

### A-10. Bảng II đổi cấu trúc

Bảng hiện là 12 cột trong `table*` toàn trang cỡ `\footnotesize`; thêm `± std` vào 11 cột số là
không thể vừa.

- [ ] Bỏ cột **Acc** (suy được: $Acc = \pi R^+ + (1-\pi)R^-$, $\pi = 0.1349$) và **RMSE**
      (cùng chiều thông tin với MAE) — ghi lý do vào chú thích bảng
- [ ] Còn 10 cột: `Mô hình | Params | BalAcc | R+ | R- | PR-AUC | MAE | r | ρ | C-idx`

### A-11. Giai đoạn 2 — từ ngày 5, sau khi Đức bàn giao số

- [ ] Viết diễn giải quanh Bảng II và Bảng IV (Đức chỉ đổ số, không viết văn xuôi)
- [ ] Ghép bản thảo, **đo số trang thật** (không ước lượng), cắt tiếp theo A-9 nếu vượt
- [ ] Phản biện chéo: người viết mục nào thì người kia đọc
- [ ] Kiểm bất biến $Acc = \pi R^+ + (1-\pi)R^-$ trên mọi dòng bảng
- [ ] Tìm `TODO-DUC` trong Overleaf ra 0 kết quả; rà chính tả; xuất PDF

---

## PHẦN B — Đức: code + chạy + sửa bài theo kết quả mới

### B-0. Ngày 1 sáng, ưu tiên cao nhất 🔴

`tools/checkpoint_robustness.py` + Bảng III (Z0). **Không cần GPU, không cần `data/`** — chỉ đọc
`results/models/*/training_metrics_test.csv` đã có trong repo, mất ~1 giờ.

Đặt đầu hàng đợi vì đây là thứ **mở khóa việc cho phần A**: nó bảo chứng cho luận điểm
"PR-AUC là chỉ số chủ đạo" (QĐ-1), mà phần A cần để viết lại IV.C. Làm sau là phần A chờ 2 ngày.

### B-1 → B-4. Sửa code (ngày 1–2, không cần GPU)

- [ ] **A-0 kế hoạch:** tách val split theo receptor + kiểm tra (số mẫu `y_aff>0` trong val ≥ 500,
      giao receptor train/val = 0, giao val/test = 0)
- [ ] **A1a–A1e:** `training.py` ghi `training_metrics_val.csv`; ghi `selection_split`,
      `best_epoch`, `best_val_score` vào `summary.json`; **chọn ngưỡng phân loại trên val**;
      `run_inference.py`/`exact_metrics.py` đọc ngưỡng đó thay vì mặc định 0.5
- [ ] **B0:** thêm `--geo_ablation {none,no_geometry,concat_fusion,no_key_bias,simple_affinity}`.
      Kiểm bắt buộc: `none` phải ra đúng **1.594.573 tham số** và load được checkpoint cũ không
      lỗi `unexpected keys`
- [ ] **A2a:** `run_overnight_valsplit.sh` thêm vòng lặp seed, `OUTDIR=..._valsplit_s${SEED}`,
      bỏ qua nếu đã có `summary.json`

### B-5 → B-7. Chạy + tổng hợp (ngày 2–7)

Xếp **theo lượt seed**, không theo model, để cắt ngang ở đâu cũng có bảng hoàn chỉnh:

| Lượt | Runs | GPU-h | Sau lượt này có gì |
|---|---|---|---|
| 1 (seed 2026) | 4 mô hình | ~8h | Bảng II val-selected, 1 seed |
| 2 (seed 2027) | 4 mô hình | ~8h | mean ± nửa biên độ |
| 3 (seed 2028) | 4 mô hình | ~8h | mean ± std, đủ R1-W4 |

Ablation (val-split, seed 2026), **thứ tự ưu tiên cắt**:
**B-1 −luồng hình học** và **B-2 −cân bằng lớp** là hai cái *không được cắt* (R1-W2 và R1-W3 đòi
đích danh); rồi `concat_fusion`, `simple_affinity`, `no_key_bias`.

Công cụ phải viết: `tools/aggregate_seeds.py` (mean ± std, **xuất thẳng thân bảng LaTeX dán
được vào Overleaf** — đừng chép tay từ TSV) và bootstrap 2 tầng (lấy mẫu lại test **và** bốc
ngẫu nhiên seed).

### B-8. Dán số vào Overleaf

Chạy `aggregate_seeds.py --latex`, dán thân bảng vào đúng 3 bảng phần A đã dựng sẵn, và xóa các
dòng `% TODO-DUC` tương ứng. **Không chạm vào văn xuôi của phần A.**

### B-9. Ba chỗ văn xuôi Đức viết

Báo cho phần A biết trước để họ không đụng vào:

1. **`tex/method.tex:139` + Mục IV.B** — mô tả chọn checkpoint/ngưỡng trên validation, kèm số
   mẫu train-split / val / test thật sau khi tách (C2-3)
2. **`tex/result.tex` Mục IV.C + IV.D** — diễn giải Bảng III (tính ổn định) và Bảng IV
   (ablation), phần "cơ chế nào tạo ra ưu thế" (C2-4, phần còn lại của C2-8)
3. **`tex/conclusion.tex` + câu kết quả trong tóm tắt** — bỏ hạn chế (i) cũ (chọn checkpoint trên
   test, dao động 0.163 — đã sửa thật nên không còn là hạn chế), thêm hạn chế mới về số seed và
   về việc tách val theo receptor chứ chưa theo cụm tương đồng (C2-9)

### B-10. Cổng cắt — ngày 6, cả nhóm

Chốt theo số lượt đã chạy xong. Chi tiết ở `revision_plan_reviews.md` §3 A4 và §8.

---

## Ba mốc bàn giao

| Ngày | Đức giao gì | Mở khóa việc gì của phần A |
|---|---|---|
| **1 chiều** | Bảng III (Z0) + kết luận "PR-AUC là chỉ số chủ đạo" | Viết lại khung Mục IV.C, câu chuyện của tóm tắt |
| **5** | Số Bảng II (lượt 1) | Diễn giải Bảng II, bắt đầu ghép bài |
| **7** | Số Bảng IV (ablation) | Diễn giải ablation, chốt câu về đóng góp #1 |

---

## Hai điều cấm

1. **Phần A không tự điền số mới.** Chỗ nào cần số thì để `% TODO-DUC:`. Không viết số cũ vào
   rồi định bụng sửa sau — đó đúng là cách sinh ra lỗi R1-W7.
2. **Phần A không viết câu khẳng định về ưu thế Balanced Accuracy** cho tới khi có số lượt 1.
   Luận điểm **PR-AUC** thì viết được ngay (Z0 bảo chứng, không phụ thuộc lượt chạy mới), nhưng
   khoảng BalAcc giữa các mô hình **chồng lấn nhau** nên đó chính là chỗ có thể phải đổi luận
   điểm (kịch bản K1 trong kế hoạch).
