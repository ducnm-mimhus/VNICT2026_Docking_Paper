# Tổng hợp Vấn đề Cần Sửa & Cải thiện
**Mục tiêu:** Chuyển đổi khóa luận → Bài báo VNICT 2026  
**Nguyên tắc:** Chỉ viết lại, không phát triển thêm (trừ ablation nhỏ nếu kịp)

> **Phiên bản 2 – đã đối chiếu toàn bộ mã nguồn `dockbench/` và `scripts/run_training.sh`.**
>
> Thay đổi so với bản 1:
> - **VĐ2 và VĐ3 đã có lời giải từ code** → chuyển sang trạng thái ✅ ĐÃ GIẢI, kèm câu trả lời
>   chính xác. Không cần Đức kiểm tra lại nữa.
> - **Thêm VĐ9–VĐ12**, trong đó **VĐ9, VĐ10, VĐ11 đều ở mức 🔴 CRITICAL** và theo đánh giá
>   là nặng ngang hoặc nặng hơn VĐ1.
> - VĐ1, VĐ4, VĐ5, VĐ7 được sửa lại lập luận / cách sửa cho khớp với code.
> - **Đánh số VĐ1–VĐ8 giữ nguyên** để không phá vỡ tham chiếu từ các tài liệu khác.

---

## Phân loại theo mức độ ưu tiên

| Mức | Nhãn | Ý nghĩa |
|---|---|---|
| 🔴 CRITICAL | Bắt buộc sửa | Reviewer sẽ reject nếu không có |
| 🟡 HIGH | Rất nên sửa | Ảnh hưởng lớn đến chất lượng bài |
| 🟢 MEDIUM | Nên sửa nếu kịp | Cải thiện điểm số nhưng không quyết định accept |
| ✅ ĐÃ GIẢI | Không còn là vấn đề mở | Chỉ cần chép kết quả vào bài |

## Bảng điều hướng nhanh (đọc theo thứ tự này)

| Thứ tự xử lý | VĐ | Mức | Tóm tắt | Chi phí |
|---|---|---|---|---|
| 1 | **VĐ11** | 🔴 | Bảng 6 mâu thuẫn số học ở 4 dòng — phải đối chiếu lại số liệu gốc | 1 giờ |
| 2 | **VĐ9** | 🔴 | Mô hình được chọn bằng chính tập test (không có validation) | 2 giờ viết |
| 3 | **VĐ10** | 🔴 | 7 mô hình **không** dùng cùng `L_reg` — GNINA dùng NLL, còn lại dùng Huber | 2 giờ viết |
| 4 | VĐ1 | 🔴 | 3 baseline đồ thị cho kết quả bằng ngẫu nhiên | 1 giờ |
| 5 | VĐ12 | 🟡 | Tài liệu kiến trúc + README mô tả sai kiến trúc/lịch train | 1 giờ |
| 6 | VĐ4 | 🟡 | Thiếu CI + C-index là đại lượng ngẫu nhiên | 1 ngày |
| 7 | VĐ5 | 🟡 | Overclaim Related Work | 30 phút |
| 8 | VĐ6 | 🟡 | Minh bạch siêu tham số | 30 phút |
| 9 | VĐ7 | 🟢 | Ablation | 1–2 ngày GPU |
| 10 | VĐ8 | 🟢 | Cắt nội dung thừa | 2 giờ |
| — | VĐ2 | ✅ | Target ái lực — đáp án là **(B)** | chỉ chép |
| — | VĐ3 | ✅ | `B_pocket` — công thức đã xác định | chỉ chép |

---

## 🔴 VẤN ĐỀ 1: Ba baseline đồ thị cho kết quả bằng ngẫu nhiên – CRITICAL

### Mô tả

Bảng 6 liệt kê EquiBind, TankBind, PotentialNet là "mạng đồ thị" / "học sâu hình học", nhưng
Balanced Accuracy = 0.503–0.513 — **đúng bằng ngẫu nhiên**.

Bản 1 giải thích là "do cài đặt lại tự viết". Giải thích đó đúng nhưng chưa đủ; đối chiếu code
cho thấy **nguyên nhân kỹ thuật cụ thể**, và đây mới là điều cần biết để quyết định giữ hay bỏ:

**(a) Toàn bộ thông tin hình học không khả vi.**
`extract_pseudo_atoms_from_grid` chọn top-K voxel bằng `topk` nhưng **chỉ giữ chỉ số nguyên**
(`common.py:88`); tọa độ `positions` được dựng từ số học nguyên trên chỉ số đó
(`common.py:94-101`), nên tensor **không có `grad_fn`**. Mọi khoảng cách, mã hóa RBF, mask
cutoff trong ba mô hình này đều là **hằng số** đối với mạng. Gradient chỉ về được stem qua
`gather` tại đúng K voxel — với K=48 trên lưới 24³ là **0.35% số voxel**. Mạng học được đặc
trưng *tại* pseudo-atom nhưng không học được cách *chọn* pseudo-atom.

**(b) TankBind: module trigonometry bị vô hiệu.**
`_TriangleMultiplicativeUpdate.forward` dùng `out.mean(dim=2, keepdim=True).expand(...)`
(`tankbind.py:70`), khiến số hạng cập nhật **giống hệt nhau ở mọi chỉ số protein `j`**. Module
đóng góp 0 tín hiệu phân biệt trên đúng trục nó được thiết kế để tinh chỉnh.

**(c) PotentialNet: đồ thị suy biến.**
`spatial_cutoff = 1.4` trên tọa độ chuẩn hóa `[−1,1]³` (khoảng cách tối đa `2√3 ≈ 3.46`), trong
khi pseudo-atom vốn cụm sát nhau quanh vùng mật độ cao → mask không gian gần như all-ones, đồ
thị suy biến thành fully-connected không có tính chọn lọc hình học. Cộng thêm
`agg = msg.sum(dim=2)` không chuẩn hóa (`potentialnet.py:62`), 48 message đổ vào một `GRUCell`
gây bão hòa.

### Rủi ro nếu không sửa

Reviewer biết lĩnh vực sẽ phân loại đây là "so sánh không trung thực" → **reject**, không phải
điểm yếu thông thường. Đồng thời, mọi kết luận dạng "nhóm mô hình đồ thị kém hơn nhóm CNN trên
bài toán này" đều **không có căn cứ** — ba dòng đó chỉ chứng minh rằng bản port sang voxel đã hỏng.

### Cách sửa

**➡️ Khuyến nghị: Phương án B — bỏ hẳn 3 baseline này.**

Chỉ giữ 3 baseline voxel thực thụ: GNINA Dense, GNINA Default2018, Pafnucy. Lý do nghiêng về B
thay vì A:

- Dù đổi tên thành "Graph-baseline-1/2/3", reviewer vẫn hỏi "tại sao bằng ngẫu nhiên?". Câu trả
  lời trung thực dẫn thẳng tới điểm (a) ở trên — tức là thừa nhận bản port có lỗi thiết kế.
- Bài 6–8 trang không đủ chỗ để trình bày và bảo vệ chuyện đó.
- Bỏ 3 dòng làm bảng gọn hơn, câu chuyện Pareto (giữa CNN voxel và mô hình đề xuất) sắc nét hơn.

**Phương án A (nếu vẫn muốn giữ để có phần phân tích hai nhóm):** đổi tên trong bảng thành
```
Graph-baseline-1 (message passing, voxel re-impl.)
Graph-baseline-2 (distance-aware attention, voxel re-impl.)
Geo-baseline (SE(3)-inspired, voxel re-impl.)
```
và **bắt buộc** kèm ghi chú nêu đúng nguyên nhân, không được nói chung chung:

> *"Do các mô hình gốc yêu cầu đầu vào tọa độ nguyên tử không tương thích với thiết lập voxel
> thống nhất, chúng tôi cài đặt lại các thành phần cốt lõi trên biểu diễn voxel thông qua một
> tập pseudo-atom được trích từ lưới. Trong cách trích này, tọa độ pseudo-atom được xác định
> bằng phép chọn top-K rời rạc nên **không khả vi**; mạng vì vậy không thể tối ưu vị trí
> pseudo-atom trong quá trình huấn luyện. Kết quả của ba mô hình này do đó phản ánh giới hạn
> của bản chuyển thể, **không** so sánh trực tiếp được với số liệu công bố của các tác giả gốc,
> và không nên dùng để kết luận về năng lực của họ mô hình đồ thị nói chung."*

Dù chọn A hay B, **Mục V (Hạn chế) phải nói rõ** không rút ra kết luận so sánh nhóm CNN vs nhóm đồ thị.

---

## ✅ VẤN ĐỀ 2: Target ái lực có dấu – ĐÃ GIẢI

### Câu trả lời: **(B) — chỉ dùng mẫu có `y_aff > 0`, cả khi huấn luyện lẫn khi đánh giá.**

Không cần Đức kiểm tra thêm. Bằng chứng:

| Giai đoạn | Cơ chế lọc | Vị trí |
|---|---|---|
| **Huấn luyện** | `mask = (affinities > 0).float()` truyền vào `CombinedAffinityLoss` | `training.py:496` |
| — cả 4 thành phần loss | mỗi module gọi `_resolve_mask(target, mask)` | `losses.py:26-30, 250, 308, 335, 352` |
| **Eval MAE / RMSE** | `output_transform_affinity_good` lọc `target > 0` | `transforms.py:131-147` |
| **Eval r / ρ / C-index** | `_AffinityAccumulator.update` lọc `target > 0` | `metrics.py:368-373` |

→ **Kịch bản xấu (2B trong bản 1) KHÔNG xảy ra.** C-index 0.789 và r 0.779 *không* phản ánh khả
năng phân biệt dấu / phân tách pose. Bảng 7 hợp lệ về mặt này. Áp dụng hướng xử lý 2A.

### Việc phải làm trong bài

**(1) Thêm phương trình tường minh vào Mục III (Phương pháp).** ⚠️ Lưu ý điều kiện là
`y_aff > 0`, **không phải** `y_pose = 1` — code hoàn toàn không đọc nhãn pose khi lọc:

```
Cả bốn thành phần của L_aff chỉ được tính trên tập
    S = { i : y_aff,i > 0 }
```

Phương trình `S = {i : y_pose_i = 1, y_aff_i ∈ ℝ⁺}` ở bản 1 là **sai** — hai tập chỉ trùng nhau
nếu mọi good pose đều có nhãn ái lực dương.

**(2) Thêm 1–2 câu vào phần Tiền xử lý:**
> *"Giá trị ái lực trong tệp `.types` được mã hóa có dấu nhằm mục đích lưu trữ kép: giá trị
> dương tương ứng pK thực nghiệm của cấu hình gần native, giá trị âm đánh dấu decoy. Nhánh hồi
> quy chỉ được giám sát trên tập con có giá trị dương; các mẫu decoy không đóng góp gradient
> cho nhánh này."*

**(3) ⚠️ BẮT BUỘC — báo cáo N của tập con.** MAE / RMSE / r / ρ / C-index **không** được tính
trên 4.618 mẫu test mà trên tập con `y_aff > 0`. Bảng 7 hiện đang ngầm ám chỉ 4.618. Phải ghi
rõ số mẫu của tập con này ngay dưới bảng, ví dụ: *"Các chỉ số ái lực được tính trên N = ___ mẫu
test có nhãn ái lực thực nghiệm (y_aff > 0)."*
👉 **Việc của Đức:** lấy con số N từ log train (`Affinity target stats (N=...)`) hoặc đếm trực
tiếp trên `ref_uff_test0.types`.

**(4) Thêm 1 câu về chuẩn hóa target** — bản 1 bỏ sót hoàn toàn:
> *"Nhãn ái lực được chuẩn hóa z-score bằng trung bình và độ lệch chuẩn ước lượng trên tập
> huấn luyện; dự đoán được đưa về thang pK gốc trước khi tính các chỉ số đánh giá."*
(`target_normalizer.py`, `training.py:1076-1081, 677-678`)

---

## ✅ VẤN ĐỀ 3: B_pocket chưa được định nghĩa – ĐÃ GIẢI

### Công thức chính xác (chép thẳng vào Mục III)

`geoformerdock.py:45-72`. Với `x_j ∈ ℝ^128` là token thứ `j` sau LayerNorm, `h` là chỉ số head
(4 head, `d_h = 32`):

$$
\text{Attention}^{(h)}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_h}} + B^{(h)}_{\text{pocket}}\right) V,
\qquad
B^{(h)}_{\text{pocket}}[i,j] = \sigma\!\left(\mathbf{w}_h^\top \mathbf{x}_j + b_h\right)
$$

trong đó `W_p = [w_1 … w_4]ᵀ ∈ ℝ^{4×128}`, `b_p ∈ ℝ^4` là tham số học được
(1.032 tham số, chiếm 0.06% mô hình).

### ⚠️ Gợi ý công thức ở bản 1 là SAI

Bản 1 đề xuất `B_pocket[i,j] = MLP_bias(φ_dist(‖c_i − c_j‖₂))`. **Code không làm điều đó.**
Nếu viết công thức khoảng cách vào bài trong khi code không có, đó là **mô tả sai kiến trúc** —
lỗi nặng hơn nhiều so với việc thừa nhận bản chất thật của cơ chế.

### Ba tính chất phải nêu trung thực

1. **Không dùng bất kỳ tọa độ 3D nào.** Bias được suy hoàn toàn từ *nội dung* token.
2. **Chỉ phụ thuộc token key `j`**, độc lập với query `i` → là **per-key bias**, không phải
   pairwise bias. Nó nâng/hạ đồng đều mức chú ý mà mọi token dành cho token `j`.
3. **Miền giá trị `(0,1)`** do sigmoid — nhỏ so với biên độ của `QKᵀ/√d_h`, nên là hiệu chỉnh
   nhẹ chứ không phải cơ chế chi phối.

### Hệ quả về cách gọi tên

- **Không viết:** *"giúp các cặp token ở vùng không gian gần nhau nhận attention cao hơn"*
  (câu này ở tài liệu kiến trúc bản 1) — sai sự thật.
- **Tên trung thực:** *content-based per-key saliency gate*.
- Nếu vẫn muốn giữ nhãn "pocket-aware" trong tên mô hình (hợp lý, vì đã dùng trong khóa luận),
  phải giải thích rằng tính pocket-aware đến **gián tiếp** qua việc token mang thông tin mật độ
  nguyên tử của vùng nó phủ, chứ không qua hình học tường minh.

---

## 🟡 VẤN ĐỀ 4: Không có khoảng tin cậy – HIGH

### Mô tả

"MAE 1.134 tốt hơn hẳn Pafnucy (1.149)" – chênh 1.3%, không có CI. Các từ "bứt phá", "áp đảo",
"vượt trội" sẽ bị gạch đỏ.

### Cách sửa (không cần chạy lại model)

**1. ⚠️ Phải dùng PAIRED bootstrap, không phải bootstrap độc lập từng mô hình.**

Bootstrap riêng từng mô hình rồi so hai khoảng CI **không** trả lời được câu hỏi cần trả lời.
Phải resample **cùng một bộ chỉ số** cho cả hai mô hình và lấy CI của **hiệu số**:

```python
import numpy as np

def paired_bootstrap_delta(y_true, pred_a, pred_b, metric_fn,
                           n_boot=2000, alpha=0.05, seed=2026):
    """CI của Δ = metric(a) − metric(b) trên cùng chỉ số resample."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    deltas = np.empty(n_boot)
    for k in range(n_boot):
        idx = rng.integers(0, n, n)                      # cùng idx cho cả a và b
        deltas[k] = metric_fn(y_true[idx], pred_a[idx]) - metric_fn(y_true[idx], pred_b[idx])
    lo, hi = np.percentile(deltas, [100*alpha/2, 100*(1-alpha/2)])
    return float(deltas.mean()), float(lo), float(hi)
```

Chỉ khi CI của Δ **không chứa 0** mới được viết "tốt hơn đáng kể".

**2. ⚠️ Nguồn nhiễu bị bỏ sót ở bản 1: C-index trong code là đại lượng NGẪU NHIÊN.**

`concordance_index` (`metrics.py:200-217`) lấy mẫu tối đa **50.000 cặp** bằng `torch.randint`
**không seed**. Với vài nghìn mẫu, số cặp thật lên tới hàng triệu → chỉ một phần nhỏ được lấy.
**Chạy lại eval trên cùng một checkpoint sẽ cho C-index khác nhau ở chữ số thập phân thứ 3.**

Trước khi báo cáo `0.789` và `0.800` như hai con số khác nhau, phải:
- Chạy `test_evaluator` **20 lần** trên cùng `best_model.pt`, ghi lại C-index mỗi lần.
- Báo cáo độ lệch chuẩn. Nếu σ ≳ 0.005 thì khoảng cách 0.011 giữa GeoFormerDock và GNINA nằm
  gọn trong nhiễu lấy mẫu và **không được diễn giải là khác biệt thực**.
- Cùng lý do, `SoftProbabilisticRankingLoss` cũng lấy cặp ngẫu nhiên — nhưng đó là loss, không
  ảnh hưởng tới việc báo cáo.

**3. Cập nhật bảng kết quả** thành dạng `0.789 [0.771–0.805]`, và thêm một cột/hàng cho
`Δ so với baseline mạnh nhất [CI]` ở các chỉ số chính (PR-AUC, Balanced Accuracy, MAE, C-index).

**4. Hạ tông ngôn ngữ:**
- "tốt hơn hẳn" → "nhỉnh hơn" (nếu CI của Δ chứa 0) hoặc "tốt hơn đáng kể" (nếu không chứa 0)
- "bứt phá" → "dẫn đầu rõ rệt"
- "áp đảo" → "cao nhất trong nhóm so sánh"
- Giữ giọng mạnh chỉ ở PR-AUC (khoảng cách +0.084 so với Pafnucy) — và **chỉ khi** paired
  bootstrap xác nhận.
- ⚠️ Đồng thời xóa câu tương tự ở tài liệu kiến trúc §8.3 bản 1: *"không thể quy về nhiễu với
  dataset 4.618 mẫu"* — đây đúng là loại khẳng định mà VĐ4 muốn loại bỏ.

---

## 🟡 VẤN ĐỀ 5: Overclaim trong phần Related Work – HIGH

### Mô tả

Câu trong khóa luận (mục 1.3.2):
> *"Từ các công trình đã công bố, có thể thấy chưa tồn tại một mô hình duy nhất giải quyết tốt
> đồng thời mọi yêu cầu của bài toán."*

Vấn đề: **GNINA 1.0** chính là mô hình đa nhiệm pose+affinity và là baseline của bài này.

### Cách sửa

> *"Các mô hình hiện có thường tối ưu riêng rẽ từng nhiệm vụ hoặc không phân tích tường minh sự
> đánh đổi giữa chất lượng phân loại cấu hình và dự đoán ái lực. GNINA là tiền lệ gần nhất khi
> huấn luyện đồng thời hai nhiệm vụ; **theo hiểu biết của chúng tôi**, sự đánh đổi giữa hai
> nhiệm vụ này chưa được định lượng tường minh trên cùng một tập dữ liệu mất cân bằng lớp với
> PR-AUC làm chỉ số chính."*

⚠️ **Lưu ý:** câu thay thế ở bản 1 (*"chưa có nghiên cứu nào đánh giá đồng thời hai nhiệm vụ...
với PR-AUC là chỉ số chính"*) vẫn là một tuyên bố "chưa có ai" và vẫn cần khảo sát để bảo vệ.
Cụm **"theo hiểu biết của chúng tôi"** là bắt buộc, hoặc bỏ hẳn vế đó.

---

## 🟡 VẤN ĐỀ 6: Siêu tham số thiên lệch về mô hình đề xuất – HIGH

### ✅ Xác nhận từ code

`run_training.sh:111-214` dùng **cùng một hàm `train_model`** cho cả 7 mô hình; mọi siêu tham số
huấn luyện (lr, batch, weight decay, lịch LR, clipping, các trọng số loss, cấu hình focal,
cân bằng minibatch, tiêu chí early stopping, seed) **giống hệt nhau**. Khác biệt duy nhất:
- `--max_pseudo_atoms 12` chỉ áp cho `geoformerdock` (`run_training.sh:131-132`);
- AMP bật cho 4 mô hình CNN nặng (`run_training.sh:127-130`).

→ Đã xác nhận: **có, dùng chung**.

### Cách sửa

Thêm vào Mục IV.2:
> *"Các siêu tham số chung (batch size, optimizer, learning rate schedule, trọng số các thành
> phần mất mát, chiến lược cân bằng lớp) được giữ nhất quán cho mọi mô hình; các siêu tham số
> kiến trúc (embedding dim, số lớp Transformer, số pseudo-atom) theo định nghĩa chỉ áp dụng cho
> mô hình đề xuất."*

### ⚠️ Hai điểm bất lợi cho tính công bằng — phải thừa nhận ở Mục V

- **(a)** `lr = 1e-3`, `batch = 1024` là cấu hình được tinh chỉnh cho mô hình đề xuất. Pafnucy
  nguyên bản dùng learning rate thấp hơn nhiều và dropout 0.5. Ép chung một cấu hình là "công
  bằng" về hình thức nhưng có thể bất lợi cho baseline.
- **(b)** AMP chỉ bật cho `gnina_dense`, `gnina_default2018`, `pafnucy`, `geoformerdock`
  (`run_training.sh:127-130`), không bật cho 3 mô hình đồ thị → không đồng nhất hoàn toàn về
  numerics giữa các mô hình.

---

## 🟢 VẤN ĐỀ 7: Thiếu Ablation Study – MEDIUM

### Mô tả

Ba đóng góp được tuyên bố (geometry encoder, `B_pocket`, `L_rank`+`L_anchor`) không có bằng
chứng thực nghiệm độc lập.

### ✅ Tin tốt: ablation geometry encoder KHÔNG cần sửa code

`--max_pseudo_atoms 0` → `extract_pseudo_atoms_from_grid` trả về K=0 (`common.py:77-82`) →
`SimpleGeometryEncoder` trả vector 0 (`geoformerdock.py:157-164`). Chạy được ngay:

```bash
GEOFORMER_MAX_PSEUDO_ATOMS=0 ONLY_MODEL=geoformerdock bash scripts/run_training.sh
```

⚠️ Hai lưu ý khi diễn giải:
- Đây là **vô hiệu hóa đầu vào hình học**, không phải "bỏ" khối — `out_proj` và
  `GatedFeatureFusion` vẫn tồn tại và vẫn nhận vector 0. Phải mô tả đúng như vậy.
- Vì tọa độ pseudo-atom **không khả vi** (xem VĐ1(a)) và geometry encoder chỉ chiếm **0.7%
  tham số**, rất có khả năng Δ ≈ 0. **Cần chuẩn bị tinh thần kết quả phủ định đóng góp #1** — và
  nếu vậy thì phải bỏ tuyên bố đóng góp riêng cho thành phần này (xem cuối mục).

### Ablation `B_pocket`

Không có flag; phải comment 3 dòng `geoformerdock.py:65-67`. ~5 phút. Cũng chỉ chiếm 0.06%
tham số → kỳ vọng Δ nhỏ.

### ⚠️ Ablation quan trọng nhất bị bỏ sót ở bản 1: bỏ cân bằng lớp trong minibatch

`run_training.sh:185-186` bật `--pose_balance_batch --pose_balance_target_per_class 32`
(mỗi minibatch được lấy mẫu lại thành 32 good + 32 bad trước khi tính pose loss —
`training.py:524-533`). Đây **nhiều khả năng là nguồn chính** của Balanced Accuracy 0.830 và
Recall(neg) 0.724, chứ không phải geometry encoder hay `B_pocket`.

Nếu reviewer hỏi *"vì sao Balanced Accuracy cao hơn hẳn?"* và câu trả lời là một thủ thuật
huấn luyện chứ không phải kiến trúc, cả ba đóng góp kiến trúc đều lung lay. **Nên chạy ablation
này trước hai cái kia** nếu chỉ đủ thời gian cho một:

```bash
# bỏ --pose_balance_batch khỏi run_training.sh dòng 185-186
```

### ⚠️ Bỏ đề xuất "chỉ cần 40 epoch" ở bản 1

Cấu hình thật chọn mô hình bằng early stopping composite trên test với patience 25 lần đánh giá
× `test_every=2` = **50 epoch không cải thiện** (`run_training.sh:75-77`). Cắt xuống 40 epoch
nghĩa là mô hình ablation được chọn theo một quy tắc **khác** với mô hình full → hai con số
không so sánh được với nhau.

Nếu bắt buộc phải cắt để tiết kiệm GPU: **cắt cả mô hình full xuống cùng số epoch** và dùng nó
làm dòng baseline riêng của bảng ablation, tách khỏi Bảng 6/7.

### Bảng ablation mục tiêu

```
Mô hình A (full)         : cấu hình đầy đủ                → PR-AUC / BalAcc / MAE / C-index
Mô hình B (no-balance)   : bỏ --pose_balance_batch        → ...
Mô hình C (no-geo)       : --max_pseudo_atoms 0           → ...
Mô hình D (no-Bpocket)   : comment geoformerdock.py:65-67 → ...
```
Tất cả các dòng phải chạy **cùng số epoch và cùng quy tắc chọn mô hình**.

**Nếu không kịp chạy:** bỏ phần tuyên bố đóng góp của từng thành phần, chỉ tuyên bố đóng góp
tổng thể của kiến trúc. Đây là lựa chọn an toàn và hoàn toàn chấp nhận được với bài 6–8 trang.

---

## 🟢 VẤN ĐỀ 8: Cấu trúc bài báo – Thông tin thừa – MEDIUM

### Nội dung cắt bỏ hoàn toàn khi viết bài báo

| Phần trong khóa luận | Lý do cắt |
|---|---|
| Chương 1 (protein, thuốc, docking truyền thống) | Kiến thức cơ sở, không phải đóng góp |
| Mục 2.3.4 – Backprop, AdamW (công thức đầy đủ) | Không ai đưa công thức Adam vào bài báo |
| Ví dụ số minh họa gradient (0.1 vs 10) | Quá chi tiết, không cần thiết |
| Bảng 3 – 28 kênh đặc trưng | Giảm còn 1 câu tham chiếu molgrid |
| Hình 13–16 (training curves chi tiết) | Giữ 1 hình tổng hợp, bỏ các hình còn lại |

⚠️ Khi cắt Bảng 3, **phải giữ lại** các con số tái lập: `C = 28`, lưới `48³`, resolution
`0.5 Å`, hộp `23.5 Å`.

### Hình ưu tiên giữ lại

1. **Hình 3** (kiến trúc tổng quan) – vẽ lại gọn hơn, ngang thay vì dọc
2. **Hình 17** (Pareto frontier) – đây là hình mạnh nhất, nên là hình chính
3. **Hình 12** (params vs C-index) – hình phụ nếu còn chỗ.
   ⚠️ Cần chú thích: 82% tham số của GNINA Default2018 nằm ở `log_sigma_head`
   (`gnina_default2018.py:69`), một nhánh mà các mô hình khác không có → trục hoành của hình này
   không so sánh được như-với-như. Xem VĐ10.
4. **Bảng 6 + 7** gộp lại kèm bootstrap CI

---

## 🔴 VẤN ĐỀ 9 (MỚI): Mô hình được chọn bằng chính tập test – CRITICAL

### Mô tả

`training.py:1528-1591`: cứ 2 epoch một lần, pipeline chạy đánh giá **trên `test_loader`**, tính
`composite = 0.5·C-index + 0.5·Balanced Accuracy`, ghi đè `best_model.pt` mỗi khi composite cải
thiện, và early-stop theo chính chỉ số đó. `summary.json` báo cáo `best_metrics`
(`training.py:1658, 1667-1682`).

**Toàn bộ pipeline không có tập validation.** Nghĩa là: mọi con số trong Bảng 6 và Bảng 7 là
**giá trị tốt nhất trên tập test qua ~50 lần đo**, cho cả 7 mô hình.

### Rủi ro nếu không sửa

Đây là lỗi phương pháp mà bất kỳ reviewer ML nào cũng nhận ra ngay, và theo đánh giá là **nặng
hơn VĐ1**: VĐ1 chỉ ảnh hưởng 3 dòng baseline, còn VĐ9 ảnh hưởng **mọi con số trong bài**, kể cả
của mô hình đề xuất. Nếu bị phát hiện mà bài không tự khai báo, uy tín toàn bộ phần thực nghiệm
sụp đổ.

### Cách sửa (không cần train lại)

**1. Khai báo tường minh trong Mục IV.2 (Thiết lập thực nghiệm).** Bắt buộc:
> *"Do hạn chế về tài nguyên tính toán, chúng tôi không tách tập validation riêng; mô hình được
> chọn bằng early stopping trên tập kiểm thử với tiêu chí tổ hợp `0.5·C-index +
> 0.5·Balanced Accuracy`, đánh giá mỗi 2 epoch. **Quy tắc này được áp dụng đồng nhất cho cả bảy
> mô hình**, nên so sánh tương đối giữa các mô hình vẫn có ý nghĩa; tuy nhiên các giá trị tuyệt
> đối là ước lượng lạc quan cho hiệu năng trên dữ liệu mới."*

**2. Nhắc lại ở Mục V (Hạn chế)** như một hạn chế được thừa nhận, cùng với "chỉ 1 fold".

**3. Bổ sung số liệu rẻ và thuyết phục — nên làm.** Từ `training_metrics_test.csv` đã có sẵn
(không cần GPU), báo cáo thêm, cạnh giá trị best:
   - giá trị tại **epoch cuối** (tương ứng `final_model.pt`), hoặc
   - **trung bình ± độ lệch chuẩn của 5 lần đánh giá cuối**.

   Việc này cho reviewer thấy biên độ dao động thật và chứng minh khoảng cách giữa các mô hình
   không phải do may mắn ở một epoch. Chi phí: một script pandas ~20 dòng.

👉 **Bổ sung vào phân công:** Đức, ưu tiên 🔴, cùng deadline với việc #5 (bootstrap CI).

---

## 🔴 VẤN ĐỀ 10 (MỚI): Bảy mô hình KHÔNG dùng cùng hàm mất mát ái lực – CRITICAL

### Mô tả

`training.py:517-519`:
```python
log_sigma = model.get_log_sigma() if hasattr(model, "get_log_sigma") else None
```
`ConfidenceAwareRegressionLoss` rẽ nhánh theo giá trị này (`losses.py:261-268`):

| Mô hình | Có `get_log_sigma()`? | → `L_reg` thực tế |
|---|---|---|
| **GNINA Dense** | ✅ `gnina_dense.py:172` | **Gaussian NLL dị phương sai** `0.5·e^{−2σ}(y−ŷ)² + σ` |
| **GNINA Default2018** | ✅ `gnina_default2018.py:87, 160` | **Gaussian NLL dị phương sai** |
| GeoFormerDock | ❌ trả `None` (`uncertainty=False`, cờ `--geoformer_uncertainty` không bật) | **Huber** |
| Pafnucy, PotentialNet, EquiBind, TankBind | ❌ không có method | **Huber** |

### Tại sao nghiêm trọng

**Hai mô hình duy nhất vượt GeoFormerDock ở MAE / RMSE / r / C-index (Bảng 7) chính là hai mô
hình duy nhất được huấn luyện bằng một hàm mất mát khác.** Gaussian NLL dị phương sai tự động
hạ trọng số các mẫu khó — đúng cơ chế cải thiện MAE và hệ số tương quan.

Điều này **phủ định trực tiếp** hai tuyên bố hiện có trong `README.md`:
- dòng 98: *"Tất cả 7 mô hình dùng **cùng** hàm mất mát affinity"*
- dòng 42: *"so sánh công bằng 7 kiến trúc trên cùng dữ liệu, **cùng loss**, cùng siêu tham số"*

Nếu bài báo lặp lại tuyên bố "cùng loss" thì đó là **phát biểu sai có thể kiểm chứng được**.

### Cách sửa (theo ràng buộc "chỉ viết lại")

**1. Không được viết "cùng hàm mất mát cho mọi mô hình".** Thay bằng phát biểu chính xác trong
Mục IV.2:
> *"Tất cả các mô hình dùng chung khung mất mát đa nhiệm và cùng bộ trọng số. Riêng thành phần
> hồi quy `L_reg` có hai biến thể tùy theo mô hình có cung cấp ước lượng phương sai hay không:
> GNINA Dense và GNINA Default2018 có nhánh `log σ` nên sử dụng dạng log-likelihood Gaussian dị
> phương sai; các mô hình còn lại, bao gồm mô hình đề xuất, sử dụng dạng Huber."*

**2. Thừa nhận confound ở Mục V (Hạn chế):**
> *"Khoảng cách về MAE và C-index giữa mô hình đề xuất và hai biến thể GNINA chịu ảnh hưởng của
> khác biệt trong dạng hàm hồi quy, không thuần túy phản ánh khác biệt kiến trúc."*

**3. Điểm này thực ra CÓ LỢI cho câu chuyện Pareto** — nó giải thích vì sao GNINA thắng ở nhánh
ái lực mà không cần thừa nhận kiến trúc đề xuất yếu hơn. Nên dùng chủ động thay vì né tránh.

**4. Hệ quả phụ về số tham số:** `log_sigma_head` của `gnina_default2018` là
`Linear(27648, 64)` = **1.769.536 tham số ≈ 82%** trong tổng 2.16M. Do đó phân tích "hiệu năng
trên mỗi tham số" (đóng góp #3, Hình 12) bị bóp méo → **nên hạ tông tuyên bố đóng góp #3** và
thêm chú thích cho Hình 12.

**5. Nếu còn thời gian GPU (không bắt buộc):** train lại GeoFormerDock với
`--geoformer_uncertainty` để có một dòng so sánh cùng dạng loss với GNINA. Đây là cách sạch nhất
để giải quyết confound, nhưng nằm ngoài ràng buộc "chỉ viết lại".

---

## 🔴 VẤN ĐỀ 11 (MỚI): Bảng 6 mâu thuẫn nội tại về mặt số học – CRITICAL

### Mô tả

Trên cùng một tập test, luôn có đẳng thức **chính xác**:
$$\text{Acc} = \pi \cdot R_{\text{pos}} + (1-\pi)\cdot R_{\text{neg}}, \qquad \pi = P(y=1)$$

Trong code, `Accuracy` (`metrics.py:620-622`) và `Pose Recall Pos/Neg` (`metrics.py:633-642`)
đều dùng ngưỡng `P(good) ≥ 0.5` trên **toàn bộ** tập test, nên đẳng thức trên phải nghiệm đúng
tuyệt đối cho mọi dòng, và `π` phải **giống nhau ở mọi dòng**.

Giải `π` ngược từ từng dòng của Bảng 6:

| Mô hình | Acc | R_pos | R_neg | **π suy ra** | |
|---|---|---|---|---|---|
| PotentialNet | 0.140 | 0.997 | 0.009 | **0.133** | ✓ |
| GNINA Default2018 | 0.631 | 0.937 | 0.585 | **0.131** | ✓ |
| Pafnucy | 0.689 | 0.928 | 0.652 | **0.134** | ✓ |
| EquiBind | 0.215 | 0.997 | 0.025 | **0.196** | ✗ |
| GNINA Dense | 0.865 | 0.989 | 0.585 | **0.693** | ✗ |
| TankBind | 0.846 | 0.998 | 0.028 | **0.843** | ✗ |
| **Mô hình đề xuất** | **0.888** | 0.936 | 0.724 | **0.774** | ✗ |

Ba dòng nhất quán ở `π ≈ 0.13` (hợp lý với tỉ lệ good pose của CrossDocked). Bốn dòng còn lại —
**bao gồm chính mô hình đề xuất** — không thể cùng thuộc một tập test.

### Rủi ro nếu không sửa

Reviewer kiểm tra được bằng máy tính bỏ túi trong 2 phút. Một bảng kết quả tự mâu thuẫn về số
học là dấu hiệu đỏ nghiêm trọng nhất có thể có trong phần thực nghiệm.

### Cách sửa

**Đối chiếu lại toàn bộ Bảng 6 với dữ liệu gốc** trước khi đưa vào bài:
- `results/models/<model>/summary.json` → các khóa `final_acc`, `final_bal_acc`,
  `final_pose_recall_pos`, `final_pose_recall_neg`, `final_pr_auc`
- `results/models/<model>/training_metrics_test.csv` → kiểm tra các giá trị này có cùng đến từ
  **một epoch duy nhất** hay không

Nghi ngờ khả dĩ nhất: một số ô được chép từ **epoch khác nhau** hoặc từ **lần chạy khác nhau**.
`summary.json` lưu `best_metrics` như một snapshot của một epoch (`training.py:1570-1572`), nên
nếu lấy đúng từ đó thì phải nhất quán.

Sau khi sửa, **chạy lại phép kiểm tra** `Acc = π·R_pos + (1−π)·R_neg` cho toàn bộ 7 dòng với
cùng một `π` — coi đây là bước bắt buộc trước khi khóa bảng.

👉 **Bổ sung vào phân công:** Phát việc #7 (kiểm tra khớp số liệu), **nâng lên ưu tiên 🔴 và
chuyển deadline lên trước ngày 6** vì Đức #8 (viết Mục IV) phụ thuộc vào bảng này.

---

## 🟡 VẤN ĐỀ 12 (MỚI): Tài liệu nội bộ mô tả sai kiến trúc và lịch huấn luyện – HIGH

### Mô tả

`model_architecture.md` (bản 1) và `README.md` chứa một số mô tả không khớp code. Vì
Mục III được viết **dựa trên tài liệu kiến trúc**, nên các lỗi này sẽ chảy
thẳng vào bài báo nếu không chặn.

`model_architecture.md` **đã được cập nhật lên bản 2** với đầy đủ tham chiếu `file:dòng` —
xem phụ lục cuối tài liệu đó để có danh sách 14 điểm sai. Bốn điểm nặng nhất:

| Bản 1 viết | Thực tế |
|---|---|
| `B_pocket` là spatial bias theo khoảng cách 3D | Per-key gate, không dùng tọa độ (VĐ3) |
| Epoch 5–14 chưa bật `L_rank`; 15–29 ramp-up | **Không tồn tại** — `L_rank` đầy đủ từ epoch 5 |
| `L_dist` dùng bình phương | Trị tuyệt đối (`losses.py:340-341`) |
| Affine Calibrator là thành phần hoạt động | Bị đóng băng ở identity khi bật `--normalize_targets` |

Về lịch `L_rank`: `run_training.sh:175-176` **có** truyền `--rank_warmup_epochs 10
--rank_rampup_epochs 15`, nhưng `CombinedAffinityLoss.__init__` nhận rồi bỏ, không gán vào
`self` (`losses.py:367-395`); `forward()` luôn tính đủ bốn thành phần (`losses.py:428-437`).
`set_epoch()` chỉ ghi `_current_epoch` mà không nơi nào đọc. Nếu viết lịch ramp-up vào bài thì
đó là **mô tả sai một thí nghiệm đã chạy** — reviewer không phát hiện được, nhưng đây là vấn đề
liêm chính khoa học.

Tương tự, các tham số **hoàn toàn không được sử dụng**: `scale_pose_coupling`,
`use_gradient_alignment`, `scale_align`, `hard_neg_fraction`. Hàm mất mát ái lực có **4** thành
phần hoạt động, **không phải 5** như `README.md:96-104` mô tả.

### Cách sửa

1. ✅ `model_architecture.md` đã cập nhật lên bản 2 — dùng bản này làm nguồn duy nhất cho Mục III.
2. Cập nhật `README.md` các mục 5.1 và 5.2 cho khớp (xóa lịch ramp-up, sửa "5 thành phần" → 4,
   xóa tuyên bố "cùng loss" theo VĐ10).
3. Khi đọc phản biện Mục III, **đối chiếu từng công thức với
   phụ lục cuối `model_architecture.md`** thay vì chỉ đọc về tính dễ hiểu.

---

## Tóm tắt Checklist

```
🔴 CRITICAL – Phải xong trước khi nộp
[ ] VĐ11: Đối chiếu lại Bảng 6 với summary.json; kiểm tra Acc = π·Rpos + (1−π)·Rneg (1 giờ) ← LÀM TRƯỚC
[ ] VĐ9 : Khai báo việc chọn mô hình trên tập test + bổ sung số liệu epoch cuối (2 giờ)
[ ] VĐ10: Viết đúng về khác biệt L_reg giữa GNINA và các mô hình còn lại (2 giờ)
[ ] VĐ1 : Bỏ (khuyến nghị) hoặc đổi tên + ghi chú 3 baseline đồ thị (1 giờ)
[ ] VĐ2 : Chép phương trình S = {i : y_aff > 0} + lấy N của tập con good-pose (30 phút)
[ ] VĐ3 : Chép công thức B_pocket + 3 tính chất; XÓA mọi diễn giải "spatial" (30 phút)

🟡 HIGH – Nên xong trước khi nộp
[ ] VĐ12: Cập nhật README.md cho khớp code; dùng model_architecture.md v2 cho Mục III (1 giờ)
[ ] VĐ4a: Paired bootstrap CI cho hiệu số Δ từ prediction files (nửa ngày code)
[ ] VĐ4b: Đo độ lệch chuẩn C-index qua 20 lần eval trên cùng checkpoint (1 giờ)
[ ] VĐ4c: Hạ tông ngôn ngữ toàn bộ phần đánh giá (2 giờ)
[ ] VĐ5 : Sửa câu overclaim Related Work, thêm "theo hiểu biết của chúng tôi" (30 phút)
[ ] VĐ6 : Thêm câu minh bạch siêu tham số + thừa nhận (a) lr chung, (b) AMP không đồng nhất (30 phút)

🟢 MEDIUM – Làm nếu còn thời gian
[ ] VĐ7 : Ablation — ưu tiên (1) bỏ pose_balance_batch, (2) max_pseudo_atoms=0, (3) bỏ B_pocket
          Cùng số epoch, cùng quy tắc chọn mô hình cho mọi dòng
[ ] VĐ8 : Cắt nội dung thừa, vẽ lại Hình 3 dạng ngang, chú thích Hình 12 theo VĐ10
```

---

## Câu hỏi còn mở cần Đức trả lời

VĐ2 và VĐ3 **đã đóng** (câu trả lời nằm trong chính tài liệu này). Các câu hỏi còn lại:

> **Q1 (🔴, phục vụ VĐ11):** Bảng 6 trong khóa luận được lấy từ đâu — `summary.json` của từng
> mô hình, hay chép tay từ log? Bốn dòng EquiBind / TankBind / GNINA Dense / Mô hình đề xuất có
> khớp với `final_acc`, `final_pose_recall_pos`, `final_pose_recall_neg` trong `summary.json`
> tương ứng không?

> **Q2 (🔴, phục vụ VĐ2):** Tập test có bao nhiêu mẫu `y_aff > 0`? (lấy từ dòng
> `[DEBUG] Affinity target stats (N=...)` trong log, hoặc đếm trực tiếp trên
> `ref_uff_test0.types`). Con số này phải xuất hiện dưới Bảng 7.

> **Q3 (🟡, phục vụ VĐ9):** Các file `training_metrics_test.csv` còn giữ được không? Nếu còn,
> trích giá trị 5 epoch đánh giá cuối cho mỗi mô hình để báo cáo kèm giá trị best.

> **Q4 (🟡, phục vụ VĐ4):** Có file dự đoán mức mẫu (`predictions_*.csv` hoặc tương đương) cho
> tập test của cả 7 mô hình không? Paired bootstrap **bắt buộc** cần dự đoán từng mẫu, không thể
> tính từ metric tổng hợp. Nếu không có, phải chạy lại inference từ `best_model.pt` (rẻ, chỉ
> forward pass trên 4.618 mẫu).
