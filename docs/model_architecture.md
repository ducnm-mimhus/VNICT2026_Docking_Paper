# Mô tả Kiến trúc Mô hình – Khóa luận Tốt nghiệp
**Tên đề tài:** Phát triển Mô hình Học sâu Đa nhiệm Khai thác Thông tin Hình học Đại phân tử cho Bài toán Liên kết Thuốc vào Protein  
**Tác giả:** Hà Vũ Minh Đức – Khoa Toán-Cơ-Tin học, ĐHKHTN – ĐHQGHN, 2026  
**Người hướng dẫn:** PGS.TS. Nguyễn Thị Hồng Minh; ThS.NCS. Tạ Văn Nhân

> **Phiên bản 2 – đã đối chiếu với mã nguồn.** Mọi công thức và siêu tham số trong tài liệu này
> đều kèm tham chiếu `file:dòng` tới repo `dockbench/`. Các mục đánh dấu ⚠️ là chỗ **phiên bản 1
> mô tả sai** so với code — không được chép lại từ bản cũ.
>
> Cấu hình chạy thực tế được lấy từ `scripts/run_training.sh` (không phải giá trị mặc định của
> `argparse` trong `dockbench/training.py` — hai bên khác nhau ở nhiều chỗ).

---

## 1. Bài toán và Đầu vào/Đầu ra

### 1.1 Bài toán

Bài toán **protein–ligand docking** được mô hình hóa như bài toán học sâu có giám sát đa nhiệm, dự đoán đồng thời:

| Nhiệm vụ | Loại | Đầu ra |
|---|---|---|
| Phân loại cấu hình liên kết (Pose Classification) | Phân loại nhị phân | `ŷ_pose ∈ ℝ²` (xác suất good/bad pose) |
| Dự đoán ái lực liên kết (Affinity Prediction) | Hồi quy | `ŷ_aff ∈ ℝ` (giá trị pKd hoặc pKi) |

### 1.2 Đầu vào

```
X ∈ ℝ^(C × D × H × W)
```
- **C = 28** kênh đặc trưng hóa-tin học (14 kênh protein + 14 kênh thuốc)
- **D = H = W = 48** (lưới voxel 3D, độ phân giải 0.5 Å, kích thước hộp 23.5 Å)
  — `training.py:203-204`, không bị `run_training.sh` ghi đè
- Được tạo bằng thư viện `molgrid` từ dữ liệu cấu trúc `.gninatypes`

### 1.3 Dữ liệu

- **Dataset:** CrossDocked2020 (Francoeur et al., 2020), split `ref_uff_train0` / `ref_uff_test0`
- **Thiết lập:** Cross-docking – phối tử gắn vào nhiều cấu trúc protein khác nhau cùng họ sinh học
- **Kích thước:** 62.335 mẫu train / 4.618 mẫu test / 3.805 phức hợp không trùng lặp
  *(con số thứ ba chưa kiểm chứng được từ code — Đức xác nhận trước khi đưa vào bài)*
- **Nhãn ái lực:** giá trị dương = pK thực nghiệm (good pose), giá trị âm = decoy (bad pose)

**Tăng cường dữ liệu (chỉ trên tập train):** xoay ngẫu nhiên đều + tịnh tiến ngẫu nhiên
**±1.0 Å** (`run_training.sh:159`). Tập test **tắt hoàn toàn** cả xoay lẫn tịnh tiến
(`training.py:1036-1037`).

### 1.4 Chuẩn hóa target (bắt buộc phải nêu — phiên bản 1 thiếu)

Cấu hình chạy dùng `--normalize_targets` (`run_training.sh:199`). Khi đó:

- μ, σ được ước lượng **chỉ trên các nhãn ái lực dương** của tập train
  (`training.py:128-156`).
- Mạng được huấn luyện trên thang z: `z = (y − μ)/σ` áp dụng cho các mẫu `y > 0`
  (`target_normalizer.py:17-23`).
- Khi đánh giá, dự đoán được đưa về thang pK gốc bằng `denormalize` trước khi tính
  MAE/RMSE/r/ρ/C-index (`training.py:677-678`).

---

## 2. Kiến trúc Tổng quan

```
Input X (28×48×48×48)
        │
        ▼
┌─────────────────────────────────────────┐
│   Shared Backbone (3D CNN)              │
│   Conv1(3³, s=1, 28→32) → BN → ReLU     │
│   Conv2(3³, s=2, 32→48) → BN → ReLU     │
│   Lightweight Residual Block (48→64)    │
│   → F_shared ∈ ℝ^(64×24×24×24)          │
└─────────────────────────────────────────┘
        │
        ├──────────────────────────────┐
        ▼                              ▼
┌──────────────────┐      ┌──────────────────────┐
│   Pose Branch    │      │   Affinity Branch    │
│ (Phân loại)      │      │ (Hồi quy)            │
└──────────────────┘      └──────────────────────┘
        │                              │
        ▼                              ▼
   ŷ_pose ∈ ℝ²                  ŷ_aff ∈ ℝ
```

**Ánh xạ chính:**
```
F_shared = Φ(X)
ŷ_pose   = g_p(F_shared)
ŷ_aff    = g_a(F_shared)
```

Hai nhánh **tách hoàn toàn** ngay sau backbone; không có đường truyền đặc trưng giữa chúng
trong cấu hình chạy (`--use_pose_to_affinity_gate` không được bật; `set_gradient_routing`
của GeoFormerDock là no-op — `geoformerdock.py:299-300`).

---

## 3. Khối Shared Backbone (Φ)

`geoformerdock.py:219-227`

### 3.1 Cấu trúc

```
X (28×48×48×48)
    → Conv Block 1: Conv3D(3³, 28→32, s=1, p=1) → StableBN → ReLU
      → F1 ∈ ℝ^(32×48×48×48)
    → Conv Block 2: Conv3D(3³, 32→48, s=2, p=1) → StableBN → ReLU
      → F2 ∈ ℝ^(48×24×24×24)
    → Lightweight Residual Block (48→64)
      → F_shared ∈ ℝ^(64×24×24×24)
```

`StableBN` = `BatchNorm3d(momentum=0.01)` — momentum nhỏ hơn mặc định PyTorch để ổn định
running statistics (`common.py:17-26`).

### 3.2 ⚠️ Residual Block (phiên bản 1 viết sai công thức)

Vì số kênh vào ≠ ra (48 → 64), khối này **có nhánh chiếu tắt** và **có ReLU sau phép cộng**
(`geoformerdock.py:112-133`):

```
G(F2)     = StableBN(Conv3D(3³, 64→64, ReLU(StableBN(Conv3D(3³, 48→64, F2)))))
S(F2)     = StableBN(Conv3D(1³, 48→64, F2))          ← nhánh chiếu tắt
F_shared  = ReLU( S(F2) + G(F2) )
```

Công thức `F_shared = F2 + G(F2)` ở bản cũ **không hợp lệ về chiều** và bỏ mất phi tuyến cuối.

### 3.3 Ý nghĩa

- Conv1 (s=1): học tương tác cục bộ rất gần kề trong lưới voxel
- Conv2 (s=2): giảm độ phân giải không gian, mở rộng receptive field, nén từ 48³ → 24³
  (mỗi voxel của `F_shared` tương ứng ~1.0 Å)
- Residual: ổn định luồng gradient, tránh suy giảm thông tin

---

## 4. Nhánh Phân loại Cấu hình (Pose Branch – g_p)

`geoformerdock.py:230-258, 307-311`

### 4.1 ⚠️ Pipeline (bản cũ vẽ sai luồng của Geometry Encoder)

Điểm khác biệt so với bản cũ: **Geometry Encoder không nhận `F_shared`**, mà nhận
`pose_vol` — đầu ra của khối CNN cục bộ (`geoformerdock.py:310`).

```
F_shared (64×24×24×24)
   │
   ▼
[CNN cục bộ]  Conv3D(3³,64→64)→BN→ReLU→MaxPool(2)→Conv3D(3³,64→64)→BN→ReLU
   → pose_vol ∈ ℝ^(64×12×12×12)
   │
   ├─→ [GAP] → φ_l(·) → BatchNorm1d → h_local ∈ ℝ^128
   │
   └─→ [Geometry Encoder] → h_geo ∈ ℝ^128
   │
   ▼
[Gated Fusion]
   a = W_a h_local,  b = W_b h_geo
   g = σ(W_g [a; b] + b_g)
   h_fuse = LayerNorm( g ⊙ a + (1−g) ⊙ b )
   │
   ▼
[MLP Head]
   z_pose  = ReLU(W₁^(p) h_fuse + b₁^(p))
   z̃_pose = Dropout(z_pose)
   o_pose  = W₂^(p) z̃_pose + b₂^(p)      ← logits ∈ ℝ²
   │
   ▼
ŷ_pose = Softmax(o_pose) ∈ ℝ²
```

Mô hình xuất `log_softmax` (`geoformerdock.py:327`); mọi hàm mất mát và metric pose đều
nhận log-xác suất.

### 4.2 ⚠️ Geometry Encoder — mô tả đầy đủ (bản cũ thiếu bước gộp)

`geoformerdock.py:138-170` + `common.py:53-103`

**Bước 1 — Trích pseudo-atom.** Chọn **K = 12** voxel có chuẩn L2 theo kênh lớn nhất từ
`pose_vol` (lưới 12³ = 1.728 voxel):
```
a_i ∈ ℝ^64   : vector kênh tại voxel được chọn
p_i ∈ [−1,1]³: tọa độ chuẩn hóa của voxel trên lưới 12³
```

> ⚠️ **`p_i` KHÔNG có đơn vị Ångström.** Đây là chỉ số lưới được ánh xạ tuyến tính về
> `[−1,1]` (`common.py:97-101`). Khoảng cách tối đa giữa hai pseudo-atom là `2√3 ≈ 3.46`.

> ⚠️ **`p_i` KHÔNG khả vi.** Phép `topk` chỉ giữ **chỉ số nguyên** (`common.py:88`), và `p_i`
> được dựng từ số học nguyên trên chỉ số đó. Gradient **không** truyền ngược qua tọa độ.
> Chỉ `a_i` là khả vi (qua `gather`), và chỉ tại đúng 12 vị trí được chọn. Hệ quả: mạng học
> được *đặc trưng tại* pseudo-atom nhưng **không học được cách chọn** pseudo-atom.

**Bước 2 — Mã hóa RBF khoảng cách.** Với M = 16 nhân Gauss có tâm `μ_m` khởi tạo
`linspace(0, 3)` và bề rộng `σ_m` khởi tạo 0.5 (cả hai **là tham số học được**):
```
d_ij      = ‖p_i − p_j‖₂                       (đơn vị lưới chuẩn hóa)
ψ^(m)_ij  = exp( −(d_ij − μ_m)² / (2σ_m²) )
```

**Bước 3 — Gộp và chiếu (bản cũ bỏ hẳn bước này).**
```
ā      = (1/K) Σ_i a_i                    ∈ ℝ^64
ψ̄^(m) = (1/K²) Σ_{i,j} ψ^(m)_ij           ∈ ℝ^16     ← gộp cả cặp i=j
h_geo  = Dropout( ReLU( W_o [ ā ; ψ̄ ] + b_o ) ) ∈ ℝ^128
```

Tức là **toàn bộ tensor RBF `12×12×16` bị nén thành 16 số vô hướng**. Đây là một bộ mô tả
thống kê bậc thấp của hình dạng đám mây pseudo-atom, **không phải message passing**. Cần nói
đúng như vậy trong bài để tránh reviewer hiểu nhầm là GNN.

### 4.3 ⚠️ Hàm mất mát – Focal Loss (bản cũ ghi sai cách lấy trung bình)

`losses.py:76-144`. Với `p_t = exp(log p[y])`:
```
ℓ_i = α_t · (1 − p_t)^γ · (−log p_t),      α_t = α nếu y_i = 1, ngược lại 1 − α
```

Cấu hình chạy bật `--pose_class_normalize` (`run_training.sh:184`), nên phép gộp **không phải**
`(1/N)Σ` mà là **trung bình cân bằng theo lớp** (`losses.py:136-144`):
```
L_pose = ½ · ( mean_{i: y_i=0} ℓ_i  +  mean_{i: y_i=1} ℓ_i )
```

| Tham số | Giá trị | Nguồn |
|---|---|---|
| γ | 2.0 | `run_training.sh:69` |
| α (good pose) | 0.75 | `run_training.sh:70` |

### 4.4 Cân bằng lớp trong minibatch (bản cũ thiếu hoàn toàn — chi tiết then chốt)

`run_training.sh:185-186` bật `--pose_balance_batch --pose_balance_target_per_class 32`.
Trước khi tính `L_pose`, mỗi minibatch được **lấy mẫu lại có hoàn lại** thành đúng
**32 good + 32 bad** (`training.py:524-533`, `training.py:407-442`).

Đây là thủ thuật huấn luyện có ảnh hưởng trực tiếp và lớn tới Balanced Accuracy /
Recall(neg). **Bắt buộc phải nêu trong phần Thiết lập thực nghiệm** — nếu không, người đọc sẽ
quy toàn bộ mức Balanced Accuracy đạt được cho kiến trúc.

Lưu ý: việc lấy mẫu lại **chỉ áp dụng cho nhánh pose**. Nhánh affinity dùng nguyên minibatch gốc.

---

## 5. Nhánh Dự đoán Ái lực (Affinity Branch – g_a)

`geoformerdock.py:261-297, 313-320`

### 5.1 Pipeline

```
F_shared (64×24×24×24)
   │
   ▼
CNN_local:  Conv3D(3³, 64→64) → StableBN → ReLU → MaxPool(2)
   → aff_vol ∈ ℝ^(64×12×12×12)
   │
   ├──→ [VoxelTokenizer]  Conv3D(4³, 64→128, stride 4) → flatten → LayerNorm
   │         → T ∈ ℝ^(N×128),  N = 3×3×3 = 27 token
   │
   │    [Transformer × 2 (Pocket-aware Attention)]
   │         → LayerNorm → h_token = φ_t( mean_i T_i ) ∈ ℝ^128
   │
   └──→ [GAP] → h_global = φ_g( GAP(aff_vol) ) ∈ ℝ^128
   │
   ▼
[Gated Fusion]  (cùng dạng với §4.1)
   h_aff = LayerNorm( g ⊙ W_a h_token + (1−g) ⊙ W_b h_global )
   │
   ▼
[MLP Head]
   ŷ_raw = W₂^(a) ReLU(W₁^(a) h_aff + b₁^(a)) + b₂^(a)
   │
   ▼
[Affine Calibrator + Clamp]  — xem §5.4, thực tế là ánh xạ đồng nhất
   ŷ_aff = clamp( s_aff · ŷ_raw + b_aff , −20, +20 )
```

> ⚠️ **Số token chỉ là 27.** `aff_vol` là lưới 12³, `patch_size = 4` → tokenizer sinh lưới
> token 3×3×3. Mỗi token phủ khoảng 8 Å. "Pocket-aware Transformer" vì vậy hoạt động trên
> một lưới rất thô — phải nói rõ con số này, nếu không người đọc hình dung hàng trăm token.

### 5.2 ⚠️ Pocket-aware Attention — định nghĩa `B_pocket` (bản cũ mô tả SAI)

`geoformerdock.py:45-72`. Cơ chế:

```python
self.pocket_gate = nn.Sequential(nn.Linear(embed_dim, num_heads), nn.Sigmoid())
pocket_scores = self.pocket_gate(x).permute(0, 2, 1).unsqueeze(-1)   # [B, H, N, 1]
pocket_bias   = pocket_scores.transpose(-2, -1)                      # [B, H, 1, N]
attn = attn + pocket_bias                                            # broadcast trên trục query
```

Công thức chính xác, với `x_j ∈ ℝ^128` là token thứ `j` **sau LayerNorm** và `h` là chỉ số head:

$$
\text{Attention}^{(h)}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_h}} + B^{(h)}_{\text{pocket}}\right) V,
\qquad
B^{(h)}_{\text{pocket}}[i,j] \;=\; \sigma\!\left(\mathbf{w}_h^\top \mathbf{x}_j + b_h\right)
$$

với `W_p ∈ ℝ^{4×128}`, `b_p ∈ ℝ^4` (`d_h = 32`, 4 head).

**Ba tính chất bắt buộc phải nêu trung thực:**

1. **Không dùng bất kỳ tọa độ 3D nào.** Bias được suy ra hoàn toàn từ *nội dung* token.
2. **Chỉ phụ thuộc token key `j`**, độc lập với query `i` → đây là **per-key bias**, không
   phải pairwise bias. Nó nâng/hạ đồng đều mức chú ý mà *mọi* token dành cho token `j`.
3. **Miền giá trị (0, 1)** do sigmoid — nhỏ so với biên độ của `QKᵀ/√d_h`, nên đây là một
   hiệu chỉnh nhẹ, không phải cơ chế chi phối.

> ⚠️ **Không được viết** "giúp các cặp token ở vùng không gian gần nhau nhận attention cao hơn"
> (câu này ở bản cũ) và **không được viết** công thức dạng `MLP(φ_dist(‖c_i − c_j‖))` — code
> không làm điều đó.
>
> **Tên gọi trung thực:** *content-based per-key saliency gate*. Nếu vẫn dùng nhãn
> "pocket-aware", phải giải thích rằng tính pocket-aware đến **gián tiếp** qua việc token mang
> thông tin mật độ nguyên tử của vùng nó phủ, chứ không qua hình học tường minh.

### 5.3 ⚠️ Hàm mất mát – Composite Loss (sửa công thức `L_dist`)

`losses.py:227-446`
```
L_aff = L_reg + L_rank + L_dist + L_anchor
```
Trọng số λ đã được gắn sẵn vào từng module (`scale`), nên tổng là phép cộng trực tiếp.

| Thành phần | Công thức thực tế trong code | λ | Nguồn |
|---|---|---|---|
| `L_reg` | Huber(y, ŷ), δ = 1.0 | 1.0 | `losses.py:264-268` |
| `L_rank` | `softplus(−sign(y_i−y_j) · clamp((ŷ_i−ŷ_j)/τ, −5, 5))`, τ=1.0, 128 cặp/batch | 0.05 | `losses.py:306-320` |
| `L_dist` | **`\|μ_ŷ − μ_y\| + \|σ_ŷ − σ_y\|`** (trị tuyệt đối, **không** bình phương) | 0.02 | `losses.py:340-341` |
| `L_anchor` | `mean \|ŷ − median(y)\|` | 0.01 | `losses.py:356-357` |

**Tập mẫu áp dụng — bắt buộc phải viết tường minh trong bài:**

$$
S = \{\, i \;:\; y_{\text{aff},i} > 0 \,\}
$$

Cả **bốn** thành phần đều chỉ tính trên `S` (`training.py:496` tạo `mask = affinities > 0`,
truyền vào `CombinedAffinityLoss`; mỗi module gọi `_resolve_mask` — `losses.py:26-30, 250,
308, 335, 352`). Mẫu có `y_aff ≤ 0` (decoy) **không đóng góp gradient nào** cho nhánh ái lực.

> ⚠️ Lưu ý điều kiện là **`y_aff > 0`**, *không phải* `y_pose = 1`. Code không hề đọc nhãn
> pose khi lọc. Hai tập này chỉ trùng nhau nếu mọi good pose đều có nhãn ái lực dương.

**Về `L_reg`:** `ConfidenceAwareRegressionLoss` có hai chế độ (`losses.py:261-268`) —
Gaussian NLL dị phương sai khi mô hình cung cấp `log_sigma`, và Huber khi không. GeoFormerDock
chạy với `uncertainty=False` (cờ `--geoformer_uncertainty` **không** được bật trong
`run_training.sh`), nên `get_log_sigma()` trả `None` và **`L_reg` là Huber**.
👉 Điểm này có hệ quả về tính công bằng của benchmark — xem `issues_and_fixes.md` VĐ10.

### 5.4 ⚠️ Affine Calibrator thực tế bị vô hiệu hóa

`AffinityCalibration` (`common.py:106-123`) và cặp tham số `aff_scale`/`aff_bias`
(`geoformerdock.py:294-295`, khởi tạo 2.0 / 6.3) **không hoạt động trong cấu hình chạy**.

Vì `--normalize_targets` được bật, `training.py:1151-1166` gán `aff_scale = 1.0`,
`aff_bias = 0.0` rồi **đóng băng** (`requires_grad_(False)`), để dự đoán ở lại thang z-score và
không xung đột với bước `denormalize` lúc đánh giá.

→ Trong bài báo, **không nên trình bày calibrator như một thành phần kiến trúc**. Nếu nhắc tới,
phải nói rõ nó bị vô hiệu hóa khi dùng chuẩn hóa target. Ngưỡng `clamp = ±20`
(`geoformerdock.py:330`) trên thang z-score gần như không bao giờ chạm tới.

---

## 6. Hàm Mất mát Tổng hợp & Huấn luyện Đa nhiệm

`training.py:540-558`

```
L_total = w_aff · L_aff + w_pose · λ_pose · s_pose · L_pose
```

| Siêu tham số | Giá trị | Vai trò | Nguồn |
|---|---|---|---|
| w_pose | 0.85 | Trọng số nhánh pose | `run_training.sh:62` |
| w_aff | 0.15 | Trọng số nhánh ái lực | `run_training.sh:63` |
| λ_pose | 1.2 | Hệ số khuếch đại pose | `run_training.sh:61` |
| s_pose | 0.5 | Chuẩn hóa thang đo pose | `run_training.sh:60` |

**Gradient theo θ (backbone):**
```
∇θ L = w_aff · ∇θ L_aff + w_pose · λ_pose · s_pose · ∇θ L_pose
```

### 6.1 ⚠️ Lịch huấn luyện thực tế (bản cũ mô tả 2 giai đoạn KHÔNG tồn tại)

Chỉ có **hai** giai đoạn trong code, quyết định bởi `POSE_ONLY_EPOCHS=4` và
`POSE_WARMUP_EPOCHS=0` (`run_training.sh:64-65`, logic tại `training.py:504-557`):

| Epoch | Hàm mục tiêu thực tế |
|---|---|
| 1 – 4 | `L = λ_pose · s_pose · L_pose` (pose-only; backbone vẫn được cập nhật) |
| 5 – 100 | `L = 0.15·L_aff + 0.85·1.2·0.5·L_pose`, với `L_aff` **đầy đủ cả 4 thành phần** |

> ⚠️ **KHÔNG có warmup / ramp-up cho `L_rank`.** Bản cũ (và `README.md:101, 117-119`) mô tả
> "epoch 5–14 chưa bật L_rank, epoch 15–29 ramp-up" — điều này **không được cài đặt**.
> `run_training.sh:175-176` có truyền `--rank_warmup_epochs 10 --rank_rampup_epochs 15`, nhưng
> `CombinedAffinityLoss.__init__` **nhận rồi bỏ**, không gán vào `self` (`losses.py:367-395`);
> `forward()` luôn tính đủ bốn thành phần (`losses.py:428-437`). `set_epoch()` chỉ ghi
> `_current_epoch` mà không có nơi nào đọc.
>
> → `L_rank` hoạt động đầy đủ **ngay từ epoch 5**. Nếu viết lịch ramp-up vào bài thì đó là mô
> tả sai một thí nghiệm đã chạy.

**Tham số bị vô hiệu hóa (nhận nhưng không dùng — không nhắc trong bài):**
`scale_pose_coupling`, `use_gradient_alignment`, `scale_align`, `hard_neg_fraction`,
`rank_warmup_epochs`, `rank_rampup_epochs`, `full_loss_epoch`.
Nói cách khác, hàm mất mát ái lực có **4 thành phần hoạt động**, không phải 5 như README mô tả.

### 6.2 Tối ưu hóa

| Hạng mục | Giá trị | Nguồn |
|---|---|---|
| Optimizer | AdamW (β=(0.9, 0.999), ε=1e-8) | `training.py:1208-1214` |
| Learning rate | 1e-3 | `run_training.sh:40` |
| Weight decay | 1e-2 | `run_training.sh:41` |
| Batch size | 1024 | `run_training.sh:45` |
| Lịch LR | warmup tuyến tính 2 epoch + cosine decay tới 1e-6 | `run_training.sh:74`, `training.py:1365-1372` |
| Gradient clipping | max-norm 5.0 | `run_training.sh:160` |
| Mixed precision (AMP) | Bật cho GeoFormerDock | `run_training.sh:127-130` |
| SAM | **Không** bật (`USE_SAM_FOR_GEO=0`) | `run_training.sh:68` |
| Số epoch tối đa | 100 | `run_training.sh:39` |
| Seed | 2026 | `run_training.sh:79` |

### 6.3 ⚠️ Lựa chọn mô hình (bản cũ thiếu — điểm phương pháp quan trọng nhất)

`training.py:1528-1591`

- Cứ **2 epoch** một lần, chạy đánh giá **trên tập test**.
- Điểm chọn mô hình: `composite = 0.5 · C-index + 0.5 · Balanced Accuracy`
  (`run_training.sh:75, 78`).
- `best_model.pt` được ghi đè mỗi khi composite trên test cải thiện; early stopping với
  patience 25 lần đánh giá (≈ 50 epoch).
- **`summary.json` báo cáo `best_metrics` — tức là giá trị test tốt nhất trong ~50 lần đo.**

**Pipeline không có tập validation.** Mọi con số ở §8 vì vậy là *max-over-epochs trên chính tập
test*. Điều này **bắt buộc phải khai báo** trong phần Thiết lập thực nghiệm và Hạn chế — xem
`issues_and_fixes.md` VĐ9.

---

## 7. Kiến trúc Chi tiết – Siêu tham số & Phân bổ tham số

| Thành phần | Giá trị |
|---|---|
| Embedding dim (Transformer token) | 128 |
| Transformer layers | 2 |
| Attention heads | 4 (head_dim = 32) |
| MLP ratio trong Transformer block | 2.0 |
| Hidden dim (MLP các nhánh) | 128 |
| Patch size (VoxelTokenizer stride) | 4 → **27 token** |
| Num RBF kernels (geometry) | 16 |
| Max pseudo-atoms K | 12 |
| Dropout | 0.1 |
| **Tổng tham số** | **1.594.573 ≈ 1.59M** |

Phân bổ tham số (đếm tay từ `geoformerdock.py`, khớp con số 1.59M trong khóa luận):

| Khối | Tham số | Tỉ lệ |
|---|---|---|
| Shared Backbone | 262.816 | 16.5% |
| Pose Branch | 323.362 | 20.3% |
| — trong đó Geometry Encoder | 10.400 | 0.7% |
| Affinity Branch | 1.008.395 | 63.2% |
| — trong đó VoxelTokenizer | 524.544 | 32.9% |
| — trong đó 2 × Transformer block | 265.224 | 16.6% |
| — trong đó `pocket_gate` (B_pocket) | 1.032 | 0.06% |
| **Tổng** | **1.594.573** | 100% |

Hai quan sát nên dùng khi viết phần thảo luận (và khi thiết kế ablation):
- **VoxelTokenizer một mình chiếm 1/3 mô hình** — chỉ là một `Conv3d(64→128, k=4, s=4)`.
- **Hai thành phần được tuyên bố là đóng góp (`B_pocket` và Geometry Encoder) cộng lại chưa
  tới 0.8% tham số.** Nếu ablation cho thấy chúng ảnh hưởng lớn thì đó là kết quả đáng nói;
  nếu ảnh hưởng nhỏ thì kết quả đó cũng nhất quán với tỉ trọng tham số — cần diễn giải cẩn thận
  theo cả hai hướng.

---

## 8. Kết quả Thực nghiệm

> ⚠️ **Cảnh báo trước khi dùng hai bảng dưới đây:**
> 1. Các con số là **giá trị tốt nhất trên tập test qua ~50 lần đánh giá** (§6.3), không phải
>    kết quả của một mô hình được chọn độc lập.
> 2. Bảng 8.1 **mâu thuẫn nội tại về mặt số học** ở 4 dòng — phải đối chiếu lại với
>    `summary.json` / `training_metrics_test.csv` trước khi đưa vào bài
>    (`issues_and_fixes.md` VĐ11).
> 3. Bảng 8.2 được tính trên **tập con `y_aff > 0`**, không phải toàn bộ 4.618 mẫu test
>    (`issues_and_fixes.md` VĐ6). Phải bổ sung N của tập con.
> 4. C-index là **đại lượng ngẫu nhiên** (lấy mẫu cặp không seed) — `issues_and_fixes.md` VĐ4.

### 8.1 Nhánh Phân loại Cấu hình

| Mô hình | Params | Acc | Bal Acc | RecallPos | RecallNeg | PR-AUC |
|---|---|---|---|---|---|---|
| PotentialNet | 0.46M | 0.140 | 0.503 | 0.997 | 0.009 | 0.162 |
| EquiBind | 0.53M | 0.215 | 0.511 | 0.997 | 0.025 | 0.171 |
| TankBind | 0.21M | 0.846 | 0.513 | 0.998 | 0.028 | 0.189 |
| GNINA Default2018 | 2.16M | 0.631 | 0.761 | 0.937 | 0.585 | 0.501 |
| GNINA Dense | 0.46M | 0.865 | 0.787 | 0.989 | 0.585 | 0.519 |
| Pafnucy | 7.99M | 0.689 | 0.790 | 0.928 | 0.652 | 0.562 |
| **Mô hình đề xuất** | **1.59M** | **0.888** | **0.830** | **0.936** | **0.724** | **0.646** |

### 8.2 Nhánh Dự đoán Ái lực (trên tập con `y_aff > 0`)

| Mô hình | Params | MAE | RMSE | r | ρ | C-index |
|---|---|---|---|---|---|---|
| TankBind | 0.21M | 1.618 | 2.012 | 0.405 | 0.394 | 0.636 |
| PotentialNet | 0.46M | 1.385 | 1.715 | 0.626 | 0.601 | 0.715 |
| EquiBind | 0.53M | 1.333 | 1.685 | 0.642 | 0.615 | 0.720 |
| Pafnucy | 7.99M | 1.149 | 1.417 | 0.768 | 0.756 | 0.785 |
| **Mô hình đề xuất** | **1.59M** | **1.134** | **1.429** | **0.779** | **0.773** | **0.789** |
| GNINA Default2018 | 2.16M | 1.018 | 1.326 | 0.796 | 0.778 | 0.800 |
| GNINA Dense | 0.46M | 1.026 | 1.314 | 0.802 | 0.786 | 0.800 |

> ⚠️ GNINA Dense và GNINA Default2018 — hai mô hình duy nhất vượt mô hình đề xuất ở bảng này —
> là **hai mô hình duy nhất được huấn luyện bằng biến thể Gaussian NLL của `L_reg`** thay vì
> Huber, do chúng có `log_sigma_head` (`issues_and_fixes.md` VĐ10). Khoảng cách trong bảng bị
> nhiễu bởi khác biệt hàm mất mát, không thuần túy là khác biệt kiến trúc.

### 8.3 Điểm mạnh cốt lõi (đã hạ tông)

- **PR-AUC 0.646** – cao nhất trong nhóm so sánh; khoảng cách +0.084 so với Pafnucy (mô hình
  voxel tốt thứ hai). *Cần bootstrap CI ghép cặp trước khi khẳng định mức ý nghĩa.*
- **Balanced Accuracy 0.830** – cao nhất, đạt được với 1.59M tham số so với 7.99M của Pafnucy.
  *Phải nêu kèm việc cân bằng lớp trong minibatch (§4.4) như một yếu tố đóng góp.*
- **Vị trí gần biên Pareto** giữa hai nhiệm vụ: không hy sinh MAE để lấy Balanced Accuracy và
  ngược lại. Đây là luận điểm trung tâm và cũng là luận điểm phòng thủ được tốt nhất.
- Hiệu quả tham số: đạt kết quả tương đương nhóm dẫn đầu ở nhánh ái lực với ~1/5 tham số của
  Pafnucy. *Lưu ý: 82% tham số của GNINA Default2018 nằm ở `log_sigma_head` — so sánh
  "hiệu năng trên mỗi tham số" giữa các kiến trúc phân bổ tham số khác nhau là yếu.*

---

## 9. Sơ đồ Dòng chảy Dữ liệu

```
.gninatypes files + .types index (cột 0 = pose label, cột 1 = affinity)
        │
        ▼
molgrid (ExampleProvider + GridMaker)
  → Voxel 48×48×48, resolution 0.5 Å, hộp 23.5 Å
  → 28 channels (14 protein + 14 ligand)
  → Train: random rotation + random translation ±1.0 Å
  → Test : không tăng cường
        │
        ▼
Tensor X ∈ ℝ^(28×48×48×48)      |   y_pose ∈ {0,1},  y_aff ∈ ℝ
        │                        │
        │                        ▼
        │                 TargetNormalizer:  z = (y_aff − μ)/σ  cho y_aff > 0
        ▼
Shared Backbone → F_shared ∈ ℝ^(64×24×24×24)
        │
   ┌────┴────┐
   ▼         ▼
Pose Br.  Aff Br.
   │         │
ŷ_pose    ŷ_aff (thang z)
   │         │
   │         ├─── eval: denormalize → thang pK
   │         │
   └────┬────┘
        ▼
   L_total = 0.15·L_aff + 0.85·1.2·0.5·L_pose  → AdamW (clip 5.0, AMP)
        │
        ▼
   Mỗi 2 epoch: đánh giá trên TEST → composite = 0.5·C-index + 0.5·BalAcc
                → lưu best_model.pt / early stopping (patience 25)
```

---

## Phụ lục: Danh mục các điểm bản 1 mô tả sai

Dùng làm checklist khi rà soát lại bản thảo — nếu bất kỳ câu nào dưới đây xuất hiện trong bài
báo, đó là lỗi cần sửa.

| # | Bản 1 viết | Thực tế trong code |
|---|---|---|
| 1 | `B_pocket` là spatial bias theo khoảng cách 3D | Per-key gate `σ(w_hᵀx_j + b_h)`, không dùng tọa độ (§5.2) |
| 2 | Epoch 5–14 chưa bật `L_rank`; 15–29 ramp-up | Không tồn tại; `L_rank` đầy đủ từ epoch 5 (§6.1) |
| 3 | `L_dist = ‖μ_ŷ−μ_y‖² + ‖σ_ŷ−σ_y‖²` | Trị tuyệt đối, không bình phương (§5.3) |
| 4 | `F_shared = F2 + G(F2)` | Có shortcut 1×1 và ReLU sau cộng (§3.2) |
| 5 | Geometry Encoder nhận `F_shared` | Nhận `pose_vol` sau CNN cục bộ (§4.1) |
| 6 | Geometry Encoder dừng ở mã hóa RBF | Có bước gộp mean thành 16 số vô hướng (§4.2) |
| 7 | `d_ij` ngầm hiểu là Ångström | Đơn vị lưới chuẩn hóa `[−1,1]` trên lưới 12³ (§4.2) |
| 8 | `L_pose = (1/N)Σ` | Trung bình cân bằng theo lớp (§4.3) |
| 9 | Affine Calibrator là thành phần hoạt động | Bị đóng băng ở identity (§5.4) |
| 10 | *(thiếu)* | Cân bằng lớp 32/32 trong minibatch (§4.4) |
| 11 | *(thiếu)* | Chuẩn hóa z-score target (§1.4) |
| 12 | *(thiếu)* | Transformer chỉ có 27 token (§5.1) |
| 13 | *(thiếu)* | Mô hình được chọn bằng chính tập test (§6.3) |
| 14 | *(thiếu)* | `p_i` không khả vi (§4.2) |
