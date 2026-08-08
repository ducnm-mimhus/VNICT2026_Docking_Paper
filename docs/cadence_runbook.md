# Chạy thực nghiệm trên JetBrains Cadence (plugin PyCharm)

**Nguồn tra cứu đã xác minh** (không phải suy đoán — đã đọc README + 3 ví dụ thật từ repo
chính thức của JetBrains):
- [JetBrains Cadence Plugin — Marketplace](https://plugins.jetbrains.com/plugin/23731-jetbrains-cadence)
- [Training Your ML Models With Cadence — PyCharm Blog](https://blog.jetbrains.com/pycharm/2025/06/training-your-ml-models-with-cadence/)
- [github.com/JetBrains/cadence-examples](https://github.com/JetBrains/cadence-examples) — README + 3 ví dụ YAML thật (đã `git clone` và đọc trực tiếp)

**Cadence là gì:** plugin PyCharm cho thuê GPU/CPU cloud theo giây (trả tiền theo phút), chạy
code local của bạn trên máy cloud mà không cần tự dựng hạ tầng. Cần **PyCharm Professional
2023.3+**. Có **$30 credit dùng thử miễn phí**.

---

## 0. Cách vận hành thực tế — 2 lớp, không phải 1

Plugin trong PyCharm (GUI) dùng để **cài đặt, đăng nhập, và chạy nhanh 1 Run Configuration đơn
giản bằng vài click**. Nhưng pipeline của dự án này là **nhiều bước tuần tự bằng bash**
(`scripts/00_download.sh` → smoke test → `scripts/run_geoformerdock_ablations.sh` →
`scripts/run_all_inference.sh` → các script phân tích) — loại việc này khớp chính xác với
**CLI + file YAML config** của Cadence hơn là 1 Run Configuration Python đơn giản. Nên hướng
dẫn dưới đây dùng **plugin để cài/đăng nhập**, và **CLI để chạy** — đây cũng là cách chính
README chính thức của Cadence mô tả.

Tôi đã viết sẵn 2 file cấu hình khớp đúng script của dự án này:
- `cadence/stage1_train.yaml` — tải data, smoke test trên GPU thật, chạy B1+B2
- `cadence/stage2_inference.yaml` — inference + A1 + paired bootstrap

Cả hai đã được kiểm tra cú pháp YAML và cú pháp bash bên trong — chưa chạy thật (không có
quyền truy cập Cadence để tự chạy).

---

## 1. Cài plugin + đăng nhập (làm trong PyCharm)

1. PyCharm → `Settings/Preferences | Plugins | Marketplace` → tìm **"Cadence"** → Install →
   Restart IDE.
2. Cài CLI (dùng cùng terminal với env `dockbench` đã dựng trước đó):
   ```bash
   pip install jetbrains-cadence
   ```
3. Đăng nhập (mở browser, xác thực, nhận $30 credit lần đầu):
   ```bash
   cadence login
   ```
4. Xác nhận workspace:
   ```bash
   cadence workspace
   ```

---

## 2. Chuẩn bị TRƯỚC khi chạy Stage 1

**Không cần tải `data/` về máy bạn** — lệnh `bash scripts/00_download.sh` sẽ chạy **trên máy
cloud của Cadence**, tải thẳng vào đó. Máy bạn chỉ cần đủ dung lượng cho code (vài chục MB),
không cần 80GB.

Kiểm tra `requirements.txt` của dự án được Cadence dùng làm nguồn cài môi trường (đã xác nhận
`pip install molgrid` chạy trực tiếp ở lượt trước — không cần sửa gì thêm).

⚠️ Trước khi bấm chạy: mở `cadence/stage1_train.yaml`, sửa `provisioning.gpu_type` nếu muốn
đổi GPU (bảng giá ở §5 dưới) — mặc định tôi chọn `l40s` (32GB vRAM, ~1.55 USD/giờ), đủ dư cho
mô hình 1.6M tham số, không cần GPU khủng như H100/H200 trong ví dụ mẫu (đó là cho fine-tune
LLM, không liên quan quy mô ở đây).

---

## 3. Chạy Stage 1 — tải data + smoke test + B1/B2

```bash
cd VNICT2026_Docking_Paper   # thư mục gốc repo, đã cd vào PyCharm project
cadence execution start --preset cadence/stage1_train.yaml
```

Lệnh này in ra **execution ID** — lưu lại, dùng để theo dõi:

```bash
cadence execution status <EXECUTION_ID>      # trạng thái
cadence execution logs <EXECUTION_ID>        # xem log trực tiếp (tương tự tail -f)
cadence execution terminal <EXECUTION_ID>    # mở terminal SSH vào máy cloud — dùng để chạy
                                              # `df -h` kiểm tra dung lượng NGAY sau khi lệnh
                                              # tải chạy được vài phút, trước khi tải hết 80GB
```

**Việc quan trọng nhất ở bước này:** dùng `cadence execution terminal <ID>` sớm (vài phút sau
khi bắt đầu) để chạy `df -h .` — xác nhận máy cloud có đủ chỗ cho 80GB, đúng như bước 0 trong
`docs/handoff_runbook.md` đã cảnh báo, chỉ khác là giờ kiểm tra trên máy cloud, không phải máy
bạn.

**Thời gian ước tính — đã sửa lại bằng số liệu thật** (bản trước ghi "~1 ngày/cái" là SAI, chỉ
là suy đoán chưa kiểm tra log). Log huấn luyện gốc (`results/models/*/training.log`, cột
`Elapsed Time` cuối cùng) cho thấy: 100 epoch của GeoFormerDock chỉ mất **1.52 giờ** thực tế
(và các model khác cũng chỉ 1.5–2.1 giờ, không phụ thuộc nhiều vào số tham số). B1/B2 dùng
đúng cấu hình đó (chỉ đổi 1 cờ), nên mỗi cái cũng nên mất **~1.5–2 giờ**, không phải 1 ngày.
**Tổng Stage 1 ước tính ~4–5 giờ** (gồm cả tải data), không phải ~2 ngày. Chi phí ước tính ở §6.

Khi xong, tải kết quả về máy bạn:
```bash
cadence execution download <EXECUTION_ID> --to ./stage1_out
```

---

## 4. Chuẩn bị TRƯỚC khi chạy Stage 2

Stage 2 cần các checkpoint **nhỏ** (vài chục MB, khác với `data/` nặng 80GB) đã có sẵn **trong
repo local của bạn** trước khi chạy — vì Cadence tự đồng bộ code+file nhỏ trong project lên
cloud (`project_sync`), không cần cơ chế tải riêng cho những file này:

```
results/models/gnina_dense/best_model.pt          ← tải từ Drive (docs/handoff_runbook.md §3.5, BẮT BUỘC)
results/models/gnina_default2018/best_model.pt    ← BẮT BUỘC
results/models/pafnucy/best_model.pt              ← BẮT BUỘC
results/models/geoformerdock/best_model.pt        ← khuyến nghị (Drive, tùy chọn)
```

Và copy kết quả B1/B2 từ Stage 1 vào đúng vị trí:
```bash
cp -r stage1_out/results/models/geoformerdock_uncertainty results/models/
cp -r stage1_out/results/models/geoformerdock_nobalance   results/models/
```

Chạy `tools/inspect_checkpoints.py` **ngay tại đây, trên máy local**, để sanity-check trước
khi tốn thêm GPU-hour ở Stage 2 (không cần data/GPU cho việc này):
```bash
python3 tools/inspect_checkpoints.py
```

---

## 5. Chạy Stage 2 — inference + A1 + paired bootstrap

```bash
cadence execution start --preset cadence/stage2_inference.yaml
cadence execution logs <EXECUTION_ID_2>
```

Stage này tải lại `data/` (chỉ để dùng cho inference trên tập test) — có chấp nhận đánh đổi
tải 2 lần thay vì giữ dữ liệu giữa 2 execution, vì băng thông cloud thường rẻ/nhanh hơn công
sức dựng cơ chế S3 riêng cho việc này. Nếu muốn tránh tải lại, gộp Stage 1 + Stage 2 thành 1
file `cmd` duy nhất — đánh đổi là nếu lỗi ở phần inference, phải chạy lại từ đầu cả phần B1/B2
đã tốn ~4-5 giờ GPU. **Khuyến nghị giữ 2 stage riêng như hiện tại** — dù thời gian B1/B2 giờ
biết là ngắn (không phải ~2 ngày như ước tính sai trước đó), tách stage vẫn an toàn hơn và chi
phí tải lại chỉ thêm ~$1.

Xong, tải kết quả:
```bash
cadence execution download <EXECUTION_ID_2> --to ./stage2_out
```
Gửi lại cho Đức đúng như mục 9 của `docs/handoff_runbook.md`.

---

## 6. Bảng giá GPU (đã xác minh từ README chính thức)

| GPU | Số lượng | vRAM | Giá/giờ | Khi nào dùng |
|---|---|---|---|---|
| **t4** | 1 | 16GB | $0.53–1.20 | Inference/metrics (Stage 2) |
| **l40s** | 1 | 32GB | $1.55 | Train B1/B2 (Stage 1) — mặc định đã chọn |
| a10g | 1 | 128GB* | $2.45 | Nếu `l40s` bị OOM ở `batch_size=1024` |
| h100/h200 | 1–8 | 160–1600GB | $2.95–28.00 | **Không cần** cho GeoFormerDock (1.6M tham số) |

*Cột vRAM cho a10g trong README có vẻ cao bất thường so với thông số phần cứng thật của card
A10G (24GB) — có thể là RAM hệ thống đi kèm, không phải vRAM GPU. Không phụ thuộc số này để
quyết định — nếu OOM ở `l40s`, thử giảm `BATCH_SIZE` (biến môi trường đã hỗ trợ sẵn trong
`scripts/run_geoformerdock_ablations.sh`) trước khi nhảy lên GPU đắt hơn.

**Ước tính chi phí toàn bộ (2 stage) — SỬA LẠI bằng số liệu thật từ log, không phải suy đoán:**

Đối chiếu `Elapsed Time` cuối cùng trong `results/models/*/training.log` cho 5 model — toàn bộ
100 epoch chỉ mất **1.5–2.1 giờ thật** (geoformerdock: 1.52h, pafnucy 8M tham số: 2.09h, không
tỉ lệ mạnh theo số tham số). B1/B2 dùng đúng cấu hình đó, nên ước tính tương tự:

- Stage 1: tải data (~1h, chưa có số thật) + B1 (~1.5-2h) + B2 (~1.5-2h) ≈ **4-5h** × $1.55/h
  ≈ **$6.2-7.8**
- Stage 2: tải lại data (~1h) + inference + bootstrap (~1h) ≈ **2h** × $0.53/h ≈ **~$1.1**
- **Tổng ước tính: ~$7.3-8.9** — nằm trong $30 credit miễn phí, kể cả phải chạy lại 1 lần do
  lỗi. Vẫn nên bật `cadence execution status`/dashboard billing để theo dõi thực tế, vì thời
  gian tải 80GB chưa có số đo thật (mới chỉ có số đo cho phần train).

---

## 7. Rủi ro chưa kiểm chứng được (khác với `docs/handoff_runbook.md`)

| Rủi ro | Vì sao chưa chắc |
|---|---|
| `pip install molgrid` chạy được trên **chính hạ tầng Cadence** | Đã xác nhận chạy được trên Linux thông thường (máy bạn), chưa chạy thử trên container/VM cụ thể của Cadence |
| Disk mặc định của VM Cadence đủ 80GB | Không thấy trường disk riêng trong schema `provisioning` (chỉ có `cpu_count`, `ram`, `gpu_type/count`) — disk có thể cố định theo GPU tier, cần `cadence execution terminal` + `df -h` kiểm tra sớm như đã nhắc ở §3 |
| `cmd:` dừng đúng khi 1 lệnh lỗi giữa chuỗi | Tôi đã thêm `set -e` ở đầu mỗi block để tự đảm bảo điều này ở tầng bash, không phụ thuộc hành vi mặc định của Cadence |

Đây là danh sách ngắn hơn nhiều so với các rủi ro đã liệt kê trong `docs/handoff_runbook.md` —
vì phần lớn rủi ro (B1/B2 chưa chạy thật, `--uncertainty` chưa test với checkpoint thật) là
rủi ro chung của cả 2 cách chạy (Cadence hay máy khác), không phải rủi ro riêng của Cadence.
