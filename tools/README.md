# tools/

Script phân tích chạy **sau** huấn luyện. Không cần GPU trừ `run_inference.py`.

| Script | Việc |
|---|---|
| `make_val_split.py` | Tách validation từ train **theo receptor** (không tách ngẫu nhiên từng dòng). Chạy trước mọi lượt train mới. |
| `run_inference.py` | Chạy một checkpoint trên tập .types, xuất dự đoán từng mẫu ra CSV. |
| `exact_metrics.py` | Tính chỉ số từ CSV dự đoán; C-index dùng **đủ cặp**, không lấy mẫu. |
| `paired_bootstrap.py` | Paired bootstrap CI cho hiệu số các chỉ số **hồi quy ái lực**. |
| `paired_bootstrap_classification.py` | Paired bootstrap CI cho **BalAcc / PR-AUC**. |
| `inspect_checkpoints.py` | Đối chiếu `summary.json` với chỉ số thật trong checkpoint (phát hiện lỗi ghép epoch). |
| `select_epoch_by_train.py` | Chọn epoch theo chỉ số **trên tập train** làm đối chứng, xuất `epoch_selection_audit.tsv`. |
| `summarize_benchmark.py` | Gom `summary.json` của các model thành một bảng. |
| `plot_training_curves.py` | Vẽ đường cong huấn luyện và các biểu đồ so sánh. |
| `dataset_report.py` | Thống kê nhanh tập dữ liệu trước khi train. |

Thứ tự dùng điển hình sau một lượt train:

```bash
bash scripts/run_all_inference.sh            # -> results/predictions/*.csv
python3 tools/exact_metrics.py --predictions results/predictions/<model>.csv
python3 tools/paired_bootstrap_classification.py --pred_a A.csv --pred_b B.csv
```
