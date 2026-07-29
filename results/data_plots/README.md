# Data Plots

Thu muc nay chua chuong trinh thong ke du lieu CrossDocked2020/PDBbind2016 tu cac file `.types` de dua vao khoa luan.

## Cach Chay

Chay tu thu muc `gnina-torch-main`:

```bash
python data_plots/plot_data_statistics.py
```

Mac dinh script doc:

```text
examples/Francoeur2020/data/types/ref_uff_train0.types
examples/Francoeur2020/data/types/ref_uff_test0.types
```

Neu chua co du lieu, tai truoc:

```bash
cd examples/Francoeur2020
bash 00_download.sh
cd ../..
```

## Dau Ra

Moi lan chay se tao mot thu muc timestamp trong:

```text
data_plots/outputs/<YYYYMMDD_HHMMSS>/
```

Trong do co:

- `summary.json`: tom tat nhanh so luong mau, ti le train/test, phan bo nhan pose, thong ke affinity/RMSD neu co.
- `tables/*.csv`: bang so lieu chi tiet, co the chen vao Excel/LaTeX.
- `plots/*.png`: bieu do san sang chen vao khoa luan.

## Cac Bieu Do Chinh

- `split_counts.png`: so luong va ti le train/test.
- `pose_class_distribution.png`: phan bo nhan pose theo split.
- `pose_class_ratio.png`: ti le tung lop pose trong moi split.
- `affinity_distribution.png`: phan bo affinity.
- `affinity_boxplot.png`: boxplot affinity theo train/test.
- `affinity_ecdf.png`: duong tich luy affinity.
- `path_resolution_status.png`: kiem tra cac duong dan structure co resolve duoc khong.
- `top_complexes_by_sample_count.png`: cac complex xuat hien nhieu nhat.
- `complex_overlap_train_test.png`: muc do trung lap complex giua train/test.
- `numeric_label_correlation.png`: tuong quan giua cac cot nhan so trong file `.types`.

## Tuy Chinh

Vi du chay voi file khac:

```bash
python data_plots/plot_data_statistics.py \
  --train-file examples/Francoeur2020/data/types/ref_uff_train0.types \
  --test-file examples/Francoeur2020/data/types/ref_uff_test0.types \
  --label-pos 0 \
  --affinity-pos 1
```

Neu file `.types` co cot RMSD, them vi tri cot do:

```bash
python data_plots/plot_data_statistics.py --rmsd-pos 2
```
