#!/usr/bin/env python3
"""
B-0 (phan bien R1-W1/W4) — Do do bat on dinh giua cac checkpoint CUOI cua moi
mo hinh, tu chinh cac file training_metrics_test.csv da co san trong repo.
KHONG can GPU, KHONG can data/ — chi doc CSV da duoc ghi lai tu cac lan train
truoc.

Muc dich: tra loi truc tiep phan bien "checkpoint tot nhat duoc chon tren
chinh tap kiem tra, va giua epoch 98/100 Balanced Accuracy dao dong toi 0.163
— lon hon ca chenh lech 0.040 giua cac kien truc".

Script nay tinh 2 thu, tren TAP CAC EPOCH CUOI (epoch >= --min_epoch, mac
dinh 70, tuc 16 lan danh gia cuoi voi test_every=2):

  1. "Bien do" (spread = max - min) cua tung chi so, tung mo hinh — de chung
     minh dao dong lon la dac tinh CHUNG cua moi mo hinh (vd BalAcc), khong
     phai rieng cua kien truc de xuat, TRONG KHI mot so chi so khac (vd
     PR-AUC) on dinh hon han.
  2. "Kiem tra roi nhau" (disjoint check): voi --primary-model (mac dinh la
     mo hinh dau tien trong --models), so [min, max] cua no voi [min, max]
     cua tung mo hinh con lai. Neu primary_min > max(moi mo hinh khac), ket
     luan ve chi so do la ON DINH VOI MOI CACH CHON CHECKPOINT — khong phu
     thuoc epoch nao duoc chon lam "best".

Cach chay (dung dung nhu trong docs/revision_plan_reviews.md Muc 0):
    python3 tools/checkpoint_robustness.py --models_dir results/models \
        --models geoformerdock,pafnucy,gnina_dense,gnina_default2018 \
        --min_epoch 70 --out results/logs/checkpoint_robustness.tsv

    # Bang bien do rong hon (Z0.1/Z0.2, tu epoch 60), kem xuat LaTeX cho Bang III:
    python3 tools/checkpoint_robustness.py --models_dir results/models \
        --models geoformerdock,pafnucy,gnina_dense,gnina_default2018 \
        --min_epoch 60 --metrics "Balanced Accuracy,PR AUC" \
        --out results/logs/checkpoint_robustness_e60.tsv \
        --latex results/logs/table3_stability.tex
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import numpy as np

# Ten hien thi cho bang LaTeX / stdout (khong doi ten cot goc trong CSV)
DISPLAY_NAME = {
    "geoformerdock": "GeoFormerDock",
    "geoformerdock_nobalance": "GeoFormerDock (-cân bằng lớp)",
    "geoformerdock_uncertainty": "GeoFormerDock (uncertainty)",
    "gnina_dense": "GNINA-Ds",
    "gnina_default2018": "GNINA-D18",
    "pafnucy": "Pafnucy",
    "potentialnet": "PotentialNet",
    "tankbind": "TankBind",
}


def read_test_metrics_csv(path: Path, metrics: List[str]) -> Dict[str, np.ndarray]:
    """Doc training_metrics_test.csv, tra ve dict {"Epoch": arr, metric: arr, ...}."""
    cols: Dict[str, List[float]] = {"Epoch": []}
    for m in metrics:
        cols[m] = []
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing = [m for m in metrics if m not in reader.fieldnames]
        if missing:
            raise KeyError(
                f"{path}: khong tim thay cot {missing} — cot co san: {reader.fieldnames}"
            )
        for row in reader:
            cols["Epoch"].append(float(row["Epoch"]))
            for m in metrics:
                cols[m].append(float(row[m]))
    return {k: np.asarray(v, dtype=float) for k, v in cols.items()}


def filter_min_epoch(data: Dict[str, np.ndarray], min_epoch: int) -> Dict[str, np.ndarray]:
    mask = data["Epoch"] >= min_epoch
    return {k: v[mask] for k, v in data.items()}


def display_name(model: str) -> str:
    return DISPLAY_NAME.get(model, model)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--models_dir", type=Path, default=Path("results/models"))
    ap.add_argument(
        "--models", type=str, required=True,
        help="Danh sach ten model (thu muc con cua --models_dir), phan cach bang dau phay. "
             "Model DAU TIEN trong danh sach duoc coi la --primary-model neu khong chi ro.",
    )
    ap.add_argument(
        "--primary-model", type=str, default=None,
        help="Model de xuat, dung lam moc cho phep kiem tra roi nhau. "
             "Mac dinh: model dau tien trong --models.",
    )
    ap.add_argument(
        "--metrics", type=str, default="Balanced Accuracy,PR AUC",
        help="Danh sach ten cot chi so (dung ten cot trong training_metrics_test.csv), "
             "phan cach bang dau phay.",
    )
    ap.add_argument(
        "--min_epoch", type=int, default=70,
        help="Chi lay cac lan danh gia co Epoch >= gia tri nay (mac dinh 70).",
    )
    ap.add_argument("--out", type=Path, default=None, help="Duong dan ghi TSV chi tiet.")
    ap.add_argument(
        "--latex", type=Path, default=None,
        help="Duong dan ghi than bang LaTeX (Bang III). Chi ho tro dung 2 chi so trong --metrics.",
    )
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    primary = args.primary_model or models[0]
    if primary not in models:
        raise SystemExit(f"--primary-model={primary!r} phai nam trong --models")

    # ---- Doc va loc du lieu ----
    per_model: Dict[str, Dict[str, np.ndarray]] = {}
    for model in models:
        csv_path = args.models_dir / model / "training_metrics_test.csv"
        if not csv_path.exists():
            print(f"[CANH BAO] Bo qua {model}: khong thay {csv_path}")
            continue
        data = read_test_metrics_csv(csv_path, metrics)
        data = filter_min_epoch(data, args.min_epoch)
        if data["Epoch"].size == 0:
            print(f"[CANH BAO] Bo qua {model}: khong co lan danh gia nao voi Epoch >= {args.min_epoch}")
            continue
        per_model[model] = data

    if primary not in per_model:
        raise SystemExit(f"Primary model {primary!r} khong co du lieu hop le — dung lai.")

    # ---- Bang 1: bien do tung mo hinh, tung chi so ----
    print("=" * 78)
    print(f"BIEN DO (spread = max - min) tren cac lan danh gia co Epoch >= {args.min_epoch}")
    print("=" * 78)
    tsv_rows = []
    for metric in metrics:
        print(f"\n--- {metric} ---")
        for model in models:
            if model not in per_model:
                continue
            v = per_model[model][metric]
            vmin, vmax = float(v.min()), float(v.max())
            spread = vmax - vmin
            mean, std = float(v.mean()), float(v.std())
            n = int(v.size)
            print(
                f"  {display_name(model):28s} n={n:2d}  "
                f"[{vmin:.4f}, {vmax:.4f}]  bien do={spread:.4f}  "
                f"mean={mean:.4f}  std={std:.4f}"
            )
            tsv_rows.append(
                {
                    "model": model,
                    "metric": metric,
                    "n_evals": n,
                    "min": vmin,
                    "max": vmax,
                    "spread": spread,
                    "mean": mean,
                    "std": std,
                }
            )

    # ---- Bang 2: kiem tra "roi nhau" so voi primary model ----
    print("\n" + "=" * 78)
    print(f"KIEM TRA ROI NHAU — moc: {display_name(primary)} (primary model)")
    print("=" * 78)
    for metric in metrics:
        print(f"\n--- {metric} ---")
        p = per_model[primary][metric]
        p_min, p_max = float(p.min()), float(p.max())
        print(f"  {display_name(primary):28s} [{p_min:.4f}, {p_max:.4f}]  (moc)")
        other_maxes = []
        for model in models:
            if model == primary or model not in per_model:
                continue
            o = per_model[model][metric]
            o_min, o_max = float(o.min()), float(o.max())
            other_maxes.append(o_max)
            verdict = "RỜI NHAU (primary luôn cao hơn)" if p_min > o_max else "CHỒNG LẤN"
            print(f"  {display_name(model):28s} [{o_min:.4f}, {o_max:.4f}]  -> {verdict}")
        if other_maxes:
            overall_max = max(other_maxes)
            overall = (
                "RỜI NHAU KHỎI MỌI BASELINE — kết luận không phụ thuộc cách chọn checkpoint"
                if p_min > overall_max
                else "CHỒNG LẤN VỚI ÍT NHẤT MỘT BASELINE — kết luận PHỤ THUỘC epoch được chọn"
            )
            print(f"  => {overall} (primary_min={p_min:.4f} vs max(others)={overall_max:.4f})")

    # ---- Ghi TSV ----
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["model", "metric", "n_evals", "min", "max", "spread", "mean", "std"]
        with args.out.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            for row in tsv_rows:
                writer.writerow(row)
        print(f"\nDa ghi: {args.out}")

    # ---- Ghi LaTeX (Bang III, san sang dan vao Overleaf) — chi ho tro dung 2 chi so ----
    # Ban .tex trong repo (docs/VNICT2026_GeoFormerDock/) chi la anh chup ban da nop,
    # KHONG sua truc tiep vao do — Overleaf moi la nguon duy nhat trong 10 ngay (xem
    # docs/phan_cong_2_phan.md). Script chi xuat mot khoi \begin{table}...\end{table}
    # hoan chinh de dan thang vao Overleaf.
    if args.latex is not None:
        if len(metrics) != 2:
            raise SystemExit("--latex chi ho tro dung 2 chi so trong --metrics (vd BalAcc + PR-AUC)")
        m1, m2 = metrics
        # Nhan cot ngan gon cho 2 chi so hay dung nhat; con lai dung nguyen ten cot CSV.
        short_label = {
            "Balanced Accuracy": "BalAcc",
            "PR AUC": "PR-AUC",
            "C-index": "C-idx",
        }
        l1 = short_label.get(m1, m1)
        l2 = short_label.get(m2, m2)

        lines = []
        lines.append(r"% Sinh tu dong boi tools/checkpoint_robustness.py — KHONG sua tay.")
        lines.append(f"% min_epoch={args.min_epoch}, models={','.join(models)}")
        lines.append(r"% Dan nguyen khoi nay vao Overleaf, ngay sau doan paired bootstrap")
        lines.append(r"% phan loai o Muc IV.C (truoc \textit{D. Phan tich danh doi}).")
        lines.append(r"\begin{table}[!t]")
        lines.append(r"\centering")
        lines.append(
            r"\caption{Khoảng giá trị (min–max) và biên độ dao động của "
            + f"{l1} và {l2}"
            + r" trên các checkpoint từ epoch "
            + f"{args.min_epoch}"
            + r" đến 100 ($N=" + str(int(next(iter(per_model.values()))["Epoch"].size))
            + r"$ lần đánh giá cuối, đánh giá mỗi 2 epoch).}"
        )
        lines.append(r"\label{tab:checkpoint_stability}")
        lines.append(r"\footnotesize")
        lines.append(r"\begin{tabular}{l ccc}")
        lines.append(r"\hline")
        lines.append(
            r"\textbf{Mô hình} & \textbf{" + l1 + r" [min, max]} & \textbf{Biên độ} \\"
        )
        lines.append(r" & \textbf{" + l2 + r" [min, max]} & \textbf{Biên độ} \\")
        lines.append(r"\hline")
        for model in models:
            if model not in per_model:
                continue
            v1 = per_model[model][m1]
            v2 = per_model[model][m2]
            name = display_name(model)
            if model == primary:
                name = r"\textbf{" + name + r"}"
            lines.append(
                f"{name} & [{v1.min():.3f}, {v1.max():.3f}] & {v1.max()-v1.min():.3f} \\\\"
            )
            lines.append(
                f" & [{v2.min():.3f}, {v2.max():.3f}] & {v2.max()-v2.min():.3f} \\\\"
            )
        lines.append(r"\hline")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")

        args.latex.parent.mkdir(parents=True, exist_ok=True)
        args.latex.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Da ghi (dan thang vao Overleaf): {args.latex}")


if __name__ == "__main__":
    main()
