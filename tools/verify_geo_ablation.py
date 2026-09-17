#!/usr/bin/env python3
"""
B0 (docs/revision_plan_reviews.md) — kiem tra NHANH cac co --geo_ablation vua
them vao GeoFormerDock, TRUOC KHI chay bat ky training/ablation that nao.

KHONG can `data/`, KHONG can molgrid, KHONG can GPU that su chay lau — chi cap
mot input gia (torch.randn) dung shape va chay 1 forward pass cho moi cau
hinh, mat vai giay. Duoc viet vi may soan code khong co torch de tu kiem thu.

Kiem tra:
  1. Ca 5 gia tri --geo_ablation deu construct duoc va forward pass ra dung
     shape (khong loi runtime).
  2. geo_ablation="none" cho DUNG 1.594.573 tham so — bang so da bao cao
     trong bai (Bang II, Tom tat).
  3. Checkpoint CU (results/models/geoformerdock/best_model.pt, train truoc
     khi co --geo_ablation) load duoc vao model "none" MOI voi strict=True,
     khong missing/unexpected keys — xac nhan sua doi khong pha checkpoint cu.

Cach chay (tren may co torch, vd server GPU sau khi ket noi lai):
    cd VNICT2026_Docking_Paper
    python3 tools/verify_geo_ablation.py

Thoat voi exit code 0 neu tat ca pass, 1 neu co loi.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch  # noqa: E402

from dockbench.models.geoformerdock import GEO_ABLATION_CHOICES, GeoFormerDock  # noqa: E402

# (C, D, H, W) — dung 28 kenh atom-typing x luoi 48^3, dung nhu bai bao cao.
INPUT_DIMS = (28, 48, 48, 48)
EXPECTED_PARAMS_NONE = 1_594_573
CHECKPOINT_PATH = _ROOT / "results" / "models" / "geoformerdock" / "best_model.pt"


def _count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def check_all_ablations() -> bool:
    ok = True
    print("--- 1+2. Construct + forward pass cho tung gia tri --geo_ablation ---")
    for ablation in GEO_ABLATION_CHOICES:
        try:
            model = GeoFormerDock(INPUT_DIMS, max_pseudo_atoms=12, geo_ablation=ablation)
            model.eval()
            n_params = _count_params(model)
            x = torch.randn(2, *INPUT_DIMS)
            with torch.no_grad():
                pose_log, affinity = model(x)
            assert pose_log.shape == (2, 2), f"pose_log shape sai: {tuple(pose_log.shape)}"
            assert affinity.shape == (2,), f"affinity shape sai: {tuple(affinity.shape)}"
            marker = "  <- BASELINE, phai dung 1.594.573" if ablation == "none" else ""
            print(f"  [OK]   geo_ablation={ablation:16s} params={n_params:>9,}{marker}")
            if ablation == "none" and n_params != EXPECTED_PARAMS_NONE:
                print(
                    f"         !!! LOI: ky vong {EXPECTED_PARAMS_NONE:,} tham so, "
                    f"duoc {n_params:,} — co su thay doi khong mong muon ve kien truc."
                )
                ok = False
        except Exception as e:  # noqa: BLE001 — muon bat MOI loi de bao cao het, khong dung lai o loi dau
            print(f"  [FAIL] geo_ablation={ablation}: {type(e).__name__}: {e}")
            ok = False
    return ok


def check_old_checkpoint_loads() -> bool:
    print("\n--- 3. Load checkpoint CU vao model 'none' MOI (strict=True) ---")
    if not CHECKPOINT_PATH.exists():
        print(f"  [BO QUA] Khong thay {CHECKPOINT_PATH} — chay script tu thu muc goc repo?")
        return True  # khong tinh la loi, chi la thieu file de kiem
    try:
        payload = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
        model = GeoFormerDock(INPUT_DIMS, max_pseudo_atoms=12, geo_ablation="none")
        result = model.load_state_dict(payload["model_state_dict"], strict=True)
        missing = list(getattr(result, "missing_keys", []))
        unexpected = list(getattr(result, "unexpected_keys", []))
        if missing or unexpected:
            print(f"  [FAIL] missing_keys={missing}, unexpected_keys={unexpected}")
            return False
        print(f"  [OK]   {CHECKPOINT_PATH.name} load sach, khong missing/unexpected keys.")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  [FAIL] {type(e).__name__}: {e}")
        return False


def main() -> None:
    ok = check_all_ablations()
    ok = check_old_checkpoint_loads() and ok
    print()
    if ok:
        print("=== TAT CA PASS — an toan de chay B-1..B-5 va training that ===")
    else:
        print("=== CO LOI — DUNG LAI, DUNG chay training that cho toi khi sua xong ===")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
