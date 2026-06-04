"""팀 공유용 OOF/test 예측 저장·로드.

규칙(중요):
- 모든 npy는 '정규 custid 순서'(train_custids.npy / test_custids.npy)에 정렬되어 있다고 가정.
- 한 모델당 3종 세트를 공유: {model}__oof.npy, {model}__test.npy, {model}__meta.json
- 팀원은 자기 {model}__*.npy 3개를 공유 oof 폴더에 떨구기만 하면 ensemble.py가 자동으로 합침.
"""
import json
import numpy as np
from . import config as C


def save_predictions(model_name: str, oof: np.ndarray, test_pred: np.ndarray, meta: dict):
    np.save(C.OOF_DIR / f"{model_name}__oof.npy", oof.astype(np.float32))
    np.save(C.OOF_DIR / f"{model_name}__test.npy", test_pred.astype(np.float32))
    with open(C.OOF_DIR / f"{model_name}__meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[oof] saved '{model_name}'  oof{oof.shape} test{test_pred.shape}  cv={meta.get('cv_auc')}")


def list_models():
    suf = "__oof.npy"
    return sorted(p.name[: -len(suf)] for p in C.OOF_DIR.glob(f"*{suf}"))


def load_oof(model_name: str):
    oof = np.load(C.OOF_DIR / f"{model_name}__oof.npy")
    test = np.load(C.OOF_DIR / f"{model_name}__test.npy")
    return oof, test
