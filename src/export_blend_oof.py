"""특정 제출(stack3 블렌드)의 oof+test를 재현 → npy 저장 + 제출CSV와 대조 검증.

blend_stack3와 동일 파이프라인을, '그 당시 존재하던 베이스 OOF'만 써서 재현.
재현 test의 rank가 제출CSV와 corr≈1이면 그 oof가 정확히 맞는 짝.
실행: python -m src.export_blend_oof <출력이름> <참조제출csv명(확장자제외)> <제외모델,콤마구분>
예:  python -m src.export_blend_oof 49_submission_stack3_hyunbin 49_submission_stack3_hyunbin first_lgbm,mega_cat
"""
import sys
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from . import config as C
from .oof_io import list_models, load_oof
from .blend_stack3 import rk, _oof_meta, _logreg, _ridge, _et, _hgb, _knn, _hillclimb

OUT = C.ROOT / "hyunbin_subs_for_minhyung"


def build(exclude):
    train_ids = np.load(C.TRAIN_IDS_NPY, allow_pickle=True)
    test_ids = np.load(C.TEST_IDS_NPY, allow_pickle=True)
    folds = np.load(C.FOLDS_NPY)
    y = pd.read_csv(C.YTRAIN_CSV).set_index(C.ID_COL).reindex(train_ids)[C.TARGET].to_numpy()
    models = [m for m in list_models() if m not in exclude]
    O = np.column_stack([rk(load_oof(m)[0]) for m in models])
    T = np.column_stack([rk(load_oof(m)[1]) for m in models])
    L1o, L1t = [], []
    for fn in [_logreg(0.05), _logreg(0.3), _logreg(1.0), _ridge, _et, _hgb, _knn]:
        o, t = _oof_meta(fn, O, T, y, folds)
        L1o.append(rk(o)); L1t.append(rk(t))
    o, t = _hillclimb(O, T, y)
    L1o.append(rk(o)); L1t.append(rk(t))
    L1o, L1t = np.column_stack(L1o), np.column_stack(L1t)
    oof, test = _hillclimb(L1o, L1t, y, n=60)
    return train_ids, test_ids, y, oof, test, models


def main():
    name = sys.argv[1]
    ref = sys.argv[2] if len(sys.argv) > 2 else None
    exclude = set(sys.argv[3].split(",")) if len(sys.argv) > 3 and sys.argv[3] else set()
    OUT.mkdir(parents=True, exist_ok=True)
    train_ids, test_ids, y, oof, test, models = build(exclude)
    auc = roc_auc_score(y, oof)
    print(f"[{name}] 재현 OOF AUC={auc:.5f}  (models={len(models)}, 제외={sorted(exclude)})")

    if ref:
        rc = pd.read_csv(C.SUB_DIR / f"{ref}.csv").set_index(C.ID_COL).reindex(test_ids)[C.TARGET].to_numpy()
        corr = np.corrcoef(rankdata(rc), rankdata(test))[0, 1]
        print(f"  ref-csv rank-corr = {corr:.6f}  {'MATCH(exact pair)' if corr > 0.9999 else 'MISMATCH-adjust exclude set'}")

    np.save(OUT / f"{name}_oof.npy", oof.astype(np.float32))
    np.save(OUT / f"{name}_test.npy", test.astype(np.float32))
    np.save(OUT / "custid_train.npy", train_ids)
    np.save(OUT / "custid_test.npy", test_ids)
    print(f"  저장: {name}_oof.npy / {name}_test.npy")


if __name__ == "__main__":
    main()
