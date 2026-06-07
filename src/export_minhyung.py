"""우리 2대 블렌드(3-level, hill-climb)의 oof+test를 npy로 export → 민형 전달용.

⚠️ 이 블렌드들은 김민형 모델을 포함하므로, 그의 Caruana에 넣으면 일부 순환.
   그래도 Caruana가 상관 고려해 가중하니 사용 가능. custid 정렬 파일 포함.
실행: python -m src.export_minhyung
"""
import os
import numpy as np
import pandas as pd

from . import config as C
from .oof_io import list_models, load_oof
from .blend_stack3 import rk, _oof_meta, _logreg, _ridge, _et, _hgb, _knn, _hillclimb

OUT = str(C.ROOT / "hyunbin_blend_for_minhyung")


def main():
    os.makedirs(OUT, exist_ok=True)
    train_ids = np.load(C.TRAIN_IDS_NPY, allow_pickle=True)
    test_ids = np.load(C.TEST_IDS_NPY, allow_pickle=True)
    folds = np.load(C.FOLDS_NPY)
    y = pd.read_csv(C.YTRAIN_CSV).set_index(C.ID_COL).reindex(train_ids)[C.TARGET].to_numpy()
    models = list_models()
    O = np.column_stack([rk(load_oof(m)[0]) for m in models])
    T = np.column_stack([rk(load_oof(m)[1]) for m in models])
    from sklearn.metrics import roc_auc_score

    # ---- 1) hill-climb (단일층) ----
    hc_o, hc_t = _hillclimb(O, T, y)
    np.save(f"{OUT}/hyunbin_hillclimb_oof.npy", hc_o.astype(np.float32))
    np.save(f"{OUT}/hyunbin_hillclimb_test.npy", hc_t.astype(np.float32))
    print(f"hillclimb OOF AUC={roc_auc_score(y, hc_o):.5f}")

    # ---- 2) 3-level ----
    L1o, L1t = [], []
    for fn in [_logreg(0.05), _logreg(0.3), _logreg(1.0), _ridge, _et, _hgb, _knn]:
        o, t = _oof_meta(fn, O, T, y, folds)
        L1o.append(rk(o)); L1t.append(rk(t))
    o, t = _hillclimb(O, T, y)
    L1o.append(rk(o)); L1t.append(rk(t))
    L1o, L1t = np.column_stack(L1o), np.column_stack(L1t)
    s3_o, s3_t = _hillclimb(L1o, L1t, y, n=60)
    np.save(f"{OUT}/hyunbin_stack3_oof.npy", s3_o.astype(np.float32))
    np.save(f"{OUT}/hyunbin_stack3_test.npy", s3_t.astype(np.float32))
    print(f"stack3 OOF AUC={roc_auc_score(y, s3_o):.5f}")

    np.save(f"{OUT}/_train_custid.npy", train_ids)
    np.save(f"{OUT}/_test_custid.npy", test_ids)
    print(f"\n저장 완료 → {OUT}")


if __name__ == "__main__":
    main()
