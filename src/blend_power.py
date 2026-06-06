"""Power-mean 블렌딩 (AUC 전용 후처리) — 3-level L1 메타들을 power-mean으로 결합.

AUC는 순위지표라 power-mean((Σ p^k)/n)^(1/k)이 순위를 재배열해 미세 게인 가능(검증사례 있음).
k 스윕. 실행: python -m src.blend_power
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from . import config as C
from .oof_io import list_models, load_oof
from .blend_stack3 import _oof_meta, _logreg, _ridge, _et, _hgb, _knn, _hillclimb


def rk(a):
    return rankdata(a) / len(a)


def pmean(P, k):
    return (np.mean(np.clip(P, 1e-6, 1) ** k, axis=1)) ** (1.0 / k)


def main():
    train_ids = np.load(C.TRAIN_IDS_NPY, allow_pickle=True)
    test_ids = np.load(C.TEST_IDS_NPY, allow_pickle=True)
    folds = np.load(C.FOLDS_NPY)
    y = pd.read_csv(C.YTRAIN_CSV).set_index(C.ID_COL).reindex(train_ids)[C.TARGET].to_numpy()
    models = list_models()
    O = np.column_stack([rk(load_oof(m)[0]) for m in models])
    T = np.column_stack([rk(load_oof(m)[1]) for m in models])

    # L1 메타 (3-level과 동일)
    L1o, L1t = [], []
    for fn in [_logreg(0.05), _logreg(0.3), _logreg(1.0), _ridge, _et, _hgb, _knn]:
        o, t = _oof_meta(fn, O, T, y, folds)
        L1o.append(rk(o)); L1t.append(rk(t))
    o, t = _hillclimb(O, T, y)
    L1o.append(rk(o)); L1t.append(rk(t))
    L1o, L1t = np.column_stack(L1o), np.column_stack(L1t)

    print("=== L2 결합 비교 ===")
    cands = {}
    cands["mean"] = (roc_auc_score(y, L1o.mean(1)), L1t.mean(1))
    o2, t2 = _hillclimb(L1o, L1t, y, n=60)
    cands["hillclimb"] = (roc_auc_score(y, o2), t2)
    for k in (2, 3, 3.5, 5, 8, 12):
        oo, tt = rk(pmean(L1o, k)), rk(pmean(L1t, k))
        cands[f"power{k}"] = (roc_auc_score(y, oo), tt)
    for nm, (a, _) in cands.items():
        print(f"  {nm:12s} OOF {a:.5f}")
    best = max(cands, key=lambda k: cands[k][0])
    print(f"\n최고: {best} = {cands[best][0]:.5f}")
    out = C.SUB_DIR / "submission_power.csv"
    pd.DataFrame({C.ID_COL: test_ids, C.TARGET: cands[best][1]}).to_csv(out, index=False)
    print(f"saved: {out.name}")


if __name__ == "__main__":
    main()
