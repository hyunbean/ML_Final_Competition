"""Ridge×10 패턴 앙상블 (Kaggle S6E2 1등 L2) — 서브셋 선별을 10번 다른 seed로 → Ridge → 평균.

L1 base OOF 풀에서 seed별 greedy 서브셋 선택(다른 시작점/순서) → 각 Ridge(fold-safe) → 10개 평균.
"선택의 분산"을 줄여 hillclimb보다 안정적인 메타 기대. AUC라 Ridge(회귀) 출력 그대로 OK.
실행: python -m src.blend_ridge10
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from . import config as C
from .oof_io import list_models, load_oof
from .blend_stack3 import rk


def _ridge_oof(O, T, y, folds, cols):
    """선택된 cols로 fold-safe Ridge OOF+test."""
    oof = np.full(len(y), np.nan); ts = np.zeros(T.shape[0])
    Os, Ts = O[:, cols], T[:, cols]
    for f in range(C.N_FOLDS):
        tr, va = np.where(folds != f)[0], np.where(folds == f)[0]
        sc = StandardScaler().fit(Os[tr])
        m = Ridge(alpha=1.0).fit(sc.transform(Os[tr]), y[tr])
        oof[va] = m.predict(sc.transform(Os[va])); ts += m.predict(sc.transform(Ts))
    return oof, ts / C.N_FOLDS


def _select(O, T, y, folds, seed, kmax=20):
    """seed별 greedy 전진선택: 랜덤 시작점 + 랜덤 순서 → 다른 서브셋."""
    rng = np.random.RandomState(seed)
    n = O.shape[1]
    aucs = [roc_auc_score(y, O[:, j]) for j in range(n)]
    top = list(np.argsort(aucs)[::-1][:8])
    start = top[rng.randint(len(top))]
    picks = [start]
    best_auc = roc_auc_score(y, _ridge_oof(O, T, y, folds, picks)[0])
    improved = True
    while improved and len(picks) < kmax:
        improved = False
        cand = [j for j in range(n) if j not in picks]
        rng.shuffle(cand)
        for j in cand:
            o, _ = _ridge_oof(O, T, y, folds, picks + [j])
            a = roc_auc_score(y, o)
            if a > best_auc + 1e-6:
                best_auc = a; picks.append(j); improved = True
                break                       # first-improvement (랜덤순서라 seed별 경로 달라짐)
    return picks, best_auc


def main():
    train_ids = np.load(C.TRAIN_IDS_NPY, allow_pickle=True)
    test_ids = np.load(C.TEST_IDS_NPY, allow_pickle=True)
    folds = np.load(C.FOLDS_NPY)
    y = pd.read_csv(C.YTRAIN_CSV).set_index(C.ID_COL).reindex(train_ids)[C.TARGET].to_numpy()
    models = list_models()
    O = np.column_stack([rk(load_oof(m)[0]) for m in models])
    T = np.column_stack([rk(load_oof(m)[1]) for m in models])
    print(f"풀: {len(models)} models")

    oofs, tests = [], []
    for seed in range(10):
        picks, a = _select(O, T, y, folds, seed)
        o, t = _ridge_oof(O, T, y, folds, picks)
        oofs.append(rk(o)); tests.append(rk(t))
        print(f"  seed {seed}: |subset|={len(picks)}  Ridge OOF={a:.5f}")
    oof = np.mean(oofs, axis=0); test = np.mean(tests, axis=0)
    auc = roc_auc_score(y, oof)
    print(f"\n==== Ridge×10 평균  OOF AUC = {auc:.5f} ====")
    out = C.SUB_DIR / "submission_ridge10.csv"
    pd.DataFrame({C.ID_COL: test_ids, C.TARGET: test}).to_csv(out, index=False)
    print(f"saved: {out.name}")


if __name__ == "__main__":
    main()
