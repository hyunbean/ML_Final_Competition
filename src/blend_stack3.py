"""3-level 스태킹 (Chris Deotte식) — 단일 스택 과적합을 다층+다양 메타로 완화.

L0: 전체 base OOF(우리+김민형) rank 정규화.
L1: 여러 메타러너(LogReg 여러 C, Ridge, ExtraTrees, Caruana hill-climb) → 각자 fold-safe OOF.
L2: L1 메타들을 평균 / hill-climb / Ridge로 결합 → 최종. (다양한 메타 평균 = 일반화↑)
OOF만 사용. 실행: python -m src.blend_stack3
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score

from . import config as C
from .oof_io import list_models, load_oof


def rk(a):
    return rankdata(a) / len(a)


def _oof_meta(fit_fn, O, T, y, folds):
    oof = np.full(len(y), np.nan); ts = np.zeros(T.shape[0])
    for f in range(C.N_FOLDS):
        tr, va = np.where(folds != f)[0], np.where(folds == f)[0]
        p_va, p_te = fit_fn(O[tr], y[tr], O[va], T)
        oof[va] = p_va; ts += p_te
    return oof, ts / C.N_FOLDS


def _logreg(Cr):
    def f(Xtr, ytr, Xva, Xte):
        sc = StandardScaler().fit(Xtr)
        m = LogisticRegression(C=Cr, max_iter=3000).fit(sc.transform(Xtr), ytr)
        return m.predict_proba(sc.transform(Xva))[:, 1], m.predict_proba(sc.transform(Xte))[:, 1]
    return f


def _ridge(Xtr, ytr, Xva, Xte):
    sc = StandardScaler().fit(Xtr)
    m = RidgeClassifier().fit(sc.transform(Xtr), ytr)
    return m.decision_function(sc.transform(Xva)), m.decision_function(sc.transform(Xte))


def _et(Xtr, ytr, Xva, Xte):
    m = ExtraTreesClassifier(n_estimators=400, min_samples_leaf=10, n_jobs=-1, random_state=C.SEED).fit(Xtr, ytr)
    return m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1]


def _hgb(Xtr, ytr, Xva, Xte):
    m = HistGradientBoostingClassifier(learning_rate=0.03, max_iter=500, max_leaf_nodes=31,
                                       l2_regularization=1.0, random_state=C.SEED).fit(Xtr, ytr)
    return m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1]


def _knn(Xtr, ytr, Xva, Xte):
    sc = StandardScaler().fit(Xtr)
    m = KNeighborsClassifier(n_neighbors=200, weights="distance", n_jobs=-1).fit(sc.transform(Xtr), ytr)
    return m.predict_proba(sc.transform(Xva))[:, 1], m.predict_proba(sc.transform(Xte))[:, 1]


def _hillclimb(O, T, y, n=120, init=3):
    aucs = [roc_auc_score(y, O[:, j]) for j in range(O.shape[1])]
    picks = list(np.argsort(aucs)[::-1][:init]); cur = O[:, picks].mean(1)
    best = (roc_auc_score(y, cur), list(picks))
    for _ in range(n):
        b = (-1, None)
        for j in range(O.shape[1]):
            a = roc_auc_score(y, (cur * len(picks) + O[:, j]) / (len(picks) + 1))
            if a > b[0]:
                b = (a, j)
        picks.append(b[1]); cur = (cur * (len(picks) - 1) + O[:, b[1]]) / len(picks)
        if roc_auc_score(y, cur) > best[0]:
            best = (roc_auc_score(y, cur), list(picks))
    w = np.zeros(O.shape[1])
    for p in best[1]:
        w[p] += 1
    w /= w.sum()
    return O @ w, T @ w


def main():
    train_ids = np.load(C.TRAIN_IDS_NPY, allow_pickle=True)
    test_ids = np.load(C.TEST_IDS_NPY, allow_pickle=True)
    folds = np.load(C.FOLDS_NPY)
    y = pd.read_csv(C.YTRAIN_CSV).set_index(C.ID_COL).reindex(train_ids)[C.TARGET].to_numpy()
    models = list_models()
    O = np.column_stack([rk(load_oof(m)[0]) for m in models])
    T = np.column_stack([rk(load_oof(m)[1]) for m in models])
    print(f"L0: {len(models)} models")

    # ---- L1 메타러너들 ----
    L1o, L1t = [], []
    metas = [("logreg0.05", _logreg(0.05)), ("logreg0.3", _logreg(0.3)), ("logreg1", _logreg(1.0)),
             ("ridge", _ridge), ("extratrees", _et), ("histgb", _hgb), ("knn", _knn)]
    for name, fn in metas:
        o, t = _oof_meta(fn, O, T, y, folds)
        L1o.append(rk(o)); L1t.append(rk(t))
        print(f"  L1 {name:12s} OOF {roc_auc_score(y, o):.5f}")
    o, t = _hillclimb(O, T, y)
    L1o.append(rk(o)); L1t.append(rk(t)); print(f"  L1 {'hillclimb':12s} OOF {roc_auc_score(y, o):.5f}")
    L1o = np.column_stack(L1o); L1t = np.column_stack(L1t)

    # ---- L2 결합 ----
    cands = {}
    avg_o, avg_t = L1o.mean(1), L1t.mean(1)
    cands["L2_avg"] = (roc_auc_score(y, avg_o), avg_t)
    o2, t2 = _hillclimb(L1o, L1t, y, n=60)
    cands["L2_hillclimb"] = (roc_auc_score(y, o2), t2)
    o3, t3 = _oof_meta(_logreg(0.5), L1o, L1t, y, folds)
    cands["L2_logreg"] = (roc_auc_score(y, o3), t3)
    print("\n=== L2 ===")
    for k, (a, _) in cands.items():
        print(f"  {k:14s} OOF {a:.5f}")

    best = max(cands, key=lambda k: cands[k][0])
    print(f"\n최고(OOF): {best} = {cands[best][0]:.5f}")
    # 모든 L2 후보 각각 저장 (logreg=OOF최고지만 신기루 위험, hillclimb=LB신뢰)
    for k, (_, t) in cands.items():
        out = C.SUB_DIR / f"submission_{k}.csv"
        pd.DataFrame({C.ID_COL: test_ids, C.TARGET: t}).to_csv(out, index=False)
        print(f"saved: {out.name}")
    pd.DataFrame({C.ID_COL: test_ids, C.TARGET: cands[best][1]}).to_csv(
        C.SUB_DIR / "submission_stack3.csv", index=False)


if __name__ == "__main__":
    main()
