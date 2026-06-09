"""make_73(김민형 3-level 스택)을 우리 환경에 포팅 + pseudo 정식 멤버 포함 재스택.

73 = hyunbin/김민형 OOF풀 → L1 메타8종 → L2 hillclimb (pseudo 없음, OOF 0.72778).
가설: pseudo 멤버(first_*_pl2 등)를 풀에 넣고 메타러너가 최적 결합하면 수동 0.7/0.3(92)보다↑?

POOL=base : pseudo 제외(73 재현)  /  POOL=full : pseudo 포함(새 스택)
실행: POOL=full python -m src.stack_make73
"""
import os
import glob
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score

from . import config as C

SEED, NF = 42, 5
POOL = os.environ.get("POOL", "full")
# pseudo/약한 실험 멤버 패턴(POOL=base면 제외 → 73 시점 재현 근사)
PSEUDO_PAT = ("_pl2", "_pl3", "_plc", "_pls", "_plcw", "_plb", "_ms", "_te", "_h95", "_w50",
              "_tk5", "cotrain", "scarf", "ft_ft", "ft_mlp", "ple_pl2", "_fs", "_unc", "kim73")


def rk(a):
    return rankdata(a) / len(a)


def auc(y, p):
    return roc_auc_score(y, p)


def main():
    tr = np.load(C.TRAIN_IDS_NPY, allow_pickle=True)
    folds = np.load(C.FOLDS_NPY)
    y = pd.read_csv("y_train.csv").set_index("custid").reindex(tr)["gender"].to_numpy().astype(int)
    N = len(y)
    all_m = sorted(p[:-len("__oof.npy")] for p in map(os.path.basename, glob.glob("artifacts/oof/*__oof.npy")))
    if POOL == "base":
        models = [m for m in all_m if not any(p in m for p in PSEUDO_PAT)]
    else:
        models = list(all_m)
    # NaN/길이 필터
    good = []
    for m in models:
        a = np.load(f"artifacts/oof/{m}__oof.npy")
        if len(a) == N and not np.isnan(a).any():
            good.append(m)
    models = good
    O = np.column_stack([rk(np.load(f"artifacts/oof/{m}__oof.npy")) for m in models])
    T = np.column_stack([rk(np.load(f"artifacts/oof/{m}__test.npy")) for m in models])
    print(f"POOL={POOL}  L0 풀 {len(models)} 멤버")

    def oof_meta(fn):
        oof = np.full(N, np.nan); ts = np.zeros(T.shape[0])
        for f in range(NF):
            tri, va = np.where(folds != f)[0], np.where(folds == f)[0]
            pv, pt = fn(O[tri], y[tri], O[va], T); oof[va] = pv; ts += pt
        return oof, ts / NF

    def lr(Cs):
        def f(Xtr, ytr, Xva, Xte):
            s = StandardScaler().fit(Xtr); m = LogisticRegression(C=Cs, max_iter=3000).fit(s.transform(Xtr), ytr)
            return m.predict_proba(s.transform(Xva))[:, 1], m.predict_proba(s.transform(Xte))[:, 1]
        return f

    def ridge(Xtr, ytr, Xva, Xte):
        s = StandardScaler().fit(Xtr); m = RidgeClassifier().fit(s.transform(Xtr), ytr)
        return m.decision_function(s.transform(Xva)), m.decision_function(s.transform(Xte))

    def et(Xtr, ytr, Xva, Xte):
        m = ExtraTreesClassifier(400, min_samples_leaf=10, n_jobs=-1, random_state=SEED).fit(Xtr, ytr)
        return m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1]

    def hgb(Xtr, ytr, Xva, Xte):
        m = HistGradientBoostingClassifier(learning_rate=0.03, max_iter=500, max_leaf_nodes=31,
                                           l2_regularization=1.0, random_state=SEED).fit(Xtr, ytr)
        return m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1]

    def knn(Xtr, ytr, Xva, Xte):
        s = StandardScaler().fit(Xtr); m = KNeighborsClassifier(200, weights="distance", n_jobs=-1).fit(s.transform(Xtr), ytr)
        return m.predict_proba(s.transform(Xva))[:, 1], m.predict_proba(s.transform(Xte))[:, 1]

    def hillclimb(Om, Tm, yy, n=120, init=3):
        a = [auc(yy, Om[:, j]) for j in range(Om.shape[1])]; pk = list(np.argsort(a)[::-1][:init])
        cur = Om[:, pk].mean(1); best = (auc(yy, cur), list(pk))
        for _ in range(n):
            b = (-1, None)
            for j in range(Om.shape[1]):
                aa = auc(yy, (cur * len(pk) + Om[:, j]) / (len(pk) + 1))
                if aa > b[0]: b = (aa, j)
            pk.append(b[1]); cur = (cur * (len(pk) - 1) + Om[:, b[1]]) / len(pk)
            if auc(yy, cur) > best[0]: best = (auc(yy, cur), list(pk))
        w = np.zeros(Om.shape[1])
        for q in best[1]: w[q] += 1
        return Om @ (w / w.sum()), Tm @ (w / w.sum())

    L1o, L1t = [], []
    for nm, fn in [("logreg0.05", lr(.05)), ("logreg0.3", lr(.3)), ("logreg1", lr(1.)),
                   ("ridge", ridge), ("extratrees", et), ("histgb", hgb), ("knn", knn)]:
        o, t = oof_meta(fn); L1o.append(rk(o)); L1t.append(rk(t)); print(f"  L1 {nm:12s} OOF {auc(y,o):.5f}")
    o, t = hillclimb(O, T, y); L1o.append(rk(o)); L1t.append(rk(t)); print(f"  L1 {'hillclimb':12s} OOF {auc(y,o):.5f}")
    L1o = np.column_stack(L1o); L1t = np.column_stack(L1t)

    cands = {}
    cands["L2_avg"] = (auc(y, L1o.mean(1)), L1t.mean(1))
    o2, t2 = hillclimb(L1o, L1t, y, n=60); cands["L2_hillclimb"] = (auc(y, o2), t2)

    def l2_logreg():
        oof = np.zeros(N); ts = np.zeros(L1t.shape[0])
        for f in range(NF):
            tri, va = np.where(folds != f)[0], np.where(folds == f)[0]
            s = StandardScaler().fit(L1o[tri]); m = LogisticRegression(C=0.5, max_iter=3000).fit(s.transform(L1o[tri]), y[tri])
            oof[va] = m.predict_proba(s.transform(L1o[va]))[:, 1]; ts += m.predict_proba(s.transform(L1t))[:, 1]
        return oof, ts / NF
    o3, t3 = l2_logreg(); cands["L2_logreg"] = (auc(y, o3), t3)
    print("=== L2 (73 base=0.72778, 92블렌드 OOF=0.72549) ===")
    from .oof_io import save_predictions
    for k, (a, tt) in cands.items():
        print(f"  {k:14s} OOF {a:.5f}")
        save_predictions(f"stack73{POOL}_{k}", (o2 if k == 'L2_hillclimb' else (o3 if k == 'L2_logreg' else L1o.mean(1))),
                         tt, meta=dict(cv_auc=float(a), seed=SEED, n_folds=NF,
                         feature_set=f"make73 3-level stack POOL={POOL} ({len(models)}멤버)", created_by="hyunbean",
                         notes=f"make_73 포팅 {POOL}(pseudo {'포함' if POOL=='full' else '제외'})"))


if __name__ == "__main__":
    main()
