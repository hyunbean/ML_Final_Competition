"""정교 블렌드 실험실 — 여러 방법 OOF 비교 → 최선 제출 후보 생성.

전체 풀(우리+김민형) rank 정규화. hill-climb / scipy 가중최적화 / 규제 스태킹 비교.
스태킹은 OOF↑지만 LB 과적합 경향 → 규제 강한 버전(작은 C)이 일반화 나을 수 있음.
실행: python -m src.blend_lab
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss

from . import config as C
from .oof_io import list_models, load_oof


def rk(a):
    return rankdata(a) / len(a)


def hill_climb(O, y, n_steps=150, init=3):
    aucs = [roc_auc_score(y, O[:, j]) for j in range(O.shape[1])]
    picks = list(np.argsort(aucs)[::-1][:init])
    cur = O[:, picks].mean(1)
    best = (roc_auc_score(y, cur), list(picks))
    for _ in range(n_steps):
        b = (-1, None)
        for j in range(O.shape[1]):
            cand = (cur * len(picks) + O[:, j]) / (len(picks) + 1)
            a = roc_auc_score(y, cand)
            if a > b[0]:
                b = (a, j)
        picks.append(b[1]); cur = (cur * (len(picks) - 1) + O[:, b[1]]) / len(picks)
        a = roc_auc_score(y, cur)
        if a > best[0]:
            best = (a, list(picks))
    w = np.zeros(O.shape[1])
    for p in best[1]:
        w[p] += 1
    return w / w.sum(), best[0]


def main():
    train_ids = np.load(C.TRAIN_IDS_NPY, allow_pickle=True)
    test_ids = np.load(C.TEST_IDS_NPY, allow_pickle=True)
    folds = np.load(C.FOLDS_NPY)
    y = pd.read_csv(C.YTRAIN_CSV).set_index(C.ID_COL).reindex(train_ids)[C.TARGET].to_numpy()
    models = list_models()
    O = np.column_stack([rk(load_oof(m)[0]) for m in models])
    T = np.column_stack([rk(load_oof(m)[1]) for m in models])
    print(f"{len(models)} models")
    cands = {}

    # 1) hill-climb
    w_hc, auc_hc = hill_climb(O, y)
    cands["hillclimb"] = (auc_hc, T @ w_hc)
    print(f"hill-climb          OOF {auc_hc:.5f}")

    # 2) scipy 가중최적화 (logloss 목적, 비음수·합1) — 좌표상승 근사
    def obj(w):
        w = np.clip(w, 0, None); s = w.sum()
        if s <= 0:
            return 9
        p = np.clip(O @ (w / s), 1e-6, 1 - 1e-6)
        return log_loss(y, p)
    res = minimize(obj, w_hc.copy(), method="Powell", options={"maxiter": 4000})
    w_sp = np.clip(res.x, 0, None); w_sp = w_sp / w_sp.sum()
    auc_sp = roc_auc_score(y, O @ w_sp)
    cands["scipy"] = (auc_sp, T @ w_sp)
    print(f"scipy(powell)       OOF {auc_sp:.5f}")

    # 3) 규제 스태킹 (작은 C = 일반화↑) — fold-safe OOF
    for Cr in (0.02, 0.05, 0.1, 0.3, 1.0):
        oof = np.full(len(y), np.nan); ts = np.zeros(len(test_ids))
        for f in range(C.N_FOLDS):
            tr, va = np.where(folds != f)[0], np.where(folds == f)[0]
            sc = StandardScaler().fit(O[tr])
            lr = LogisticRegression(C=Cr, max_iter=3000).fit(sc.transform(O[tr]), y[tr])
            oof[va] = lr.predict_proba(sc.transform(O[va]))[:, 1]
            ts += lr.predict_proba(sc.transform(T))[:, 1]
        a = roc_auc_score(y, oof)
        cands[f"stackC{Cr}"] = (a, ts / C.N_FOLDS)
        print(f"stack C={Cr:<4}        OOF {a:.5f}")

    best = max(cands, key=lambda k: cands[k][0])
    print(f"\n최고 OOF: {best} = {cands[best][0]:.5f}")
    for nm in ("hillclimb", best):
        out = C.SUB_DIR / f"blend_{nm}.csv"
        pd.DataFrame({C.ID_COL: test_ids, C.TARGET: cands[nm][1]}).to_csv(out, index=False)
        print(f"saved: {out.name}  (OOF {cands[nm][0]:.5f})")


if __name__ == "__main__":
    main()
