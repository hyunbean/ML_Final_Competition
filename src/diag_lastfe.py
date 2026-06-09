"""진단: GPT 막판 lastfe(금액 quantile/거래간격/일단위 다양성)가 521에 흡수되나 검증.

lf 단독 AUC + corr(73) + 521 vs 521+lf CV(흡수 여부). GPU 불필요.
실행: python -m src.diag_lastfe
"""
import os
os.environ["KML_LASTFE"] = "1"
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

from . import config as C
from .train_first import build_all


def _cv(X, y, folds):
    oof = np.full(len(y), np.nan)
    for f in range(C.N_FOLDS):
        tv, vv = folds != f, folds == f
        m = lgb.LGBMClassifier(n_estimators=600, num_leaves=31, learning_rate=0.03,
                               subsample=0.8, colsample_bytree=0.7, random_state=42, n_jobs=-1, verbose=-1)
        m.fit(X[tv], y[tv]); oof[vv] = m.predict_proba(X[vv])[:, 1]
    return roc_auc_score(y, oof), oof


def main():
    allf, ydf, _, _ = build_all()
    tr = np.load(C.TRAIN_IDS_NPY, allow_pickle=True)
    folds = np.load(C.FOLDS_NPY)
    X = allf.reindex(tr).fillna(0.0)
    y = ydf.set_index("custid").reindex(tr)["gender"].to_numpy()
    lf = [c for c in X.columns if c.startswith("lf_")]
    base = [c for c in X.columns if not c.startswith("lf_")]
    print(f"lf 피처 {len(lf)}개: {lf}")
    a_lf, oof_lf = _cv(X[lf].values, y, folds)
    print(f"### lf 단독 AUC = {a_lf:.5f} (>0.6이면 신호 있음)")
    for m in ["kim73", "first_xgb_pl2"]:
        p = f"artifacts/oof/{m}__oof.npy"
        if os.path.exists(p):
            print(f"    corr(lf, {m}) = {np.corrcoef(rankdata(oof_lf), rankdata(np.load(p)))[0,1]:.4f}")
    a_base, _ = _cv(X[base].values, y, folds)
    a_full, _ = _cv(X.values, y, folds)
    print(f"### 521 단독 CV = {a_base:.5f}  →  521+lf CV = {a_full:.5f}  (증분 {a_full-a_base:+.5f})")
    print("    증분>0 = 흡수 안됨(추가가치) / <=0 = 흡수")


if __name__ == "__main__":
    main()
