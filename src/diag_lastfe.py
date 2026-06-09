"""진단: 두 AI가 제안한 신규 피처군(lf/gi/mg/lb)이 521에 흡수되나 전수 검증.

각 그룹 단독 AUC + corr(73) + 521 대비 증분. GPU 불필요.
실행: python -m src.diag_lastfe
"""
import os
for e in ["KML_LASTFE", "KML_GENINT", "KML_MEGA", "KML_LABELFE"]:
    os.environ[e] = "1"
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

from . import config as C
from .train_first import build_all

GROUPS = {"lf": "lastfe(quantile/gap/daily)", "gi": "genint(성별score interaction)",
          "mg": "mega(goodcd계층/transition/basket/drift)", "lb": "labelfe(basket충돌/대리구매/tdecay)"}


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
    newpref = tuple(GROUPS.keys())
    base = [c for c in X.columns if not c.startswith(tuple(p + "_" for p in newpref))]
    pl2 = np.load("artifacts/oof/first_xgb_pl2__oof.npy") if os.path.exists("artifacts/oof/first_xgb_pl2__oof.npy") else None
    print(f"\n=== 신규 그룹별 단독 검증 (전체 {X.shape[1]} / base {len(base)}) ===")
    for pre, desc in GROUPS.items():
        cols = [c for c in X.columns if c.startswith(pre + "_")]
        if not cols:
            print(f"  [{pre}] 없음"); continue
        a, oof = _cv(X[cols].values, y, folds)
        cs = f" corr(pl2)={np.corrcoef(rankdata(oof), rankdata(pl2))[0,1]:.3f}" if pl2 is not None else ""
        print(f"  [{pre}] {desc}: {len(cols)}개  단독AUC={a:.5f}{cs}")
    a_base, _ = _cv(X[base].values, y, folds)
    print(f"\n=== 증분 (base CV={a_base:.5f}) ===")
    # 그룹별 누적 증분
    cum = list(base)
    for pre in newpref:
        cols = [c for c in X.columns if c.startswith(pre + "_")]
        if not cols:
            continue
        cum = cum + cols
        a, _ = _cv(X[cum].values, y, folds)
        print(f"  base+{pre} ({len(cols)}개): CV={a:.5f}  (vs base {a-a_base:+.5f})")
    a_all, _ = _cv(X.values, y, folds)
    print(f"  base+ALL: CV={a_all:.5f}  (vs base {a_all-a_base:+.5f})")


if __name__ == "__main__":
    main()
