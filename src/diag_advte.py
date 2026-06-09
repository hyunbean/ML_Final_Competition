"""진단: (1) adversarial validation(train/test shift) + (2) smoothed TE 신호/직교성.

딥리서치 #1·#2 검증용. GPU 불필요(lgbm CPU). 결과 숫자만 출력.
실행: python -m src.diag_advte
"""
import os
os.environ["KML_TESMOOTH"] = "1"   # build_all에 tesm 피처 포함시켜서 한 번에 검증
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

from . import config as C
from .train_first import build_all


def main():
    allf, ydf, _, _ = build_all()
    tr = np.load(C.TRAIN_IDS_NPY, allow_pickle=True)
    te = np.load(C.TEST_IDS_NPY, allow_pickle=True)
    folds = np.load(C.FOLDS_NPY)
    X = allf.reindex(tr).fillna(0.0)
    Xt = allf.reindex(te).fillna(0.0)
    y = ydf.set_index("custid").reindex(tr)["gender"].to_numpy()
    tcols = [c for c in allf.columns if c.startswith("tesm_")]
    feat = [c for c in allf.columns if c not in tcols]   # 실제 모델 피처(521)

    # ---- (1) adversarial validation : train vs test 구분 가능한가 ----
    Xa = pd.concat([X[feat], Xt[feat]], ignore_index=True)
    ya = np.r_[np.zeros(len(X)), np.ones(len(Xt))]
    m = lgb.LGBMClassifier(n_estimators=400, num_leaves=31, random_state=42, n_jobs=-1, verbose=-1)
    adv = cross_val_score(m, Xa, ya, cv=5, scoring="roc_auc").mean()
    print(f"\n### ADV AUC = {adv:.4f}   (0.5=shift없음 / 0.6+=시프트존재=sample-weight 기회)")
    m.fit(Xa, ya)
    top = [c for _, c in sorted(zip(m.feature_importances_, Xa.columns), reverse=True)[:10]]
    print(f"    shift 주도 top10: {top}")

    # ---- (2) smoothed TE : gender 신호 있나 + 기존 pseudo와 직교한가 ----
    print(f"\n    tesm 피처 {len(tcols)}개: {tcols}")
    E = X[tcols].to_numpy()
    oof = np.full(len(y), np.nan)
    for f in range(C.N_FOLDS):
        tv, vv = folds != f, folds == f
        mm = lgb.LGBMClassifier(n_estimators=300, num_leaves=15, random_state=42, verbose=-1)
        mm.fit(E[tv], y[tv])
        oof[vv] = mm.predict_proba(E[vv])[:, 1]
    auc = roc_auc_score(y, oof)
    print(f"### tesm 단독 AUC = {auc:.4f}   (>0.6이면 goodcd TE에 gender 신호 있음)")
    for p in ["first_xgb_pl2", "mh_bestblend69"]:
        fp = f"artifacts/oof/{p}__oof.npy"
        if os.path.exists(fp):
            c = np.corrcoef(rankdata(oof), rankdata(np.load(fp)))[0, 1]
            print(f"    corr(tesm, {p}) = {c:.4f}   (<0.9면 직교 → 투입 가치)")


if __name__ == "__main__":
    main()
