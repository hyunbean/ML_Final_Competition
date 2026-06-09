"""Phase1 깨끗한 FS: LGBM/XGB/Cat importance rank aggregation + null importance → 핵심 피처 선별.

이나경/박진성 교훈(양→질): 단일 gain 아닌 3모델 합의 + null로 노이즈 제거 + 상관 prune.
선별 피처 리스트를 artifacts/fs_clean_feats.json에 저장 → Phase2에서 모델 학습에 사용.
실행: FS_TOPN=180 python -m src.train_fs_clean
"""
import os
import json
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from . import config as C
from .train_first import build_all

TOPN = int(os.environ.get("FS_TOPN", "180"))
NULL_REP = int(os.environ.get("FS_NULLREP", "5"))
CORR_THR = float(os.environ.get("FS_CORR", "0.95"))


def _imp_lgb(X, y, seed=42):
    import lightgbm as lgb
    return lgb.LGBMClassifier(n_estimators=500, num_leaves=31, learning_rate=0.05,
                              random_state=seed, n_jobs=-1, verbose=-1).fit(X, y).feature_importances_


def _imp_xgb(X, y):
    import xgboost as xgb
    m = xgb.XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05, random_state=42,
                          tree_method="hist", verbosity=0).fit(X, y)
    return m.feature_importances_


def _imp_cat(X, y):
    from catboost import CatBoostClassifier
    return CatBoostClassifier(iterations=400, depth=6, learning_rate=0.05, random_seed=42,
                              verbose=0, allow_writing_files=False).fit(X, y).feature_importances_


def _cv(X, y, folds, cols):
    import lightgbm as lgb
    oof = np.full(len(y), np.nan)
    for f in range(C.N_FOLDS):
        tv, vv = folds != f, folds == f
        m = lgb.LGBMClassifier(n_estimators=800, num_leaves=31, learning_rate=0.03, subsample=0.8,
                               colsample_bytree=0.7, random_state=42, n_jobs=-1, verbose=-1)
        m.fit(X[cols].values[tv], y[tv]); oof[vv] = m.predict_proba(X[cols].values[vv])[:, 1]
    return roc_auc_score(y, oof)


def main():
    allf, ydf, _, _ = build_all()
    tr = np.load(C.TRAIN_IDS_NPY, allow_pickle=True)
    folds = np.load(C.FOLDS_NPY)
    X = allf.reindex(tr).fillna(0.0)
    y = ydf.set_index("custid").reindex(tr)["gender"].to_numpy()
    cols = list(X.columns)
    print(f"전체 피처 {len(cols)}")

    # 1) 3모델 importance rank aggregation
    score = np.zeros(len(cols))
    for fn, w, nm in [(_imp_lgb, 0.4, "lgb"), (_imp_xgb, 0.3, "xgb"), (_imp_cat, 0.3, "cat")]:
        imp = fn(X.values, y); score += w * rankdata(imp)
        print(f"  {nm} importance 완료")
    real = pd.Series(score, index=cols)

    # 2) null importance (타깃 셔플 대비)
    rng = np.random.default_rng(42)
    nullmax = np.zeros(len(cols))
    for i in range(NULL_REP):
        yp = rng.permutation(y)
        nullmax = np.maximum(nullmax, rankdata(_imp_lgb(X.values, yp, seed=100 + i)))
    passed = real.values > nullmax     # 실제 rank가 null 최대 rank 초과
    print(f"  null importance 통과: {passed.sum()}/{len(cols)}")

    # 3) 상관 prune (높은 중요도 우선 유지)
    ranked = real[passed].sort_values(ascending=False)
    cand = list(ranked.index)
    corr = X[cand].corr().abs().fillna(0)
    drop = set()
    for i, a in enumerate(cand):
        if a in drop:
            continue
        for b in cand[i + 1:]:
            if b not in drop and corr.loc[a, b] > CORR_THR:
                drop.add(b)
    cand = [c for c in cand if c not in drop]
    top = cand[:TOPN]
    print(f"  상관제거 후 {len(cand)} → top{TOPN}: {len(top)}개")

    # 4) CV 비교 (전체 vs 선별)
    cv_all = _cv(X, y, folds, cols)
    cv_top = _cv(X, y, folds, top)
    print(f"\n=== CV 비교 ===")
    print(f"  전체 {len(cols)}개: {cv_all:.5f}")
    print(f"  선별 {len(top)}개: {cv_top:.5f}  (증분 {cv_top-cv_all:+.5f})")
    json.dump(top, open("artifacts/fs_clean_feats.json", "w"), ensure_ascii=False)
    print(f"  → artifacts/fs_clean_feats.json 저장 ({len(top)}개)")


if __name__ == "__main__":
    main()
