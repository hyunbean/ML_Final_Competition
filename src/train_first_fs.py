"""교수님 힌트 #2: 중요도 하위 feature + 상관(유사) feature 제거 → 노이즈제거 (+0.002 기대).

521피처 중 (a)중요도 하위 제거 (b)상관>thr 쌍에서 낮은중요도 제거 → 깨끗한 셋으로 재학습.
실행(GPU): FS_DROP=0.4 FS_CORR=0.95 XGB_GPU=1 python -m src.train_first_fs [xgb|lgbm|cat]
"""
import os
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

from . import config as C
from .oof_io import save_predictions
from .train_first import build_all

DROP_FRAC = float(os.environ.get("FS_DROP", "0.4"))    # 중요도 하위 비율 제거
CORR_THR = float(os.environ.get("FS_CORR", "0.95"))    # 이 상관 이상 쌍은 낮은중요도 제거
GPU = os.environ.get("XGB_GPU", "1") == "1"
CB = os.environ.get("CB_TASK_TYPE", "GPU")


def select_features(X, y):
    m = lgb.LGBMClassifier(n_estimators=600, num_leaves=63, learning_rate=0.05,
                           random_state=C.SEED, n_jobs=-1, verbose=-1).fit(X, y)
    imp = pd.Series(m.feature_importances_, index=X.columns).sort_values(ascending=False)
    n_keep = int(len(imp) * (1 - DROP_FRAC))
    keep = list(imp.index[:n_keep])                    # (a) 중요도 상위만
    corr = X[keep].corr().abs().fillna(0)
    drop = set()
    for i, a in enumerate(keep):                        # (b) 상관 유사쌍 → 낮은중요도(뒤) 제거
        if a in drop:
            continue
        for b in keep[i + 1:]:
            if b not in drop and corr.loc[a, b] > CORR_THR:
                drop.add(b)
    final = [c for c in keep if c not in drop]
    print(f"  feature select: {len(X.columns)} -> 중요도상위 {len(keep)} -> 상관제거후 {len(final)}")
    return final


def _fit(kind, Xtr, ytr, Xva, yva, Xt):
    import xgboost as xgb
    from catboost import CatBoostClassifier
    if kind == "xgb":
        m = xgb.XGBClassifier(n_estimators=4000, early_stopping_rounds=150, objective="binary:logistic",
                              eval_metric="auc", learning_rate=0.02, max_depth=7, min_child_weight=5, gamma=0.1,
                              subsample=0.8, colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=5.0,
                              random_state=C.SEED, tree_method="hist", device="cuda" if GPU else "cpu")
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    elif kind == "lgbm":
        m = lgb.LGBMClassifier(n_estimators=4000, objective="binary", metric="auc", learning_rate=0.02,
                               num_leaves=64, min_child_samples=40, subsample=0.8, subsample_freq=1,
                               colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=5.0, n_jobs=-1,
                               random_state=C.SEED, verbose=-1)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[lgb.early_stopping(150, verbose=False)])
    else:
        m = CatBoostClassifier(iterations=1658, learning_rate=0.0107, depth=7, l2_leaf_reg=13.9, subsample=0.708,
                               colsample_bylevel=0.818, min_data_in_leaf=39, random_strength=3.15, eval_metric="AUC",
                               random_seed=C.SEED, verbose=0, allow_writing_files=False, task_type=CB)
        m.fit(Xtr, ytr)
    return m.predict_proba(Xva)[:, 1], m.predict_proba(Xt)[:, 1]


def main():
    kind = sys.argv[1] if len(sys.argv) > 1 else "xgb"
    train_ids = np.load(C.TRAIN_IDS_NPY, allow_pickle=True)
    test_ids = np.load(C.TEST_IDS_NPY, allow_pickle=True)
    folds = np.load(C.FOLDS_NPY)
    allf, ydf, _, _ = build_all()
    X = allf.reindex(train_ids).fillna(0.0); Xt = allf.reindex(test_ids).fillna(0.0)
    y = ydf.set_index("custid").reindex(train_ids)["gender"].to_numpy()
    feats = select_features(X, y)
    X = X[feats]; Xt = Xt[feats]
    oof = np.full(len(y), np.nan); test_sum = np.zeros(len(test_ids))
    for f in range(C.N_FOLDS):
        tri, va = np.where(folds != f)[0], np.where(folds == f)[0]
        vp, tp = _fit(kind, X.iloc[tri], y[tri], X.iloc[va], y[va], Xt)
        oof[va] = vp; test_sum += tp
        print(f"[fold {f}] AUC={roc_auc_score(y[va], oof[va]):.5f}")
    cv = float(roc_auc_score(y, oof)); name = f"first_{kind}_fs"
    print(f"\n==== {name}  CV={cv:.5f}  (feats={len(feats)}, DROP={DROP_FRAC} CORR={CORR_THR}) ====")
    save_predictions(name, oof, test_sum / C.N_FOLDS, meta=dict(cv_auc=cv, seed=C.SEED, n_folds=C.N_FOLDS,
                     feature_set=f"1등FE 중요도/상관 선택 {len(feats)}feat", created_by="hyunbean",
                     notes=f"교수힌트#2 feature selection DROP{DROP_FRAC} CORR{CORR_THR}"))


if __name__ == "__main__":
    main()
