"""다중시드 GBDT 평균 (first_{xgb,lgbm,cat}_5s) — 같은 강모델을 5시드로 OOF/test 평균.

분산감소 → private LB robust (셰이크업 방어). 신기루 아님(같은 모델 평균이라 진짜 안정화).
실행: pip install xgboost lightgbm catboost gensim
  XGB_GPU=1 CB_TASK_TYPE=GPU python -m src.train_seedbag [xgb|lgbm|cat|all]
"""
import os
import sys
import numpy as np
from sklearn.metrics import roc_auc_score

from . import config as C
from .oof_io import save_predictions
from .train_first import build_all

SEEDS = [42, 7, 123, 2024, 99]
GPU_XGB = os.environ.get("XGB_GPU", "0") == "1"
CB_TASK = os.environ.get("CB_TASK_TYPE", "CPU")


def _fit_oof(kind, X, Xt, y, folds, seed):
    import xgboost as xgb
    import lightgbm as lgb
    from catboost import CatBoostClassifier
    oof = np.full(len(y), np.nan); test = np.zeros(Xt.shape[0])
    for f in range(C.N_FOLDS):
        tri, va = np.where(folds != f)[0], np.where(folds == f)[0]
        Xtr, ytr, Xva, yva = X.iloc[tri], y[tri], X.iloc[va], y[va]
        if kind == "xgb":
            p = dict(objective="binary:logistic", eval_metric="auc", learning_rate=0.02, max_depth=7,
                     min_child_weight=5, gamma=0.1, subsample=0.8, colsample_bytree=0.7,
                     reg_alpha=1.0, reg_lambda=5.0, random_state=seed)
            if GPU_XGB:
                p.update(tree_method="hist", device="cuda")
            m = xgb.XGBClassifier(n_estimators=4000, early_stopping_rounds=150, **p)
            m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        elif kind == "lgbm":
            p = dict(objective="binary", metric="auc", learning_rate=0.02, num_leaves=64, max_depth=-1,
                     min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                     reg_alpha=1.0, reg_lambda=5.0, n_jobs=-1, random_state=seed, verbose=-1)
            m = lgb.LGBMClassifier(n_estimators=4000, **p)
            m.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[lgb.early_stopping(150, verbose=False)])
        else:
            p = dict(iterations=1658, learning_rate=0.0107, depth=7, l2_leaf_reg=13.9, subsample=0.708,
                     colsample_bylevel=0.818, min_data_in_leaf=39, random_strength=3.15, eval_metric="AUC",
                     random_seed=seed, verbose=0, allow_writing_files=False, task_type=CB_TASK)
            m = CatBoostClassifier(**p)
            m.fit(Xtr, ytr)
        oof[va] = m.predict_proba(Xva)[:, 1]; test += m.predict_proba(Xt)[:, 1] / C.N_FOLDS
    return oof, test


def run(kind, X, Xt, y, folds, test_ids):
    oof_sum = np.zeros(len(y)); test_sum = np.zeros(len(test_ids))
    for s in SEEDS:
        oof, test = _fit_oof(kind, X, Xt, y, folds, s)
        oof_sum += oof / len(SEEDS); test_sum += test / len(SEEDS)
        print(f"  [{kind} seed {s}] CV={roc_auc_score(y, oof):.5f}")
    cv = float(roc_auc_score(y, oof_sum))
    name = f"first_{kind}_5s"
    print(f"==== {name}  5seed평균 CV = {cv:.5f} ====")
    save_predictions(name, oof_sum, test_sum, meta=dict(
        cv_auc=cv, seed=C.SEED, n_folds=C.N_FOLDS, feature_set=f"작년1등 FE, {len(SEEDS)}seed 평균",
        created_by="hyunbean", notes=f"{kind} {len(SEEDS)}-seed bag (분산감소 robust)"))


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    train_ids = np.load(C.TRAIN_IDS_NPY, allow_pickle=True)
    test_ids = np.load(C.TEST_IDS_NPY, allow_pickle=True)
    folds = np.load(C.FOLDS_NPY)
    allf, ydf, _, _ = build_all()
    X = allf.reindex(train_ids).fillna(0)
    Xt = allf.reindex(test_ids).fillna(0)
    y = ydf.set_index("custid").reindex(train_ids)["gender"].to_numpy()
    print(f"X={X.shape}  seeds={SEEDS}  which={which}")
    for kind in (["xgb", "lgbm", "cat"] if which == "all" else [which]):
        print(f"\n--- {kind} seedbag ---")
        run(kind, X, Xt, y, folds, test_ids)


if __name__ == "__main__":
    main()
