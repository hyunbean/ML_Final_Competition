"""1등 노트북 피처셋(497) + XGBoost → 'first_xgb'. first_cat/first_lgbm과 또 다른 모델.

1등 피처셋은 우리 블렌드를 가장 크게 올린 셋(first_cat/lgbm +0.0006). 세 번째 모델(XGB)로 다양성.
실행(GPU): pip install xgboost gensim → python -m src.folds → XGB_GPU=1 python -m src.train_first_xgb
"""
import os
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score

from . import config as C
from .oof_io import save_predictions
from .train_first import build_all

MODEL_NAME = "first_xgb"
GPU = os.environ.get("XGB_GPU", "0") == "1"


def main():
    train_ids = np.load(C.TRAIN_IDS_NPY, allow_pickle=True)
    test_ids = np.load(C.TEST_IDS_NPY, allow_pickle=True)
    folds = np.load(C.FOLDS_NPY)
    allf, y_df, _, _ = build_all()
    X = allf.reindex(train_ids).fillna(0.0)
    Xt = allf.reindex(test_ids).fillna(0.0)
    y = y_df.set_index("custid").reindex(train_ids)["gender"].to_numpy()
    print(f"X={X.shape} GPU={GPU}")

    params = dict(objective="binary:logistic", eval_metric="auc", learning_rate=0.02,
                  max_depth=7, min_child_weight=5, gamma=0.1, subsample=0.8,
                  colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=5.0, random_state=C.SEED)
    if GPU:
        params.update(tree_method="hist", device="cuda")
    oof = np.full(len(y), np.nan); test_sum = np.zeros(len(test_ids))
    for f in range(C.N_FOLDS):
        tri, va = np.where(folds != f)[0], np.where(folds == f)[0]
        m = xgb.XGBClassifier(n_estimators=4000, early_stopping_rounds=150, **params)
        m.fit(X.iloc[tri], y[tri], eval_set=[(X.iloc[va], y[va])], verbose=False)
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]; test_sum += m.predict_proba(Xt)[:, 1]
        print(f"[fold {f}] AUC={roc_auc_score(y[va], oof[va]):.5f}")
    cv = float(roc_auc_score(y, oof))
    print(f"\n==== {MODEL_NAME}  CV AUC = {cv:.5f} ====")
    save_predictions(MODEL_NAME, oof, test_sum / C.N_FOLDS, meta=dict(
        cv_auc=cv, seed=C.SEED, n_folds=C.N_FOLDS, feature_set="작년1등 FE(497) XGBoost",
        created_by="hyunbean", notes="1st-place FE + XGBoost 5fold OOF"))


if __name__ == "__main__":
    main()
