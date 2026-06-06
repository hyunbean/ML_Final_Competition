"""김민형 mega 572피처 + XGBoost → 'mega_xgb'. mega_cat/mega_ag와 또 다른 모델.

실행: pip install xgboost pyarrow → python -m src.folds → python -m src.train_mega_xgb
GPU 런타임: XGB_GPU=1 환경변수
"""
import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

from . import config as C
from .oof_io import save_predictions

MODEL_NAME = "mega_xgb"
MEGA_DIR = os.environ.get("MEGA_DIR", str(C.ROOT / "민형_mega"))
GPU = os.environ.get("XGB_GPU", "0") == "1"


def main():
    train_ids = np.load(C.TRAIN_IDS_NPY, allow_pickle=True)
    test_ids = np.load(C.TEST_IDS_NPY, allow_pickle=True)
    folds = np.load(C.FOLDS_NPY)
    y = pd.read_csv(C.YTRAIN_CSV).set_index(C.ID_COL).reindex(train_ids)[C.TARGET].to_numpy()

    tr = pd.read_parquet(f"{MEGA_DIR}/mega_train.parquet").set_index(C.ID_COL).reindex(train_ids).reset_index(drop=True)
    te = pd.read_parquet(f"{MEGA_DIR}/mega_test.parquet").set_index(C.ID_COL).reindex(test_ids).reset_index(drop=True)
    tr.columns = [f"c{i}" for i in range(tr.shape[1])]; te.columns = tr.columns
    tr = tr.fillna(0.0); te = te.fillna(0.0)
    print(f"mega_xgb X={tr.shape} GPU={GPU}")

    params = dict(objective="binary:logistic", eval_metric="auc", learning_rate=0.02,
                  max_depth=8, min_child_weight=5, gamma=0.1, subsample=0.8,
                  colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=5.0, random_state=C.SEED)
    if GPU:
        params.update(tree_method="hist", device="cuda")
    oof = np.full(len(y), np.nan); test_sum = np.zeros(len(test_ids))
    for f in range(C.N_FOLDS):
        tri, va = np.where(folds != f)[0], np.where(folds == f)[0]
        m = xgb.XGBClassifier(n_estimators=4000, early_stopping_rounds=150, **params)
        m.fit(tr.iloc[tri], y[tri], eval_set=[(tr.iloc[va], y[va])], verbose=False)
        oof[va] = m.predict_proba(tr.iloc[va])[:, 1]; test_sum += m.predict_proba(te)[:, 1]
        print(f"[fold {f}] AUC={roc_auc_score(y[va], oof[va]):.5f}")
    cv = float(roc_auc_score(y, oof))
    print(f"\n==== {MODEL_NAME}  CV AUC = {cv:.5f} ====")
    save_predictions(MODEL_NAME, oof, test_sum / C.N_FOLDS, meta=dict(
        cv_auc=cv, seed=C.SEED, n_folds=C.N_FOLDS, feature_set="mega572 XGBoost",
        created_by="hyunbean", notes="mega 572feat + XGBoost 5fold OOF"))


if __name__ == "__main__":
    main()
