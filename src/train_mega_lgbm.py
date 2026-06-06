"""김민형 mega 피처 + LightGBM → 'mega_lgbm'. mega_cat/xgb와 또 다른 모델(같은피처 다른에러).

김민형 업데이트 피처 받으면 mega_cat/xgb/lgbm 3종 다 돌려 다양성 확보.
실행: pip install lightgbm pyarrow → python -m src.folds → python -m src.train_mega_lgbm
"""
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

from . import config as C
from .oof_io import save_predictions

MODEL_NAME = "mega_lgbm"
MEGA_DIR = os.environ.get("MEGA_DIR", str(C.ROOT / "민형_mega"))


def main():
    train_ids = np.load(C.TRAIN_IDS_NPY, allow_pickle=True)
    test_ids = np.load(C.TEST_IDS_NPY, allow_pickle=True)
    folds = np.load(C.FOLDS_NPY)
    y = pd.read_csv(C.YTRAIN_CSV).set_index(C.ID_COL).reindex(train_ids)[C.TARGET].to_numpy()

    tr = pd.read_parquet(f"{MEGA_DIR}/mega_train.parquet").set_index(C.ID_COL).reindex(train_ids).reset_index(drop=True)
    te = pd.read_parquet(f"{MEGA_DIR}/mega_test.parquet").set_index(C.ID_COL).reindex(test_ids).reset_index(drop=True)
    tr.columns = [f"c{i}" for i in range(tr.shape[1])]; te.columns = tr.columns
    tr = tr.fillna(0.0); te = te.fillna(0.0)
    print(f"mega_lgbm X={tr.shape}")

    params = dict(objective="binary", metric="auc", learning_rate=0.02, num_leaves=63,
                  min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.6,
                  reg_alpha=1.0, reg_lambda=5.0, n_jobs=-1, random_state=C.SEED, verbose=-1)
    oof = np.full(len(y), np.nan); test_sum = np.zeros(len(test_ids))
    for f in range(C.N_FOLDS):
        tri, va = np.where(folds != f)[0], np.where(folds == f)[0]
        m = lgb.LGBMClassifier(n_estimators=4000, **params)
        m.fit(tr.iloc[tri], y[tri], eval_set=[(tr.iloc[va], y[va])], callbacks=[lgb.early_stopping(150, verbose=False)])
        oof[va] = m.predict_proba(tr.iloc[va])[:, 1]; test_sum += m.predict_proba(te)[:, 1]
        print(f"[fold {f}] AUC={roc_auc_score(y[va], oof[va]):.5f}")
    cv = float(roc_auc_score(y, oof))
    print(f"\n==== {MODEL_NAME}  CV AUC = {cv:.5f} ====")
    save_predictions(MODEL_NAME, oof, test_sum / C.N_FOLDS, meta=dict(
        cv_auc=cv, seed=C.SEED, n_folds=C.N_FOLDS, feature_set="mega LightGBM",
        created_by="hyunbean", notes="mega 피처 + LightGBM 5fold OOF"))


if __name__ == "__main__":
    main()
