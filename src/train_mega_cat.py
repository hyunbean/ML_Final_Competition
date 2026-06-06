"""김민형 mega 572피처 + 튜닝 CatBoost → 'mega_cat'. mega_ag(AutoGluon)와 다른 모델, 빠름.

mega 피처셋은 우리/1등과 또 다른 구성 → 새 멤버 기대. 폴드 동일(folds.npy).
실행: pip install catboost pyarrow → python -m src.folds → python -m src.train_mega_cat
"""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier

from . import config as C
from .oof_io import save_predictions

MODEL_NAME = "mega_cat"
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
    print(f"mega_cat X={tr.shape}")

    params = dict(iterations=2000, learning_rate=0.0107, depth=7, l2_leaf_reg=13.9,
                  subsample=0.708, colsample_bylevel=0.818, min_data_in_leaf=39,
                  random_strength=3.15, eval_metric="AUC", random_seed=C.SEED,
                  verbose=0, allow_writing_files=False)
    oof = np.full(len(y), np.nan); test_sum = np.zeros(len(test_ids))
    for f in range(C.N_FOLDS):
        tri, va = np.where(folds != f)[0], np.where(folds == f)[0]
        m = CatBoostClassifier(**params)
        m.fit(tr.iloc[tri], y[tri], eval_set=(tr.iloc[va], y[va]), early_stopping_rounds=150)
        oof[va] = m.predict_proba(tr.iloc[va])[:, 1]; test_sum += m.predict_proba(te)[:, 1]
        print(f"[fold {f}] AUC={roc_auc_score(y[va], oof[va]):.5f}")
    cv = float(roc_auc_score(y, oof))
    print(f"\n==== {MODEL_NAME}  CV AUC = {cv:.5f} ====")
    save_predictions(MODEL_NAME, oof, test_sum / C.N_FOLDS, meta=dict(
        cv_auc=cv, seed=C.SEED, n_folds=C.N_FOLDS, feature_set="mega572 CatBoost",
        created_by="hyunbean", notes="mega 572feat + tuned CatBoost 5fold OOF"))


if __name__ == "__main__":
    main()
