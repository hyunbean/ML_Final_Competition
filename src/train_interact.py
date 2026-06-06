"""우리 r7 피처 + 김민형 interact 108(top25 2-way곱, null-imp선택) → CatBoost → 'interact_cat'.

명시적 2-way 상호작용을 GBT가 자동으로 못 잡는 부분 보강. 김민형 강한base서 +0.00075(⚠️선택과적합).
interact 블록 custid정렬 = 그의 mega 순서 → 우리 정규순서로 reindex.
실행(GPU): pip install catboost → python -m src.folds → python -m src.train_interact
"""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier

from . import config as C
from .oof_io import save_predictions
from .features import build_features

MODEL_NAME = "interact_cat"
MH = C.ROOT / "mega피처_김민형"
CB_TASK = os.environ.get("CB_TASK_TYPE", "CPU")


def main():
    train_ids = np.load(C.TRAIN_IDS_NPY, allow_pickle=True)
    test_ids = np.load(C.TEST_IDS_NPY, allow_pickle=True)
    folds = np.load(C.FOLDS_NPY)
    X, y, Xt = build_features()

    htr = pd.read_parquet(MH / "mega_train.parquet", columns=[C.ID_COL])[C.ID_COL].to_numpy()
    hte = pd.read_parquet(MH / "mega_test.parquet", columns=[C.ID_COL])[C.ID_COL].to_numpy()
    itr = pd.read_parquet(MH / "feature_attempts/block_s1_interact_tr.parquet")
    ite = pd.read_parquet(MH / "feature_attempts/block_s1_interact_te.parquet")
    itr.index = htr; ite.index = hte
    itr = itr.reindex(train_ids).reset_index(drop=True)
    ite = ite.reindex(test_ids).reset_index(drop=True)

    Xf = pd.concat([X.reset_index(drop=True), itr], axis=1).fillna(0.0)
    Xtf = pd.concat([Xt.reset_index(drop=True), ite], axis=1).fillna(0.0)
    print(f"interact_cat X={Xf.shape} (r7 {X.shape[1]} + interact {itr.shape[1]}) task={CB_TASK}")

    params = dict(iterations=2000, learning_rate=0.03, depth=6, l2_leaf_reg=5.0,
                  random_strength=3.0, eval_metric="AUC", random_seed=C.SEED,
                  verbose=0, allow_writing_files=False, task_type=CB_TASK)
    if CB_TASK == "GPU":
        params["devices"] = "0"
    oof = np.full(len(y), np.nan); test_sum = np.zeros(len(test_ids))
    for f in range(C.N_FOLDS):
        tri, va = np.where(folds != f)[0], np.where(folds == f)[0]
        m = CatBoostClassifier(**params)
        m.fit(Xf.iloc[tri], y[tri], eval_set=(Xf.iloc[va], y[va]), early_stopping_rounds=150)
        oof[va] = m.predict_proba(Xf.iloc[va])[:, 1]; test_sum += m.predict_proba(Xtf)[:, 1]
        print(f"[fold {f}] AUC={roc_auc_score(y[va], oof[va]):.5f}")
    cv = float(roc_auc_score(y, oof))
    print(f"\n==== {MODEL_NAME}  CV AUC = {cv:.5f} ====")
    save_predictions(MODEL_NAME, oof, test_sum / C.N_FOLDS, meta=dict(
        cv_auc=cv, seed=C.SEED, n_folds=C.N_FOLDS, feature_set=f"r7({X.shape[1]})+interact108 CatBoost",
        created_by="hyunbean", notes="우리 r7 + 김민형 interact 108 2-way곱, 5fold OOF"))


if __name__ == "__main__":
    main()
