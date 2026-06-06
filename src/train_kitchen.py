"""Kitchen-sink — 우리 r7 + 1등 497 + mega 572 피처 전부 합쳐 LightGBM → 'kitchen_lgbm'.

세 강한 피처셋을 한 모델에 → 우리 최강 단일 후보. 셋 다 custid 정렬 일치시켜 concat.
실행(GPU권장): pip install lightgbm catboost gensim pyarrow
     python -m src.folds → python -m src.train_kitchen
"""
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

from . import config as C
from .oof_io import save_predictions
from .features import build_features
from .train_first import build_all

MODEL_NAME = "kitchen_lgbm"
MEGA_DIR = os.environ.get("MEGA_DIR", str(C.ROOT / "민형_mega"))
GPU = os.environ.get("LGB_GPU", "0") == "1"


def main():
    train_ids = np.load(C.TRAIN_IDS_NPY, allow_pickle=True)
    test_ids = np.load(C.TEST_IDS_NPY, allow_pickle=True)
    folds = np.load(C.FOLDS_NPY)

    # 1) 우리 r7 (train_ids/test_ids 순서로 정렬되어 반환)
    Xo, y, Xot = build_features()
    Xo = Xo.add_prefix("our_"); Xot = Xot.add_prefix("our_")

    # 2) 1등 497
    allf, _, _, _ = build_all()
    A_tr = allf.reindex(train_ids).reset_index(drop=True).add_prefix("f1_")
    A_te = allf.reindex(test_ids).reset_index(drop=True).add_prefix("f1_")

    # 3) mega 572
    mtr = pd.read_parquet(f"{MEGA_DIR}/mega_train.parquet").set_index(C.ID_COL).reindex(train_ids).reset_index(drop=True)
    mte = pd.read_parquet(f"{MEGA_DIR}/mega_test.parquet").set_index(C.ID_COL).reindex(test_ids).reset_index(drop=True)
    mtr.columns = [f"mg_{i}" for i in range(mtr.shape[1])]; mte.columns = mtr.columns

    X = pd.concat([Xo, A_tr, mtr], axis=1).fillna(0.0)
    Xt = pd.concat([Xot, A_te, mte], axis=1).fillna(0.0)
    Xt = Xt.reindex(columns=X.columns, fill_value=0.0)
    print(f"kitchen X={X.shape}  (우리+1등+mega 합본) GPU={GPU}")

    params = dict(objective="binary", metric="auc", learning_rate=0.02, num_leaves=80,
                  min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
                  reg_alpha=2.0, reg_lambda=8.0, n_jobs=-1, random_state=C.SEED, verbose=-1)
    if GPU:
        params["device"] = "gpu"
    oof = np.full(len(y), np.nan); test_sum = np.zeros(len(test_ids))
    for f in range(C.N_FOLDS):
        tri, va = np.where(folds != f)[0], np.where(folds == f)[0]
        m = lgb.LGBMClassifier(n_estimators=5000, **params)
        m.fit(X.iloc[tri], y[tri], eval_set=[(X.iloc[va], y[va])],
              callbacks=[lgb.early_stopping(200, verbose=False)])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]; test_sum += m.predict_proba(Xt)[:, 1]
        print(f"[fold {f}] AUC={roc_auc_score(y[va], oof[va]):.5f}")
    cv = float(roc_auc_score(y, oof))
    print(f"\n==== {MODEL_NAME}  CV AUC = {cv:.5f} ====")
    save_predictions(MODEL_NAME, oof, test_sum / C.N_FOLDS, meta=dict(
        cv_auc=cv, seed=C.SEED, n_folds=C.N_FOLDS, feature_set=f"우리r7+1등497+mega572 합본({X.shape[1]}) LGBM",
        created_by="hyunbean", notes="kitchen-sink: 3개 피처셋 전부 합쳐 LightGBM 5fold OOF"))


if __name__ == "__main__":
    main()
