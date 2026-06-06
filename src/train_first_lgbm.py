"""1등 노트북 피처셋(497) + LightGBM → 'first_lgbm'. first_cat(CatBoost)와 모델 달라 결 다름.

실행(Colab/DLPC): pip install lightgbm → python -m src.folds → python -m src.train_first_lgbm
"""
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

from . import config as C
from .oof_io import save_predictions
from .train_first import build_all

MODEL_NAME = "first_lgbm"


def main():
    train_ids = np.load(C.TRAIN_IDS_NPY, allow_pickle=True)
    test_ids = np.load(C.TEST_IDS_NPY, allow_pickle=True)
    folds = np.load(C.FOLDS_NPY)
    allf, y_df, _, _ = build_all()
    X = allf.reindex(train_ids).fillna(0.0)
    Xt = allf.reindex(test_ids).fillna(0.0)
    y = y_df.set_index("custid").reindex(train_ids)["gender"].to_numpy()
    print(f"X={X.shape} Xt={Xt.shape}")

    params = dict(objective="binary", metric="auc", learning_rate=0.02, num_leaves=64,
                  max_depth=-1, min_child_samples=40, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=5.0, n_jobs=-1,
                  random_state=C.SEED, verbose=-1)
    oof = np.full(len(y), np.nan); test_sum = np.zeros(len(test_ids))
    for f in range(C.N_FOLDS):
        tri, va = np.where(folds != f)[0], np.where(folds == f)[0]
        m = lgb.LGBMClassifier(n_estimators=4000, **params)
        m.fit(X.iloc[tri], y[tri], eval_set=[(X.iloc[va], y[va])],
              callbacks=[lgb.early_stopping(150, verbose=False)])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]; test_sum += m.predict_proba(Xt)[:, 1]
        print(f"[fold {f}] AUC={roc_auc_score(y[va], oof[va]):.5f}")
    cv = float(roc_auc_score(y, oof))
    print(f"\n==== {MODEL_NAME}  CV AUC = {cv:.5f} ====")
    save_predictions(MODEL_NAME, oof, test_sum / C.N_FOLDS, meta=dict(
        cv_auc=cv, seed=C.SEED, n_folds=C.N_FOLDS, feature_set=f"작년1등 FE ({X.shape[1]}feat) LightGBM",
        created_by="hyunbean", notes="1st-place notebook FE + LightGBM 5fold OOF"))


if __name__ == "__main__":
    main()
