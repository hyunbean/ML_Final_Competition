"""Teacher-uncertainty features ('first_xgb_unc') — GPT조언 transductive 레버.

teacher들의 예측 통계(mean/std/entropy/min/max)를 피처로 추가 → test 분포(불확실성) 직접 학습.
teacher OOF=fold-safe(우리 folds), test=teacher test예측. pseudo와 다른 test-side 신호.
실행(GPU): XGB_GPU=1 python -m src.train_unc
"""
import os
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from . import config as C
from .oof_io import save_predictions
from .train_first import build_all

GPU = os.environ.get("XGB_GPU", "1") == "1"
TEACH = ["mh_bestblend69", "mh_05_AutoGluon_megamax", "mh_07_AutoGluon_mega572",
         "mh_09_XGBoost_mega", "first_lgbm", "first_cat", "first_xgb"]


def _stats(mat):
    """(n, k) 예측행렬 → mean/std/entropy/min/max/spread 피처."""
    p = np.clip(mat, 1e-6, 1 - 1e-6)
    ent = -(p * np.log(p) + (1 - p) * np.log(1 - p)).mean(1)
    return pd.DataFrame({
        "tu_mean": mat.mean(1), "tu_std": mat.std(1), "tu_entropy": ent,
        "tu_min": mat.min(1), "tu_max": mat.max(1), "tu_spread": mat.max(1) - mat.min(1),
        "tu_n_high": (mat >= 0.5).sum(1) / mat.shape[1],
    })


def main():
    train_ids = np.load(C.TRAIN_IDS_NPY, allow_pickle=True)
    test_ids = np.load(C.TEST_IDS_NPY, allow_pickle=True)
    folds = np.load(C.FOLDS_NPY)
    allf, ydf, _, _ = build_all()
    y = ydf.set_index("custid").reindex(train_ids)["gender"].to_numpy()
    avail = [m for m in TEACH if os.path.exists(f"artifacts/oof/{m}__oof.npy")]
    print(f"teacher uncertainty from {len(avail)}: {avail}")
    oof_mat = np.column_stack([np.load(f"artifacts/oof/{m}__oof.npy") for m in avail])     # fold-safe OOF
    te_mat = np.column_stack([np.load(f"artifacts/oof/{m}__test.npy") for m in avail])
    Su = _stats(oof_mat); Su.index = train_ids
    St = _stats(te_mat); St.index = test_ids

    X = allf.reindex(train_ids).fillna(0.0).join(Su)
    Xt = allf.reindex(test_ids).fillna(0.0).join(St)
    print(f"X={X.shape} (+{Su.shape[1]} uncertainty feats)")
    params = dict(objective="binary:logistic", eval_metric="auc", learning_rate=0.02, max_depth=7,
                  min_child_weight=5, gamma=0.1, subsample=0.8, colsample_bytree=0.7, reg_alpha=1.0,
                  reg_lambda=5.0, random_state=C.SEED, tree_method="hist", device="cuda" if GPU else "cpu")
    oof = np.full(len(y), np.nan); test_sum = np.zeros(len(test_ids))
    for f in range(C.N_FOLDS):
        tri, va = np.where(folds != f)[0], np.where(folds == f)[0]
        m = xgb.XGBClassifier(n_estimators=4000, early_stopping_rounds=150, **params)
        m.fit(X.iloc[tri], y[tri], eval_set=[(X.iloc[va], y[va])], verbose=False)
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]; test_sum += m.predict_proba(Xt)[:, 1]
        print(f"[fold {f}] AUC={roc_auc_score(y[va], oof[va]):.5f}")
    cv = float(roc_auc_score(y, oof))
    print(f"\n==== first_xgb_unc  CV={cv:.5f} ====")
    save_predictions("first_xgb_unc", oof, test_sum / C.N_FOLDS, meta=dict(cv_auc=cv, seed=C.SEED,
                     n_folds=C.N_FOLDS, feature_set="1등FE + teacher uncertainty(mean/std/entropy)",
                     created_by="hyunbean", notes="GPT transductive: teacher 예측통계 피처"))


if __name__ == "__main__":
    main()
