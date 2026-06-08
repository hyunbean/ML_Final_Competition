"""Soft pseudo-labeling ('first_xgb_pls') — hard 0/1 대신 teacher 확률을 soft target으로.

hard는 0.91·0.99 둘다 '1'(정보손실,과신). soft는 teacher 확률 그대로 타겟 → 불확실성 보존, mirage 덜함.
xgb.train(binary:logistic)은 float label 허용. val fold 순수(OOF정직). teacher=student격리.
실행(GPU): PL_CONF=0.15 XGB_GPU=1 python -m src.train_pseudo_soft
"""
import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

from . import config as C
from .oof_io import save_predictions
from .train_first import build_all

GPU = os.environ.get("XGB_GPU", "1") == "1"
CONF = float(os.environ.get("PL_CONF", "0.15"))   # |p-0.5|>=CONF 인 test만 soft pseudo (0이면 전체)
TEACH = ["mh_bestblend69", "mh_05_AutoGluon_megamax", "mh_07_AutoGluon_mega572",
         "mh_09_XGBoost_mega", "first_lgbm", "first_cat"]


def main():
    train_ids = np.load(C.TRAIN_IDS_NPY, allow_pickle=True)
    test_ids = np.load(C.TEST_IDS_NPY, allow_pickle=True)
    folds = np.load(C.FOLDS_NPY)
    allf, ydf, _, _ = build_all()
    X = allf.reindex(train_ids).fillna(0.0)
    Xt = allf.reindex(test_ids).fillna(0.0)
    y = ydf.set_index("custid").reindex(train_ids)["gender"].to_numpy().astype(float)

    avail = [m for m in TEACH if os.path.exists(f"artifacts/oof/{m}__test.npy")]
    soft = np.mean([np.load(f"artifacts/oof/{m}__test.npy") for m in avail], axis=0)   # teacher 평균확률(soft)
    sel = np.abs(soft - 0.5) >= CONF
    Xp = Xt[sel.tolist()]; soft_y = soft[sel]
    print(f"teacher={avail} | soft pseudo {sel.sum()}/{len(soft)} (CONF>={CONF})")

    params = dict(objective="binary:logistic", eval_metric="auc", eta=0.02, max_depth=7,
                  min_child_weight=5, gamma=0.1, subsample=0.8, colsample_bytree=0.7,
                  reg_alpha=1.0, reg_lambda=5.0, seed=C.SEED, tree_method="hist",
                  device="cuda" if GPU else "cpu")
    oof = np.full(len(y), np.nan); test_sum = np.zeros(len(test_ids))
    for f in range(C.N_FOLDS):
        tri, va = np.where(folds != f)[0], np.where(folds == f)[0]
        Xtr = pd.concat([X.iloc[tri], Xp], axis=0)
        ytr = np.concatenate([y[tri], soft_y])                      # 진짜라벨(0/1) + soft(확률)
        dtr = xgb.DMatrix(Xtr, label=ytr); dva = xgb.DMatrix(X.iloc[va], label=y[va])
        dt = xgb.DMatrix(Xt)
        bst = xgb.train(params, dtr, num_boost_round=4000, evals=[(dva, "v")],
                        early_stopping_rounds=150, verbose_eval=False)
        oof[va] = bst.predict(xgb.DMatrix(X.iloc[va])); test_sum += bst.predict(dt)
        print(f"[fold {f}] AUC={roc_auc_score(y[va], oof[va]):.5f}")
    cv = float(roc_auc_score(y, oof))
    print(f"\n==== first_xgb_pls  CV={cv:.5f} ====")
    save_predictions("first_xgb_pls", oof, test_sum / C.N_FOLDS, meta=dict(cv_auc=cv, seed=C.SEED,
                     n_folds=C.N_FOLDS, feature_set="1등FE + soft pseudo(teacher확률 타겟)",
                     created_by="hyunbean", notes="soft pseudo-labeling, 불확실성 보존"))


if __name__ == "__main__":
    main()
