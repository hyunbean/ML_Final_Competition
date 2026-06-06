"""Knowledge distillation ('distill_nn') — 앙상블 OOF(soft target)를 raw 피처로 모방하는 단일모델.

teacher = 강한 멤버들 rank평균(fold-safe OOF). student = r-features로 teacher 확률 회귀(MSE) NN/LGBM.
앙상블 지식을 단일에 압축 → 다른 에러구조 가능(보통 ensemble과 상관높아 기여 작음). 기록/실험용.
실행: pip install lightgbm gensim → python -m src.train_distill
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from . import config as C
from .oof_io import save_predictions
from .features import build_features

MODEL_NAME = "distill_lgbm"
TEACHERS = ["first_xgb", "first_lgbm", "first_cat", "kitchen_lgbm", "mh_s1_augdeep_mlp_adv"]


def main():
    folds = np.load(C.FOLDS_NPY)
    test_ids = np.load(C.TEST_IDS_NPY, allow_pickle=True)
    X, y, Xt = build_features()
    # teacher soft target = 강한멤버 rank평균 (OOF)
    t = np.mean([rankdata(np.load(f"artifacts/oof/{m}__oof.npy")) / len(y) for m in TEACHERS], axis=0)
    print(f"teacher AUC={roc_auc_score(y, t):.5f}, student X={X.shape}")
    oof = np.full(len(y), np.nan); test_sum = np.zeros(len(test_ids))
    params = dict(objective="regression", metric="rmse", learning_rate=0.02, num_leaves=63,
                  min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.6,
                  reg_alpha=1.0, reg_lambda=5.0, n_jobs=-1, random_state=C.SEED, verbose=-1)
    for f in range(C.N_FOLDS):
        tri, va = np.where(folds != f)[0], np.where(folds == f)[0]
        m = lgb.LGBMRegressor(n_estimators=2000, **params)
        m.fit(X.iloc[tri], t[tri], eval_set=[(X.iloc[va], t[va])], callbacks=[lgb.early_stopping(120, verbose=False)])
        oof[va] = m.predict(X.iloc[va]); test_sum += m.predict(Xt)
        print(f"[fold {f}] AUC={roc_auc_score(y[va], oof[va]):.5f}")
    cv = float(roc_auc_score(y, oof))
    print(f"\n==== {MODEL_NAME}  CV AUC = {cv:.5f} ====")
    save_predictions(MODEL_NAME, oof, test_sum / C.N_FOLDS, meta=dict(
        cv_auc=cv, seed=C.SEED, n_folds=C.N_FOLDS, feature_set="KD: 앙상블 soft target 모방",
        created_by="hyunbean", notes="knowledge distillation from strong ensemble OOF"))


if __name__ == "__main__":
    main()
