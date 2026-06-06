"""GBDT leaf-index → LogReg ('leaf_lr'). 트리 잎인덱스 원핫 → 선형 = 다른 기하/에러.

LightGBM이 각 샘플을 어떤 잎에 보내는지(leaf index)를 원핫→LogReg. 트리가 못 만드는
'잎 공기 조합의 선형결합'을 학습 → 기존 GBDT와 결 다른 멤버 (거의 공짜). fold-safe.
실행: pip install lightgbm → python -m src.folds → python -m src.train_leaf
"""
import numpy as np
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score

from . import config as C
from .oof_io import save_predictions
from .features import build_features

MODEL_NAME = "leaf_lr"


def main():
    folds = np.load(C.FOLDS_NPY)
    test_ids = np.load(C.TEST_IDS_NPY, allow_pickle=True)
    X, y, Xt = build_features()
    print(f"X={X.shape}")
    oof = np.full(len(y), np.nan); test_sum = np.zeros(len(test_ids))
    for f in range(C.N_FOLDS):
        tri, va = np.where(folds != f)[0], np.where(folds == f)[0]
        gbm = lgb.LGBMClassifier(n_estimators=300, num_leaves=31, learning_rate=0.05,
                                 subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
                                 random_state=C.SEED, verbose=-1).fit(X.iloc[tri], y[tri])
        b = gbm.booster_
        Ltr = b.predict(X.iloc[tri], pred_leaf=True)
        Lva = b.predict(X.iloc[va], pred_leaf=True)
        Lte = b.predict(Xt, pred_leaf=True)
        ohe = OneHotEncoder(handle_unknown="ignore").fit(Ltr)
        lr = LogisticRegression(C=0.3, max_iter=3000).fit(ohe.transform(Ltr), y[tri])
        oof[va] = lr.predict_proba(ohe.transform(Lva))[:, 1]
        test_sum += lr.predict_proba(ohe.transform(Lte))[:, 1]
        print(f"[fold {f}] AUC={roc_auc_score(y[va], oof[va]):.5f}")
    cv = float(roc_auc_score(y, oof))
    print(f"\n==== {MODEL_NAME}  CV AUC = {cv:.5f} ====")
    save_predictions(MODEL_NAME, oof, test_sum / C.N_FOLDS, meta=dict(
        cv_auc=cv, seed=C.SEED, n_folds=C.N_FOLDS, feature_set="r7 GBDT leaf-index → LogReg",
        created_by="hyunbean", notes="Facebook GBDT+LR: 잎인덱스 원핫→LogReg, 5fold OOF"))


if __name__ == "__main__":
    main()
