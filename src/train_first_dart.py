"""DART LGBM (1등 피처셋) → 'first_lgbm_dart'. 트리 드롭아웃 부스팅 = 일반 GBDT와 결 다름(decorrelate).
CPU 전용. 실행: pip install lightgbm gensim catboost → python -m src.train_first_dart
"""
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from . import config as C
from .oof_io import save_predictions
from .train_first import build_all

MODEL_NAME = "first_lgbm_dart"


def main():
    train_ids = np.load(C.TRAIN_IDS_NPY, allow_pickle=True)
    test_ids = np.load(C.TEST_IDS_NPY, allow_pickle=True)
    folds = np.load(C.FOLDS_NPY)
    allf, ydf, _, _ = build_all()
    X = allf.reindex(train_ids).fillna(0.0)
    Xt = allf.reindex(test_ids).fillna(0.0)
    y = ydf.set_index("custid").reindex(train_ids)["gender"].to_numpy()
    print(f"DART X={X.shape}")
    params = dict(objective="binary", metric="auc", boosting_type="dart", learning_rate=0.05,
                  num_leaves=63, min_child_samples=40, subsample=0.8, colsample_bytree=0.7,
                  reg_alpha=1.0, reg_lambda=5.0, drop_rate=0.1, n_jobs=-1, random_state=C.SEED, verbose=-1)
    oof = np.full(len(y), np.nan); ts = np.zeros(len(test_ids))
    for f in range(C.N_FOLDS):
        tri, va = np.where(folds != f)[0], np.where(folds == f)[0]
        m = lgb.LGBMClassifier(n_estimators=1500, **params)   # dart는 early stop 안 됨 → 고정
        m.fit(X.iloc[tri], y[tri])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]; ts += m.predict_proba(Xt)[:, 1]
        print(f"[fold {f}] AUC={roc_auc_score(y[va], oof[va]):.5f}")
    cv = float(roc_auc_score(y, oof)); print(f"\n==== {MODEL_NAME}  CV AUC = {cv:.5f} ====")
    save_predictions(MODEL_NAME, oof, ts / C.N_FOLDS, meta=dict(cv_auc=cv, seed=C.SEED, n_folds=C.N_FOLDS,
                     feature_set="1등 FE DART LGBM", created_by="hyunbean", notes="DART boosting (decorrelated GBDT)"))


if __name__ == "__main__":
    main()
