"""엄격한 Pseudo-labeling ('first_xgb_pl2') — 천장돌파 도박(고변동).

이전 pseudo 실패원인=teacher 자기순환. 이번엔 teacher를 student(first_xgb)와 완전격리:
  teacher = 김민형 독립피처(mega) 모델 + first_lgbm/cat (first_xgb 제외) test rank-avg.
val fold은 절대 pseudo 아님(순수 train) → OOF 정직. 고신뢰(>=0.90/<=0.10) test만 pseudo.
실행(GPU): pip install xgboost gensim catboost → python -m src.train_pseudo_strict
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

MODEL_NAME = "first_xgb_pl2"
HI, LO = 0.90, 0.10
# student(first_xgb)와 격리된 teacher (김민형 mega 독립피처 + 우리 lgbm/cat, xgb계열 제외)
TEACH = ["mh_bestblend69", "mh_05_AutoGluon_megamax", "mh_07_AutoGluon_mega572",
         "mh_09_XGBoost_mega", "first_lgbm", "first_cat"]


def main():
    train_ids = np.load(C.TRAIN_IDS_NPY, allow_pickle=True)
    test_ids = np.load(C.TEST_IDS_NPY, allow_pickle=True)
    folds = np.load(C.FOLDS_NPY)
    allf, ydf, _, _ = build_all()
    X = allf.reindex(train_ids).fillna(0.0)
    Xt = allf.reindex(test_ids).fillna(0.0)
    y = ydf.set_index("custid").reindex(train_ids)["gender"].to_numpy()

    avail = [m for m in TEACH if os.path.exists(f"artifacts/oof/{m}__test.npy")]
    print(f"teacher({len(avail)}): {avail}")
    sub = np.mean([rankdata(np.load(f"artifacts/oof/{m}__test.npy")) / len(test_ids) for m in avail], axis=0)
    conf = (sub >= HI) | (sub <= LO)
    pl_y = (sub[conf] >= 0.5).astype(int)
    Xp = Xt[conf.tolist()]
    print(f"고신뢰 pseudo test: {conf.sum()}/{len(sub)} (pos {pl_y.mean():.2f})")

    params = dict(objective="binary:logistic", eval_metric="auc", learning_rate=0.02, max_depth=7,
                  min_child_weight=5, gamma=0.1, subsample=0.8, colsample_bytree=0.7,
                  reg_alpha=1.0, reg_lambda=5.0, random_state=C.SEED, tree_method="hist",
                  device="cuda" if os.environ.get("XGB_GPU", "1") == "1" else "cpu")
    oof = np.full(len(y), np.nan); test_sum = np.zeros(len(test_ids))
    for f in range(C.N_FOLDS):
        tri, va = np.where(folds != f)[0], np.where(folds == f)[0]
        Xtr = pd.concat([X.iloc[tri], Xp], axis=0)              # train(다른fold) + pseudo test
        ytr = np.concatenate([y[tri], pl_y])
        m = xgb.XGBClassifier(n_estimators=4000, early_stopping_rounds=150, **params)
        m.fit(Xtr, ytr, eval_set=[(X.iloc[va], y[va])], verbose=False)   # val = 순수 train (누수X)
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]; test_sum += m.predict_proba(Xt)[:, 1]
        print(f"[fold {f}] AUC={roc_auc_score(y[va], oof[va]):.5f}")
    cv = float(roc_auc_score(y, oof))
    print(f"\n==== {MODEL_NAME}  CV AUC = {cv:.5f} ====")
    save_predictions(MODEL_NAME, oof, test_sum / C.N_FOLDS, meta=dict(
        cv_auc=cv, seed=C.SEED, n_folds=C.N_FOLDS, feature_set="1등FE + strict pseudo(격리teacher)",
        created_by="hyunbean", notes="strict pseudo-label, teacher=김민형mega+lgbm/cat(xgb격리), val순수"))


if __name__ == "__main__":
    main()
