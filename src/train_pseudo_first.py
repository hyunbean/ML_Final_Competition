"""Pseudo-labeling ('first_xgb_pl') — 현재 최고 블렌드로 test 고신뢰 라벨 → train에 추가 재학습.

test 20k(라벨없음)를 준지도로 활용. 비지도 representation(0.66캡)과 다른 차원 = 천장돌파 후보.
OOF-safe: val fold은 절대 pseudo 아님(진짜 train만). pseudo는 test에서만(train누수 없음).
실행(GPU): pip install xgboost gensim catboost → python -m src.train_pseudo_first
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

from . import config as C
from .oof_io import save_predictions
from .train_first import build_all

MODEL_NAME = "first_xgb_pl"
HI, LO = 0.85, 0.15   # 고신뢰 임계


def main():
    train_ids = np.load(C.TRAIN_IDS_NPY, allow_pickle=True)
    test_ids = np.load(C.TEST_IDS_NPY, allow_pickle=True)
    folds = np.load(C.FOLDS_NPY)
    allf, ydf, _, _ = build_all()
    X = allf.reindex(train_ids).fillna(0.0)
    Xt = allf.reindex(test_ids).fillna(0.0)
    y = ydf.set_index("custid").reindex(train_ids)["gender"].to_numpy()

    # teacher = 현재 최고 블렌드 test 예측 (submission_stack3 = 최신 블렌드)
    sub = pd.read_csv(C.SUB_DIR / "submission_stack3.csv").set_index(C.ID_COL).reindex(test_ids)[C.TARGET].to_numpy()
    conf = (sub >= HI) | (sub <= LO)
    pl_y = (sub[conf] >= 0.5).astype(int)
    Xp = Xt[conf]
    print(f"pseudo 고신뢰 test: {conf.sum()}/{len(sub)} (pos {pl_y.mean():.2f})")

    params = dict(objective="binary:logistic", eval_metric="auc", learning_rate=0.02, max_depth=7,
                  min_child_weight=5, gamma=0.1, subsample=0.8, colsample_bytree=0.7,
                  reg_alpha=1.0, reg_lambda=5.0, random_state=C.SEED, tree_method="hist", device="cuda")
    oof = np.full(len(y), np.nan); test_sum = np.zeros(len(test_ids))
    for f in range(C.N_FOLDS):
        tri, va = np.where(folds != f)[0], np.where(folds == f)[0]
        # 학습 = train(다른fold) + pseudo test (val fold은 진짜 train만, 누수X)
        Xtr = pd.concat([X.iloc[tri], Xp], axis=0)
        ytr = np.concatenate([y[tri], pl_y])
        m = xgb.XGBClassifier(n_estimators=4000, early_stopping_rounds=150, **params)
        m.fit(Xtr, ytr, eval_set=[(X.iloc[va], y[va])], verbose=False)
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]; test_sum += m.predict_proba(Xt)[:, 1]
        print(f"[fold {f}] AUC={roc_auc_score(y[va], oof[va]):.5f}")
    cv = float(roc_auc_score(y, oof))
    print(f"\n==== {MODEL_NAME}  CV AUC = {cv:.5f} ====")
    save_predictions(MODEL_NAME, oof, test_sum / C.N_FOLDS, meta=dict(
        cv_auc=cv, seed=C.SEED, n_folds=C.N_FOLDS, feature_set="1등 FE + pseudo-label(test 준지도)",
        created_by="hyunbean", notes="pseudo-labeling on 1st-place features, 준지도 천장돌파 시도"))


if __name__ == "__main__":
    main()
