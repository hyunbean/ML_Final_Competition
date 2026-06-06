"""AUC 직접 최적화 — XGBoost rank:pairwise + LGBM focal loss (다른 손실 = decorrelated).

지표가 AUC(순위)인데 그동안 logloss만 썼음. 순위손실/focal로 학습하면
logloss 모델과 상관 낮아 블렌드 다양성. 우리 풀피처, 5-fold OOF.
실행: python -m src.train_rank
"""
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

from . import config as C
from .features import build_features
from .oof_io import save_predictions


def _focal_obj(gamma=2.0, alpha=0.7):
    def obj(y_pred, dtrain):
        y = dtrain.get_label()
        p = 1.0 / (1.0 + np.exp(-y_pred))
        # focal gradient/hessian (binary)
        pt = np.where(y == 1, p, 1 - p)
        a = np.where(y == 1, alpha, 1 - alpha)
        g = a * (1 - pt) ** gamma * (gamma * pt * np.log(np.clip(pt, 1e-6, 1)) + pt - 1) * np.where(y == 1, 1, -1)
        h = np.abs(a * (1 - pt) ** gamma) * pt * (1 - pt) * (1 + gamma)  # 근사
        return g, np.clip(h, 1e-6, None)
    return obj


def _run_xgb_rank(X, y, Xtest, folds):
    oof = np.full(len(y), np.nan); ts = np.zeros(len(Xtest))
    Xv = np.nan_to_num(X.values.astype(np.float32), posinf=0, neginf=0)
    Xtv = np.nan_to_num(Xtest.values.astype(np.float32), posinf=0, neginf=0)
    dtest = xgb.DMatrix(Xtv)
    for f in range(C.N_FOLDS):
        tr, va = np.where(folds != f)[0], np.where(folds == f)[0]
        dtr = xgb.DMatrix(Xv[tr], label=y[tr]); dtr.set_group([len(tr)])  # 전체 1그룹 = AUC근사
        dva = xgb.DMatrix(Xv[va], label=y[va])
        p = dict(objective="rank:pairwise", eval_metric="auc", tree_method="hist",
                 eta=0.03, max_depth=7, subsample=0.8, colsample_bytree=0.6, min_child_weight=5)
        bst = xgb.train(p, dtr, num_boost_round=1200)
        oof[va] = bst.predict(dva); ts += bst.predict(dtest)
        print(f"  [rank fold {f}] AUC={roc_auc_score(y[va], oof[va]):.5f}")
    return oof, ts / C.N_FOLDS


def _run_lgb_focal(X, y, Xtest, folds):
    oof = np.full(len(y), np.nan); ts = np.zeros(len(Xtest))
    Xv = np.nan_to_num(X.values.astype(np.float32), posinf=0, neginf=0)
    Xtv = np.nan_to_num(Xtest.values.astype(np.float32), posinf=0, neginf=0)
    for f in range(C.N_FOLDS):
        tr, va = np.where(folds != f)[0], np.where(folds == f)[0]
        dtr = lgb.Dataset(Xv[tr], label=y[tr])
        m = lgb.train(dict(learning_rate=0.03, num_leaves=63, feature_fraction=0.6,
                           bagging_fraction=0.8, bagging_freq=1, verbosity=-1,
                           objective=_focal_obj()),
                      dtr, num_boost_round=1500)
        raw = m.predict(Xv[va]); oof[va] = 1 / (1 + np.exp(-raw))
        ts += 1 / (1 + np.exp(-m.predict(Xtv)))
        print(f"  [focal fold {f}] AUC={roc_auc_score(y[va], oof[va]):.5f}")
    return oof, ts / C.N_FOLDS


def main():
    X, y, Xtest = build_features()
    folds = np.load(C.FOLDS_NPY)
    print(f"X={X.shape}")

    # xgb_rank는 0.58로 실패(전체1그룹 pairwise가 글로벌AUC에 안 맞음) → 스킵
    print("=== LGBM focal loss ===")
    oof, ts = _run_lgb_focal(X, y, Xtest, folds)
    cv = float(roc_auc_score(y, oof)); print(f"==== lgbm_focal CV = {cv:.5f} ====")
    save_predictions("lgbm_focal", oof, ts, meta=dict(cv_auc=cv, seed=C.SEED, n_folds=C.N_FOLDS,
                     feature_set="full", created_by="hyunbean", notes="LGBM focal loss (gamma2)"))


if __name__ == "__main__":
    main()
