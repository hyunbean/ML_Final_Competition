"""XGBoost 트레이너 (GPU) — train_lgbm/catboost와 동일 패턴: 체크포인트 + OOF·test npy.

실행:
  python -m src.folds          # (선행)
  python -m src.train_xgb      # GPU 학습
"""
import numpy as np
from sklearn.metrics import roc_auc_score
import xgboost as xgb

from . import config as C
from .data import build_xy
from .features import build_features
from .checkpoint import Checkpoint
from .oof_io import save_predictions

USE_FULL_FEATURES = True
DEVICE = "cuda"          # GPU 없으면 "cpu"
MODEL_NAME = "xgb_full" if USE_FULL_FEATURES else "xgb_baseline"
CREATED_BY = "hyunbean"

PARAMS = dict(
    objective="binary:logistic", eval_metric="auc",
    tree_method="hist", device=DEVICE,
    n_estimators=6000, learning_rate=0.03, max_depth=6,
    subsample=0.8, colsample_bytree=0.8,
    reg_lambda=1.0, reg_alpha=0.0, min_child_weight=5, n_jobs=-1,
)


def main():
    X, y, Xtest = build_features() if USE_FULL_FEATURES else build_xy()
    folds = np.load(C.FOLDS_NPY)
    n_tr, n_te = len(X), len(Xtest)
    print(f"X_train={X.shape}  X_test={Xtest.shape}  device={DEVICE}")

    ckpt = Checkpoint(MODEL_NAME, C.CKPT_DIR)
    state = ckpt.load(default=dict(
        done=[], oof=np.full(n_tr, np.nan, dtype=np.float64),
        test_sum=np.zeros(n_te, dtype=np.float64), fold_auc={}))

    for f in range(C.N_FOLDS):
        if f in state["done"]:
            continue
        tr, va = np.where(folds != f)[0], np.where(folds == f)[0]
        model = xgb.XGBClassifier(**PARAMS, early_stopping_rounds=200)
        model.fit(X.iloc[tr], y[tr], eval_set=[(X.iloc[va], y[va])], verbose=False)
        state["oof"][va] = model.predict_proba(X.iloc[va])[:, 1]
        state["test_sum"] += model.predict_proba(Xtest)[:, 1]
        auc = roc_auc_score(y[va], state["oof"][va])
        state["fold_auc"][str(f)] = float(auc)
        state["done"].append(f)
        ckpt.save(state)
        print(f"[fold {f}] AUC={auc:.5f}  best_iter={model.best_iteration}  done={state['done']}")

    oof = state["oof"]
    test_pred = state["test_sum"] / C.N_FOLDS
    cv = float(roc_auc_score(y, oof))
    print(f"\n==== {MODEL_NAME}  CV AUC = {cv:.5f} ====")

    save_predictions(MODEL_NAME, oof, test_pred, meta=dict(
        cv_auc=cv, fold_auc=state["fold_auc"], seed=C.SEED, n_folds=C.N_FOLDS,
        params=dict(PARAMS),
        feature_set="full(te+emb+div+grp)" if USE_FULL_FEATURES else "baseline_agg",
        created_by=CREATED_BY, notes=f"XGBoost ({DEVICE})"))
    ckpt.cleanup()


if __name__ == "__main__":
    main()
