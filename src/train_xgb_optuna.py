"""XGBoost Optuna 튜닝 (GPU) — sqlite 재개. 결과: xgb_optuna OOF.
실행: python -m src.train_xgb_optuna [n_trials]   (기본 40)
GPU 없으면 DEVICE="cpu"."""
import sys
import numpy as np
import optuna
import xgboost as xgb
from sklearn.metrics import roc_auc_score

from . import config as C
from .features import build_features
from .oof_io import save_predictions

def _has_gpu():
    import os
    import shutil
    import subprocess
    if os.path.exists("/dev/nvidia0"):
        return True
    if shutil.which("nvidia-smi"):
        try:
            return subprocess.run(["nvidia-smi"], capture_output=True).returncode == 0
        except Exception:
            pass
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


MODEL_NAME = "xgb_optuna"
DEVICE = "cuda" if _has_gpu() else "cpu"
N_TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 40


def _oof(params, X, y, folds, Xtest=None, es=200):
    oof = np.full(len(y), np.nan)
    ts = np.zeros(len(Xtest)) if Xtest is not None else None
    for f in range(C.N_FOLDS):
        tr, va = np.where(folds != f)[0], np.where(folds == f)[0]
        m = xgb.XGBClassifier(**params, early_stopping_rounds=es)
        m.fit(X.iloc[tr], y[tr], eval_set=[(X.iloc[va], y[va])], verbose=False)
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
        if ts is not None:
            ts += m.predict_proba(Xtest)[:, 1]
    return oof, (ts / C.N_FOLDS if ts is not None else None)


def main():
    X, y, Xtest = build_features()
    folds = np.load(C.FOLDS_NPY)

    def obj(t):
        p = dict(objective="binary:logistic", eval_metric="auc", tree_method="hist",
                 device=DEVICE, n_jobs=-1, n_estimators=4000,
                 learning_rate=t.suggest_float("learning_rate", 0.01, 0.1, log=True),
                 max_depth=t.suggest_int("max_depth", 3, 10),
                 min_child_weight=t.suggest_int("min_child_weight", 1, 20),
                 subsample=t.suggest_float("subsample", 0.5, 1.0),
                 colsample_bytree=t.suggest_float("colsample_bytree", 0.4, 1.0),
                 reg_lambda=t.suggest_float("reg_lambda", 1e-2, 30.0, log=True),
                 reg_alpha=t.suggest_float("reg_alpha", 1e-3, 10.0, log=True))
        oof, _ = _oof(p, X, y, folds)
        return roc_auc_score(y, oof)

    st = optuna.create_study(direction="maximize", study_name="xgb",
                             storage=f"sqlite:///{C.ARTIFACTS / 'optuna_xgb.db'}", load_if_exists=True)
    st.optimize(obj, n_trials=N_TRIALS)
    print(f"\nbest CV={st.best_value:.5f}\nbest={st.best_params}")

    best = dict(objective="binary:logistic", eval_metric="auc", tree_method="hist",
                device=DEVICE, n_jobs=-1, n_estimators=8000, **st.best_params)
    oof, test = _oof(best, X, y, folds, Xtest)
    cv = float(roc_auc_score(y, oof))
    print(f"\n==== {MODEL_NAME}  final CV AUC = {cv:.5f} ====")
    save_predictions(MODEL_NAME, oof, test, meta=dict(
        cv_auc=cv, seed=C.SEED, n_folds=C.N_FOLDS, params=best, n_trials=N_TRIALS,
        feature_set="full(te+emb+div+grp)", created_by="hyunbean", notes=f"XGBoost Optuna {N_TRIALS}"))


if __name__ == "__main__":
    main()
