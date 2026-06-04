"""LGBM Optuna 튜닝 (CPU) — 정규화/하이퍼파라미터 자동탐색 → best로 5-fold 학습 → OOF 저장.

목적: 과적합 방어(정규화 lambda_l1/l2·min_child·feature_fraction 탐색) + CV 향상.
- Optuna study는 sqlite(artifacts/optuna_lgbm.db)에 저장 → 중단해도 재실행 시 이어서 탐색.
- tmux 야간 실행 추천.

실행: python -m src.train_lgbm_optuna [n_trials]   (기본 50)
"""
import sys
import numpy as np
import optuna
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

from . import config as C
from .features import build_features
from .oof_io import save_predictions

MODEL_NAME = "lgbm_optuna"
N_TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 50


def _oof_cv(params, X, y, folds, Xtest=None, es=150):
    oof = np.full(len(y), np.nan)
    test_sum = np.zeros(len(Xtest)) if Xtest is not None else None
    for f in range(C.N_FOLDS):
        tr, va = np.where(folds != f)[0], np.where(folds == f)[0]
        m = lgb.LGBMClassifier(**params)
        m.fit(X.iloc[tr], y[tr], eval_set=[(X.iloc[va], y[va])], eval_metric="auc",
              callbacks=[lgb.early_stopping(es, verbose=False)])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
        if test_sum is not None:
            test_sum += m.predict_proba(Xtest)[:, 1]
    return oof, (test_sum / C.N_FOLDS if test_sum is not None else None)


def main():
    X, y, Xtest = build_features()
    folds = np.load(C.FOLDS_NPY)
    print(f"X_train={X.shape}  features={X.shape[1]}  trials={N_TRIALS}")

    def objective(trial):
        params = dict(
            objective="binary", metric="auc", n_jobs=-1, verbosity=-1, bagging_freq=1,
            n_estimators=4000,
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            num_leaves=trial.suggest_int("num_leaves", 15, 255),
            max_depth=trial.suggest_int("max_depth", 3, 12),
            min_child_samples=trial.suggest_int("min_child_samples", 20, 200),
            feature_fraction=trial.suggest_float("feature_fraction", 0.4, 1.0),
            bagging_fraction=trial.suggest_float("bagging_fraction", 0.5, 1.0),
            lambda_l1=trial.suggest_float("lambda_l1", 1e-3, 10.0, log=True),
            lambda_l2=trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
        )
        oof, _ = _oof_cv(params, X, y, folds)
        return roc_auc_score(y, oof)

    storage = f"sqlite:///{C.ARTIFACTS / 'optuna_lgbm.db'}"
    study = optuna.create_study(direction="maximize", study_name="lgbm",
                                storage=storage, load_if_exists=True)
    study.optimize(objective, n_trials=N_TRIALS)
    print(f"\nbest CV AUC = {study.best_value:.5f}\nbest params = {study.best_params}")

    # 최종: best params로 OOF/test 생성 (n_estimators 늘리고 early stopping)
    best = dict(objective="binary", metric="auc", n_jobs=-1, verbosity=-1,
                bagging_freq=1, n_estimators=8000, **study.best_params)
    oof, test_pred = _oof_cv(best, X, y, folds, Xtest=Xtest, es=200)
    cv = float(roc_auc_score(y, oof))
    print(f"\n==== {MODEL_NAME}  final CV AUC = {cv:.5f} ====")

    save_predictions(MODEL_NAME, oof, test_pred, meta=dict(
        cv_auc=cv, seed=C.SEED, n_folds=C.N_FOLDS, params=best,
        n_trials=N_TRIALS, feature_set="full(te+emb+div+grp)",
        created_by="hyunbean", notes=f"LGBM Optuna {N_TRIALS} trials"))


if __name__ == "__main__":
    main()
