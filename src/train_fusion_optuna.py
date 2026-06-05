"""밤샘용: fused(our+mega) 피처에 LGBM Deep Optuna 튜닝 → fusion_opt.

fusion_lgbm(기본파라미터 0.712)을 깊게 튜닝하면 더 오르나? sqlite 재개(밤샘 안전).
폴드 동일(TE 안전). 실행: python -m src.train_fusion_optuna [n_trials=150]
"""
import sys
import numpy as np
import optuna
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

from . import config as C
from .features import build_features
from .train_fusion import _merge_mega, MEGA_DIR
from .oof_io import save_predictions

MODEL_NAME = "fusion_opt"
N_TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 150


def main():
    X, y, Xtest = build_features()
    train_ids = np.load(C.TRAIN_IDS_NPY, allow_pickle=True)
    test_ids = np.load(C.TEST_IDS_NPY, allow_pickle=True)
    folds = np.load(C.FOLDS_NPY)
    Xf = _merge_mega(X, train_ids, f"{MEGA_DIR}/mega_train.parquet").reset_index(drop=True)
    Xtf = _merge_mega(Xtest, test_ids, f"{MEGA_DIR}/mega_test.parquet").reset_index(drop=True)
    Xtf = Xtf.reindex(columns=Xf.columns, fill_value=0.0)
    Xv = np.nan_to_num(Xf.values.astype(np.float32), posinf=0, neginf=0)
    Xtv = np.nan_to_num(Xtf.values.astype(np.float32), posinf=0, neginf=0)
    print(f"fused {Xv.shape} Optuna {N_TRIALS} trials")

    def oof_auc(params, full_test=None):
        oof = np.full(len(y), np.nan)
        ts = np.zeros(len(Xtv)) if full_test is not None else None
        for f in range(C.N_FOLDS):
            tr, va = np.where(folds != f)[0], np.where(folds == f)[0]
            m = lgb.LGBMClassifier(**params)
            m.fit(Xv[tr], y[tr], eval_set=[(Xv[va], y[va])], callbacks=[lgb.early_stopping(120, verbose=False)])
            oof[va] = m.predict_proba(Xv[va])[:, 1]
            if ts is not None:
                ts += m.predict_proba(Xtv)[:, 1]
        return roc_auc_score(y, oof), oof, (ts / C.N_FOLDS if ts is not None else None)

    def obj(t):
        p = dict(objective="binary", metric="auc", n_jobs=4, verbosity=-1, n_estimators=3000,
                 learning_rate=t.suggest_float("learning_rate", 0.01, 0.05, log=True),
                 num_leaves=t.suggest_int("num_leaves", 32, 200),
                 min_child_samples=t.suggest_int("min_child_samples", 40, 300),
                 feature_fraction=t.suggest_float("feature_fraction", 0.2, 0.7),
                 bagging_fraction=t.suggest_float("bagging_fraction", 0.6, 1.0), bagging_freq=1,
                 lambda_l1=t.suggest_float("lambda_l1", 1e-3, 5.0, log=True),
                 lambda_l2=t.suggest_float("lambda_l2", 1e-2, 10.0, log=True))
        return oof_auc(p)[0]

    st = optuna.create_study(direction="maximize", study_name="fusion",
                             storage=f"sqlite:///{C.ARTIFACTS / 'optuna_fusion.db'}", load_if_exists=True)
    st.optimize(obj, n_trials=N_TRIALS)
    print(f"\nbest CV={st.best_value:.5f}\nbest={st.best_params}")

    best = dict(objective="binary", metric="auc", n_jobs=4, verbosity=-1, n_estimators=5000,
                bagging_freq=1, **st.best_params)
    cv, oof, test = oof_auc(best, full_test=True)
    print(f"\n==== {MODEL_NAME}  final CV AUC = {cv:.5f} ====")
    save_predictions(MODEL_NAME, oof, test, meta=dict(
        cv_auc=cv, seed=C.SEED, n_folds=C.N_FOLDS, params=best, n_trials=N_TRIALS,
        feature_set="fused our+mega Optuna", created_by="hyunbean", notes=f"fused Deep Optuna {N_TRIALS}"))


if __name__ == "__main__":
    main()
