"""1등 피처셋(497) 핵심 모델 Optuna 튜닝 (overnight) — early stopping 적용 버전.

first_cat/lgbm/xgb이 블렌드 高weight라 강화하면 블렌드 직접↑. 모델별 OPT_TIME초 탐색
→ best params로 5fold OOF(early stopping) 재생성. 저장: first_{lgbm,xgb,cat}_opt2.
실행(GPU): pip install optuna lightgbm xgboost catboost gensim
   OPT_TIME=7200 XGB_GPU=1 CB_TASK_TYPE=GPU python -m src.train_first_optuna [lgbm|xgb|cat|all]
"""
import os
import sys
import numpy as np
import optuna
from sklearn.metrics import roc_auc_score

from . import config as C
from .oof_io import save_predictions
from .train_first import build_all

OPT_TIME = int(os.environ.get("OPT_TIME", "7200"))
GPU_XGB = os.environ.get("XGB_GPU", "0") == "1"
CB_TASK = os.environ.get("CB_TASK_TYPE", "CPU")
optuna.logging.set_verbosity(optuna.logging.WARNING)

_train_ids = _test_ids = _folds = _X = _Xt = _y = None


def _oof_es(kind, params, want_test=False):
    """5fold OOF with early stopping (과적합 방지). want_test=False면 oof만(탐색용 빠르게)."""
    import lightgbm as lgb
    import xgboost as xgb
    from catboost import CatBoostClassifier
    oof = np.full(len(_y), np.nan); test = np.zeros(len(_test_ids))
    for f in range(C.N_FOLDS):
        tri, va = np.where(_folds != f)[0], np.where(_folds == f)[0]
        Xtr, ytr, Xva, yva = _X.iloc[tri], _y[tri], _X.iloc[va], _y[va]
        if kind == "lgbm":
            m = lgb.LGBMClassifier(n_estimators=6000, **params)
            m.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[lgb.early_stopping(150, verbose=False)])
        elif kind == "xgb":
            m = xgb.XGBClassifier(n_estimators=6000, early_stopping_rounds=150, **params)
            m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        else:
            m = CatBoostClassifier(iterations=6000, **params)
            m.fit(Xtr, ytr, eval_set=(Xva, yva), early_stopping_rounds=150)
        oof[va] = m.predict_proba(Xva)[:, 1]
        if want_test:
            test += m.predict_proba(_Xt)[:, 1]
    return oof, (test / C.N_FOLDS if want_test else None)


def _space(kind, t):
    if kind == "lgbm":
        return dict(objective="binary", metric="auc", n_jobs=-1, verbose=-1, random_state=C.SEED, subsample_freq=1,
                    learning_rate=t.suggest_float("lr", 0.01, 0.05, log=True), num_leaves=t.suggest_int("nl", 31, 160),
                    min_child_samples=t.suggest_int("mcs", 20, 120), subsample=t.suggest_float("ss", 0.6, 1.0),
                    colsample_bytree=t.suggest_float("cs", 0.4, 0.9), reg_alpha=t.suggest_float("ra", 1e-3, 10, log=True),
                    reg_lambda=t.suggest_float("rl", 1e-3, 30, log=True))
    if kind == "xgb":
        p = dict(objective="binary:logistic", eval_metric="auc", random_state=C.SEED,
                 learning_rate=t.suggest_float("lr", 0.01, 0.05, log=True), max_depth=t.suggest_int("md", 5, 10),
                 min_child_weight=t.suggest_int("mcw", 1, 20), gamma=t.suggest_float("g", 1e-3, 5, log=True),
                 subsample=t.suggest_float("ss", 0.6, 1.0), colsample_bytree=t.suggest_float("cs", 0.4, 0.9),
                 reg_alpha=t.suggest_float("ra", 1e-3, 10, log=True), reg_lambda=t.suggest_float("rl", 1e-3, 30, log=True))
        if GPU_XGB:
            p.update(tree_method="hist", device="cuda")
        return p
    return dict(loss_function="Logloss", eval_metric="AUC", random_seed=C.SEED, verbose=0, allow_writing_files=False,
                task_type=CB_TASK, learning_rate=t.suggest_float("lr", 0.01, 0.05, log=True),
                depth=t.suggest_int("d", 5, 9), l2_leaf_reg=t.suggest_float("l2", 1, 30, log=True),
                random_strength=t.suggest_float("rs", 0.5, 5))


def main():
    global _train_ids, _test_ids, _folds, _X, _Xt, _y
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    _train_ids = np.load(C.TRAIN_IDS_NPY, allow_pickle=True)
    _test_ids = np.load(C.TEST_IDS_NPY, allow_pickle=True)
    _folds = np.load(C.FOLDS_NPY)
    allf, ydf, _, _ = build_all()
    _X = allf.reindex(_train_ids).fillna(0.0).reset_index(drop=True)
    _Xt = allf.reindex(_test_ids).fillna(0.0).reset_index(drop=True)
    _y = ydf.set_index("custid").reindex(_train_ids)["gender"].to_numpy()
    print(f"X={_X.shape} OPT_TIME={OPT_TIME}s/model which={which}")

    todo = ["lgbm", "xgb", "cat"] if which == "all" else [which]
    for kind in todo:
        print(f"\n--- Optuna {kind} (early stopping) ---")
        st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
        st.optimize(lambda t: roc_auc_score(_y, _oof_es(kind, _space(kind, t))[0]), timeout=OPT_TIME)
        print(f"  best CV={st.best_value:.5f}, params={st.best_params}")

        class _FT:  # best_params를 _space에 다시 먹이기 위한 가짜 trial
            def __init__(s, p): s.p = p
            def suggest_float(s, k, *a, **kw): return s.p[k]
            def suggest_int(s, k, *a, **kw): return s.p[k]
        params = _space(kind, _FT(st.best_params))
        oof, test = _oof_es(kind, params, want_test=True)
        cv = float(roc_auc_score(_y, oof))
        name = f"first_{kind}_opt2"
        print(f"==== {name}  CV AUC = {cv:.5f} ====")
        save_predictions(name, oof, test, meta=dict(cv_auc=cv, seed=C.SEED, n_folds=C.N_FOLDS,
                         feature_set="작년1등 FE(497) Optuna+earlystop", created_by="hyunbean",
                         notes=f"1st-place FE + Optuna {kind} (early stopping, overnight)"))


if __name__ == "__main__":
    main()
