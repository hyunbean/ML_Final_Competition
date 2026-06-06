"""1등 피처셋(497) 핵심 모델 Optuna 튜닝 (overnight). first_cat/lgbm/xgb이 블렌드 高weight라
이들을 강화하면 블렌드가 직접 상승. 모델별 OPT_TIME초 탐색 → best로 5fold OOF 재생성.

저장: first_lgbm_opt / first_xgb_opt / first_cat_opt (기존 default판과 별개 멤버)
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

OPT_TIME = int(os.environ.get("OPT_TIME", "7200"))   # 모델당 초
GPU_XGB = os.environ.get("XGB_GPU", "0") == "1"
CB_TASK = os.environ.get("CB_TASK_TYPE", "CPU")
optuna.logging.set_verbosity(optuna.logging.WARNING)

_train_ids = _test_ids = _folds = _X = _Xt = _y = None


def _oof_cv(make_model, fit_kw=None):
    fit_kw = fit_kw or {}
    oof = np.full(len(_y), np.nan); test = np.zeros(len(_test_ids))
    for f in range(C.N_FOLDS):
        tri, va = np.where(_folds != f)[0], np.where(_folds == f)[0]
        m = make_model()
        m.fit(_X.iloc[tri], _y[tri], **({k: v(va) for k, v in fit_kw.items()} if fit_kw else {}))
        oof[va] = m.predict_proba(_X.iloc[va])[:, 1]; test += m.predict_proba(_Xt)[:, 1]
    return oof, test / C.N_FOLDS


def _run_lgbm():
    import lightgbm as lgb

    def obj(t):
        p = dict(objective="binary", metric="auc", n_jobs=-1, verbose=-1, random_state=C.SEED,
                 learning_rate=t.suggest_float("lr", 0.01, 0.05, log=True),
                 num_leaves=t.suggest_int("nl", 31, 160),
                 min_child_samples=t.suggest_int("mcs", 20, 120),
                 subsample=t.suggest_float("ss", 0.6, 1.0), subsample_freq=1,
                 colsample_bytree=t.suggest_float("cs", 0.4, 0.9),
                 reg_alpha=t.suggest_float("ra", 1e-3, 10, log=True),
                 reg_lambda=t.suggest_float("rl", 1e-3, 30, log=True))
        oof, _ = _oof_cv(lambda: lgb.LGBMClassifier(n_estimators=1500, **p))
        return roc_auc_score(_y, oof)
    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    st.optimize(obj, timeout=OPT_TIME)
    bp = dict(objective="binary", metric="auc", n_jobs=-1, verbose=-1, random_state=C.SEED,
              subsample_freq=1, n_estimators=3000)
    bp.update({"learning_rate": st.best_params["lr"], "num_leaves": st.best_params["nl"],
               "min_child_samples": st.best_params["mcs"], "subsample": st.best_params["ss"],
               "colsample_bytree": st.best_params["cs"], "reg_alpha": st.best_params["ra"],
               "reg_lambda": st.best_params["rl"]})
    return lambda: lgb.LGBMClassifier(**bp), st.best_value


def _run_xgb():
    import xgboost as xgb

    def obj(t):
        p = dict(objective="binary:logistic", eval_metric="auc", random_state=C.SEED, n_estimators=1500,
                 learning_rate=t.suggest_float("lr", 0.01, 0.05, log=True),
                 max_depth=t.suggest_int("md", 5, 10), min_child_weight=t.suggest_int("mcw", 1, 20),
                 gamma=t.suggest_float("g", 1e-3, 5, log=True), subsample=t.suggest_float("ss", 0.6, 1.0),
                 colsample_bytree=t.suggest_float("cs", 0.4, 0.9),
                 reg_alpha=t.suggest_float("ra", 1e-3, 10, log=True),
                 reg_lambda=t.suggest_float("rl", 1e-3, 30, log=True))
        if GPU_XGB:
            p.update(tree_method="hist", device="cuda")
        oof, _ = _oof_cv(lambda: xgb.XGBClassifier(**p))
        return roc_auc_score(_y, oof)
    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    st.optimize(obj, timeout=OPT_TIME)
    bp = dict(objective="binary:logistic", eval_metric="auc", random_state=C.SEED, n_estimators=3000, **{
        "learning_rate": st.best_params["lr"], "max_depth": st.best_params["md"],
        "min_child_weight": st.best_params["mcw"], "gamma": st.best_params["g"],
        "subsample": st.best_params["ss"], "colsample_bytree": st.best_params["cs"],
        "reg_alpha": st.best_params["ra"], "reg_lambda": st.best_params["rl"]})
    if GPU_XGB:
        bp.update(tree_method="hist", device="cuda")
    return lambda: xgb.XGBClassifier(**bp), st.best_value


def _run_cat():
    from catboost import CatBoostClassifier

    def obj(t):
        p = dict(loss_function="Logloss", eval_metric="AUC", random_seed=C.SEED, verbose=0,
                 allow_writing_files=False, iterations=1500, task_type=CB_TASK,
                 learning_rate=t.suggest_float("lr", 0.01, 0.05, log=True),
                 depth=t.suggest_int("d", 5, 9), l2_leaf_reg=t.suggest_float("l2", 1, 30, log=True),
                 random_strength=t.suggest_float("rs", 0.5, 5))
        oof, _ = _oof_cv(lambda: CatBoostClassifier(**p))
        return roc_auc_score(_y, oof)
    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    st.optimize(obj, timeout=OPT_TIME)
    bp = dict(loss_function="Logloss", eval_metric="AUC", random_seed=C.SEED, verbose=0,
              allow_writing_files=False, iterations=3000, task_type=CB_TASK, **{
        "learning_rate": st.best_params["lr"], "depth": st.best_params["d"],
        "l2_leaf_reg": st.best_params["l2"], "random_strength": st.best_params["rs"]})
    return lambda: CatBoostClassifier(**bp), st.best_value


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

    runners = {"lgbm": ("first_lgbm_opt", _run_lgbm), "xgb": ("first_xgb_opt", _run_xgb),
               "cat": ("first_cat_opt", _run_cat)}
    todo = list(runners) if which == "all" else [which]
    for key in todo:
        name, runner = runners[key]
        print(f"\n--- Optuna {key} ---")
        make, best = runner()
        print(f"  best CV={best:.5f} → 재학습(n_est 2x)")
        oof, test = _oof_cv(make)
        cv = float(roc_auc_score(_y, oof))
        print(f"==== {name}  CV AUC = {cv:.5f} ====")
        save_predictions(name, oof, test, meta=dict(cv_auc=cv, seed=C.SEED, n_folds=C.N_FOLDS,
                         feature_set="작년1등 FE(497) Optuna-tuned", created_by="hyunbean",
                         notes=f"1st-place FE + Optuna {key} (overnight)"))


if __name__ == "__main__":
    main()
