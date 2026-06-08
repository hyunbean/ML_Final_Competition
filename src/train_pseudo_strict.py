"""엄격한 Pseudo-labeling (first_{xgb,lgbm,cat}_pl2) — test-side 신호 (OOF가 못보는 차원).

pl2(xgb) w20이 LB 0.73468->0.73502 돌파 확인! pseudo test-side 진짜. → lgbm/cat도 만들어 앙상블.
teacher = student계열 격리(김민형 mega 독립피처 + 타모델). val fold 순수(OOF정직).
실행(GPU): PL_HI=0.85 PL_LO=0.15 python -m src.train_pseudo_strict [xgb|lgbm|cat|all]
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from . import config as C
from .oof_io import save_predictions
from .train_first import build_all

HI = float(os.environ.get("PL_HI", "0.90"))
LO = float(os.environ.get("PL_LO", "0.10"))
GPU = os.environ.get("XGB_GPU", "1") == "1"
ITER = os.environ.get("PL_ITER", "0") == "1"   # 반복pseudo: teacher에 1라운드 pl2 추가(라벨품질↑) → pl3
SUF = "_pl3" if ITER else "_pl2"
# teacher: 각 student와 격리되게 구성 (mega 독립피처 위주 + 타계열). ITER이면 pl2도 teacher에.
_EXTRA = ["first_xgb_pl2", "first_lgbm_pl2"] if ITER else []
TEACH = {
    "xgb": ["mh_bestblend69", "mh_05_AutoGluon_megamax", "mh_07_AutoGluon_mega572", "mh_09_XGBoost_mega", "first_lgbm", "first_cat"] + _EXTRA,
    "lgbm": ["mh_bestblend69", "mh_05_AutoGluon_megamax", "mh_07_AutoGluon_mega572", "mh_11_CatBoost_mega", "first_xgb", "first_cat"] + _EXTRA,
    "cat": ["mh_bestblend69", "mh_05_AutoGluon_megamax", "mh_07_AutoGluon_mega572", "mh_09_XGBoost_mega", "first_xgb", "first_lgbm"] + _EXTRA,
}


def _teacher(kind, test_ids):
    avail = [m for m in TEACH[kind] if os.path.exists(f"artifacts/oof/{m}__test.npy")]
    return np.mean([rankdata(np.load(f"artifacts/oof/{m}__test.npy")) / len(test_ids) for m in avail], axis=0), avail


def _fit(kind, Xtr, ytr, Xva, yva, Xt):
    import xgboost as xgb, lightgbm as lgb
    from catboost import CatBoostClassifier
    if kind == "xgb":
        m = xgb.XGBClassifier(n_estimators=4000, early_stopping_rounds=150, objective="binary:logistic",
                              eval_metric="auc", learning_rate=0.02, max_depth=7, min_child_weight=5, gamma=0.1,
                              subsample=0.8, colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=5.0,
                              random_state=C.SEED, tree_method="hist", device="cuda" if GPU else "cpu")
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    elif kind == "lgbm":
        m = lgb.LGBMClassifier(n_estimators=4000, objective="binary", metric="auc", learning_rate=0.02,
                               num_leaves=64, min_child_samples=40, subsample=0.8, subsample_freq=1,
                               colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=5.0, n_jobs=-1,
                               random_state=C.SEED, verbose=-1)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[lgb.early_stopping(150, verbose=False)])
    else:
        m = CatBoostClassifier(iterations=2000, learning_rate=0.03, depth=7, l2_leaf_reg=5, eval_metric="AUC",
                               random_seed=C.SEED, verbose=0, allow_writing_files=False,
                               task_type="GPU" if GPU else "CPU")
        m.fit(Xtr, ytr, eval_set=(Xva, yva), early_stopping_rounds=150)
    return m.predict_proba(Xva)[:, 1], m.predict_proba(Xt)[:, 1]


def run(kind, X, Xt, y, folds, test_ids):
    sub, avail = _teacher(kind, test_ids)
    conf = (sub >= HI) | (sub <= LO)
    pl_y = (sub[conf] >= 0.5).astype(int)
    Xp = Xt[conf.tolist()]
    print(f"[{kind}] teacher={avail} | pseudo {conf.sum()}/{len(sub)} (pos {pl_y.mean():.2f}) HI/LO={HI}/{LO}")
    oof = np.full(len(y), np.nan); test_sum = np.zeros(len(test_ids))
    for f in range(C.N_FOLDS):
        tri, va = np.where(folds != f)[0], np.where(folds == f)[0]
        Xtr = pd.concat([X.iloc[tri], Xp], axis=0); ytr = np.concatenate([y[tri], pl_y])
        vp, tp = _fit(kind, Xtr, ytr, X.iloc[va], y[va], Xt)
        oof[va] = vp; test_sum += tp
        print(f"  [fold {f}] AUC={roc_auc_score(y[va], vp):.5f}")
    cv = float(roc_auc_score(y, oof)); name = f"first_{kind}{SUF}"
    print(f"==== {name}  CV={cv:.5f} ====")
    save_predictions(name, oof, test_sum / C.N_FOLDS, meta=dict(cv_auc=cv, seed=C.SEED, n_folds=C.N_FOLDS,
                     feature_set="1등FE + strict pseudo", created_by="hyunbean",
                     notes=f"{kind} strict pseudo(격리teacher HI/LO={HI}/{LO}), test-side 신호"))


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    train_ids = np.load(C.TRAIN_IDS_NPY, allow_pickle=True)
    test_ids = np.load(C.TEST_IDS_NPY, allow_pickle=True)
    folds = np.load(C.FOLDS_NPY)
    allf, ydf, _, _ = build_all()
    X = allf.reindex(train_ids).fillna(0.0); Xt = allf.reindex(test_ids).fillna(0.0)
    y = ydf.set_index("custid").reindex(train_ids)["gender"].to_numpy()
    for kind in (["xgb", "lgbm", "cat"] if which == "all" else [which]):
        run(kind, X, Xt, y, folds, test_ids)


if __name__ == "__main__":
    main()
