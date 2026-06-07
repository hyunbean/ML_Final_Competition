"""Feature-bagging GBDT ('first_lgbm_bag','first_cat_bag') — 1등피처 랜덤 부분집합으로 여러 GBDT 평균.
기존 GBDT는 corr 0.95(다양성X). 모델마다 다른 50~65% 피처만 보게 해 decorrelate → NN클러스터처럼 블렌드 기여.
실행(GPU): pip install lightgbm catboost gensim → python -m src.train_first_bag [lgbm|cat]
"""
import os, sys
import numpy as np
from sklearn.metrics import roc_auc_score
from . import config as C
from .oof_io import save_predictions
from .train_first import build_all

KIND = sys.argv[1] if len(sys.argv) > 1 else "lgbm"
N_BAG = 5
FRAC = 0.6
CB_TASK = os.environ.get("CB_TASK_TYPE", "CPU")


def main():
    import lightgbm as lgb
    from catboost import CatBoostClassifier
    train_ids = np.load(C.TRAIN_IDS_NPY, allow_pickle=True)
    test_ids = np.load(C.TEST_IDS_NPY, allow_pickle=True)
    folds = np.load(C.FOLDS_NPY)
    allf, ydf, _, _ = build_all()
    X = allf.reindex(train_ids).fillna(0.0).reset_index(drop=True)
    Xt = allf.reindex(test_ids).fillna(0.0).reset_index(drop=True)
    y = ydf.set_index("custid").reindex(train_ids)["gender"].to_numpy()
    ncol = X.shape[1]
    print(f"first_{KIND}_bag X={X.shape}  bag={N_BAG} frac={FRAC}")
    oof = np.full(len(y), np.nan); ts = np.zeros(len(test_ids))
    for f in range(C.N_FOLDS):
        tri, va = np.where(folds != f)[0], np.where(folds == f)[0]
        vp = np.zeros(len(va)); tp = np.zeros(len(test_ids))
        for b in range(N_BAG):
            rng = np.random.RandomState(100 + b)
            cols = rng.choice(ncol, int(ncol * FRAC), replace=False)
            Xs, Xvs, Xts = X.iloc[tri, cols], X.iloc[va, cols], Xt.iloc[:, cols]
            if KIND == "lgbm":
                m = lgb.LGBMClassifier(n_estimators=3000, learning_rate=0.02, num_leaves=63,
                                       subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                       reg_lambda=5.0, n_jobs=-1, random_state=b, verbose=-1)
                m.fit(Xs, y[tri], eval_set=[(Xvs, y[va])], callbacks=[lgb.early_stopping(120, verbose=False)])
            else:
                p = dict(iterations=3000, learning_rate=0.02, depth=7, l2_leaf_reg=5.0, eval_metric="AUC",
                         random_seed=b, verbose=0, allow_writing_files=False, task_type=CB_TASK)
                if CB_TASK == "GPU": p["devices"] = "0"
                m = CatBoostClassifier(**p)
                m.fit(Xs, y[tri], eval_set=(Xvs, y[va]), early_stopping_rounds=120)
            vp += m.predict_proba(Xvs)[:, 1] / N_BAG; tp += m.predict_proba(Xts)[:, 1] / N_BAG
        oof[va] = vp; ts += tp
        print(f"[fold {f}] AUC={roc_auc_score(y[va], oof[va]):.5f}")
    cv = float(roc_auc_score(y, oof)); name = f"first_{KIND}_bag"
    print(f"\n==== {name}  CV AUC = {cv:.5f} ====")
    save_predictions(name, oof, ts / C.N_FOLDS, meta=dict(cv_auc=cv, seed=C.SEED, n_folds=C.N_FOLDS,
                     feature_set=f"1등 FE feature-bag({FRAC}) {KIND}", created_by="hyunbean", notes="feature-bagging GBDT for decorrelation"))


if __name__ == "__main__":
    main()
