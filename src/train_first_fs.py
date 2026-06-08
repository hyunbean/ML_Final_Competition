"""교수님 힌트 #2: 중요도 하위 feature + 상관(유사) feature 제거 → 노이즈제거 (+0.002 기대).

521피처 중 (a)중요도 하위 제거 (b)상관>thr 쌍에서 낮은중요도 제거 → 깨끗한 셋으로 재학습.
실행(GPU): FS_DROP=0.4 FS_CORR=0.95 XGB_GPU=1 python -m src.train_first_fs [xgb|lgbm|cat]
"""
import os
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

from . import config as C
from .oof_io import save_predictions
from .train_first import build_all

DROP_FRAC = float(os.environ.get("FS_DROP", "0.4"))    # 중요도 하위 비율 제거
CORR_THR = float(os.environ.get("FS_CORR", "0.95"))    # 이 상관 이상 쌍은 낮은중요도 제거
GPU = os.environ.get("XGB_GPU", "1") == "1"
CB = os.environ.get("CB_TASK_TYPE", "GPU")


def select_features(X, y):
    def _imp(yy):
        return lgb.LGBMClassifier(n_estimators=600, num_leaves=63, learning_rate=0.05,
                                  random_state=C.SEED, n_jobs=-1, verbose=-1).fit(X, yy).feature_importances_
    imp_real = pd.Series(_imp(y), index=X.columns)
    if os.environ.get("FS_NULL") == "1":               # #7 null-importance: 랜덤타겟 대비
        rng = np.random.default_rng(C.SEED)
        nulls = np.max([_imp(rng.permutation(y)) for _ in range(3)], axis=0)   # 셔플 최대 중요도
        keep = list(imp_real[imp_real.values > nulls].sort_values(ascending=False).index)  # 노이즈 초과만
        imp = imp_real[keep].sort_values(ascending=False)
        print(f"  null-importance: {len(X.columns)} -> {len(keep)} (랜덤 초과)")
    else:
        imp = imp_real.sort_values(ascending=False)
        n_keep = int(len(imp) * (1 - DROP_FRAC))
        keep = list(imp.index[:n_keep])                # (a) 중요도 상위만
    corr = X[keep].corr().abs().fillna(0)
    drop = set()
    for i, a in enumerate(keep):                        # (b) 상관 유사쌍 → 낮은중요도(뒤) 제거
        if a in drop:
            continue
        for b in keep[i + 1:]:
            if b not in drop and corr.loc[a, b] > CORR_THR:
                drop.add(b)
    final = [c for c in keep if c not in drop]
    print(f"  feature select: {len(X.columns)} -> 중요도상위 {len(keep)} -> 상관제거후 {len(final)}")
    return final


def disc_features(X, Xt, y, folds):
    """#5 임베딩 기반 성별-판별 피처 (OOF fold-safe). 남성/여성 임베딩 중심방향 투영 + 각 중심거리."""
    pref = ("w2vwm_gc_", "ftwm_gc_", "d2v_gc_", "w2v_brd_nm_")
    cols = [c for c in X.columns if any(c.startswith(p) for p in pref)]
    if not cols:
        return X, Xt
    E = X[cols].to_numpy(np.float32); Et = Xt[cols].to_numpy(np.float32)
    proj = np.zeros(len(y)); proj_t = np.zeros(len(Et))
    df = np.zeros(len(y)); dm = np.zeros(len(y)); dft = np.zeros(len(Et)); dmt = np.zeros(len(Et))
    for f in range(C.N_FOLDS):
        tri, va = np.where(folds != f)[0], np.where(folds == f)[0]
        cf = E[tri][y[tri] == 1].mean(0); cm = E[tri][y[tri] == 0].mean(0)   # 여성/남성 중심(train fold)
        d = cf - cm; d = d / (np.linalg.norm(d) + 1e-9)
        proj[va] = E[va] @ d
        df[va] = np.linalg.norm(E[va] - cf, axis=1); dm[va] = np.linalg.norm(E[va] - cm, axis=1)
        proj_t += (Et @ d) / C.N_FOLDS
        dft += np.linalg.norm(Et - cf, axis=1) / C.N_FOLDS; dmt += np.linalg.norm(Et - cm, axis=1) / C.N_FOLDS
    X = X.copy(); Xt = Xt.copy()
    X["disc_proj"] = proj; X["disc_dfem"] = df; X["disc_dmal"] = dm; X["disc_dgap"] = dm - df
    Xt["disc_proj"] = proj_t; Xt["disc_dfem"] = dft; Xt["disc_dmal"] = dmt; Xt["disc_dgap"] = dmt - dft
    print(f"  +disc features (임베딩 성별판별축 4개, fold-safe)")
    return X, Xt


def _fit(kind, Xtr, ytr, Xva, yva, Xt):
    import xgboost as xgb
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
        cp = dict(iterations=1658, learning_rate=0.0107, depth=7, l2_leaf_reg=13.9, subsample=0.708,
                  bootstrap_type="Bernoulli", min_data_in_leaf=39, random_strength=3.15, eval_metric="AUC",
                  random_seed=C.SEED, verbose=0, allow_writing_files=False, task_type=CB)
        if CB != "GPU":
            cp["colsample_bylevel"] = 0.818            # rsm은 GPU 미지원
        m = CatBoostClassifier(**cp)
        m.fit(Xtr, ytr)
    return m.predict_proba(Xva)[:, 1], m.predict_proba(Xt)[:, 1]


def main():
    kind = sys.argv[1] if len(sys.argv) > 1 else "xgb"
    train_ids = np.load(C.TRAIN_IDS_NPY, allow_pickle=True)
    test_ids = np.load(C.TEST_IDS_NPY, allow_pickle=True)
    folds = np.load(C.FOLDS_NPY)
    allf, ydf, _, _ = build_all()
    X = allf.reindex(train_ids).fillna(0.0); Xt = allf.reindex(test_ids).fillna(0.0)
    y = ydf.set_index("custid").reindex(train_ids)["gender"].to_numpy()
    if os.environ.get("FS_DISC", "1") == "1":          # #5 임베딩 성별판별 피처 (있을때만)
        X, Xt = disc_features(X, Xt, y, folds)
    feats = select_features(X, y)
    X = X[feats]; Xt = Xt[feats]
    oof = np.full(len(y), np.nan); test_sum = np.zeros(len(test_ids))
    for f in range(C.N_FOLDS):
        tri, va = np.where(folds != f)[0], np.where(folds == f)[0]
        vp, tp = _fit(kind, X.iloc[tri], y[tri], X.iloc[va], y[va], Xt)
        oof[va] = vp; test_sum += tp
        print(f"[fold {f}] AUC={roc_auc_score(y[va], oof[va]):.5f}")
    cv = float(roc_auc_score(y, oof)); name = f"first_{kind}_fs"
    print(f"\n==== {name}  CV={cv:.5f}  (feats={len(feats)}, DROP={DROP_FRAC} CORR={CORR_THR}) ====")
    save_predictions(name, oof, test_sum / C.N_FOLDS, meta=dict(cv_auc=cv, seed=C.SEED, n_folds=C.N_FOLDS,
                     feature_set=f"1등FE 중요도/상관 선택 {len(feats)}feat", created_by="hyunbean",
                     notes=f"교수힌트#2 feature selection DROP{DROP_FRAC} CORR{CORR_THR}"))


if __name__ == "__main__":
    main()
