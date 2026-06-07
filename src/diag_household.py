"""household/conflict 진단 (GPT 요청): 3개 트리(xgb/cat/lgbm) BASE vs +conflict,
간이 stack, 공통오답군 AUC, conflict/proxy_male2 importance rank.

핵심 판정: 3개 트리가 '동시에' 오르고 stack +0.001 이상이면 진짜. 일부만/노이즈면 폐기.
실행(GPU): pip install xgboost lightgbm catboost gensim → python -m src.diag_household
"""
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

from . import config as C
from .train_first import build_all, build_household, _load

NF = C.N_FOLDS
import os
DEV = os.environ.get("DIAG_DEV", "cuda")          # cpu로 로컬 실행 가능
CBT = "GPU" if DEV == "cuda" else "CPU"


def _oof(kind, X, y, folds):
    import xgboost as xgb, lightgbm as lgb
    from catboost import CatBoostClassifier
    oof = np.full(len(y), np.nan); imp = np.zeros(X.shape[1])
    for f in range(NF):
        tri, va = np.where(folds != f)[0], np.where(folds == f)[0]
        if kind == "xgb":
            m = xgb.XGBClassifier(n_estimators=4000, early_stopping_rounds=150, objective="binary:logistic",
                                  eval_metric="auc", learning_rate=0.02, max_depth=7, min_child_weight=5,
                                  gamma=0.1, subsample=0.8, colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=5.0,
                                  random_state=C.SEED, tree_method="hist", device=DEV)
            m.fit(X.iloc[tri], y[tri], eval_set=[(X.iloc[va], y[va])], verbose=False)
            imp += m.feature_importances_
        elif kind == "lgbm":
            m = lgb.LGBMClassifier(n_estimators=6000, objective="binary", metric="auc", learning_rate=0.02,
                                   num_leaves=63, min_child_samples=40, subsample=0.8, subsample_freq=1,
                                   colsample_bytree=0.8, reg_alpha=1.0, reg_lambda=5.0, n_jobs=-1,
                                   random_state=C.SEED, verbose=-1)
            m.fit(X.iloc[tri], y[tri], eval_set=[(X.iloc[va], y[va])],
                  callbacks=[lgb.early_stopping(150, verbose=False)])
            imp += m.feature_importances_
        else:
            m = CatBoostClassifier(iterations=6000, loss_function="Logloss", eval_metric="AUC", learning_rate=0.03,
                                   depth=7, l2_leaf_reg=5, random_seed=C.SEED, verbose=0, allow_writing_files=False,
                                   task_type=CBT)
            m.fit(X.iloc[tri], y[tri], eval_set=(X.iloc[va], y[va]), early_stopping_rounds=150)
            imp += m.get_feature_importance()
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
    return oof, imp / NF


def _stack(oofs, y, folds):
    """3 OOF -> fold-safe logreg stack OOF AUC."""
    Z = np.column_stack(oofs); s = np.full(len(y), np.nan)
    for f in range(NF):
        tri, va = np.where(folds != f)[0], np.where(folds == f)[0]
        lr = LogisticRegression(C=1.0, max_iter=1000).fit(Z[tri], y[tri])
        s[va] = lr.predict_proba(Z[va])[:, 1]
    return roc_auc_score(y, s)


def main():
    train_ids = np.load(C.TRAIN_IDS_NPY, allow_pickle=True)
    folds = np.load(C.FOLDS_NPY)
    allf, ydf, _, _ = build_all()
    _, _, _, full = _load()
    y = ydf.set_index("custid").reindex(train_ids)["gender"].to_numpy()
    Xb = allf.reindex(train_ids).fillna(0.0).reset_index(drop=True)
    hh = build_household(full)
    Xa = allf.join(hh, how="left").reindex(train_ids).fillna(0.0).reset_index(drop=True)
    print(f"BASE feat={Xb.shape[1]}, +conflict feat={Xa.shape[1]} (+{Xa.shape[1]-Xb.shape[1]})\n")

    base, aug, imps = {}, {}, {}
    print("========== 모델별 CV (동일 파라미터, BASE vs +conflict) ==========")
    for k in ["xgb", "cat", "lgbm"]:
        ob, _ = _oof(k, Xb, y, folds)
        oa, ia = _oof(k, Xa, y, folds)
        base[k], aug[k], imps[k] = ob, oa, ia
        cb, ca = roc_auc_score(y, ob), roc_auc_score(y, oa)
        print(f"  first_{k:4s}  {cb:.5f} -> {ca:.5f}   ({ca-cb:+.5f})")

    sb = _stack([base["xgb"], base["cat"], base["lgbm"]], y, folds)
    sa = _stack([aug["xgb"], aug["cat"], aug["lgbm"]], y, folds)
    print(f"  stack(3)   {sb:.5f} -> {sa:.5f}   ({sa-sb:+.5f})")

    print(f"\n========== conflict/proxy importance rank (+conflict 모델) ==========")
    cols = list(Xa.columns); n = len(cols)
    for k in ["xgb", "cat", "lgbm"]:
        order = [cols[j] for j in np.argsort(imps[k])[::-1]]
        rk = {c: i + 1 for i, c in enumerate(order)}
        s = "  ".join(f"{f}=#{rk[f]}" for f in ["hh_conflict", "hh_proxy_male2"] if f in rk)
        print(f"  {k:4s} (/{n}):  {s}")

    print(f"\n========== 공통오답군 AUC (3모델 base 평균 틀린 고객) ==========")
    base_avg = (base["xgb"] + base["cat"] + base["lgbm"]) / 3
    aug_avg = (aug["xgb"] + aug["cat"] + aug["lgbm"]) / 3
    wrong = ((base_avg >= 0.5).astype(int) != y)
    print(f"  공통오답 = {wrong.sum()}명 ({wrong.mean()*100:.1f}%)")
    print(f"  오답군 AUC   BASE {roc_auc_score(y[wrong], base_avg[wrong]):.5f} -> "
          f"+conflict {roc_auc_score(y[wrong], aug_avg[wrong]):.5f}")
    print(f"  오답군 recall(y=1) BASE {(base_avg[wrong & (y==1)]>=0.5).mean():.4f} -> "
          f"+conflict {(aug_avg[wrong & (y==1)]>=0.5).mean():.4f}")


if __name__ == "__main__":
    main()
