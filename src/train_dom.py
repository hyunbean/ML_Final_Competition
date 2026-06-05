"""P4: 도메인 파생 보강 모델 — 우리 피처 + (mega가 약하게 가진) 시퀀스/주기 신호.

target-free(누수 0): ① 구매주기 규칙성(간격 통계) ② 카테고리 전이 bigram(순서) ③ 시간×카테고리 교호.
집계 mega와 다른 '전이/규칙성' 각도. build_features에 덧붙여 LGBM. 5-fold OOF.
실행: python -m src.folds → python -m src.train_dom
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

from . import config as C
from . import data as D
from .features import build_features
from .oof_io import save_predictions

MODEL_NAME = "dom_lgbm"
SEQCOL = "corner_nm"
PARAMS = dict(objective="binary", metric="auc", learning_rate=0.02, num_leaves=64,
              min_child_samples=80, feature_fraction=0.6, bagging_fraction=0.8,
              bagging_freq=1, n_estimators=3000, n_jobs=4, verbosity=-1)


def _periodicity(df, ids):
    d = df[[C.ID_COL, "sales_datetime"]].copy()
    d["sales_datetime"] = pd.to_datetime(d["sales_datetime"])
    d["day"] = d["sales_datetime"].dt.normalize()
    g = d.groupby(C.ID_COL)["day"]

    def stats(s):
        u = np.sort(s.unique())
        if len(u) < 2:
            return pd.Series({"ip_mean": 0, "ip_std": 0, "ip_cv": 0, "ip_max": 0, "ip_reg": 0})
        diff = np.diff(u).astype("timedelta64[D]").astype(float)
        m, sd = diff.mean(), diff.std()
        return pd.Series({"ip_mean": m, "ip_std": sd, "ip_cv": sd / (m + 1e-6),
                          "ip_max": diff.max(), "ip_reg": 1.0 / (sd / (m + 1e-6) + 1.0)})
    return g.apply(stats).unstack().reindex(ids).fillna(0.0)


def _transitions(df, ids):
    d = df[[C.ID_COL, "sales_datetime", SEQCOL]].copy()
    d["sales_datetime"] = pd.to_datetime(d["sales_datetime"])
    d = d.sort_values([C.ID_COL, "sales_datetime"])
    d["prev"] = d.groupby(C.ID_COL)[SEQCOL].shift(1)
    d = d.dropna(subset=["prev"])
    d["same"] = (d[SEQCOL].astype(str) == d["prev"].astype(str)).astype(float)
    d["pair"] = d["prev"].astype(str) + ">" + d[SEQCOL].astype(str)
    g = d.groupby(C.ID_COL)
    out = pd.DataFrame({
        "tr_n": g.size(),
        "tr_same_ratio": g["same"].mean(),       # 같은 코너 연속 비율(집중쇼핑)
        "tr_uniq": g["pair"].nunique(),          # 전이 다양성
    })
    out["tr_uniq_ratio"] = out["tr_uniq"] / (out["tr_n"] + 1)
    return out.reindex(ids).fillna(0.0)


def main():
    X, y, Xtest = build_features()
    train_ids = np.load(C.TRAIN_IDS_NPY, allow_pickle=True)
    test_ids = np.load(C.TEST_IDS_NPY, allow_pickle=True)
    folds = np.load(C.FOLDS_NPY)
    tr, te, _ = D.load_raw()

    blocks_tr = [_periodicity(tr, train_ids), _transitions(tr, train_ids)]
    blocks_te = [_periodicity(te, test_ids), _transitions(te, test_ids)]
    Dtr = pd.concat(blocks_tr, axis=1); Dte = pd.concat(blocks_te, axis=1)
    print(f"도메인 파생 {Dtr.shape[1]}개: {list(Dtr.columns)}")

    Xa = np.hstack([np.nan_to_num(X.values, posinf=0, neginf=0), Dtr.values]).astype(np.float32)
    Xta = np.hstack([np.nan_to_num(Xtest.values, posinf=0, neginf=0), Dte.values]).astype(np.float32)

    oof = np.full(len(y), np.nan); test_sum = np.zeros(len(test_ids))
    for f in range(C.N_FOLDS):
        trn, va = np.where(folds != f)[0], np.where(folds == f)[0]
        m = lgb.LGBMClassifier(**PARAMS)
        m.fit(Xa[trn], y[trn], eval_set=[(Xa[va], y[va])], callbacks=[lgb.early_stopping(120, verbose=False)])
        oof[va] = m.predict_proba(Xa[va])[:, 1]; test_sum += m.predict_proba(Xta)[:, 1]
        print(f"[fold {f}] AUC={roc_auc_score(y[va], oof[va]):.5f}")
    cv = float(roc_auc_score(y, oof))
    print(f"\n==== {MODEL_NAME}  CV AUC = {cv:.5f} ====")
    save_predictions(MODEL_NAME, oof, test_sum / C.N_FOLDS, meta=dict(
        cv_auc=cv, seed=C.SEED, n_folds=C.N_FOLDS, feature_set="our + 구매주기+전이bigram",
        created_by="hyunbean", notes="domain: periodicity + category transition bigram (target-free)"))


if __name__ == "__main__":
    main()
