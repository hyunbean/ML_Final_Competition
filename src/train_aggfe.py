"""가설검증: 숫자-범주화 TE + count/frequency 인코딩 (우승자 단골, mega에 없는 각도).

H: 금액/빈도를 '구간(범주)'으로 보고 fold-safe TE + 카테고리 희소도(count) 인코딩하면
   mega(연속 TE/W2V)가 못 잡은 비선형 신호 → 블렌드 weight 먹는다.
우리 build_features에 새 블록 추가 → LGBM. weight 0이면 가설 기각.
실행: python -m src.train_aggfe
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

from . import config as C
from . import data as D
from .features import build_features, _add_time
from .oof_io import save_predictions

MODEL_NAME = "aggfe_lgbm"
PARAMS = dict(objective="binary", metric="auc", learning_rate=0.02, num_leaves=64,
              min_child_samples=80, feature_fraction=0.6, bagging_fraction=0.8,
              bagging_freq=1, n_estimators=3000, n_jobs=4, verbosity=-1)
ALPHA = 20.0


def _num_as_cat_te(tr, te, train_ids, test_ids, folds, y):
    """per-txn 금액/시간 구간을 범주로 → fold-safe 성별 TE → 고객별 mean/std."""
    gser = pd.Series(y, index=train_ids); gmean = float(y.mean())
    fold_of = pd.Series(folds, index=train_ids)
    # 구간 토큰 (전 거래 공통 경계)
    feats_tr, feats_te = {}, {}
    for col, nbin in [("net_amt", 20), ("hour", 24)]:
        if col == "hour":
            tr["_b"] = tr["hour"].astype(int); te["_b"] = te["hour"].astype(int)
        else:
            edges = pd.qcut(tr[col], nbin, retbins=True, duplicates="drop")[1]
            edges[0], edges[-1] = -np.inf, np.inf
            tr["_b"] = pd.cut(tr[col], edges, labels=False); te["_b"] = pd.cut(te[col], edges, labels=False)
        tt = tr[[C.ID_COL, "_b"]].copy(); tt["_g"] = tt[C.ID_COL].map(gser); tt["_f"] = tt[C.ID_COL].map(fold_of)
        lo = pd.Series(np.nan, index=tt.index)
        for f in range(C.N_FOLDS):
            fit = tt[tt["_f"] != f]
            rate = (fit.groupby("_b")["_g"].sum() + ALPHA * gmean) / (fit.groupby("_b")["_g"].count() + ALPHA)
            m = tt["_f"] == f
            lo[m] = tt.loc[m, "_b"].map(rate)
        tt["r"] = lo.fillna(gmean)
        g = tt.groupby(C.ID_COL)["r"]
        feats_tr[f"te_{col}bin_mean"] = g.mean(); feats_tr[f"te_{col}bin_std"] = g.std().fillna(0)
        # test: 전체 train rate
        rate_all = (tt.groupby("_b")["_g"].sum() + ALPHA * gmean) / (tt.groupby("_b")["_g"].count() + ALPHA)
        te2 = te[[C.ID_COL, "_b"]].copy(); te2["r"] = te2["_b"].map(rate_all).fillna(gmean)
        ge = te2.groupby(C.ID_COL)["r"]
        feats_te[f"te_{col}bin_mean"] = ge.mean(); feats_te[f"te_{col}bin_std"] = ge.std().fillna(0)
    return (pd.DataFrame(feats_tr).reindex(train_ids).fillna(0.0),
            pd.DataFrame(feats_te).reindex(test_ids).fillna(0.0))


def _count_enc(tr, te, train_ids, test_ids):
    """카테고리 전역 빈도(희소도) → 고객별 평균 log-count (니치 vs 대중)."""
    out_tr, out_te = {}, {}
    both = pd.concat([tr[[C.ID_COL, "brd_nm", "goodcd"]], te[[C.ID_COL, "brd_nm", "goodcd"]]])
    for col in ["brd_nm", "goodcd"]:
        cnt = both[col].value_counts()
        for df, ids, out in [(tr, train_ids, out_tr), (te, test_ids, out_te)]:
            lc = np.log1p(df[col].map(cnt))
            g = lc.groupby(df[C.ID_COL])
            out[f"cnt_{col}_mean"] = g.mean(); out[f"cnt_{col}_min"] = g.min()
    return (pd.DataFrame(out_tr).reindex(train_ids).fillna(0.0),
            pd.DataFrame(out_te).reindex(test_ids).fillna(0.0))


def main():
    X, y, Xtest = build_features()
    train_ids = np.load(C.TRAIN_IDS_NPY, allow_pickle=True)
    test_ids = np.load(C.TEST_IDS_NPY, allow_pickle=True)
    folds = np.load(C.FOLDS_NPY)
    tr, te, _ = D.load_raw(); tr = _add_time(tr); te = _add_time(te)

    a1, b1 = _num_as_cat_te(tr, te, train_ids, test_ids, folds, y)
    a2, b2 = _count_enc(tr, te, train_ids, test_ids)
    Dtr = pd.concat([a1, a2], axis=1); Dte = pd.concat([b1, b2], axis=1)
    print(f"새 피처 {Dtr.shape[1]}개: {list(Dtr.columns)}")

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
        cv_auc=cv, seed=C.SEED, n_folds=C.N_FOLDS, feature_set="our + num-as-cat TE + count enc",
        created_by="hyunbean", notes="hypothesis: num-as-categorical TE + frequency encoding"))


if __name__ == "__main__":
    main()
