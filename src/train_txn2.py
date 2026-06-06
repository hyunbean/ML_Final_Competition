"""txn_lgbm 개선판 — 블렌드 1등 기여자(txn_lgbm 0.680)를 강화. goodcd+금액비율+튜닝.

블렌드에 실제 weight 먹는 모델을 올리면 블렌드가 직접 상승. 거래단위, fold-safe(고객단위).
실행: python -m src.train_txn2
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

from . import config as C
from .oof_io import save_predictions

MODEL_NAME = "txn_lgbm2"
CAT = ["brd_nm", "part_nm", "pc_nm", "corner_nm", "team_nm", "str_nm", "buyer_nm",
       "goodcd", "season", "time_zone"]                       # +goodcd
NUM = ["net_amt", "tot_amt", "dis_amt", "inst_mon", "import_flg",
       "hour", "weekday", "month", "is_weekend",
       "dis_ratio", "net_ratio"]                               # +금액비율
SEASON = {3: "봄", 4: "봄", 5: "봄", 6: "여름", 7: "여름", 8: "여름",
          9: "가을", 10: "가을", 11: "가을", 12: "겨울", 1: "겨울", 2: "겨울"}
PARAMS = dict(objective="binary", metric="auc", learning_rate=0.03, num_leaves=300,
              min_child_samples=200, feature_fraction=0.6, bagging_fraction=0.7,
              bagging_freq=1, lambda_l2=2.0, n_estimators=700, n_jobs=-1, verbosity=-1)


def _prep(df):
    df = df.copy()
    df["sales_datetime"] = pd.to_datetime(df["sales_datetime"])
    df["hour"] = df["sales_datetime"].dt.hour
    df["weekday"] = df["sales_datetime"].dt.weekday
    df["month"] = df["sales_datetime"].dt.month
    df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)
    df["season"] = df["month"].map(SEASON)
    df["time_zone"] = pd.cut(df["hour"], bins=[-1, 11, 14, 17, 24],
                             labels=["오전", "점심", "오후", "저녁"]).astype(str)
    tot = df["tot_amt"].replace(0, np.nan)
    df["dis_ratio"] = (df["dis_amt"] / tot).fillna(0.0)
    df["net_ratio"] = (df["net_amt"] / tot).fillna(1.0)
    return df


def main():
    train_ids = np.load(C.TRAIN_IDS_NPY, allow_pickle=True)
    test_ids = np.load(C.TEST_IDS_NPY, allow_pickle=True)
    folds = np.load(C.FOLDS_NPY)
    fold_of = pd.Series(folds, index=train_ids)
    y = pd.read_csv(C.YTRAIN_CSV).set_index(C.ID_COL).reindex(train_ids)[C.TARGET]
    gender = pd.Series(y.values, index=train_ids)

    tr = _prep(pd.read_csv(C.TRAIN_CSV, encoding=C.ENCODING))
    te = _prep(pd.read_csv(C.TEST_CSV, encoding=C.ENCODING))
    for c in CAT:
        cats = pd.Index(pd.concat([tr[c].astype(str), te[c].astype(str)]).unique())
        tr[c] = pd.Categorical(tr[c].astype(str), categories=cats).codes
        te[c] = pd.Categorical(te[c].astype(str), categories=cats).codes
    tr["_g"] = tr[C.ID_COL].map(gender)
    tr["_fold"] = tr[C.ID_COL].map(fold_of)
    feats = CAT + NUM
    print(f"train txns={len(tr):,}  feats={len(feats)} (+goodcd,비율)")

    oof = pd.Series(np.nan, index=train_ids)
    for f in range(C.N_FOLDS):
        trn, val = tr[tr["_fold"] != f], tr[tr["_fold"] == f]
        m = lgb.LGBMClassifier(**PARAMS)
        m.fit(trn[feats], trn["_g"], categorical_feature=CAT)
        vp = m.predict_proba(val[feats])[:, 1]
        agg = pd.Series(vp, index=val[C.ID_COL].values).groupby(level=0).mean()
        oof.loc[agg.index] = agg.values
        print(f"[fold {f}] cust-AUC={roc_auc_score(gender.reindex(agg.index).values, agg.values):.5f}")

    m = lgb.LGBMClassifier(**PARAMS)
    m.fit(tr[feats], tr["_g"], categorical_feature=CAT)
    tp = m.predict_proba(te[feats])[:, 1]
    test_pred = pd.Series(tp, index=te[C.ID_COL].values).groupby(level=0).mean().reindex(test_ids).fillna(gender.mean())

    oof = oof.reindex(train_ids).fillna(gender.mean())
    cv = float(roc_auc_score(y.values, oof.values))
    print(f"\n==== {MODEL_NAME}  CV AUC = {cv:.5f} (기존 txn_lgbm 0.680) ====")
    save_predictions(MODEL_NAME, oof.values, test_pred.values, meta=dict(
        cv_auc=cv, seed=C.SEED, n_folds=C.N_FOLDS, feature_set="txn-level +goodcd+ratios+tuned",
        created_by="hyunbean", notes="improved transaction-level LGBM (goodcd, amount ratios, tuned)"))


if __name__ == "__main__":
    main()
