"""거래 단위(transaction-level) CatBoost — txn_lgbm과 같은 결, 다른 알고리즘.

txn_lgbm이 단독 0.68인데 블렌드 weight 0.22(상위)를 먹은 '노다지 표현'.
CatBoost native 범주형(문자열 그대로)으로 한 번 더 → txn_lgbm과 또 살짝 decorrelated.
fold-safe(고객 단위). features 캐시 불필요.

실행: python -m src.train_txn_cat
"""
import os
import shutil
import subprocess
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score

from . import config as C
from .oof_io import save_predictions


def _has_gpu():
    if os.path.exists("/dev/nvidia0"):
        return True
    if shutil.which("nvidia-smi"):
        try:
            return subprocess.run(["nvidia-smi"], capture_output=True).returncode == 0
        except Exception:
            pass
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


MODEL_NAME = "txn_catboost"
TASK = "GPU" if _has_gpu() else "CPU"
CAT = ["brd_nm", "part_nm", "pc_nm", "corner_nm", "team_nm", "str_nm", "buyer_nm", "season", "time_zone"]
NUM = ["net_amt", "tot_amt", "dis_amt", "inst_mon", "import_flg", "hour", "weekday", "month", "is_weekend"]
SEASON = {3: "봄", 4: "봄", 5: "봄", 6: "여름", 7: "여름", 8: "여름",
          9: "가을", 10: "가을", 11: "가을", 12: "겨울", 1: "겨울", 2: "겨울"}
PARAMS = dict(loss_function="Logloss", eval_metric="AUC", task_type=TASK, devices="0",
              learning_rate=0.05, depth=8, l2_leaf_reg=5.0, iterations=500,
              random_seed=C.SEED, verbose=False)


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
    for c in CAT:
        df[c] = df[c].astype(str)
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
    tr["_g"] = tr[C.ID_COL].map(gender)
    tr["_fold"] = tr[C.ID_COL].map(fold_of)
    feats = CAT + NUM
    print(f"train txns={len(tr):,}  test txns={len(te):,}  device={TASK}")

    oof = pd.Series(np.nan, index=train_ids)
    for f in range(C.N_FOLDS):
        trn, val = tr[tr["_fold"] != f], tr[tr["_fold"] == f]
        m = CatBoostClassifier(**PARAMS)
        m.fit(Pool(trn[feats], trn["_g"], cat_features=CAT))
        vp = m.predict_proba(val[feats])[:, 1]
        agg = pd.Series(vp, index=val[C.ID_COL].values).groupby(level=0).mean()
        oof.loc[agg.index] = agg.values
        print(f"[fold {f}] cust-AUC={roc_auc_score(gender.reindex(agg.index).values, agg.values):.5f}")

    m = CatBoostClassifier(**PARAMS)
    m.fit(Pool(tr[feats], tr["_g"], cat_features=CAT))
    tp = m.predict_proba(te[feats])[:, 1]
    test_pred = pd.Series(tp, index=te[C.ID_COL].values).groupby(level=0).mean().reindex(test_ids).fillna(gender.mean())

    oof = oof.reindex(train_ids).fillna(gender.mean())
    cv = float(roc_auc_score(y.values, oof.values))
    print(f"\n==== {MODEL_NAME}  CV AUC = {cv:.5f} ====")
    save_predictions(MODEL_NAME, oof.values, test_pred.values, meta=dict(
        cv_auc=cv, seed=C.SEED, n_folds=C.N_FOLDS, feature_set="transaction-level",
        created_by="hyunbean", notes="transaction-level CatBoost (native cat), mean-agg per customer"))


if __name__ == "__main__":
    main()
