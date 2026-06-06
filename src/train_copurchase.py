"""동시구매 transductive 피처 ('copurchase') — 같은 분+점포+(goodcd/brd)에 함께 산 이웃의 gender 전파.

content모델이 못 가진 '관계형' 정보 (test 이웃엔 train 라벨 들어감 = 준지도). OOF-safe:
train은 다른 fold 이웃만, test는 전체 train 이웃. 단독 AUC~0.6이나 직교 정보원.
실행: pip install lightgbm → python -m src.folds → python -m src.train_copurchase
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

from . import config as C
from .oof_io import save_predictions

MODEL_NAME = "copurchase"
KEYS = [["sales_datetime", "str_nm", "goodcd"], ["sales_datetime", "str_nm", "brd_nm"]]


def _neighbor_feats(tr, te, y, folds, train_ids, test_ids, keycols):
    tag = keycols[-1]
    cols = [C.ID_COL] + keycols
    a_tr = tr[cols].copy(); a_te = te[cols].copy()
    a_tr["k"] = a_tr[keycols].astype(str).agg("|".join, axis=1)
    a_te["k"] = a_te[keycols].astype(str).agg("|".join, axis=1)
    fold_of = pd.Series(folds, index=train_ids)
    a_tr["fold"] = a_tr[C.ID_COL].map(fold_of)
    a_tr["g"] = a_tr[C.ID_COL].map(pd.Series(y, index=train_ids))
    # key별 전체 train gender합/수 + fold별
    tot = a_tr.groupby("k").agg(gs=("g", "sum"), n=("g", "size"))
    perfold = a_tr.groupby(["k", "fold"]).agg(gsf=("g", "sum"), nf=("g", "size")).reset_index()

    # train OOF: 각 거래의 이웃 = 전체 - 자기fold
    m = a_tr.merge(tot, on="k").merge(perfold, on=["k", "fold"])
    m["nb_sum"] = m["gs"] - m["gsf"]; m["nb_n"] = m["n"] - m["nf"]
    m = m[m["nb_n"] > 0]
    m["rate"] = m["nb_sum"] / m["nb_n"]
    tr_feat = m.groupby(C.ID_COL).agg(**{f"cp_{tag}_rate": ("rate", "mean"),
                                         f"cp_{tag}_cnt": ("rate", "size")})
    # test: 이웃 = 전체 train
    mt = a_te.merge(tot, on="k")
    mt["rate"] = mt["gs"] / mt["n"]
    te_feat = mt.groupby(C.ID_COL).agg(**{f"cp_{tag}_rate": ("rate", "mean"),
                                          f"cp_{tag}_cnt": ("rate", "size")})
    gm = float(y.mean())
    return (tr_feat.reindex(train_ids).fillna({f"cp_{tag}_rate": gm, f"cp_{tag}_cnt": 0}),
            te_feat.reindex(test_ids).fillna({f"cp_{tag}_rate": gm, f"cp_{tag}_cnt": 0}))


def main():
    train_ids = np.load(C.TRAIN_IDS_NPY, allow_pickle=True)
    test_ids = np.load(C.TEST_IDS_NPY, allow_pickle=True)
    folds = np.load(C.FOLDS_NPY)
    y = pd.read_csv(C.YTRAIN_CSV).set_index(C.ID_COL).reindex(train_ids)[C.TARGET].to_numpy()
    use = [C.ID_COL, "sales_datetime", "str_nm", "goodcd", "brd_nm"]
    tr = pd.read_csv(C.TRAIN_CSV, encoding=C.ENCODING, usecols=use)
    te = pd.read_csv(C.TEST_CSV, encoding=C.ENCODING, usecols=use)

    Xtr, Xte = [], []
    for kc in KEYS:
        a, b = _neighbor_feats(tr, te, y, folds, train_ids, test_ids, kc)
        Xtr.append(a); Xte.append(b)
    Xtr = pd.concat(Xtr, axis=1); Xte = pd.concat(Xte, axis=1)
    Xtr["cp_diff"] = Xtr.filter(like="_rate").mean(axis=1)   # 종합
    Xte["cp_diff"] = Xte.filter(like="_rate").mean(axis=1)
    print(f"copurchase X={Xtr.shape}\n{Xtr.describe().T[['mean','std']]}")

    params = dict(objective="binary", metric="auc", learning_rate=0.02, num_leaves=15,
                  min_child_samples=80, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                  reg_alpha=1.0, reg_lambda=5.0, n_jobs=-1, random_state=C.SEED, verbose=-1)
    oof = np.full(len(y), np.nan); test_sum = np.zeros(len(test_ids))
    for f in range(C.N_FOLDS):
        tri, va = np.where(folds != f)[0], np.where(folds == f)[0]
        m = lgb.LGBMClassifier(n_estimators=600, **params)
        m.fit(Xtr.iloc[tri], y[tri], eval_set=[(Xtr.iloc[va], y[va])], callbacks=[lgb.early_stopping(60, verbose=False)])
        oof[va] = m.predict_proba(Xtr.iloc[va])[:, 1]; test_sum += m.predict_proba(Xte)[:, 1]
        print(f"[fold {f}] AUC={roc_auc_score(y[va], oof[va]):.5f}")
    cv = float(roc_auc_score(y, oof))
    print(f"\n==== {MODEL_NAME}  CV AUC = {cv:.5f} ====")
    save_predictions(MODEL_NAME, oof, test_sum / C.N_FOLDS, meta=dict(
        cv_auc=cv, seed=C.SEED, n_folds=C.N_FOLDS, feature_set="동시구매 이웃 gender 전파(transductive)",
        created_by="hyunbean", notes="co-purchase neighbor gender propensity, 관계형 직교 정보원"))


if __name__ == "__main__":
    main()
