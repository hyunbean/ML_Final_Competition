"""ALS 협업필터 임베딩 ('als_lgbm') — customer×goodcd 행렬 implicit ALS → 고객 64d factor → LGBM.

LightGCN의 핵심(협업필터 '비슷한 상품 사는 고객')을 1/10 비용으로. goodcd 최finest, 정크제외.
node2vec(0.666)보다 CF-최적화라 강할 수 있음. 단독 약해도 직교 멤버 후보.
실행: pip install implicit lightgbm scipy → python -m src.folds → python -m src.train_als
"""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

from . import config as C
from .oof_io import save_predictions

MODEL_NAME = "als_lgbm"
FACTORS = 64
JUNK = {"2700000000000"}


def main():
    import implicit
    train_ids = np.load(C.TRAIN_IDS_NPY, allow_pickle=True)
    test_ids = np.load(C.TEST_IDS_NPY, allow_pickle=True)
    folds = np.load(C.FOLDS_NPY)
    y = pd.read_csv(C.YTRAIN_CSV).set_index(C.ID_COL).reindex(train_ids)[C.TARGET].to_numpy()
    use = [C.ID_COL, "goodcd", "net_amt", "pc_nm", "corner_nm"]
    tr = pd.read_csv(C.TRAIN_CSV, encoding=C.ENCODING, usecols=use)
    te = pd.read_csv(C.TEST_CSV, encoding=C.ENCODING, usecols=use)
    a = pd.concat([tr, te], ignore_index=True)
    a["goodcd"] = a["goodcd"].astype(str)
    junk = a["goodcd"].isin(JUNK) | (a["pc_nm"].astype(str) == "미확인pc") | (a["corner_nm"].astype(str) == "용기보증")
    a = a[~junk]

    all_ids = np.concatenate([train_ids, test_ids])
    cust_idx = {c: i for i, c in enumerate(all_ids)}
    a["ci"] = a[C.ID_COL].map(cust_idx)
    a = a.dropna(subset=["ci"]); a["ci"] = a["ci"].astype(int)
    gcodes, _ = pd.factorize(a["goodcd"])
    a["gi"] = gcodes
    val = np.log1p(a["net_amt"].clip(lower=0).to_numpy() + 1)
    M = csr_matrix((val, (a["ci"], a["gi"])), shape=(len(all_ids), a["gi"].max() + 1))
    print(f"customer×goodcd = {M.shape}, ALS factors={FACTORS}")

    als = implicit.als.AlternatingLeastSquares(factors=FACTORS, regularization=0.05, iterations=20, random_state=C.SEED)
    als.fit(M)
    UF = als.user_factors  # (n_cust, factors)
    if hasattr(UF, "to_numpy"):
        UF = UF.to_numpy()
    UF = np.asarray(UF)[:, :FACTORS]
    Xtr = UF[:len(train_ids)]; Xte = UF[len(train_ids):]
    print(f"als X={Xtr.shape}")

    params = dict(objective="binary", metric="auc", learning_rate=0.02, num_leaves=31,
                  min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                  reg_alpha=1.0, reg_lambda=5.0, n_jobs=-1, random_state=C.SEED, verbose=-1)
    oof = np.full(len(y), np.nan); test_sum = np.zeros(len(test_ids))
    for f in range(C.N_FOLDS):
        tri, va = np.where(folds != f)[0], np.where(folds == f)[0]
        m = lgb.LGBMClassifier(n_estimators=2000, **params)
        m.fit(Xtr[tri], y[tri], eval_set=[(Xtr[va], y[va])], callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[va] = m.predict_proba(Xtr[va])[:, 1]; test_sum += m.predict_proba(Xte)[:, 1]
        print(f"[fold {f}] AUC={roc_auc_score(y[va], oof[va]):.5f}")
    cv = float(roc_auc_score(y, oof))
    print(f"\n==== {MODEL_NAME}  CV AUC = {cv:.5f} ====")
    save_predictions(MODEL_NAME, oof, test_sum / C.N_FOLDS, meta=dict(
        cv_auc=cv, seed=C.SEED, n_folds=C.N_FOLDS, feature_set=f"ALS customer×goodcd {FACTORS}d",
        created_by="hyunbean", notes="implicit ALS 협업필터 임베딩 (LightGCN-lite)"))


if __name__ == "__main__":
    main()
