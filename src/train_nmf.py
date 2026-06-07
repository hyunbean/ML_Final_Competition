"""NMF 협업표현 ('nmf_lgbm') — customer×pc_nm 양수분해 → 고객 페르소나 factor → LGBM. 빠름.

SVD(음수허용)와 달리 NMF는 양수조합만 → 국소 소비페르소나 명확. 미시도.
실행: pip install scikit-learn lightgbm → python -m src.folds → python -m src.train_nmf
"""
import numpy as np, pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import NMF
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from . import config as C
from .oof_io import save_predictions

MODEL_NAME = "nmf_lgbm"
K = 24


def main():
    train_ids = np.load(C.TRAIN_IDS_NPY, allow_pickle=True)
    test_ids = np.load(C.TEST_IDS_NPY, allow_pickle=True)
    folds = np.load(C.FOLDS_NPY)
    y = pd.read_csv(C.YTRAIN_CSV).set_index(C.ID_COL).reindex(train_ids)[C.TARGET].to_numpy()
    a = pd.concat([pd.read_csv(C.TRAIN_CSV, encoding=C.ENCODING, usecols=[C.ID_COL, "pc_nm", "net_amt"]),
                   pd.read_csv(C.TEST_CSV, encoding=C.ENCODING, usecols=[C.ID_COL, "pc_nm", "net_amt"])], ignore_index=True)
    all_ids = np.concatenate([train_ids, test_ids]); ci = {c: i for i, c in enumerate(all_ids)}
    a["ci"] = a[C.ID_COL].map(ci); a = a.dropna(subset=["ci"]); a["ci"] = a["ci"].astype(int)
    pcode, _ = pd.factorize(a["pc_nm"].astype(str))
    M = csr_matrix((np.log1p(a["net_amt"].clip(lower=0) + 1), (a["ci"], pcode)), shape=(len(all_ids), pcode.max() + 1))
    W = NMF(n_components=K, init="nndsvda", random_state=C.SEED, max_iter=300).fit_transform(M)
    Xtr, Xte = W[:len(train_ids)], W[len(train_ids):]
    print(f"NMF X={Xtr.shape}")
    params = dict(objective="binary", metric="auc", learning_rate=0.02, num_leaves=31, min_child_samples=40,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, n_jobs=-1, random_state=C.SEED, verbose=-1)
    oof = np.full(len(y), np.nan); ts = np.zeros(len(test_ids))
    for f in range(C.N_FOLDS):
        tri, va = np.where(folds != f)[0], np.where(folds == f)[0]
        m = lgb.LGBMClassifier(n_estimators=1500, **params)
        m.fit(Xtr[tri], y[tri], eval_set=[(Xtr[va], y[va])], callbacks=[lgb.early_stopping(80, verbose=False)])
        oof[va] = m.predict_proba(Xtr[va])[:, 1]; ts += m.predict_proba(Xte)[:, 1]
        print(f"[fold {f}] AUC={roc_auc_score(y[va], oof[va]):.5f}")
    cv = float(roc_auc_score(y, oof)); print(f"\n==== {MODEL_NAME}  CV AUC = {cv:.5f} ====")
    save_predictions(MODEL_NAME, oof, ts / C.N_FOLDS, meta=dict(cv_auc=cv, seed=C.SEED, n_folds=C.N_FOLDS,
                     feature_set=f"NMF customer×pc {K}d", created_by="hyunbean", notes="NMF 양수분해 페르소나"))


if __name__ == "__main__":
    main()
