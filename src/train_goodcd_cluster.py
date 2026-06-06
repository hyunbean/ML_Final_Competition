"""goodcd 잠재군집 ('goodcd_cluster') — item2vec(goodcd)→KMeans → 고객별 군집 지출비중 단독 LGBM.

goodcd 11k(브랜드보다 finer) 상품수준 취향. 정크코드(용기보증/미확인 22.6%) 제외.
수치/TE 피처와 직교 → 스택서 살아남는 멤버 후보. (GPT/Gemini 공통 1순위)
실행: pip install gensim lightgbm scikit-learn → python -m src.folds → python -m src.train_goodcd_cluster
"""
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

from . import config as C
from .oof_io import save_predictions

MODEL_NAME = "goodcd_cluster"
N_CLUST = 180
VEC = 64
JUNK = {"2700000000000"}   # 용기보증/미확인 (22.6%)


def main():
    train_ids = np.load(C.TRAIN_IDS_NPY, allow_pickle=True)
    test_ids = np.load(C.TEST_IDS_NPY, allow_pickle=True)
    folds = np.load(C.FOLDS_NPY)
    y = pd.read_csv(C.YTRAIN_CSV).set_index(C.ID_COL).reindex(train_ids)[C.TARGET].to_numpy()
    use = [C.ID_COL, "sales_datetime", "goodcd", "net_amt", "pc_nm", "corner_nm"]
    tr = pd.read_csv(C.TRAIN_CSV, encoding=C.ENCODING, usecols=use)
    te = pd.read_csv(C.TEST_CSV, encoding=C.ENCODING, usecols=use)
    allx = pd.concat([tr, te], ignore_index=True)
    allx["goodcd"] = allx["goodcd"].astype(str)
    # 정크코드 제외 (용기보증/미확인pc)
    junk_mask = allx["goodcd"].isin(JUNK) | (allx["pc_nm"].astype(str) == "미확인pc") | (allx["corner_nm"].astype(str) == "용기보증")
    print(f"정크 제외: {junk_mask.mean()*100:.1f}% 거래")
    clean = allx[~junk_mask].copy()

    # item2vec: 고객별 goodcd 시퀀스(시간순)
    clean = clean.sort_values([C.ID_COL, "sales_datetime"])
    seqs = clean.groupby(C.ID_COL)["goodcd"].apply(list)
    w2v = Word2Vec(sentences=seqs.tolist(), vector_size=VEC, window=5, min_count=5, sg=1, workers=4, epochs=10, seed=C.SEED)
    vocab = [g for g in w2v.wv.key_to_index]
    vecs = np.vstack([w2v.wv[g] for g in vocab])
    km = KMeans(n_clusters=N_CLUST, random_state=C.SEED, n_init=5).fit(vecs)
    g2c = dict(zip(vocab, km.labels_))
    print(f"goodcd {len(vocab)} → 군집 {N_CLUST}")

    # 고객별 군집 지출비중
    def cluster_share(df, ids):
        d = df[~df["goodcd"].isin(JUNK)].copy()
        d["cl"] = d["goodcd"].map(g2c)
        d = d.dropna(subset=["cl"]); d["cl"] = d["cl"].astype(int)
        d["a"] = d["net_amt"].clip(lower=0)
        amt = d.groupby([C.ID_COL, "cl"])["a"].sum().unstack(fill_value=0.0).reindex(columns=range(N_CLUST), fill_value=0.0)
        sh = amt.div(amt.sum(axis=1).replace(0, 1), axis=0)
        sh.columns = [f"gcl_{c}" for c in sh.columns]
        return sh.reindex(ids).fillna(0.0)

    Xtr = cluster_share(allx[allx[C.ID_COL].isin(set(train_ids))], train_ids)
    Xte = cluster_share(allx[allx[C.ID_COL].isin(set(test_ids))], test_ids)
    print(f"goodcd_cluster X={Xtr.shape}")

    params = dict(objective="binary", metric="auc", learning_rate=0.02, num_leaves=63,
                  min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.6,
                  reg_alpha=1.0, reg_lambda=5.0, n_jobs=-1, random_state=C.SEED, verbose=-1)
    oof = np.full(len(y), np.nan); test_sum = np.zeros(len(test_ids))
    for f in range(C.N_FOLDS):
        tri, va = np.where(folds != f)[0], np.where(folds == f)[0]
        m = lgb.LGBMClassifier(n_estimators=3000, **params)
        m.fit(Xtr.iloc[tri], y[tri], eval_set=[(Xtr.iloc[va], y[va])], callbacks=[lgb.early_stopping(120, verbose=False)])
        oof[va] = m.predict_proba(Xtr.iloc[va])[:, 1]; test_sum += m.predict_proba(Xte)[:, 1]
        print(f"[fold {f}] AUC={roc_auc_score(y[va], oof[va]):.5f}")
    cv = float(roc_auc_score(y, oof))
    print(f"\n==== {MODEL_NAME}  CV AUC = {cv:.5f} ====")
    save_predictions(MODEL_NAME, oof, test_sum / C.N_FOLDS, meta=dict(
        cv_auc=cv, seed=C.SEED, n_folds=C.N_FOLDS, feature_set=f"goodcd item2vec→KMeans{N_CLUST} 지출비중",
        created_by="hyunbean", notes="goodcd 잠재군집(정크제외) 단독 LGBM, 직교 멤버 후보"))


if __name__ == "__main__":
    main()
