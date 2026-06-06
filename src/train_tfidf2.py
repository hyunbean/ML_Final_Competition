"""TF-IDF 강화판 — goodcd 코드계층(prefix 4/6/8) + 바이그램 + 필드 토큰. 단독↑ + 블렌드 기여↑ 노림.

tfidf_lr(0.660)보다 어휘 풍부하게. 희소표현이라 여전히 decorrelated. sklearn만.
실행: python -m src.folds → python -m src.train_tfidf2
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from . import config as C
from .oof_io import save_predictions

MODEL_NAME = "tfidf2_lr"
CATS = ["corner_nm", "brd_nm", "pc_nm", "part_nm"]


def _docs(df, ids, edges):
    df = df.copy()
    ab = pd.cut(df["net_amt"], edges, labels=False).fillna(0).astype(int)
    parts = ab.map(lambda v: f"a{v}")
    for c in CATS:
        parts = parts + " " + c[0] + df[c].astype(str).str.replace(r"\s+", "", regex=True)
    gc = df["goodcd"].astype(str)
    for n in (4, 6, 8):                     # goodcd 코드계층 prefix
        parts = parts + " g" + str(n) + gc.str[:n]
    doc = parts.groupby(df[C.ID_COL]).apply(lambda s: " ".join(s))
    return doc.reindex(ids).fillna("").tolist()


def main():
    train_ids = np.load(C.TRAIN_IDS_NPY, allow_pickle=True)
    test_ids = np.load(C.TEST_IDS_NPY, allow_pickle=True)
    folds = np.load(C.FOLDS_NPY)
    y = pd.read_csv(C.YTRAIN_CSV).set_index(C.ID_COL).reindex(train_ids)[C.TARGET].to_numpy()
    use = [C.ID_COL, "net_amt", "goodcd"] + CATS
    tr = pd.read_csv(C.TRAIN_CSV, encoding=C.ENCODING, usecols=use)
    te = pd.read_csv(C.TEST_CSV, encoding=C.ENCODING, usecols=use)
    edges = pd.qcut(tr["net_amt"], 20, retbins=True, duplicates="drop")[1]
    edges[0], edges[-1] = -np.inf, np.inf
    tr_docs, te_docs = _docs(tr, train_ids, edges), _docs(te, test_ids, edges)

    vec = TfidfVectorizer(token_pattern=r"\S+", min_df=5, sublinear_tf=True, ngram_range=(1, 2))
    Xtr = vec.fit_transform(tr_docs); Xte = vec.transform(te_docs)
    print(f"TF-IDF2 vocab={len(vec.vocabulary_)}  X={Xtr.shape}")

    oof = np.full(len(y), np.nan); test_sum = np.zeros(len(test_ids))
    for f in range(C.N_FOLDS):
        trn, va = np.where(folds != f)[0], np.where(folds == f)[0]
        m = LogisticRegression(C=3.0, max_iter=3000, solver="liblinear").fit(Xtr[trn], y[trn])
        oof[va] = m.predict_proba(Xtr[va])[:, 1]; test_sum += m.predict_proba(Xte)[:, 1]
        print(f"[fold {f}] AUC={roc_auc_score(y[va], oof[va]):.5f}")
    cv = float(roc_auc_score(y, oof))
    print(f"\n==== {MODEL_NAME}  CV AUC = {cv:.5f} ====")
    save_predictions(MODEL_NAME, oof, test_sum / C.N_FOLDS, meta=dict(
        cv_auc=cv, seed=C.SEED, n_folds=C.N_FOLDS, feature_set="TF-IDF + goodcd prefix + bigram",
        created_by="hyunbean", notes="enhanced TF-IDF (goodcd hierarchy + ngram12)"))


if __name__ == "__main__":
    main()
