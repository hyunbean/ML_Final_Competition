"""multilingual-e5 거래텍스트 임베딩 → LGBM ('e5_lgbm'). 한국어 상품/코너명에 BERT보다 강함.

고객별 corner/brd/pc 시퀀스를 문서로 → e5(대조학습 검색임베더)로 인코딩 → SVD64 → LGBM.
비지도(누수없음). 우리 BERT(text_roberta)와 다른 임베딩공간 = 결 다름.
실행(GPU): pip install sentence-transformers lightgbm → python -m src.train_e5
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import roc_auc_score

from . import config as C
from .oof_io import save_predictions

MODEL_NAME = "e5_lgbm"
TXT = ["corner_nm", "brd_nm", "pc_nm"]


def _docs(df, ids):
    parts = df[TXT[0]].astype(str)
    for c in TXT[1:]:
        parts = parts + " " + df[c].astype(str)
    s = parts.groupby(df[C.ID_COL]).apply(lambda x: " ".join(x)).reindex(ids).fillna("")
    return ("query: " + s).tolist()


def main():
    from sentence_transformers import SentenceTransformer
    train_ids = np.load(C.TRAIN_IDS_NPY, allow_pickle=True)
    test_ids = np.load(C.TEST_IDS_NPY, allow_pickle=True)
    folds = np.load(C.FOLDS_NPY)
    y = pd.read_csv(C.YTRAIN_CSV).set_index(C.ID_COL).reindex(train_ids)[C.TARGET].to_numpy()
    use = [C.ID_COL] + TXT
    tr = pd.read_csv(C.TRAIN_CSV, encoding=C.ENCODING, usecols=use)
    te = pd.read_csv(C.TEST_CSV, encoding=C.ENCODING, usecols=use)

    enc = SentenceTransformer("intfloat/multilingual-e5-base")
    Etr = enc.encode(_docs(tr, train_ids), normalize_embeddings=True, batch_size=128, show_progress_bar=True)
    Ete = enc.encode(_docs(te, test_ids), normalize_embeddings=True, batch_size=128, show_progress_bar=True)
    svd = TruncatedSVD(64, random_state=C.SEED).fit(np.vstack([Etr, Ete]))
    Xtr = svd.transform(Etr); Xte = svd.transform(Ete)
    print(f"e5 emb→svd X={Xtr.shape}")

    params = dict(objective="binary", metric="auc", learning_rate=0.02, num_leaves=31,
                  min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                  reg_alpha=1.0, reg_lambda=5.0, n_jobs=-1, random_state=C.SEED, verbose=-1)
    oof = np.full(len(y), np.nan); test_sum = np.zeros(len(test_ids))
    for f in range(C.N_FOLDS):
        tri, va = np.where(folds != f)[0], np.where(folds == f)[0]
        m = lgb.LGBMClassifier(n_estimators=3000, **params)
        m.fit(Xtr[tri], y[tri], eval_set=[(Xtr[va], y[va])], callbacks=[lgb.early_stopping(120, verbose=False)])
        oof[va] = m.predict_proba(Xtr[va])[:, 1]; test_sum += m.predict_proba(Xte)[:, 1]
        print(f"[fold {f}] AUC={roc_auc_score(y[va], oof[va]):.5f}")
    cv = float(roc_auc_score(y, oof))
    print(f"\n==== {MODEL_NAME}  CV AUC = {cv:.5f} ====")
    save_predictions(MODEL_NAME, oof, test_sum / C.N_FOLDS, meta=dict(
        cv_auc=cv, seed=C.SEED, n_folds=C.N_FOLDS, feature_set="multilingual-e5 emb→SVD64 LGBM",
        created_by="hyunbean", notes="e5 다국어 거래텍스트 임베딩 5fold OOF"))


if __name__ == "__main__":
    main()
