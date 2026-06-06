"""CatBoost native text features — goodcd/brd/corner 토큰을 CatBoost 내부 BoW/BM25로 처리.

우리 TF-IDF(외부 sklearn)와 달리 CatBoost가 텍스트를 내부 토큰화→트리에 통합 = 다른 처리/오류구조.
우리 수치피처 + 고객별 토큰문서(text_features) 함께. 5-fold OOF.
실행: pip install catboost gensim → python -m src.folds → python -m src.train_cat_text
"""
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score

from . import config as C
from .features import build_features
from .oof_io import save_predictions

MODEL_NAME = "cat_text"
TXTCOLS = ["corner_nm", "brd_nm", "goodcd", "pc_nm"]


def _doc(df, ids):
    df = df.copy()
    parts = df[TXTCOLS[0]].astype(str).str.replace(r"\s+", "", regex=True)
    for c in TXTCOLS[1:]:
        parts = parts + " " + c[0] + df[c].astype(str).str.replace(r"\s+", "", regex=True)
    return parts.groupby(df[C.ID_COL]).apply(lambda s: " ".join(s)).reindex(ids).fillna("")


def main():
    X, y, Xtest = build_features()
    train_ids = np.load(C.TRAIN_IDS_NPY, allow_pickle=True)
    test_ids = np.load(C.TEST_IDS_NPY, allow_pickle=True)
    folds = np.load(C.FOLDS_NPY)
    use = [C.ID_COL] + TXTCOLS
    tr = pd.read_csv(C.TRAIN_CSV, encoding=C.ENCODING, usecols=use)
    te = pd.read_csv(C.TEST_CSV, encoding=C.ENCODING, usecols=use)

    Xtr = X.copy().reset_index(drop=True); Xtr.columns = [f"n{i}" for i in range(X.shape[1])]
    Xte = Xtest.copy().reset_index(drop=True); Xte.columns = Xtr.columns
    Xtr["doc"] = _doc(tr, train_ids).values
    Xte["doc"] = _doc(te, test_ids).values
    Xtr = Xtr.fillna(0.0); Xte = Xte.fillna(0.0)
    print(f"X={Xtr.shape} (+text doc)")

    params = dict(loss_function="Logloss", eval_metric="AUC", iterations=2000, learning_rate=0.03,
                  depth=6, l2_leaf_reg=5.0, random_seed=C.SEED, verbose=False,
                  text_features=["doc"], task_type="CPU")
    oof = np.full(len(y), np.nan); test_sum = np.zeros(len(test_ids))
    for f in range(C.N_FOLDS):
        tri, va = np.where(folds != f)[0], np.where(folds == f)[0]
        m = CatBoostClassifier(**params)
        m.fit(Pool(Xtr.iloc[tri], y[tri], text_features=["doc"]),
              eval_set=Pool(Xtr.iloc[va], y[va], text_features=["doc"]), early_stopping_rounds=150)
        oof[va] = m.predict_proba(Xtr.iloc[va])[:, 1]; test_sum += m.predict_proba(Xte)[:, 1]
        print(f"[fold {f}] AUC={roc_auc_score(y[va], oof[va]):.5f}")
    cv = float(roc_auc_score(y, oof))
    print(f"\n==== {MODEL_NAME}  CV AUC = {cv:.5f} ====")
    save_predictions(MODEL_NAME, oof, test_sum / C.N_FOLDS, meta=dict(
        cv_auc=cv, seed=C.SEED, n_folds=C.N_FOLDS, feature_set="our + CatBoost text(goodcd/brd)",
        created_by="hyunbean", notes="CatBoost native text features (internal BoW)"))


if __name__ == "__main__":
    main()
