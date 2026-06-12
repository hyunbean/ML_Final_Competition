"""임베딩 전용 모델 ('emb_lgbm') — W2V(+SVD/LDA) 임베딩 컬럼만으로 학습.

수치/TE/집계 피처를 완전히 배제 → 수치기반 GBT들과 직교 → 스택서 안 죽는 멤버(두 AI 강조).
실행(GPU/CPU): pip install lightgbm gensim → python -m src.folds → python -m src.train_emb_only
"""
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

from . import config as C
from .oof_io import save_predictions
from .features import build_features

MODEL_NAME = "emb_lgbm"


def main():
    folds = np.load(C.FOLDS_NPY)
    test_ids = np.load(C.TEST_IDS_NPY, allow_pickle=True)
    X, y, Xt = build_features()   # r10 캐시 재사용(W2V 임베딩 포함)
    emb = [c for c in X.columns if ("_w2v_" in c) or c.startswith("svd_") or c.startswith("lda_")]
    Xe, Xte = X[emb], Xt[emb]
    print(f"임베딩 전용 X={Xe.shape} (전체 {X.shape[1]} 중 임베딩 {len(emb)})")

    params = dict(objective="binary", metric="auc", learning_rate=0.02, num_leaves=63,
                  min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.6,
                  reg_alpha=1.0, reg_lambda=5.0, n_jobs=-1, random_state=C.SEED, verbose=-1)
    oof = np.full(len(y), np.nan); test_sum = np.zeros(len(test_ids))
    for f in range(C.N_FOLDS):
        tri, va = np.where(folds != f)[0], np.where(folds == f)[0]
        m = lgb.LGBMClassifier(n_estimators=3000, **params)
        m.fit(Xe.iloc[tri], y[tri], eval_set=[(Xe.iloc[va], y[va])], callbacks=[lgb.early_stopping(150, verbose=False)])
        oof[va] = m.predict_proba(Xe.iloc[va])[:, 1]; test_sum += m.predict_proba(Xte)[:, 1]
        print(f"[fold {f}] AUC={roc_auc_score(y[va], oof[va]):.5f}")
    cv = float(roc_auc_score(y, oof))
    print(f"\n==== {MODEL_NAME}  CV AUC = {cv:.5f} ====")
    save_predictions(MODEL_NAME, oof, test_sum / C.N_FOLDS, meta=dict(
        cv_auc=cv, seed=C.SEED, n_folds=C.N_FOLDS, feature_set=f"임베딩전용 W2V+SVD/LDA ({len(emb)}) LGBM",
        created_by="hyunbean", notes="disjoint-L1: 임베딩 컬럼만 → 수치GBT와 직교 멤버"))


if __name__ == "__main__":
    main()
