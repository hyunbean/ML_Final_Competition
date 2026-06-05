"""TabPFN v2 — 사전학습 트랜스포머(in-context learning) tabular 파운데이션 모델.

아무도(우리/팀/1등) 안 쓴 완전히 다른 inductive bias → 블렌드 다양성.
TabPFN은 행/피처 제한이 있어 ① lgbm gain으로 top-K 피처 선택 ② 폴드train을 10k 서브샘플
앙상블(여러 번 fit 평균)로 처리. 5-fold OOF.

실행(GPU): pip install tabpfn lightgbm gensim → python -m src.folds → python -m src.train_tabpfn
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

from . import config as C
from .features import build_features
from .oof_io import save_predictions

MODEL_NAME = "tabpfn"
TOPK = 100          # TabPFN 피처 상한
SUB = 10000         # fit당 서브샘플
N_BAG = 4           # 서브샘플 앙상블 횟수


def _topk_feats(X, y, folds):
    imp = np.zeros(X.shape[1])
    for f in range(C.N_FOLDS):
        tr = np.where(folds != f)[0]
        m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=63, n_jobs=-1, verbosity=-1)
        m.fit(X.iloc[tr], y[tr])
        imp += m.feature_importances_
    cols = pd.Series(imp, index=X.columns).sort_values(ascending=False).head(TOPK).index
    return list(cols)


def main():
    from tabpfn import TabPFNClassifier
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    X, y, Xtest = build_features()
    folds = np.load(C.FOLDS_NPY)
    cols = _topk_feats(X, y, folds)
    Xv = np.nan_to_num(X[cols].values.astype(np.float32), posinf=0, neginf=0)
    Xtv = np.nan_to_num(Xtest[cols].values.astype(np.float32), posinf=0, neginf=0)
    print(f"TabPFN top{TOPK} feats, device={dev}, bag={N_BAG}x{SUB}")

    oof = np.full(len(y), np.nan)
    test_sum = np.zeros(len(Xtv))
    for f in range(C.N_FOLDS):
        tr, va = np.where(folds != f)[0], np.where(folds == f)[0]
        vp = np.zeros(len(va)); tp = np.zeros(len(Xtv))
        for b in range(N_BAG):
            rng = np.random.default_rng(C.SEED + b)
            sub = rng.choice(tr, size=min(SUB, len(tr)), replace=False)
            clf = TabPFNClassifier(device=dev)
            clf.fit(Xv[sub], y[sub])
            vp += clf.predict_proba(Xv[va])[:, 1]
            tp += clf.predict_proba(Xtv)[:, 1]
        oof[va] = vp / N_BAG
        test_sum += tp / N_BAG
        print(f"[fold {f}] AUC={roc_auc_score(y[va], oof[va]):.5f}")

    cv = float(roc_auc_score(y, oof))
    print(f"\n==== {MODEL_NAME}  CV AUC = {cv:.5f} ====")
    save_predictions(MODEL_NAME, oof, test_sum / C.N_FOLDS, meta=dict(
        cv_auc=cv, seed=C.SEED, n_folds=C.N_FOLDS, feature_set=f"top{TOPK} + TabPFN bag{N_BAG}",
        created_by="hyunbean", notes="TabPFN v2 in-context, top-K feats, subsample-bagged"))


if __name__ == "__main__":
    main()
