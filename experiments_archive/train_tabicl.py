"""TabICL ('tabicl') — 표형 in-context 학습 foundation 모델 (TabPFN 계열, 다른 구조).

우리 r7 피처에 TabICL 5fold OOF. TabPFN과 다른 모델클래스 = 결 다름.
실행(GPU): pip install tabicl → python -m src.folds → python -m src.train_tabicl
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import config as C
from .oof_io import save_predictions
from .features import build_features

MODEL_NAME = "tabicl"


def main():
    from tabicl import TabICLClassifier
    folds = np.load(C.FOLDS_NPY)
    test_ids = np.load(C.TEST_IDS_NPY, allow_pickle=True)
    X, y, Xt = build_features()
    Xv = np.nan_to_num(X.values.astype(np.float32), posinf=0, neginf=0)
    Xtv = np.nan_to_num(Xt.values.astype(np.float32), posinf=0, neginf=0)
    print(f"TabICL X={Xv.shape}")
    oof = np.full(len(y), np.nan); test_sum = np.zeros(len(test_ids))
    for f in range(C.N_FOLDS):
        tri, va = np.where(folds != f)[0], np.where(folds == f)[0]
        m = TabICLClassifier(random_state=C.SEED)
        m.fit(Xv[tri], y[tri])
        oof[va] = m.predict_proba(Xv[va])[:, 1]; test_sum += m.predict_proba(Xtv)[:, 1]
        print(f"[fold {f}] AUC={roc_auc_score(y[va], oof[va]):.5f}")
    cv = float(roc_auc_score(y, oof))
    print(f"\n==== {MODEL_NAME}  CV AUC = {cv:.5f} ====")
    save_predictions(MODEL_NAME, oof, test_sum / C.N_FOLDS, meta=dict(
        cv_auc=cv, seed=C.SEED, n_folds=C.N_FOLDS, feature_set="r7 TabICL foundation",
        created_by="hyunbean", notes="TabICL in-context tabular foundation, 5fold OOF"))


if __name__ == "__main__":
    main()
