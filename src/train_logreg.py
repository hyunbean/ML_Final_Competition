"""Logistic Regression (CPU) — 앙상블 다양성용 (선형, 트리/NN과 다른 bias).

표준화(StandardScaler) + 5-fold OOF. 단일 AUC는 낮아도 '다른 류'라 블렌드에 기여.
결과: logreg_full OOF/test npy.

실행: python -m src.train_logreg
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from . import config as C
from .features import build_features
from .oof_io import save_predictions

MODEL_NAME = "logreg_full"


def main():
    X, y, Xtest = build_features()
    folds = np.load(C.FOLDS_NPY)
    Xv = np.nan_to_num(X.values.astype(np.float32), posinf=0.0, neginf=0.0)
    Xtv = np.nan_to_num(Xtest.values.astype(np.float32), posinf=0.0, neginf=0.0)
    print(f"X={X.shape}")

    oof = np.full(len(y), np.nan)
    test_sum = np.zeros(len(Xtv))
    for f in range(C.N_FOLDS):
        tr, va = np.where(folds != f)[0], np.where(folds == f)[0]
        sc = StandardScaler().fit(Xv[tr])
        m = LogisticRegression(C=1.0, max_iter=3000, n_jobs=-1)
        m.fit(sc.transform(Xv[tr]), y[tr])
        oof[va] = m.predict_proba(sc.transform(Xv[va]))[:, 1]
        test_sum += m.predict_proba(sc.transform(Xtv))[:, 1]
        print(f"[fold {f}] AUC={roc_auc_score(y[va], oof[va]):.5f}")

    cv = float(roc_auc_score(y, oof))
    print(f"\n==== {MODEL_NAME}  CV AUC = {cv:.5f} ====")
    save_predictions(MODEL_NAME, oof, test_sum / C.N_FOLDS, meta=dict(
        cv_auc=cv, seed=C.SEED, n_folds=C.N_FOLDS, feature_set="full",
        created_by="hyunbean", notes="LogisticRegression (scaled, C=1)"))


if __name__ == "__main__":
    main()
