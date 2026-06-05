"""ExtraTrees (CPU) — 앙상블 다양성용 (배깅 트리, 부스팅과 다른 bias).
5-fold OOF. 결과: et_full.
실행: python -m src.train_et
"""
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score

from . import config as C
from .features import build_features
from .oof_io import save_predictions

MODEL_NAME = "et_full"


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
        m = ExtraTreesClassifier(n_estimators=800, max_features="sqrt",
                                 min_samples_leaf=5, n_jobs=-1, random_state=C.SEED)
        m.fit(Xv[tr], y[tr])
        oof[va] = m.predict_proba(Xv[va])[:, 1]
        test_sum += m.predict_proba(Xtv)[:, 1]
        print(f"[fold {f}] AUC={roc_auc_score(y[va], oof[va]):.5f}")

    cv = float(roc_auc_score(y, oof))
    print(f"\n==== {MODEL_NAME}  CV AUC = {cv:.5f} ====")
    save_predictions(MODEL_NAME, oof, test_sum / C.N_FOLDS, meta=dict(
        cv_auc=cv, seed=C.SEED, n_folds=C.N_FOLDS, feature_set="full",
        created_by="hyunbean", notes="ExtraTrees 800"))


if __name__ == "__main__":
    main()
