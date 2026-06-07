"""RealMLP (1등 피처셋) → 'first_realmlp'. 강한 NN(tuned defaults), 우리 미시도(tabm만 약식).

작년 1등도 RealMLP 씀. pytabkit RealMLP_TD = 튜닝불필요. 1등 497피처에 5fold OOF.
실행(GPU): pip install pytabkit gensim catboost → python -m src.folds → python -m src.train_realmlp
"""
import numpy as np
from sklearn.metrics import roc_auc_score

from . import config as C
from .oof_io import save_predictions
from .train_first import build_all

MODEL_NAME = "first_realmlp"


def main():
    from pytabkit import RealMLP_TD_Classifier
    train_ids = np.load(C.TRAIN_IDS_NPY, allow_pickle=True)
    test_ids = np.load(C.TEST_IDS_NPY, allow_pickle=True)
    folds = np.load(C.FOLDS_NPY)
    allf, ydf, _, _ = build_all()
    X = allf.reindex(train_ids).fillna(0.0).to_numpy(np.float32)
    Xt = allf.reindex(test_ids).fillna(0.0).to_numpy(np.float32)
    X = np.nan_to_num(X, posinf=0, neginf=0); Xt = np.nan_to_num(Xt, posinf=0, neginf=0)
    y = ydf.set_index("custid").reindex(train_ids)["gender"].to_numpy()
    print(f"RealMLP X={X.shape}")
    oof = np.full(len(y), np.nan); test_sum = np.zeros(len(test_ids))
    for f in range(C.N_FOLDS):
        tri, va = np.where(folds != f)[0], np.where(folds == f)[0]
        m = RealMLP_TD_Classifier(n_cv=1, n_refinement_epochs=0, random_state=C.SEED, device="cuda", val_metric_name="cross_entropy")
        m.fit(X[tri], y[tri])
        oof[va] = m.predict_proba(X[va])[:, 1]; test_sum += m.predict_proba(Xt)[:, 1]
        print(f"[fold {f}] AUC={roc_auc_score(y[va], oof[va]):.5f}")
    cv = float(roc_auc_score(y, oof))
    print(f"\n==== {MODEL_NAME}  CV AUC = {cv:.5f} ====")
    save_predictions(MODEL_NAME, oof, test_sum / C.N_FOLDS, meta=dict(
        cv_auc=cv, seed=C.SEED, n_folds=C.N_FOLDS, feature_set="작년1등 FE(497) RealMLP",
        created_by="hyunbean", notes="RealMLP tuned-defaults, 강한 NN 미시도"))


if __name__ == "__main__":
    main()
