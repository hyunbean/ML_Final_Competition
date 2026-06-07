"""RealTabR_D_Classifier (1등 피처셋) → 'first_tabr'. 강한모델×1등 방향, RealMLP/TabM과 다른 구조.
실행(GPU): pip install pytabkit gensim catboost → python -m src.train_first_tabr
"""
import numpy as np
from sklearn.metrics import roc_auc_score
from . import config as C
from .oof_io import save_predictions
from .train_first import build_all

MODEL_NAME = "first_tabr"


def main():
    from pytabkit import RealTabR_D_Classifier
    train_ids = np.load(C.TRAIN_IDS_NPY, allow_pickle=True)
    test_ids = np.load(C.TEST_IDS_NPY, allow_pickle=True)
    folds = np.load(C.FOLDS_NPY)
    allf, ydf, _, _ = build_all()
    X = np.nan_to_num(allf.reindex(train_ids).fillna(0.0).to_numpy(np.float32), posinf=0, neginf=0)
    Xt = np.nan_to_num(allf.reindex(test_ids).fillna(0.0).to_numpy(np.float32), posinf=0, neginf=0)
    y = ydf.set_index("custid").reindex(train_ids)["gender"].to_numpy()
    print(f" X={X.shape}")
    oof = np.full(len(y), np.nan); ts = np.zeros(len(test_ids))
    for f in range(C.N_FOLDS):
        tri, va = np.where(folds != f)[0], np.where(folds == f)[0]
        m = RealTabR_D_Classifier(n_cv=1, random_state=C.SEED, device="cuda")
        m.fit(X[tri], y[tri])
        oof[va] = m.predict_proba(X[va])[:, 1]; ts += m.predict_proba(Xt)[:, 1]
        print(f"[fold {f}] AUC={roc_auc_score(y[va], oof[va]):.5f}")
    cv = float(roc_auc_score(y, oof)); print(f"\n====   CV AUC = {cv:.5f} ====")
    save_predictions(MODEL_NAME, oof, ts / C.N_FOLDS, meta=dict(cv_auc=cv, seed=C.SEED, n_folds=C.N_FOLDS,
                     feature_set="1등 FE RealTabR_D_Classifier", created_by="hyunbean", notes="RealTabR_D_Classifier on 1st-place feats"))


if __name__ == "__main__":
    main()
