"""RealMLP multi-seed 평균 ('first_realmlp_s') — RealMLP를 3 seed로 각 fold 학습→평균. robust+미세게인.
실행(GPU): pip install pytabkit gensim catboost → python -m src.train_realmlp_seeds
"""
import numpy as np
from sklearn.metrics import roc_auc_score
from . import config as C
from .oof_io import save_predictions
from .train_first import build_all

MODEL_NAME = "first_realmlp_s"
SEEDS = [42, 7, 123]


def main():
    from pytabkit import RealMLP_TD_Classifier
    train_ids = np.load(C.TRAIN_IDS_NPY, allow_pickle=True)
    test_ids = np.load(C.TEST_IDS_NPY, allow_pickle=True)
    folds = np.load(C.FOLDS_NPY)
    allf, ydf, _, _ = build_all()
    X = np.nan_to_num(allf.reindex(train_ids).fillna(0.0).to_numpy(np.float32), posinf=0, neginf=0)
    Xt = np.nan_to_num(allf.reindex(test_ids).fillna(0.0).to_numpy(np.float32), posinf=0, neginf=0)
    y = ydf.set_index("custid").reindex(train_ids)["gender"].to_numpy()
    print(f"RealMLP-seeds X={X.shape}")
    oof = np.full(len(y), np.nan); ts = np.zeros(len(test_ids))
    for f in range(C.N_FOLDS):
        tri, va = np.where(folds != f)[0], np.where(folds == f)[0]
        vp = np.zeros(len(va)); tp = np.zeros(len(test_ids))
        for s in SEEDS:
            m = RealMLP_TD_Classifier(n_cv=1, random_state=s, device="cuda")
            m.fit(X[tri], y[tri])
            vp += m.predict_proba(X[va])[:, 1] / len(SEEDS)
            tp += m.predict_proba(Xt)[:, 1] / len(SEEDS)
        oof[va] = vp; ts += tp
        print(f"[fold {f}] AUC={roc_auc_score(y[va], oof[va]):.5f}")
    cv = float(roc_auc_score(y, oof)); print(f"\n==== {MODEL_NAME}  CV AUC = {cv:.5f} ====")
    save_predictions(MODEL_NAME, oof, ts / C.N_FOLDS, meta=dict(cv_auc=cv, seed=C.SEED, n_folds=C.N_FOLDS,
                     feature_set="1등 FE RealMLP 3seed avg", created_by="hyunbean", notes="RealMLP seed-averaged"))


if __name__ == "__main__":
    main()
