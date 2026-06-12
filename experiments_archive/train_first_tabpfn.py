"""TabPFN (1등 피처셋) → 'first_tabpfn'. foundation 모델, 강한모델×1등피처 방향.
30k는 TabPFN엔 크니 fold마다 stratified 서브샘플(여러번)→평균. 실행(GPU): pip install tabpfn
"""
import numpy as np
from sklearn.metrics import roc_auc_score
from . import config as C
from .oof_io import save_predictions
from .train_first import build_all

MODEL_NAME = "first_tabpfn"
SUB = 9000      # fold당 서브샘플 크기
BAG = 4         # 서브샘플 반복


def main():
    from tabpfn import TabPFNClassifier
    train_ids = np.load(C.TRAIN_IDS_NPY, allow_pickle=True)
    test_ids = np.load(C.TEST_IDS_NPY, allow_pickle=True)
    folds = np.load(C.FOLDS_NPY)
    allf, ydf, _, _ = build_all()
    X = np.nan_to_num(allf.reindex(train_ids).fillna(0.0).to_numpy(np.float32), posinf=0, neginf=0)
    Xt = np.nan_to_num(allf.reindex(test_ids).fillna(0.0).to_numpy(np.float32), posinf=0, neginf=0)
    y = ydf.set_index("custid").reindex(train_ids)["gender"].to_numpy()
    print(f"TabPFN X={X.shape}")
    oof = np.full(len(y), np.nan); ts = np.zeros(len(test_ids))
    for f in range(C.N_FOLDS):
        tri, va = np.where(folds != f)[0], np.where(folds == f)[0]
        vp = np.zeros(len(va)); tp = np.zeros(len(test_ids))
        for b in range(BAG):
            rng = np.random.RandomState(C.SEED + b)
            idx = rng.choice(tri, size=min(SUB, len(tri)), replace=False)
            m = TabPFNClassifier(device="cuda")
            m.fit(X[idx], y[idx])
            vp += m.predict_proba(X[va])[:, 1] / BAG
            tp += m.predict_proba(Xt)[:, 1] / BAG
        oof[va] = vp; ts += tp
        print(f"[fold {f}] AUC={roc_auc_score(y[va], oof[va]):.5f}")
    cv = float(roc_auc_score(y, oof)); print(f"\n==== {MODEL_NAME}  CV AUC = {cv:.5f} ====")
    save_predictions(MODEL_NAME, oof, ts / C.N_FOLDS, meta=dict(cv_auc=cv, seed=C.SEED, n_folds=C.N_FOLDS,
                     feature_set="1등 FE TabPFN(subsample bag)", created_by="hyunbean", notes="TabPFN on 1st-place feats"))


if __name__ == "__main__":
    main()
