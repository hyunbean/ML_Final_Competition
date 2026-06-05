"""kNN (CPU) — 앙상블 다양성용 (거리 기반, 트리/선형과 완전 다른 bias).
표준화 + PCA(30) 후 kNN. 5-fold OOF. 결과: knn_full.
실행: python -m src.train_knn
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score

from . import config as C
from .features import build_features
from .oof_io import save_predictions

MODEL_NAME = "knn_full"
N_NEIGHBORS = 200
N_PCA = 30


def main():
    X, y, Xtest = build_features()
    folds = np.load(C.FOLDS_NPY)
    Xv = np.nan_to_num(X.values.astype(np.float32), posinf=0.0, neginf=0.0)
    Xtv = np.nan_to_num(Xtest.values.astype(np.float32), posinf=0.0, neginf=0.0)
    print(f"X={X.shape}  k={N_NEIGHBORS} pca={N_PCA}")

    oof = np.full(len(y), np.nan)
    test_sum = np.zeros(len(Xtv))
    for f in range(C.N_FOLDS):
        tr, va = np.where(folds != f)[0], np.where(folds == f)[0]
        sc = StandardScaler().fit(Xv[tr])
        pca = PCA(n_components=N_PCA, random_state=C.SEED).fit(sc.transform(Xv[tr]))
        Ztr = pca.transform(sc.transform(Xv[tr]))
        Zva = pca.transform(sc.transform(Xv[va]))
        Zte = pca.transform(sc.transform(Xtv))
        m = KNeighborsClassifier(n_neighbors=N_NEIGHBORS, weights="distance", n_jobs=-1)
        m.fit(Ztr, y[tr])
        oof[va] = m.predict_proba(Zva)[:, 1]
        test_sum += m.predict_proba(Zte)[:, 1]
        print(f"[fold {f}] AUC={roc_auc_score(y[va], oof[va]):.5f}")

    cv = float(roc_auc_score(y, oof))
    print(f"\n==== {MODEL_NAME}  CV AUC = {cv:.5f} ====")
    save_predictions(MODEL_NAME, oof, test_sum / C.N_FOLDS, meta=dict(
        cv_auc=cv, seed=C.SEED, n_folds=C.N_FOLDS, feature_set=f"full+pca{N_PCA}",
        created_by="hyunbean", notes=f"kNN k={N_NEIGHBORS} distance, PCA{N_PCA}"))


if __name__ == "__main__":
    main()
