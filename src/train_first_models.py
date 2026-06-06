"""1등 피처셋(497)에 다양한 모델 4종 → first_et/first_rf/first_mlp/first_lr.

first_cat/lgbm/xgb이 다 블렌드 올렸음 → 1등 피처셋 × 모델 다양성이 우리 금광.
doc(1등 kaggle)도 RF/ET/MLP/Linear 다 씀. sklearn이라 CPU로 빠름.
실행: pip install gensim catboost scikit-learn → python -m src.folds → python -m src.train_first_models
"""
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from . import config as C
from .oof_io import save_predictions
from .train_first import build_all


def _et(): return ExtraTreesClassifier(n_estimators=700, min_samples_leaf=5, max_features="sqrt", n_jobs=-1, random_state=C.SEED)
def _rf(): return RandomForestClassifier(n_estimators=700, min_samples_leaf=5, max_features="sqrt", n_jobs=-1, random_state=C.SEED)
def _mlp(): return MLPClassifier(hidden_layer_sizes=(256, 128), alpha=1e-3, learning_rate_init=1e-3,
                                 batch_size=256, max_iter=300, early_stopping=True, n_iter_no_change=12, random_state=C.SEED)
def _lr(): return LogisticRegression(C=0.5, max_iter=2000)

MODELS = {"first_et": (_et, False), "first_rf": (_rf, False),
          "first_mlp": (_mlp, True), "first_lr": (_lr, True)}


def main():
    train_ids = np.load(C.TRAIN_IDS_NPY, allow_pickle=True)
    test_ids = np.load(C.TEST_IDS_NPY, allow_pickle=True)
    folds = np.load(C.FOLDS_NPY)
    allf, y_df, _, _ = build_all()
    X = allf.reindex(train_ids).fillna(0.0).to_numpy(np.float32)
    Xt = allf.reindex(test_ids).fillna(0.0).to_numpy(np.float32)
    X = np.nan_to_num(X, posinf=0, neginf=0); Xt = np.nan_to_num(Xt, posinf=0, neginf=0)
    y = y_df.set_index("custid").reindex(train_ids)["gender"].to_numpy()
    print(f"X={X.shape}")

    for name, (ctor, scale) in MODELS.items():
        oof = np.full(len(y), np.nan); test_sum = np.zeros(len(test_ids))
        for f in range(C.N_FOLDS):
            tri, va = np.where(folds != f)[0], np.where(folds == f)[0]
            Xtr, Xva, Xte = X[tri], X[va], Xt
            if scale:
                sc = StandardScaler().fit(Xtr)
                Xtr, Xva, Xte = sc.transform(Xtr), sc.transform(Xva), sc.transform(Xt)
            m = ctor().fit(Xtr, y[tri])
            oof[va] = m.predict_proba(Xva)[:, 1]; test_sum += m.predict_proba(Xte)[:, 1]
        cv = float(roc_auc_score(y, oof))
        print(f"==== {name}  CV AUC = {cv:.5f} ====")
        save_predictions(name, oof, test_sum / C.N_FOLDS, meta=dict(
            cv_auc=cv, seed=C.SEED, n_folds=C.N_FOLDS, feature_set="작년1등 FE(497)",
            created_by="hyunbean", notes=f"1st-place FE + {name} 5fold OOF"))


if __name__ == "__main__":
    main()
