"""TabNet (pytorch-tabnet, GPU) — 앙상블 다양성용 (attention 기반 tabular NN).

표준화(StandardScaler) + 5-fold OOF + 체크포인트. 결과: tabnet_full OOF/test npy.
MLP와 또 다른 구조라 앙상블 다양성에 기여.

설치: pip install pytorch-tabnet
실행(Colab GPU): python -m src.train_tabnet
"""
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from pytorch_tabnet.tab_model import TabNetClassifier

from . import config as C
from .features import build_features
from .checkpoint import Checkpoint
from .oof_io import save_predictions

MODEL_NAME = "tabnet_full"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _fold(Xtr, ytr, Xva, yva, Xte):
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xva_s, Xte_s = sc.transform(Xtr), sc.transform(Xva), sc.transform(Xte)
    clf = TabNetClassifier(
        seed=C.SEED, verbose=0, device_name=DEVICE,
        n_d=16, n_a=16, n_steps=4, gamma=1.5,
        optimizer_params=dict(lr=2e-2),
    )
    clf.fit(
        Xtr_s, ytr.astype(int), eval_set=[(Xva_s, yva.astype(int))],
        eval_metric=["auc"], max_epochs=100, patience=15,
        batch_size=1024, virtual_batch_size=128,
    )
    va = clf.predict_proba(Xva_s)[:, 1]
    te = clf.predict_proba(Xte_s)[:, 1]
    return va, te, roc_auc_score(yva, va)


def main():
    X, y, Xtest = build_features()
    folds = np.load(C.FOLDS_NPY)
    Xv, Xtv = X.values.astype(np.float32), Xtest.values.astype(np.float32)
    n_tr, n_te = len(X), len(Xtest)
    print(f"X={X.shape}  device={DEVICE}")

    ckpt = Checkpoint(MODEL_NAME, C.CKPT_DIR)
    state = ckpt.load(default=dict(
        done=[], oof=np.full(n_tr, np.nan, dtype=np.float64),
        test_sum=np.zeros(n_te, dtype=np.float64), fold_auc={}))

    for f in range(C.N_FOLDS):
        if f in state["done"]:
            continue
        tr, va = np.where(folds != f)[0], np.where(folds == f)[0]
        va_p, te_p, auc = _fold(Xv[tr], y[tr], Xv[va], y[va], Xtv)
        state["oof"][va] = va_p
        state["test_sum"] += te_p
        state["fold_auc"][str(f)] = float(auc)
        state["done"].append(f)
        ckpt.save(state)
        print(f"[fold {f}] AUC={auc:.5f}  done={state['done']}")

    oof = state["oof"]
    test_pred = state["test_sum"] / C.N_FOLDS
    cv = float(roc_auc_score(y, oof))
    print(f"\n==== {MODEL_NAME}  CV AUC = {cv:.5f} ====")

    save_predictions(MODEL_NAME, oof, test_pred, meta=dict(
        cv_auc=cv, fold_auc=state["fold_auc"], seed=C.SEED, n_folds=C.N_FOLDS,
        feature_set="full(te+emb+div+grp)", created_by="hyunbean",
        notes=f"TabNet ({DEVICE})"))
    ckpt.cleanup()


if __name__ == "__main__":
    main()
