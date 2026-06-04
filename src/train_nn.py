"""Tabular MLP (PyTorch, GPU) — 앙상블 다양성용 (GBDT와 다른 계열).

표준화(StandardScaler) + MLP(BatchNorm/Dropout) + 5-fold OOF + 체크포인트.
결과: mlp_full OOF/test npy. NN은 단독 성능보다 '다양성'으로 앙상블에 기여.

실행(Colab GPU 권장):
  python -m src.folds        # (선행)
  python -m src.train_nn
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from . import config as C
from .features import build_features
from .checkpoint import Checkpoint
from .oof_io import save_predictions

MODEL_NAME = "mlp_full"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS, BATCH, PATIENCE, LR = 60, 512, 10, 1e-3


class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(d),
            nn.Linear(d, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def _train_fold(Xtr, ytr, Xva, yva, Xte):
    torch.manual_seed(C.SEED)
    sc = StandardScaler().fit(Xtr)
    Xtr_t = torch.tensor(sc.transform(Xtr), dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.float32)
    Xva_t = torch.tensor(sc.transform(Xva), dtype=torch.float32, device=DEVICE)
    Xte_t = torch.tensor(sc.transform(Xte), dtype=torch.float32, device=DEVICE)
    dl = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(Xtr_t, ytr_t), batch_size=BATCH, shuffle=True)

    model = MLP(Xtr.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    lossf = nn.BCEWithLogitsLoss()

    best_auc, best_va, best_te, wait = -1.0, None, None, 0
    for _ in range(EPOCHS):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            lossf(model(xb), yb).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            va_p = torch.sigmoid(model(Xva_t)).cpu().numpy()
        auc = roc_auc_score(yva, va_p)
        if auc > best_auc:
            best_auc, wait = auc, 0
            with torch.no_grad():
                best_va = va_p
                best_te = torch.sigmoid(model(Xte_t)).cpu().numpy()
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    return best_va, best_te, best_auc


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
        va_p, te_p, auc = _train_fold(Xv[tr], y[tr], Xv[va], y[va], Xtv)
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
        notes=f"PyTorch MLP 256-128 ({DEVICE})"))
    ckpt.cleanup()


if __name__ == "__main__":
    main()
