"""DAE 표현 + NN 헤드 (Jahrer 원본 방식) — dae_lgbm(LGBM헤드)과 다른 헤드로 decorrelated.

train_dae의 DAE를 재사용해 은닉피처 추출 → MLP 헤드로 5-fold OOF.
실행(GPU): pip install lightgbm gensim → python -m src.folds → python -m src.train_dae_nn
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from . import config as C
from .features import build_features
from .oof_io import save_predictions
from .train_dae import gaussrank, DAE, swap_noise, HID, LAYERS, EPOCHS, BATCH, LR, DEVICE

MODEL_NAME = "dae_nn"
HEAD_EPOCHS, HEAD_BATCH, HEAD_LR = 40, 256, 1e-3


def _extract_dae(allX):
    torch.manual_seed(C.SEED)
    rng = np.random.default_rng(C.SEED)
    data = torch.tensor(allX, device=DEVICE)
    model = DAE(allX.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossf = nn.MSELoss()
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(data), device=DEVICE)
        for i in range(0, len(data), BATCH):
            xb = data[perm[i:i + BATCH]]
            opt.zero_grad(); lossf(model(swap_noise(xb, rng)), xb).backward(); opt.step()
    model.eval()
    with torch.no_grad():
        feats = np.zeros((len(data), HID * LAYERS), dtype=np.float32)
        for i in range(0, len(data), 1024):
            feats[i:i + 1024] = model(data[i:i + 1024], return_hidden=True).cpu().numpy()
    return feats


class Head(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
                                 nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x).squeeze(1)


@torch.no_grad()
def _pred(m, X, bs=1024):
    m.eval(); out = []
    for i in range(0, len(X), bs):
        out.append(torch.sigmoid(m(torch.tensor(X[i:i + bs], device=DEVICE))).cpu().numpy())
    return np.concatenate(out)


def main():
    X, y, Xtest = build_features()
    folds = np.load(C.FOLDS_NPY)
    n_tr = len(X)
    allX = np.vstack([np.nan_to_num(X.values, posinf=0, neginf=0),
                      np.nan_to_num(Xtest.values, posinf=0, neginf=0)]).astype(np.float32)
    allX = gaussrank(allX)
    print(f"DAE 추출 d={allX.shape[1]} device={DEVICE}")
    feats = _extract_dae(allX)
    Ftr, Fte = feats[:n_tr], feats[n_tr:]
    mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-6
    Ftr = (Ftr - mu) / sd; Fte = (Fte - mu) / sd

    oof = np.full(len(y), np.nan); test_sum = np.zeros(len(Fte))
    for f in range(C.N_FOLDS):
        tr, va = np.where(folds != f)[0], np.where(folds == f)[0]
        torch.manual_seed(C.SEED + f)
        dl = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.tensor(Ftr[tr]), torch.tensor(y[tr], dtype=torch.float32)),
            batch_size=HEAD_BATCH, shuffle=True)
        m = Head(Ftr.shape[1]).to(DEVICE)
        opt = torch.optim.Adam(m.parameters(), lr=HEAD_LR, weight_decay=1e-5)
        lossf = nn.BCEWithLogitsLoss()
        best = (-1, None, None); wait = 0
        for ep in range(HEAD_EPOCHS):
            m.train()
            for xb, yb in dl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad(); lossf(m(xb), yb).backward(); opt.step()
            vp = _pred(m, Ftr[va]); a = roc_auc_score(y[va], vp)
            if a > best[0]:
                best = (a, vp, _pred(m, Fte)); wait = 0
            else:
                wait += 1
                if wait >= 6:
                    break
        oof[va] = best[1]; test_sum += best[2]
        print(f"[fold {f}] AUC={best[0]:.5f}")
    cv = float(roc_auc_score(y, oof))
    print(f"\n==== {MODEL_NAME}  CV AUC = {cv:.5f} ====")
    save_predictions(MODEL_NAME, oof, test_sum / C.N_FOLDS, meta=dict(
        cv_auc=cv, seed=C.SEED, n_folds=C.N_FOLDS, feature_set="DAE hidden + MLP head",
        created_by="hyunbean", notes="DAE features + NN head (Jahrer style)"))


if __name__ == "__main__":
    main()
