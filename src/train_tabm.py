"""TabM (BatchEnsemble MLP) — 한 MLP 안에 k개 멤버 효율 앙상블. mean-of-loss 학습.

TabM 핵심(ICLR'25)을 직접 구현: 공유 backbone + 멤버별 rank-1 어댑터(r,s) → k개 예측의
평균. 단일 MLP보다 일관되게↑, GBDT와 대등. GaussRank 입력. 우리 피처, 5-fold OOF.
실행(GPU): python -m src.folds → python -m src.train_tabm
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.special import erfinv
from sklearn.metrics import roc_auc_score

from . import config as C
from .features import build_features
from .oof_io import save_predictions

MODEL_NAME = "tabm"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
K, HID, EPOCHS, BATCH, PAT, LR = 8, 256, 60, 256, 8, 2e-3


def gaussrank(X):
    out = np.empty_like(X, dtype=np.float32); n = len(X)
    for j in range(X.shape[1]):
        r = pd.Series(X[:, j]).rank(method="average").to_numpy()
        out[:, j] = (np.sqrt(2) * erfinv(np.clip(2 * (r - 0.5) / n - 1, -1 + 1e-6, 1 - 1e-6))).astype(np.float32)
    return out


class BELinear(nn.Module):
    def __init__(self, din, dout, k):
        super().__init__()
        self.lin = nn.Linear(din, dout, bias=False)
        self.r = nn.Parameter(torch.ones(k, din) + 0.1 * torch.randn(k, din))
        self.s = nn.Parameter(torch.ones(k, dout) + 0.1 * torch.randn(k, dout))
        self.b = nn.Parameter(torch.zeros(k, dout))

    def forward(self, x):                       # x: (k,B,din)
        return self.lin(x * self.r[:, None, :]) * self.s[:, None, :] + self.b[:, None, :]


class TabM(nn.Module):
    def __init__(self, d, k=K):
        super().__init__()
        self.k = k
        self.l1 = BELinear(d, HID, k); self.l2 = BELinear(HID, HID, k); self.out = BELinear(HID, 1, k)
        self.act = nn.ReLU(); self.dp = nn.Dropout(0.2)

    def forward(self, x):                        # x: (B,d)
        x = x.unsqueeze(0).expand(self.k, -1, -1)  # (k,B,d)
        h = self.dp(self.act(self.l1(x)))
        h = self.dp(self.act(self.l2(h)))
        return self.out(h).squeeze(-1)           # (k,B)


@torch.no_grad()
def _pred(m, X, bs=2048):
    m.eval(); out = []
    for i in range(0, len(X), bs):
        xb = torch.tensor(X[i:i + bs], device=DEVICE)
        out.append(torch.sigmoid(m(xb)).mean(0).cpu().numpy())   # k멤버 평균
    return np.concatenate(out)


def main():
    X, y, Xtest = build_features()
    folds = np.load(C.FOLDS_NPY)
    Xv = gaussrank(np.nan_to_num(X.values.astype(np.float32), posinf=0, neginf=0))
    Xtv = gaussrank(np.nan_to_num(Xtest.values.astype(np.float32), posinf=0, neginf=0))
    print(f"TabM X={Xv.shape} k={K} device={DEVICE}")

    oof = np.full(len(y), np.nan); test_sum = np.zeros(len(Xtv))
    lossf = nn.BCEWithLogitsLoss()
    for f in range(C.N_FOLDS):
        tr, va = np.where(folds != f)[0], np.where(folds == f)[0]
        torch.manual_seed(C.SEED + f)
        dl = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.tensor(Xv[tr]), torch.tensor(y[tr], dtype=torch.float32)),
            batch_size=BATCH, shuffle=True)
        m = TabM(Xv.shape[1]).to(DEVICE)
        opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=1e-4)
        best = (-1, None, None); wait = 0
        for ep in range(EPOCHS):
            m.train()
            for xb, yb in dl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                pred = m(xb)                                   # (k,B)
                loss = lossf(pred, yb.unsqueeze(0).expand_as(pred))  # mean-of-loss
                opt.zero_grad(); loss.backward(); opt.step()
            vp = _pred(m, Xv[va]); a = roc_auc_score(y[va], vp)
            if a > best[0]:
                best = (a, vp, _pred(m, Xtv)); wait = 0
            else:
                wait += 1
                if wait >= PAT:
                    break
        oof[va] = best[1]; test_sum += best[2]
        print(f"[fold {f}] AUC={best[0]:.5f}")
    cv = float(roc_auc_score(y, oof))
    print(f"\n==== {MODEL_NAME}  CV AUC = {cv:.5f} ====")
    save_predictions(MODEL_NAME, oof, test_sum / C.N_FOLDS, meta=dict(
        cv_auc=cv, seed=C.SEED, n_folds=C.N_FOLDS, feature_set=f"TabM BatchEnsemble k={K}",
        created_by="hyunbean", notes="TabM-style (BatchEnsemble MLP, mean-of-loss, GaussRank)"))


if __name__ == "__main__":
    main()
