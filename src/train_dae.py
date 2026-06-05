"""DAE (Denoising Autoencoder) 표현학습 — Porto Seguro 1등(Michael Jahrer) 기법.

핵심: 약한 모델을 더 만드는 게 아니라, 풀피처를 GaussRank+swap noise로 손상→복원 학습한
은닉활성(강하면서 원본과 decorrelated한 새 피처공간)을 만들어 그 위에서 LGBM 학습.
→ 기존 모델과 상관 낮은데 성능 강함 = 블렌드에서 weight 먹는 다양성.

비지도 DAE는 train+test 전체로 학습(누수 없음). 지도 헤드만 fold-safe.
실행: pip install lightgbm gensim → python -m src.folds → python -m src.train_dae
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.special import erfinv
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

from . import config as C
from .features import build_features
from .oof_io import save_predictions

MODEL_NAME = "dae_lgbm"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HID, LAYERS = 1500, 3
SWAP, EPOCHS, BATCH, LR = 0.15, 150, 128, 3e-3
LGB_PARAMS = dict(objective="binary", metric="auc", learning_rate=0.02, num_leaves=63,
                  feature_fraction=0.3, bagging_fraction=0.7, bagging_freq=1,
                  min_child_samples=50, n_estimators=2000, n_jobs=-1, verbosity=-1)


def gaussrank(X):
    n = len(X)
    out = np.empty_like(X, dtype=np.float32)
    for j in range(X.shape[1]):
        r = pd.Series(X[:, j]).rank(method="average").to_numpy()
        r = np.clip((r - 0.5) / n, 1e-6, 1 - 1e-6)
        out[:, j] = (np.sqrt(2) * erfinv(2 * r - 1)).astype(np.float32)
    return out


class DAE(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.enc, prev = nn.ModuleList(), d
        for _ in range(LAYERS):
            self.enc.append(nn.Linear(prev, HID)); prev = HID
        self.out = nn.Linear(HID, d)
        self.act = nn.ReLU()

    def forward(self, x, return_hidden=False):
        h, hs = x, []
        for l in self.enc:
            h = self.act(l(h)); hs.append(h)
        return torch.cat(hs, 1) if return_hidden else self.out(h)


def swap_noise(x, rng):
    B, d = x.shape
    mask = torch.rand(B, d, device=x.device) < SWAP
    idx = torch.randint(0, B, (B, d), device=x.device)
    cols = torch.arange(d, device=x.device).repeat(B, 1)
    return torch.where(mask, x[idx, cols], x)


def main():
    X, y, Xtest = build_features()
    folds = np.load(C.FOLDS_NPY)
    n_tr = len(X)
    allX = np.vstack([np.nan_to_num(X.values, posinf=0, neginf=0),
                      np.nan_to_num(Xtest.values, posinf=0, neginf=0)]).astype(np.float32)
    allX = gaussrank(allX)
    d = allX.shape[1]
    print(f"DAE: rows={len(allX)} d={d} device={DEVICE}")

    torch.manual_seed(C.SEED)
    rng = np.random.default_rng(C.SEED)
    data = torch.tensor(allX, device=DEVICE)
    model = DAE(d).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossf = nn.MSELoss()
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(data), device=DEVICE)
        tot = 0.0
        for i in range(0, len(data), BATCH):
            xb = data[perm[i:i + BATCH]]
            xc = swap_noise(xb, rng)
            opt.zero_grad()
            loss = lossf(model(xc), xb)
            loss.backward(); opt.step()
            tot += loss.item() * len(xb)
        if ep % 20 == 0 or ep == EPOCHS - 1:
            print(f"  epoch {ep:3d}  recon MSE={tot / len(data):.4f}")

    model.eval()
    with torch.no_grad():
        feats = np.zeros((len(data), HID * LAYERS), dtype=np.float32)
        for i in range(0, len(data), 1024):
            feats[i:i + 1024] = model(data[i:i + 1024], return_hidden=True).cpu().numpy()
    Ftr, Fte = feats[:n_tr], feats[n_tr:]
    print(f"DAE 피처 {feats.shape} → LGBM 학습")

    oof = np.full(len(y), np.nan)
    test_sum = np.zeros(len(Fte))
    for f in range(C.N_FOLDS):
        tr, va = np.where(folds != f)[0], np.where(folds == f)[0]
        m = lgb.LGBMClassifier(**LGB_PARAMS)
        m.fit(Ftr[tr], y[tr], eval_set=[(Ftr[va], y[va])],
              callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[va] = m.predict_proba(Ftr[va])[:, 1]
        test_sum += m.predict_proba(Fte)[:, 1]
        print(f"[fold {f}] AUC={roc_auc_score(y[va], oof[va]):.5f}")

    cv = float(roc_auc_score(y, oof))
    print(f"\n==== {MODEL_NAME}  CV AUC = {cv:.5f} ====")
    save_predictions(MODEL_NAME, oof, test_sum / C.N_FOLDS, meta=dict(
        cv_auc=cv, seed=C.SEED, n_folds=C.N_FOLDS, feature_set=f"DAE({LAYERS}x{HID}) hidden + LGBM",
        created_by="hyunbean", notes="Denoising AutoEncoder (swap noise + GaussRank) features → LGBM"))


if __name__ == "__main__":
    main()
