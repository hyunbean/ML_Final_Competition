"""SCARF: tabular contrastive representation (GPT #3) — pseudo teacher diversity용.

self-supervised pretrain을 train+test 전체로(transductive) → feature-corruption contrastive(NT-Xent)
→ encoder embedding → 분류기. GBDT가 못 보는 표현 + test 구조 반영 → corr 수렴을 깰 후보.

verify-first: emb 분류기 OOF/test 만들고 AUC + 기존 pseudo와 corr 즉시 출력.
실행(GPU): SCARF_GPU=1 python -m src.train_scarf
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

from . import config as C
from .oof_io import save_predictions
from .train_first import build_all

GPU = os.environ.get("SCARF_GPU", "1") == "1"
EPOCHS = int(os.environ.get("SCARF_EPOCHS", "150"))
BATCH = int(os.environ.get("SCARF_BATCH", "1024"))
EMB = int(os.environ.get("SCARF_EMB", "128"))
CORRUPT = float(os.environ.get("SCARF_CORRUPT", "0.6"))
TEMP = float(os.environ.get("SCARF_TEMP", "0.5"))
HEAD = os.environ.get("SCARF_HEAD", "lgbm")   # emb 위 분류기: lgbm | mlp


def _encoder(d):
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(d, 256), nn.BatchNorm1d(256), nn.ReLU(),
        nn.Linear(256, EMB), nn.BatchNorm1d(EMB), nn.ReLU(),
    )


def _proj():
    import torch.nn as nn
    return nn.Sequential(nn.Linear(EMB, EMB), nn.ReLU(), nn.Linear(EMB, 64))


def _corrupt(x, rate):
    import torch
    B, D = x.shape
    mask = torch.rand(B, D, device=x.device) < rate
    perm = torch.randint(0, B, (B, D), device=x.device)         # 배치 내 marginal 샘플
    cols = torch.arange(D, device=x.device).expand(B, D)
    x_rand = x[perm, cols]
    return torch.where(mask, x_rand, x)


def _ntxent(z1, z2, temp):
    import torch
    import torch.nn.functional as F
    z = torch.cat([z1, z2], 0)
    z = F.normalize(z, dim=1)
    n = z1.shape[0]
    sim = z @ z.T / temp
    sim.fill_diagonal_(-1e9)
    targets = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(sim, targets)


def _pretrain(Xall, dev):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    enc, proj = _encoder(Xall.shape[1]).to(dev), _proj().to(dev)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(proj.parameters()), lr=1e-3, weight_decay=1e-5)
    dl = DataLoader(TensorDataset(torch.tensor(Xall)), batch_size=BATCH, shuffle=True, drop_last=True)
    for ep in range(EPOCHS):
        enc.train(); proj.train(); tot = 0.0
        for (xb,) in dl:
            xb = xb.to(dev)
            v1, v2 = _corrupt(xb, CORRUPT), _corrupt(xb, CORRUPT)
            loss = _ntxent(proj(enc(v1)), proj(enc(v2)), TEMP)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        if ep % 20 == 0 or ep == EPOCHS - 1:
            print(f"  [pretrain ep{ep}] loss={tot/len(dl):.4f}")
    enc.eval()
    with torch.no_grad():
        emb = []
        Xt = torch.tensor(Xall).to(dev)
        for i in range(0, len(Xt), 8192):
            emb.append(enc(Xt[i:i + 8192]).cpu().numpy())
    return np.concatenate(emb).astype(np.float32)


def _head_oof(E, Et, y, folds):
    oof = np.full(len(y), np.nan, np.float32); test_sum = np.zeros(len(Et))
    for f in range(C.N_FOLDS):
        tri, va = np.where(folds != f)[0], np.where(folds == f)[0]
        if HEAD == "lgbm":
            import lightgbm as lgb
            m = lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.02, num_leaves=48,
                                   min_child_samples=40, subsample=0.8, subsample_freq=1,
                                   colsample_bytree=0.8, reg_lambda=5.0, random_state=C.SEED,
                                   n_jobs=-1, verbose=-1)
            m.fit(E[tri], y[tri], eval_set=[(E[va], y[va])],
                  callbacks=[lgb.early_stopping(120, verbose=False)])
            oof[va] = m.predict_proba(E[va])[:, 1]; test_sum += m.predict_proba(Et)[:, 1]
        else:
            from sklearn.linear_model import LogisticRegression
            m = LogisticRegression(max_iter=2000, C=1.0).fit(E[tri], y[tri])
            oof[va] = m.predict_proba(E[va])[:, 1]; test_sum += m.predict_proba(Et)[:, 1]
        print(f"  [fold {f}] AUC={roc_auc_score(y[va], oof[va]):.5f}")
    return oof, test_sum / C.N_FOLDS


def main():
    import torch
    dev = "cuda" if (GPU and torch.cuda.is_available()) else "cpu"
    train_ids = np.load(C.TRAIN_IDS_NPY, allow_pickle=True)
    test_ids = np.load(C.TEST_IDS_NPY, allow_pickle=True)
    folds = np.load(C.FOLDS_NPY)
    allf, ydf, _, _ = build_all()
    X = allf.reindex(train_ids).fillna(0.0).to_numpy(np.float32)
    Xt = allf.reindex(test_ids).fillna(0.0).to_numpy(np.float32)
    y = ydf.set_index("custid").reindex(train_ids)["gender"].to_numpy().astype(np.float32)
    # quantile fit on train+test (transductive pretrain이니 함께)
    qt = QuantileTransformer(output_distribution="normal", subsample=1_000_000,
                             random_state=C.SEED).fit(np.vstack([X, Xt]))
    Xall = qt.transform(np.vstack([X, Xt])).astype(np.float32)
    print(f"[scarf] dev={dev} Xall={Xall.shape} emb={EMB} head={HEAD}")
    emb = _pretrain(Xall, dev)
    E, Et = emb[:len(X)], emb[len(X):]
    oof, test = _head_oof(E, Et, y, folds)
    cv = float(roc_auc_score(y, oof)); name = f"scarf_{HEAD}"
    print(f"==== {name}  CV={cv:.5f} ====")
    for m in ["first_xgb_pl2", "first_lgbm_pl2", "mh_bestblend69"]:
        p = f"artifacts/oof/{m}__oof.npy"
        if os.path.exists(p):
            print(f"  corr(oof, {m})={np.corrcoef(rankdata(oof), rankdata(np.load(p)))[0,1]:.4f}")
    save_predictions(name, oof, test, meta=dict(cv_auc=cv, seed=C.SEED, n_folds=C.N_FOLDS,
                     feature_set="SCARF contrastive emb(transductive pretrain)", created_by="hyunbean",
                     notes="SCARF self-supervised(train+test) → emb 분류기, teacher diversity용"))


if __name__ == "__main__":
    main()
