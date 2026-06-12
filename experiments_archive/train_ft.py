"""Neural tabular teacher (FT-Transformer / MLP) — pseudo teacher diversity용.

GPT/딥리서치 결론: 우리 pseudo teacher는 전부 GBDT(축정렬 split)라 어떤 변형도 corr~1.0 수렴.
FT-Transformer는 고차 상호작용을 다르게 봐서 test에서 *다른 샘플*을 고신뢰로 뽑음 → 다른 pseudo set.
단독 모델로는 캡(0.69~0.71)일 수 있으나 **teacher diversity**가 목적.

verify-first: 먼저 OOF/test 만들고 AUC + 기존 pseudo와 corr 확인 → 직교면 teacher로 투입.

실행(GPU): FT_ARCH=ft FT_GPU=1 python -m src.train_ft
            FT_ARCH=mlp ... (FT-Transformer 미설치 시 자동 fallback)
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # 단편화 완화
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import roc_auc_score

from . import config as C
from .oof_io import save_predictions
from .train_first import build_all

ARCH = os.environ.get("FT_ARCH", "ft")
GPU = os.environ.get("FT_GPU", "1") == "1"
EPOCHS = int(os.environ.get("FT_EPOCHS", "120"))
PATIENCE = int(os.environ.get("FT_PATIENCE", "16"))
BATCH = int(os.environ.get("FT_BATCH", "128"))   # 521피처 attention O(n^2) → batch 작게(OOM 방지)
FT_BLOCKS = int(os.environ.get("FT_BLOCKS", "2"))
FT_DBLOCK = int(os.environ.get("FT_DBLOCK", "128"))
LR = float(os.environ.get("FT_LR", "1e-4"))
WD = float(os.environ.get("FT_WD", "1e-5"))


def _make_model(d, arch):
    import torch.nn as nn
    if arch == "ft":
        from rtdl_revisiting_models import FTTransformer
        return FTTransformer(
            n_cont_features=d, cat_cardinalities=[], d_out=1,
            n_blocks=FT_BLOCKS, d_block=FT_DBLOCK, attention_n_heads=8,
            attention_dropout=0.2, ffn_d_hidden_multiplier=4 / 3,
            ffn_dropout=0.1, residual_dropout=0.0,
        ), "ft"
    # fallback: numerical-embedding MLP (여전히 GBDT와 다른 표현)
    class MLP(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(256, 64), nn.BatchNorm1d(64), nn.ReLU(),
                nn.Linear(64, 1),
            )

        def forward(self, x_cont, x_cat=None):
            return self.net(x_cont).squeeze(-1)
    return MLP(d), "mlp"


def _fit_fold(Xtr, ytr, Xva, yva, Xt, arch, dev):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    model, kind = _make_model(Xtr.shape[1], arch)
    model = model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    lossf = torch.nn.BCEWithLogitsLoss()
    tl = DataLoader(TensorDataset(torch.tensor(Xtr), torch.tensor(ytr)),
                    batch_size=BATCH, shuffle=True, drop_last=True)
    Xva_t = torch.tensor(Xva).to(dev); Xt_t = torch.tensor(Xt).to(dev)

    def _pred(M, X):
        M.eval()
        out = []
        ib = 512 if kind == "ft" else 4096          # FT는 attention O(n_feat^2)라 inference도 작게(OOM방지)
        with torch.no_grad():
            for i in range(0, len(X), ib):
                xb = X[i:i + ib]
                logit = M(xb, None) if kind == "ft" else M(xb)
                out.append(torch.sigmoid(logit).float().cpu().numpy().ravel())
        return np.concatenate(out)

    best_auc, best_va, best_t, bad = -1, None, None, 0
    for ep in range(EPOCHS):
        model.train()
        for xb, yb in tl:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            logit = model(xb, None) if kind == "ft" else model(xb)
            loss = lossf(logit.ravel(), yb)
            loss.backward()
            opt.step()
        va = _pred(model, Xva_t)
        auc = roc_auc_score(yva, va)
        if auc > best_auc + 1e-5:
            best_auc, best_va, best_t, bad = auc, va, _pred(model, Xt_t), 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    return best_va, best_t, best_auc


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
    print(f"[ft] dev={dev} arch={ARCH} X={X.shape}")

    oof = np.full(len(y), np.nan, np.float32)
    test_sum = np.zeros(len(test_ids), np.float64)
    used_arch = ARCH
    for f in range(C.N_FOLDS):
        tri, va = np.where(folds != f)[0], np.where(folds == f)[0]
        qt = QuantileTransformer(output_distribution="normal", subsample=1_000_000,
                                 random_state=C.SEED).fit(X[tri])      # neural엔 quantile이 표준
        Xtr, Xva, Xte = qt.transform(X[tri]).astype(np.float32), qt.transform(X[va]).astype(np.float32), qt.transform(Xt).astype(np.float32)
        try:
            vp, tp, auc = _fit_fold(Xtr, y[tri], Xva, y[va], Xte, ARCH, dev)
        except ImportError as e:
            print(f"  (FT-Transformer 미설치: {e} → MLP fallback)")
            used_arch = "mlp"
            vp, tp, auc = _fit_fold(Xtr, y[tri], Xva, y[va], Xte, "mlp", dev)
        oof[va] = vp; test_sum += tp
        print(f"  [fold {f}] AUC={auc:.5f}")
    cv = float(roc_auc_score(y, oof))
    name = f"ft_{used_arch}"
    print(f"==== {name}  CV={cv:.5f} ====")
    # 기존 pseudo와 corr (직교성 즉시 확인)
    from scipy.stats import rankdata
    for m in ["first_xgb_pl2", "first_lgbm_pl2", "mh_bestblend69"]:
        p = f"artifacts/oof/{m}__oof.npy"
        if os.path.exists(p):
            c = np.corrcoef(rankdata(oof), rankdata(np.load(p)))[0, 1]
            print(f"  corr(oof, {m})={c:.4f}")
    save_predictions(name, oof, test_sum / C.N_FOLDS, meta=dict(cv_auc=cv, seed=C.SEED,
                     n_folds=C.N_FOLDS, feature_set="1등FE 521 + quantile", created_by="hyunbean",
                     notes=f"neural tabular teacher({used_arch}), pseudo teacher diversity용"))


if __name__ == "__main__":
    main()
