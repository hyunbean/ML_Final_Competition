"""P2: 강한 멀티필드 트랜잭션 시퀀스 트랜스포머 (CASPR/Transformers4Rec식).

mega(집계)와 근본적으로 직교한 '순서' 표현 → 블렌드 weight 1순위 후보.
필드 7개(brd+part+corner+pc+amount+hour+weekday)를 토큰화해 강한 트랜스포머로 학습.
seq_transformer(0.664)보다 크고 깊게 → 0.70급 목표. raw 거래만, 5-fold OOF.
실행(GPU): python -m src.folds → python -m src.train_seq_caspr
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from . import config as C
from .oof_io import save_predictions

MODEL_NAME = "seq_caspr"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CAT = ["brd_nm", "part_nm", "corner_nm", "pc_nm"]
EMB = {"brd_nm": 32, "part_nm": 12, "corner_nm": 16, "pc_nm": 16, "amt": 8, "hour": 8, "wd": 6}
AMT_BINS, MAXLEN, DMODEL, HEADS, LAYERS = 12, 80, 128, 8, 3
EPOCHS, BATCH, PATIENCE, LR = 30, 256, 6, 1e-3


def _encode(tr, te):
    for df in (tr, te):
        df["sales_datetime"] = pd.to_datetime(df["sales_datetime"])
        df["hour"] = df["sales_datetime"].dt.hour
        df["wd"] = df["sales_datetime"].dt.weekday
    vocabs = {}
    for c in CAT:
        vocab = {v: i + 1 for i, v in enumerate(pd.concat([tr[c], te[c]]).astype(str).unique())}
        tr[c] = tr[c].astype(str).map(vocab).astype(int); te[c] = te[c].astype(str).map(vocab).astype(int)
        vocabs[c] = len(vocab)
    edges = pd.qcut(tr["net_amt"], AMT_BINS, retbins=True, duplicates="drop")[1]
    edges[0], edges[-1] = -np.inf, np.inf
    for df in (tr, te):
        df["amt"] = pd.cut(df["net_amt"], edges, labels=False).fillna(0).astype(int) + 1
        df["hour"] = df["hour"] + 1
        df["wd"] = df["wd"] + 1
    vocabs.update(amt=AMT_BINS + 2, hour=26, wd=9)
    return tr, te, vocabs


ALL = CAT + ["amt", "hour", "wd"]


def _pad(df, ids):
    df = df.sort_values([C.ID_COL, "sales_datetime"])
    arrs = {c: np.zeros((len(ids), MAXLEN), dtype=np.int64) for c in ALL}
    pos = {cid: i for i, cid in enumerate(ids)}
    for cid, g in df.groupby(C.ID_COL, sort=False):
        if cid not in pos:
            continue
        i = pos[cid]
        for c in ALL:
            s = g[c].to_numpy()[-MAXLEN:]
            arrs[c][i, -len(s):] = s
    return np.stack([arrs[c] for c in ALL], axis=2)


class CASPR(nn.Module):
    def __init__(self, vocabs):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(vocabs[c] + 1, EMB[c], padding_idx=0) for c in ALL])
        self.proj = nn.Linear(sum(EMB[c] for c in ALL), DMODEL)
        self.cls = nn.Parameter(torch.zeros(1, 1, DMODEL))
        self.pos = nn.Parameter(torch.zeros(1, MAXLEN + 1, DMODEL))
        layer = nn.TransformerEncoderLayer(DMODEL, HEADS, DMODEL * 4, dropout=0.2,
                                           batch_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, LAYERS)
        self.head = nn.Sequential(nn.LayerNorm(DMODEL), nn.Linear(DMODEL, 64), nn.GELU(),
                                  nn.Dropout(0.3), nn.Linear(64, 1))

    def forward(self, x):
        pad = (x[:, :, 0] == 0)
        z = torch.cat([emb(x[:, :, i]) for i, emb in enumerate(self.embs)], dim=2)
        z = self.proj(z)
        B = z.size(0)
        z = torch.cat([self.cls.expand(B, -1, -1), z], dim=1) + self.pos
        mask = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=x.device), pad], dim=1)
        h = self.enc(z, src_key_padding_mask=mask)
        return self.head(h[:, 0]).squeeze(1)


@torch.no_grad()
def _pred(model, X, bs=256):
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        xb = torch.tensor(X[i:i + bs], device=DEVICE)
        out.append(torch.sigmoid(model(xb)).cpu().numpy())
        del xb
    return np.concatenate(out)


def _train(Xtr, ytr, Xva, yva, Xte, vocabs):
    torch.manual_seed(C.SEED)
    dl = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.tensor(Xtr), torch.tensor(ytr, dtype=torch.float32)),
        batch_size=BATCH, shuffle=True)
    model = CASPR(vocabs).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()
    best_auc, best_va, best_te, wait = -1.0, None, None, 0
    for _ in range(EPOCHS):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(); lossf(model(xb), yb).backward(); opt.step()
        vp = _pred(model, Xva)              # 배치 추론(OOM 방지)
        auc = roc_auc_score(yva, vp)
        if auc > best_auc:
            best_te = _pred(model, Xte)
            best_auc, best_va, wait = auc, vp, 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    return best_va, best_te


def main():
    train_ids = np.load(C.TRAIN_IDS_NPY, allow_pickle=True)
    test_ids = np.load(C.TEST_IDS_NPY, allow_pickle=True)
    folds = np.load(C.FOLDS_NPY)
    y = pd.read_csv(C.YTRAIN_CSV).set_index(C.ID_COL).reindex(train_ids)[C.TARGET].to_numpy()
    use = [C.ID_COL, "sales_datetime", "net_amt"] + CAT
    tr = pd.read_csv(C.TRAIN_CSV, encoding=C.ENCODING, usecols=use)
    te = pd.read_csv(C.TEST_CSV, encoding=C.ENCODING, usecols=use)
    tr, te, vocabs = _encode(tr, te)
    Xtr, Xte = _pad(tr, train_ids), _pad(te, test_ids)
    print(f"vocabs={vocabs} X={Xtr.shape} device={DEVICE}")

    oof = np.full(len(y), np.nan); test_sum = np.zeros(len(test_ids))
    for f in range(C.N_FOLDS):
        trn, val = np.where(folds != f)[0], np.where(folds == f)[0]
        va_p, te_p = _train(Xtr[trn], y[trn], Xtr[val], y[val], Xte, vocabs)
        oof[val] = va_p; test_sum += te_p
        print(f"[fold {f}] AUC={roc_auc_score(y[val], va_p):.5f}")
    cv = float(roc_auc_score(y, oof))
    print(f"\n==== {MODEL_NAME}  CV AUC = {cv:.5f} ====")
    save_predictions(MODEL_NAME, oof, test_sum / C.N_FOLDS, meta=dict(
        cv_auc=cv, seed=C.SEED, n_folds=C.N_FOLDS, feature_set="seq 7fields CASPR-transformer",
        created_by="hyunbean", notes=f"CASPR-style multi-field tx transformer (maxlen{MAXLEN}, {LAYERS}L)"))


if __name__ == "__main__":
    main()
