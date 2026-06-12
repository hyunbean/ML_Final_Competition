"""멀티채널 시퀀스 GRU (PyTorch GPU) — brd_nm + part_nm + 금액버킷 3채널.

seq_gru(brd 1채널, 0.656)보다 신호↑ 기대. 여전히 '순서' 표현이라 집계모델과 decorrelated.
features 캐시 불필요(raw 거래만). 5-fold 고객 단위.

실행: python -m src.folds (선행) → python -m src.train_seq2
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from . import config as C
from .oof_io import save_predictions

MODEL_NAME = "seq_gru2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHANNELS = ["brd_nm", "part_nm"]          # 범주형 채널
AMT_BINS = 12
MAXLEN, HID = 60, 96
EMB = {"brd_nm": 24, "part_nm": 12, "amt": 6}
EPOCHS, BATCH, PATIENCE, LR = 25, 256, 5, 2e-3


def _encode(tr, te):
    tr = tr.copy(); te = te.copy()
    tr["sales_datetime"] = pd.to_datetime(tr["sales_datetime"])
    te["sales_datetime"] = pd.to_datetime(te["sales_datetime"])
    vocabs = {}
    for c in CHANNELS:
        vocab = {v: i + 1 for i, v in enumerate(pd.concat([tr[c], te[c]]).astype(str).unique())}
        tr[c] = tr[c].astype(str).map(vocab).astype(int)
        te[c] = te[c].astype(str).map(vocab).astype(int)
        vocabs[c] = len(vocab)
    edges = pd.qcut(tr["net_amt"], AMT_BINS, retbins=True, duplicates="drop")[1]
    edges[0], edges[-1] = -np.inf, np.inf
    tr["amt"] = pd.cut(tr["net_amt"], edges, labels=False).astype(int) + 1
    te["amt"] = pd.cut(te["net_amt"], edges, labels=False).fillna(0).astype(int) + 1
    vocabs["amt"] = AMT_BINS + 2
    return tr, te, vocabs


def _pad(df, ids):
    cols = CHANNELS + ["amt"]
    df = df.sort_values([C.ID_COL, "sales_datetime"])
    arrs = {c: np.zeros((len(ids), MAXLEN), dtype=np.int64) for c in cols}
    pos = {cid: i for i, cid in enumerate(ids)}
    for cid, g in df.groupby(C.ID_COL, sort=False):
        if cid not in pos:
            continue
        i = pos[cid]
        for c in cols:
            s = g[c].to_numpy()[-MAXLEN:]
            arrs[c][i, -len(s):] = s
    return np.stack([arrs[c] for c in cols], axis=2)   # (N, MAXLEN, 3)


class SeqNet(nn.Module):
    def __init__(self, vocabs):
        super().__init__()
        self.e_brd = nn.Embedding(vocabs["brd_nm"] + 1, EMB["brd_nm"], padding_idx=0)
        self.e_part = nn.Embedding(vocabs["part_nm"] + 1, EMB["part_nm"], padding_idx=0)
        self.e_amt = nn.Embedding(vocabs["amt"] + 1, EMB["amt"], padding_idx=0)
        d = sum(EMB.values())
        self.gru = nn.GRU(d, HID, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(nn.Linear(HID * 2, 96), nn.ReLU(), nn.Dropout(0.3), nn.Linear(96, 1))

    def forward(self, x):
        z = torch.cat([self.e_brd(x[:, :, 0]), self.e_part(x[:, :, 1]), self.e_amt(x[:, :, 2])], dim=2)
        o, _ = self.gru(z)
        return self.fc(o.mean(1)).squeeze(1)


def _train(Xtr, ytr, Xva, yva, Xte, vocabs):
    torch.manual_seed(C.SEED)
    dl = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.tensor(Xtr), torch.tensor(ytr, dtype=torch.float32)),
        batch_size=BATCH, shuffle=True)
    Xva_t, Xte_t = torch.tensor(Xva, device=DEVICE), torch.tensor(Xte, device=DEVICE)
    model = SeqNet(vocabs).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossf = nn.BCEWithLogitsLoss()
    best_auc, best_va, best_te, wait = -1.0, None, None, 0
    for _ in range(EPOCHS):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(); lossf(model(xb), yb).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vp = torch.sigmoid(model(Xva_t)).cpu().numpy()
        auc = roc_auc_score(yva, vp)
        if auc > best_auc:
            with torch.no_grad():
                best_te = torch.sigmoid(model(Xte_t)).cpu().numpy()
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

    use = [C.ID_COL, "sales_datetime", "net_amt"] + CHANNELS
    tr = pd.read_csv(C.TRAIN_CSV, encoding=C.ENCODING, usecols=use)
    te = pd.read_csv(C.TEST_CSV, encoding=C.ENCODING, usecols=use)
    tr, te, vocabs = _encode(tr, te)
    Xtr, Xte = _pad(tr, train_ids), _pad(te, test_ids)
    print(f"vocabs={vocabs}  X={Xtr.shape}  device={DEVICE}")

    oof = np.full(len(y), np.nan)
    test_sum = np.zeros(len(test_ids))
    for f in range(C.N_FOLDS):
        trn, val = np.where(folds != f)[0], np.where(folds == f)[0]
        va_p, te_p = _train(Xtr[trn], y[trn], Xtr[val], y[val], Xte, vocabs)
        oof[val] = va_p
        test_sum += te_p
        print(f"[fold {f}] AUC={roc_auc_score(y[val], va_p):.5f}")

    cv = float(roc_auc_score(y, oof))
    print(f"\n==== {MODEL_NAME}  CV AUC = {cv:.5f} ====")
    save_predictions(MODEL_NAME, oof, test_sum / C.N_FOLDS, meta=dict(
        cv_auc=cv, seed=C.SEED, n_folds=C.N_FOLDS, feature_set="seq:brd+part+amt",
        created_by="hyunbean", notes=f"multi-channel Bi-GRU (brd+part+amt, maxlen{MAXLEN})"))


if __name__ == "__main__":
    main()
