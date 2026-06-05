"""시퀀스 NN (Bi-GRU, PyTorch GPU) — 고객 구매 시퀀스(brd_nm 순서)로 학습.

입력이 '집계'가 아니라 '순서있는 시퀀스' → 기존 모델과 상관 낮음 = 큰 다양성.
features 캐시 불필요(raw 거래만). 5-fold 고객 단위.

실행: python -m src.folds (선행) → python -m src.train_seq
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from . import config as C
from .oof_io import save_predictions

MODEL_NAME = "seq_gru"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEQ_COL = "brd_nm"
MAXLEN, EMB, HID = 50, 32, 64
EPOCHS, BATCH, PATIENCE, LR = 25, 256, 5, 2e-3


def _seqs(df, vocab):
    df = df[[C.ID_COL, "sales_datetime", SEQ_COL]].copy()
    df["sales_datetime"] = pd.to_datetime(df["sales_datetime"])
    df = df.sort_values([C.ID_COL, "sales_datetime"])
    df[SEQ_COL] = df[SEQ_COL].astype(str).map(vocab).fillna(0).astype(int)
    return df.groupby(C.ID_COL)[SEQ_COL].apply(lambda s: s.to_numpy()[-MAXLEN:])


def _pad(seqs, ids):
    arr = np.zeros((len(ids), MAXLEN), dtype=np.int64)
    for i, cid in enumerate(ids):
        s = seqs.get(cid, np.array([], dtype=int))
        if len(s):
            arr[i, -len(s):] = s
    return arr


class GRUNet(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.emb = nn.Embedding(vocab + 1, EMB, padding_idx=0)
        self.gru = nn.GRU(EMB, HID, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(nn.Linear(HID * 2, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1))

    def forward(self, x):
        o, _ = self.gru(self.emb(x))
        return self.fc(o.mean(1)).squeeze(1)


def _train(Xtr, ytr, Xva, yva, Xte, V):
    torch.manual_seed(C.SEED)
    dl = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.tensor(Xtr), torch.tensor(ytr, dtype=torch.float32)),
        batch_size=BATCH, shuffle=True)
    Xva_t = torch.tensor(Xva, device=DEVICE)
    Xte_t = torch.tensor(Xte, device=DEVICE)
    model = GRUNet(V).to(DEVICE)
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

    tr = pd.read_csv(C.TRAIN_CSV, encoding=C.ENCODING, usecols=[C.ID_COL, "sales_datetime", SEQ_COL])
    te = pd.read_csv(C.TEST_CSV, encoding=C.ENCODING, usecols=[C.ID_COL, "sales_datetime", SEQ_COL])
    vocab = {v: i + 1 for i, v in enumerate(pd.concat([tr[SEQ_COL], te[SEQ_COL]]).astype(str).unique())}
    Xtr = _pad(_seqs(tr, vocab), train_ids)
    Xte = _pad(_seqs(te, vocab), test_ids)
    V = len(vocab)
    print(f"vocab={V}  maxlen={MAXLEN}  device={DEVICE}")

    oof = np.full(len(y), np.nan)
    test_sum = np.zeros(len(test_ids))
    for f in range(C.N_FOLDS):
        trn, val = np.where(folds != f)[0], np.where(folds == f)[0]
        va_p, te_p = _train(Xtr[trn], y[trn], Xtr[val], y[val], Xte, V)
        oof[val] = va_p
        test_sum += te_p
        print(f"[fold {f}] AUC={roc_auc_score(y[val], va_p):.5f}")

    cv = float(roc_auc_score(y, oof))
    print(f"\n==== {MODEL_NAME}  CV AUC = {cv:.5f} ====")
    save_predictions(MODEL_NAME, oof, test_sum / C.N_FOLDS, meta=dict(
        cv_auc=cv, seed=C.SEED, n_folds=C.N_FOLDS, feature_set=f"seq:{SEQ_COL}",
        created_by="hyunbean", notes=f"Bi-GRU on {SEQ_COL} seq (maxlen{MAXLEN})"))


if __name__ == "__main__":
    main()
