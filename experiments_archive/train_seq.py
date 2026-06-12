"""거래 시퀀스 GRU ('seq_gru') — 집계 안 한 원본 거래 시간순 나열 → GRU+attention → 성별.

우리 모든 모델은 custid 집계 피처 기반. 이건 순서/타이밍을 직접 먹는 유일한 미시도 모델.
집계가 버린 신호 → 직교 AND 충분히 강함(0.70+) 노림 = frontier-law 통과 후보.
8 카테고리(part/pc/corner/brd/buyer/goodcd/inst/import) + 6 수치(amt/dis/hour/dow/month/delta) + attention.
실행(GPU): pip install torch → python -m src.folds → python -m src.train_seq
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import config as C

MODEL_NAME = "seq_gru"
MAXLEN = 100
CATS = ["part_nm", "pc_nm", "corner_nm", "brd_nm", "buyer_nm", "goodcd", "inst_mon", "import_flg"]
BATCH = 256
EPOCHS = 18
PATIENCE = 3


def _load_seqs():
    use = ["custid", "sales_datetime", "tot_amt", "dis_amt", "net_amt"] + CATS
    use = list(dict.fromkeys(use))
    tr = pd.read_csv(C.TRAIN_CSV, encoding=C.ENCODING, usecols=use)
    te = pd.read_csv(C.TEST_CSV, encoding=C.ENCODING, usecols=use)
    tr["_ds"] = "tr"; te["_ds"] = "te"
    a = pd.concat([tr, te], ignore_index=True)
    a["dt"] = pd.to_datetime(a["sales_datetime"], errors="coerce")
    a = a.sort_values(["custid", "dt"]).reset_index(drop=True)
    cardin = {}
    for c in CATS:
        codes, _ = pd.factorize(a[c].astype(str))
        a[c + "_i"] = codes + 1                       # 0 = pad
        cardin[c] = int(codes.max()) + 2
    a["amt_log"] = np.sign(a["net_amt"]) * np.log1p(a["net_amt"].abs())
    a["dis_ratio"] = (a["dis_amt"] / (a["tot_amt"].abs() + 1.0)).clip(0, 1)
    a["hour"] = a["dt"].dt.hour.fillna(12) / 23.0
    a["dow"] = a["dt"].dt.dayofweek.fillna(0) / 6.0
    a["month"] = (a["dt"].dt.month.fillna(1) - 1) / 11.0
    a["delta"] = a.groupby("custid")["dt"].diff().dt.total_seconds().fillna(0) / 86400.0
    a["delta_log"] = np.log1p(a["delta"].clip(lower=0))
    NUM = ["amt_log", "dis_ratio", "hour", "dow", "month", "delta_log"]
    trmask = a["_ds"] == "tr"
    for c in ["amt_log", "delta_log"]:
        mu, sd = a.loc[trmask, c].mean(), a.loc[trmask, c].std() + 1e-6
        a[c] = (a[c] - mu) / sd
    return a, cardin, NUM


def _build_arrays(a, ids, NUM):
    cat_cols = [c + "_i" for c in CATS]
    nC, nN = len(CATS), len(NUM)
    Xc = np.zeros((len(ids), MAXLEN, nC), np.int64)
    Xn = np.zeros((len(ids), MAXLEN, nN), np.float32)
    L = np.zeros(len(ids), np.int64)
    pos = {cid: i for i, cid in enumerate(ids)}
    catmat = a[cat_cols].to_numpy(np.int64); nummat = a[NUM].to_numpy(np.float32)
    for cid, idx in a.groupby("custid").indices.items():
        if cid not in pos:
            continue
        if len(idx) > MAXLEN:
            idx = idx[-MAXLEN:]
        i = pos[cid]; n = len(idx)
        Xc[i, :n] = catmat[idx]; Xn[i, :n] = nummat[idx]; L[i] = n
    return Xc, Xn, L


def main():
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(C.SEED)
    print(f"device={dev}")
    train_ids = np.load(C.TRAIN_IDS_NPY, allow_pickle=True)
    test_ids = np.load(C.TEST_IDS_NPY, allow_pickle=True)
    folds = np.load(C.FOLDS_NPY)
    y = pd.read_csv(C.YTRAIN_CSV).set_index(C.ID_COL).reindex(train_ids)[C.TARGET].to_numpy(np.float32)

    a, cardin, NUM = _load_seqs()
    print(f"거래 {len(a)}행, cardin={cardin}")
    Xc_tr, Xn_tr, L_tr = _build_arrays(a, list(train_ids), NUM)
    Xc_te, Xn_te, L_te = _build_arrays(a, list(test_ids), NUM)
    print(f"train seq {Xc_tr.shape}, 평균길이 {L_tr.mean():.1f}")

    class Net(nn.Module):
        def __init__(s):
            super().__init__()
            s.embs = nn.ModuleList([nn.Embedding(cardin[c], min(32, (cardin[c] + 1) // 2), padding_idx=0) for c in CATS])
            edim = sum(e.embedding_dim for e in s.embs)
            s.gru = nn.GRU(edim + len(NUM), 96, batch_first=True, bidirectional=True)
            s.att = nn.Linear(192, 1)
            s.head = nn.Sequential(nn.Linear(192, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1))

        def forward(s, xc, xn, length):
            e = torch.cat([emb(xc[:, :, i]) for i, emb in enumerate(s.embs)], -1)
            h, _ = s.gru(torch.cat([e, xn], -1))
            mask = (torch.arange(xc.size(1), device=xc.device)[None, :] < length[:, None]).float().unsqueeze(-1)
            sc = s.att(h).masked_fill(mask == 0, -1e9)
            w = torch.softmax(sc, 1)
            return s.head((h * w).sum(1)).squeeze(-1)

    def to_ds(Xc, Xn, L, yy=None):
        t = [torch.from_numpy(Xc), torch.from_numpy(Xn), torch.from_numpy(L)]
        if yy is not None:
            t.append(torch.from_numpy(yy))
        return TensorDataset(*t)

    te_dl = DataLoader(to_ds(Xc_te, Xn_te, L_te), batch_size=512)
    oof = np.full(len(y), np.nan); test_sum = np.zeros(len(test_ids))
    for f in range(C.N_FOLDS):
        tri, va = np.where(folds != f)[0], np.where(folds == f)[0]
        tr_dl = DataLoader(to_ds(Xc_tr[tri], Xn_tr[tri], L_tr[tri], y[tri]), batch_size=BATCH, shuffle=True)
        va_dl = DataLoader(to_ds(Xc_tr[va], Xn_tr[va], L_tr[va], y[va]), batch_size=512)
        net = Net().to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
        lossf = nn.BCEWithLogitsLoss()
        best_auc, best_oof, best_test, bad = 0, None, None, 0
        for ep in range(EPOCHS):
            net.train()
            for xc, xn, L, yy in tr_dl:
                opt.zero_grad()
                loss = lossf(net(xc.to(dev), xn.to(dev), L.to(dev)), yy.to(dev))
                loss.backward(); opt.step()
            net.eval(); ps = []
            with torch.no_grad():
                for xc, xn, L, yy in va_dl:
                    ps.append(torch.sigmoid(net(xc.to(dev), xn.to(dev), L.to(dev))).cpu().numpy())
            p = np.concatenate(ps); auc = roc_auc_score(y[va], p)
            print(f"  fold{f} ep{ep} val={auc:.5f}")
            if auc > best_auc:
                best_auc, best_oof, bad = auc, p, 0
                tps = []
                with torch.no_grad():
                    for xc, xn, L in te_dl:
                        tps.append(torch.sigmoid(net(xc.to(dev), xn.to(dev), L.to(dev))).cpu().numpy())
                best_test = np.concatenate(tps)
            else:
                bad += 1
                if bad >= PATIENCE:
                    break
        oof[va] = best_oof; test_sum += best_test
        print(f"[fold {f}] AUC={best_auc:.5f}")
    cv = float(roc_auc_score(y, oof))
    print(f"\n==== {MODEL_NAME}  CV AUC = {cv:.5f} ====")
    from .oof_io import save_predictions
    save_predictions(MODEL_NAME, oof, test_sum / C.N_FOLDS, meta=dict(
        cv_auc=cv, seed=C.SEED, n_folds=C.N_FOLDS, feature_set="원본 거래시퀀스 GRU+attention(8cat+6num+timing)",
        created_by="hyunbean", notes="transaction-sequence GRU, 집계 안한 순서/타이밍 직접학습 (미시도 직교축)"))


if __name__ == "__main__":
    main()
