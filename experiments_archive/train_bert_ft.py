"""Fine-tuned 한국어 BERT (TabLLM식) — 고객 구매를 '풍부한 문장'으로 만들어 end-to-end 파인튜닝.

frozen(text_roberta 0.652 실패)과 달리 ① 수량·금액 포함 자연문장 ② 분류 헤드까지 학습.
교수님 베이스라인 방식 + 리서치(TabLLM)에서 deep tabular 능가 보고. 표현이 완전히 달라 팀블렌드 기여 기대.
features 캐시 불필요. 5-fold OOF. 실행(GPU): pip install transformers → python -m src.train_bert_ft
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import roc_auc_score

from . import config as C
from .oof_io import save_predictions

MODEL_NAME = "bert_ft"
HF_MODEL = "klue/roberta-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CATCOL = "corner_nm"            # 서술적 한국어 카테고리
TOPK, MAXLEN = 20, 128
EPOCHS, BATCH, LR = 3, 32, 2e-5


def _docs(df, ids):
    """고객별 '풍부한 문장': '화장품 12회 35만원 · 여성캐주얼 8회 20만원 · ...'"""
    d = df[[C.ID_COL, CATCOL, "net_amt"]].copy()
    d[CATCOL] = d[CATCOL].astype(str)
    g = d.groupby([C.ID_COL, CATCOL]).agg(cnt=("net_amt", "size"), amt=("net_amt", "sum")).reset_index()
    g = g.sort_values([C.ID_COL, "amt"], ascending=[True, False])
    g["txt"] = g[CATCOL] + " " + g["cnt"].astype(int).astype(str) + "회 " + (g["amt"] / 10000).round().astype(int).astype(str) + "만원"
    docs = g.groupby(C.ID_COL)["txt"].apply(lambda s: " · ".join(s.to_numpy()[:TOPK]))
    return docs.reindex(ids).fillna("").tolist()


def _encode(texts, tok):
    return tok(texts, padding="max_length", truncation=True, max_length=MAXLEN, return_tensors="pt")


@torch.no_grad()
def _predict(model, enc, idx):
    model.eval()
    out = np.zeros(len(idx))
    for i in range(0, len(idx), 64):
        b = idx[i:i + 64]
        logit = model(input_ids=enc["input_ids"][b].to(DEVICE),
                      attention_mask=enc["attention_mask"][b].to(DEVICE)).logits
        out[i:i + len(b)] = torch.softmax(logit, 1)[:, 1].cpu().numpy()
    return out


def main():
    train_ids = np.load(C.TRAIN_IDS_NPY, allow_pickle=True)
    test_ids = np.load(C.TEST_IDS_NPY, allow_pickle=True)
    folds = np.load(C.FOLDS_NPY)
    y = pd.read_csv(C.YTRAIN_CSV).set_index(C.ID_COL).reindex(train_ids)[C.TARGET].to_numpy()

    use = [C.ID_COL, CATCOL, "net_amt"]
    tr = pd.read_csv(C.TRAIN_CSV, encoding=C.ENCODING, usecols=use)
    te = pd.read_csv(C.TEST_CSV, encoding=C.ENCODING, usecols=use)
    tok = AutoTokenizer.from_pretrained(HF_MODEL)
    print(f"예시: {_docs(tr, train_ids[:1])[0][:90]}")
    enc_tr = _encode(_docs(tr, train_ids), tok)
    enc_te = _encode(_docs(te, test_ids), tok)
    print(f"{HF_MODEL} 파인튜닝 on {DEVICE}")

    oof = np.full(len(y), np.nan)
    test_sum = np.zeros(len(test_ids))
    lossf = nn.CrossEntropyLoss()
    for f in range(C.N_FOLDS):
        trn = np.where(folds != f)[0]
        val = np.where(folds == f)[0]
        torch.manual_seed(C.SEED)
        model = AutoModelForSequenceClassification.from_pretrained(HF_MODEL, num_labels=2).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=LR)
        yt = torch.tensor(y, dtype=torch.long)
        for ep in range(EPOCHS):
            model.train()
            perm = np.random.default_rng(C.SEED + ep).permutation(trn)
            for i in range(0, len(perm), BATCH):
                b = perm[i:i + BATCH]
                opt.zero_grad()
                logit = model(input_ids=enc_tr["input_ids"][b].to(DEVICE),
                              attention_mask=enc_tr["attention_mask"][b].to(DEVICE)).logits
                lossf(logit, yt[b].to(DEVICE)).backward()
                opt.step()
        oof[val] = _predict(model, enc_tr, val)
        test_sum += _predict(model, enc_te, np.arange(len(test_ids)))
        print(f"[fold {f}] AUC={roc_auc_score(y[val], oof[val]):.5f}")
        del model; torch.cuda.empty_cache()

    cv = float(roc_auc_score(y, oof))
    print(f"\n==== {MODEL_NAME}  CV AUC = {cv:.5f} ====")
    save_predictions(MODEL_NAME, oof, test_sum / C.N_FOLDS, meta=dict(
        cv_auc=cv, seed=C.SEED, n_folds=C.N_FOLDS, feature_set=f"{HF_MODEL} fine-tuned on rich text",
        created_by="hyunbean", notes=f"TabLLM-style fine-tune ({CATCOL} 수량+금액 문장)"))


if __name__ == "__main__":
    main()
