"""사전학습 한국어 LM 임베딩 모델 (교수님 베이스라인 방식, GPU).

고객의 구매 카테고리(한국어 단어)를 '문장'으로 만들어 사전학습 KLUE/RoBERTa로 인코딩.
→ 사전학습 의미지식(립스틱→여성 등)을 활용 = W2V가 못 잡는 신호 + 완전히 다른 representation.
frozen 임베딩(mean-pool) → 5fold StandardScaler+LogReg 헤드 → OOF. features 캐시 불필요.

실행: pip install transformers
     python -m src.folds
     python -m src.train_text_emb [HF모델명]   # 기본 klue/roberta-base
예: python -m src.train_text_emb monologg/koelectra-base-v3-discriminator
    python -m src.train_text_emb klue/bert-base
모델마다 다른 OOF(text_roberta / text_koelectra ...)로 저장 → 앙상블 다양성.
"""
import sys
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from . import config as C
from .oof_io import save_predictions

HF_MODEL = sys.argv[1] if len(sys.argv) > 1 else "klue/roberta-base"
_TAG = HF_MODEL.split("/")[-1].split("-")[0]      # roberta / koelectra / bert ...
MODEL_NAME = f"text_{_TAG}"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOPK = 40            # 고객당 상위 카테고리 토큰 수
MAXTOK = 64          # 토크나이저 max_length
BATCH = 64
TEXT_COLS = ["corner_nm", "part_nm", "pc_nm"]   # 한국어 의미 카테고리


def _docs(df, ids):
    """고객별 '주로 산 카테고리' 문장 — 빈도 상위 TOPK 한국어 토큰을 빈도순 나열."""
    df = df[[C.ID_COL] + TEXT_COLS].copy()
    long = df.melt(C.ID_COL, value_name="tok")[[C.ID_COL, "tok"]].dropna()
    long["tok"] = long["tok"].astype(str).str.replace(r"\s+", "", regex=True)
    vc = long.groupby([C.ID_COL, "tok"]).size().reset_index(name="n")
    vc = vc.sort_values([C.ID_COL, "n"], ascending=[True, False])
    docs = vc.groupby(C.ID_COL)["tok"].apply(lambda s: " ".join(s.to_numpy()[:TOPK]))
    return docs.reindex(ids).fillna("").tolist()


@torch.no_grad()
def _embed(texts, tok, model):
    out = np.zeros((len(texts), model.config.hidden_size), dtype=np.float32)
    for i in range(0, len(texts), BATCH):
        batch = texts[i:i + BATCH]
        enc = tok(batch, padding=True, truncation=True, max_length=MAXTOK, return_tensors="pt").to(DEVICE)
        h = model(**enc).last_hidden_state                       # (B,L,H)
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1.0)  # mean-pool
        out[i:i + len(batch)] = pooled.cpu().numpy()
        if i % (BATCH * 20) == 0:
            print(f"  embed {i}/{len(texts)}")
    return out


def main():
    train_ids = np.load(C.TRAIN_IDS_NPY, allow_pickle=True)
    test_ids = np.load(C.TEST_IDS_NPY, allow_pickle=True)
    folds = np.load(C.FOLDS_NPY)
    y = pd.read_csv(C.YTRAIN_CSV).set_index(C.ID_COL).reindex(train_ids)[C.TARGET].to_numpy()

    use = [C.ID_COL] + TEXT_COLS
    tr = pd.read_csv(C.TRAIN_CSV, encoding=C.ENCODING, usecols=use)
    te = pd.read_csv(C.TEST_CSV, encoding=C.ENCODING, usecols=use)
    tr_docs, te_docs = _docs(tr, train_ids), _docs(te, test_ids)
    print(f"예시 문장: {tr_docs[0][:80]}...")

    tok = AutoTokenizer.from_pretrained(HF_MODEL)
    model = AutoModel.from_pretrained(HF_MODEL).to(DEVICE).eval()
    print(f"{HF_MODEL} on {DEVICE} — 임베딩 추출...")
    Xtr, Xte = _embed(tr_docs, tok, model), _embed(te_docs, tok, model)

    oof = np.full(len(y), np.nan)
    test_sum = np.zeros(len(test_ids))
    for f in range(C.N_FOLDS):
        trn, val = np.where(folds != f)[0], np.where(folds == f)[0]
        sc = StandardScaler().fit(Xtr[trn])
        lr = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(Xtr[trn]), y[trn])
        oof[val] = lr.predict_proba(sc.transform(Xtr[val]))[:, 1]
        test_sum += lr.predict_proba(sc.transform(Xte))[:, 1]
        print(f"[fold {f}] AUC={roc_auc_score(y[val], oof[val]):.5f}")

    cv = float(roc_auc_score(y, oof))
    print(f"\n==== {MODEL_NAME}  CV AUC = {cv:.5f} ====")
    save_predictions(MODEL_NAME, oof, test_sum / C.N_FOLDS, meta=dict(
        cv_auc=cv, seed=C.SEED, n_folds=C.N_FOLDS, feature_set=f"{HF_MODEL} mean-pool",
        created_by="hyunbean", notes=f"pretrained KLUE/RoBERTa emb on category text (top{TOPK})"))


if __name__ == "__main__":
    main()
