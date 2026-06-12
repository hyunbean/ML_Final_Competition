"""node2vec(DeepWalk) 그래프 임베딩 모델 (CPU, gensim).

고객↔브랜드 이분그래프에서 랜덤워크 → Word2Vec로 노드 임베딩 → 고객벡터로 성별예측.
'그래프 구조'(누구와 누가 같은 브랜드를 사는가의 고차 연결) = 집계/순서/텍스트와 또 다른 표현.
그래프는 train+test 전체로 구성(비지도, 누수 없음). features 캐시 불필요.

실행: pip install gensim → python -m src.folds → python -m src.train_node2vec
"""
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from . import config as C
from .oof_io import save_predictions

MODEL_NAME = "node2vec"
ITEM_COL = "brd_nm"
DIM, WALK_LEN, NUM_WALKS, WINDOW, EPOCHS = 64, 20, 10, 5, 5


def _walks(cust2items, item2cust, rng):
    custs = list(cust2items.keys())
    walks = []
    for _ in range(NUM_WALKS):
        rng.shuffle(custs)
        for c in custs:
            typ, val, walk = "c", c, [f"c_{c}"]
            for _ in range(WALK_LEN):
                if typ == "c":
                    items = cust2items[val]
                    val = items[rng.integers(len(items))]
                    walk.append(f"b_{val}"); typ = "b"
                else:
                    users = item2cust[val]
                    val = users[rng.integers(len(users))]
                    walk.append(f"c_{val}"); typ = "c"
            walks.append(walk)
    return walks


def main():
    train_ids = np.load(C.TRAIN_IDS_NPY, allow_pickle=True)
    test_ids = np.load(C.TEST_IDS_NPY, allow_pickle=True)
    folds = np.load(C.FOLDS_NPY)
    y = pd.read_csv(C.YTRAIN_CSV).set_index(C.ID_COL).reindex(train_ids)[C.TARGET].to_numpy()

    use = [C.ID_COL, ITEM_COL]
    df = pd.concat([pd.read_csv(C.TRAIN_CSV, encoding=C.ENCODING, usecols=use),
                    pd.read_csv(C.TEST_CSV, encoding=C.ENCODING, usecols=use)], ignore_index=True)
    df[ITEM_COL] = df[ITEM_COL].astype(str)
    cust2items = {c: g.to_numpy() for c, g in df.groupby(C.ID_COL)[ITEM_COL]}
    item2cust = {b: g.to_numpy() for b, g in df.groupby(ITEM_COL)[C.ID_COL]}
    print(f"고객={len(cust2items):,}  브랜드={len(item2cust):,}  랜덤워크 생성...")

    rng = np.random.default_rng(C.SEED)
    walks = _walks(cust2items, item2cust, rng)
    print(f"walks={len(walks):,} → Word2Vec 학습(dim={DIM})...")
    w2v = Word2Vec(walks, vector_size=DIM, window=WINDOW, min_count=0, sg=1,
                   workers=4, epochs=EPOCHS, seed=C.SEED)

    def emb(ids):
        X = np.zeros((len(ids), DIM), dtype=np.float32)
        for i, c in enumerate(ids):
            k = f"c_{c}"
            if k in w2v.wv:
                X[i] = w2v.wv[k]
        return X
    Xtr, Xte = emb(train_ids), emb(test_ids)

    oof = np.full(len(y), np.nan)
    test_sum = np.zeros(len(test_ids))
    for f in range(C.N_FOLDS):
        tr, va = np.where(folds != f)[0], np.where(folds == f)[0]
        sc = StandardScaler().fit(Xtr[tr])
        lr = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(Xtr[tr]), y[tr])
        oof[va] = lr.predict_proba(sc.transform(Xtr[va]))[:, 1]
        test_sum += lr.predict_proba(sc.transform(Xte))[:, 1]
        print(f"[fold {f}] AUC={roc_auc_score(y[va], oof[va]):.5f}")

    cv = float(roc_auc_score(y, oof))
    print(f"\n==== {MODEL_NAME}  CV AUC = {cv:.5f} ====")
    save_predictions(MODEL_NAME, oof, test_sum / C.N_FOLDS, meta=dict(
        cv_auc=cv, seed=C.SEED, n_folds=C.N_FOLDS, feature_set=f"node2vec({ITEM_COL},dim{DIM})",
        created_by="hyunbean", notes=f"DeepWalk on customer-{ITEM_COL} bipartite graph"))


if __name__ == "__main__":
    main()
