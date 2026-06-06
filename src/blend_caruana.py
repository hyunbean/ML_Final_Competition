"""Bagged Caruana ensemble selection — 단순 hill-climb의 과적합을 줄인 정교 블렌드.

전체 OOF(우리+김민형) rank 정규화 → 모델 라이브러리의 랜덤 부분집합(bag)마다 greedy
forward selection(중복허용) 반복 → 가중치 평균. mh_22 몰빵을 완화해 LB 일반화↑.
실행: python -m src.blend_caruana
"""
import numpy as np
import pandas as pd
from collections import Counter
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from . import config as C
from .oof_io import list_models, load_oof

BAGS = 60
BAG_FRAC = 0.5
N_STEPS = 100
INIT = 3          # 개별 AUC 상위 init개로 시작


def _rank(a):
    return rankdata(a) / len(a)


def _greedy(oof_r, y, pool, n_steps, init):
    aucs = {m: roc_auc_score(y, oof_r[m]) for m in pool}
    start = sorted(pool, key=lambda m: -aucs[m])[:init]
    picks = list(start)
    cur = np.mean([oof_r[m] for m in picks], axis=0)
    best = (roc_auc_score(y, cur), list(picks))
    for _ in range(n_steps):
        b = (-1, None)
        for m in pool:
            cand = (cur * len(picks) + oof_r[m]) / (len(picks) + 1)
            a = roc_auc_score(y, cand)
            if a > b[0]:
                b = (a, m)
        picks.append(b[1])
        cur = (cur * (len(picks) - 1) + oof_r[b[1]]) / len(picks)
        a = roc_auc_score(y, cur)
        if a > best[0]:
            best = (a, list(picks))
    return best[1]


def main():
    train_ids = np.load(C.TRAIN_IDS_NPY, allow_pickle=True)
    test_ids = np.load(C.TEST_IDS_NPY, allow_pickle=True)
    y = pd.read_csv(C.YTRAIN_CSV).set_index(C.ID_COL).reindex(train_ids)[C.TARGET].to_numpy()
    models = list_models()
    oof_r = {m: _rank(load_oof(m)[0]) for m in models}
    test_r = {m: _rank(load_oof(m)[1]) for m in models}
    print(f"{len(models)} models, bagged Caruana ({BAGS} bags x {BAG_FRAC})")

    rng = np.random.default_rng(C.SEED)
    counts = Counter()
    for b in range(BAGS):
        k = max(INIT + 1, int(len(models) * BAG_FRAC))
        pool = list(rng.choice(models, size=k, replace=False))
        picks = _greedy(oof_r, y, pool, N_STEPS, INIT)
        counts.update(picks)
    tot = sum(counts.values())
    w = {m: c / tot for m, c in counts.items()}

    oof_blend = sum(w[m] * oof_r[m] for m in w)
    test_blend = sum(w[m] * test_r[m] for m in w)
    cv = roc_auc_score(y, oof_blend)
    print(f"\n==== Bagged Caruana  OOF AUC = {cv:.5f} ====")
    print("  top weights:", {m: round(v, 3) for m, v in sorted(w.items(), key=lambda x: -x[1])[:10]})
    out = C.SUB_DIR / "submission_caruana.csv"
    pd.DataFrame({C.ID_COL: test_ids, C.TARGET: test_blend}).to_csv(out, index=False)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
