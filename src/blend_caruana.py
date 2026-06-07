"""Stability Caruana — 행 부트스트랩 100회 → 멤버 선택빈도 → 진짜 vs 신기루 분리 → 큐레이션 블렌드.

GPT 진단: 94멤버 전부+선형메타 = 메타 과적합(OOF↑ LB↓). 강멤버 corr 0.90~0.98 = 실제 독립신호 5~10개뿐.
방법: 매 bag마다 고객 80% 재표집 → 전체멤버 hillclimb → 선택된 멤버 기록. 100회 빈도 집계.
빈도 높은 멤버=robust(진짜), 낮은 멤버=신기루. MIN_FREQ 이상만 큐레이션해 최종 블렌드.
실행: python -m src.blend_caruana
"""
import numpy as np
import pandas as pd
from collections import Counter
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from . import config as C
from .oof_io import list_models, load_oof

BAGS = 100
SUBSAMPLE = 0.8
N_STEPS = 40
INIT = 3
MIN_FREQ = 0.5      # 이 빈도 이상 선택된 멤버만 최종 블렌드에 채택
POOL_MIN_CV = 0.70  # 사전제외: CV 0.70 미만 약멤버는 어차피 안뽑힘(stability 후보풀서 제외)


def _rank(a):
    return rankdata(a) / len(a)


def fast_auc(y, s):
    """벡터화 AUC (roc_auc_score보다 3~5배 빠름). y=0/1, s=점수."""
    o = np.argsort(s, kind="stable")
    yy = y[o]
    n_pos = yy.sum(); n_neg = len(yy) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    cum_neg = np.cumsum(1.0 - yy)
    return float((yy * cum_neg).sum() / (n_pos * n_neg))


def _greedy(oof, y, pool, n_steps, init):
    aucs = {m: fast_auc(y, oof[m]) for m in pool}
    picks = sorted(pool, key=lambda m: -aucs[m])[:init]
    cur = np.mean([oof[m] for m in picks], axis=0)
    best = (fast_auc(y, cur), list(picks))
    for _ in range(n_steps):
        b = (-1, None)
        k = len(picks)
        for m in pool:
            a = fast_auc(y, (cur * k + oof[m]) / (k + 1))
            if a > b[0]:
                b = (a, m)
        picks.append(b[1])
        cur = (cur * (len(picks) - 1) + oof[b[1]]) / len(picks)
        a = fast_auc(y, cur)
        if a > best[0]:
            best = (a, list(picks))
    return best[1]


def main():
    train_ids = np.load(C.TRAIN_IDS_NPY, allow_pickle=True)
    test_ids = np.load(C.TEST_IDS_NPY, allow_pickle=True)
    y = pd.read_csv(C.YTRAIN_CSV).set_index(C.ID_COL).reindex(train_ids)[C.TARGET].to_numpy()
    allmodels = list_models()
    oof_r = {m: _rank(load_oof(m)[0]) for m in allmodels}
    test_r = {m: _rank(load_oof(m)[1]) for m in allmodels}
    N = len(y)
    cvs = {m: fast_auc(y, oof_r[m]) for m in allmodels}
    models = [m for m in allmodels if cvs[m] >= POOL_MIN_CV]
    print(f"{len(allmodels)}개 중 CV>={POOL_MIN_CV} {len(models)}개로 Stability Caruana ({BAGS} bags x {SUBSAMPLE} row-bootstrap)")
    print(f"  제외(약멤버 {len(allmodels)-len(models)}개): {[m for m in allmodels if cvs[m] < POOL_MIN_CV]}")

    rng = np.random.default_rng(C.SEED)
    freq = Counter(); wsum = Counter()
    for b in range(BAGS):
        idx = rng.choice(N, size=int(N * SUBSAMPLE), replace=False)
        yb = y[idx]; oofb = {m: oof_r[m][idx] for m in models}
        picks = _greedy(oofb, yb, models, N_STEPS, INIT)
        for m in set(picks):
            freq[m] += 1
        for m in picks:
            wsum[m] += 1
        if (b + 1) % 20 == 0:
            print(f"  bag {b+1}/{BAGS}")

    freq_pct = {m: freq[m] / BAGS for m in models}
    print("\n=== 멤버 선택빈도 (robust=진짜 / 낮음=신기루) ===")
    for m, f in sorted(freq_pct.items(), key=lambda x: -x[1]):
        mark = "✓" if f >= MIN_FREQ else " "
        print(f"  {mark} {f*100:>5.0f}%  CV={cvs[m]:.4f}  {m}")

    keep = [m for m in models if freq_pct[m] >= MIN_FREQ]
    w = {m: wsum[m] for m in keep}; tot = sum(w.values())
    w = {m: v / tot for m, v in w.items()}
    oof_blend = sum(w[m] * oof_r[m] for m in w)
    test_blend = sum(w[m] * test_r[m] for m in w)
    cv = roc_auc_score(y, oof_blend)
    print(f"\n==== 큐레이션 블렌드 ({len(keep)}멤버, freq>={MIN_FREQ})  OOF AUC = {cv:.5f} ====")
    print("  가중치:", {m: round(v, 3) for m, v in sorted(w.items(), key=lambda x: -x[1])})
    out = C.SUB_DIR / "submission_caruana.csv"
    pd.DataFrame({C.ID_COL: test_ids, C.TARGET: test_blend}).to_csv(out, index=False)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
