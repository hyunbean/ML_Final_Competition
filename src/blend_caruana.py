"""Stability Caruana (벡터화) — 행 부트스트랩 100회 → 멤버 선택빈도 → 신기루 분리 → 큐레이션 블렌드.

GPT 진단: 94멤버+선형메타=과적합(OOF↑LB↓). 강멤버 corr 0.90~0.98=실제 독립신호 5~10개뿐.
매 bag: 고객 80% 재표집 → 전체멤버 greedy hillclimb(중복허용) → 선택멤버 기록. 100회 빈도집계.
빈도 높음=robust(진짜), 낮음=신기루. MIN_FREQ 이상만 큐레이션.
AUC는 벡터화(67후보 동시 argsort)로 roc_auc 47만콜 병목 제거.
실행: python -m src.blend_caruana
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from . import config as C
from .oof_io import list_models, load_oof

BAGS = 25
SUBSAMPLE = 0.8
N_STEPS = 25
INIT = 3
MIN_FREQ = 0.5
POOL_MIN_CV = 0.70


def _aucs(mat, ypos):
    """mat (n,k) 각 열의 AUC. 단일 argsort + 양성 랭크합 (동률무시 근사)."""
    n = mat.shape[0]
    npos = int(ypos.sum()); nneg = n - npos
    if npos == 0 or nneg == 0:
        return np.full(mat.shape[1], 0.5)
    order = np.argsort(mat, axis=0)                # (n,k) 점수 오름차순 인덱스
    ys = ypos[order]                               # 정렬된 양성마스크
    rpos = (np.arange(n)[:, None] * ys).sum(0)     # 양성 0-indexed 랭크합
    return (rpos - npos * (npos - 1) / 2.0) / (npos * nneg)


def _greedy(OOF, ypos, n_steps, init):
    """OOF (n,M) rank행렬 → greedy forward(중복허용) → 선택된 열인덱스 리스트."""
    base = _aucs(OOF, ypos)
    picks = list(np.argsort(base)[::-1][:init])
    cur = OOF[:, picks].mean(1)
    k = len(picks)
    best_auc = _aucs(cur[:, None], ypos)[0]; best = list(picks)
    for _ in range(n_steps):
        cand = (cur[:, None] * k + OOF) / (k + 1)     # (n, M)
        a = _aucs(cand, ypos)
        j = int(a.argmax())
        picks.append(j); k += 1
        cur = (cur * (k - 1) + OOF[:, j]) / k
        if a[j] > best_auc:
            best_auc = a[j]; best = list(picks)
    return best


def main():
    train_ids = np.load(C.TRAIN_IDS_NPY, allow_pickle=True)
    test_ids = np.load(C.TEST_IDS_NPY, allow_pickle=True)
    y = pd.read_csv(C.YTRAIN_CSV).set_index(C.ID_COL).reindex(train_ids)[C.TARGET].to_numpy()
    ypos = (y == 1)
    allm = list_models()
    oof = {m: rankdata(load_oof(m)[0]) / len(y) for m in allm}
    test = {m: rankdata(load_oof(m)[1]) / len(test_ids) for m in allm}
    cvs = {m: _aucs(oof[m][:, None], ypos)[0] for m in allm}
    models = [m for m in allm if cvs[m] >= POOL_MIN_CV]
    excl = [m for m in allm if cvs[m] < POOL_MIN_CV]
    M = len(models)
    OOF = np.column_stack([oof[m] for m in models])
    N = len(y)
    print(f"{len(allm)}개 중 CV>={POOL_MIN_CV} {M}개 (제외 {len(excl)}개: {excl})")

    rng = np.random.default_rng(C.SEED)
    freq = np.zeros(M); wsum = np.zeros(M)
    for b in range(BAGS):
        idx = rng.choice(N, size=int(N * SUBSAMPLE), replace=False)
        picks = _greedy(OOF[idx], ypos[idx], N_STEPS, INIT)
        for j in set(picks):
            freq[j] += 1
        for j in picks:
            wsum[j] += 1
        if (b + 1) % 25 == 0:
            print(f"  bag {b+1}/{BAGS}")

    freq_pct = freq / BAGS
    print("\n=== 멤버 선택빈도 (✓ robust / 낮음 신기루) ===")
    for j in np.argsort(freq_pct)[::-1]:
        mark = "✓" if freq_pct[j] >= MIN_FREQ else " "
        print(f"  {mark} {freq_pct[j]*100:>5.0f}%  CV={cvs[models[j]]:.4f}  {models[j]}")

    keep = [j for j in range(M) if freq_pct[j] >= MIN_FREQ]
    w = wsum[keep] / wsum[keep].sum()
    oof_blend = sum(w[i] * oof[models[keep[i]]] for i in range(len(keep)))
    test_blend = sum(w[i] * test[models[keep[i]]] for i in range(len(keep)))
    cv = _aucs(oof_blend[:, None], ypos)[0]
    print(f"\n==== 큐레이션 블렌드 ({len(keep)}멤버, freq>={MIN_FREQ})  OOF AUC = {cv:.5f} ====")
    print("  가중치:", {models[keep[i]]: round(float(w[i]), 3) for i in np.argsort(w)[::-1]})
    out = C.SUB_DIR / "submission_caruana.csv"
    pd.DataFrame({C.ID_COL: test_ids, C.TARGET: test_blend}).to_csv(out, index=False)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
