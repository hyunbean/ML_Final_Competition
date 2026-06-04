"""공유된 OOF를 모아 블렌딩 → 제출 파일 생성.

- 각 모델 OOF를 rank 정규화(AUC는 순위 기반).
- rank-average(베이스라인)와 **Hill-Climbing 가중 블렌드**(Caruana) 둘 다 평가,
  Hill-Climbing 가중치를 test에 적용해 제출 파일 생성.
- 팀원들이 artifacts/oof/ 에 {model}__oof.npy / __test.npy 를 넣으면 자동 포함.

실행: python -m src.ensemble
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from . import config as C
from .oof_io import list_models, load_oof


def _rank(a):
    return rankdata(a) / len(a)


def hill_climb(oof_r, y, models, n_steps=100):
    """Caruana hill-climbing: 매 스텝 OOF AUC를 가장 올리는 모델을 (중복 허용) 선택."""
    n = len(y)
    cur = np.zeros(n)
    picks = []
    best_overall = (-1.0, [])
    for _ in range(n_steps):
        best = (-1.0, None)
        for m in models:
            cand = (cur * len(picks) + oof_r[m]) / (len(picks) + 1)
            a = roc_auc_score(y, cand)
            if a > best[0]:
                best = (a, m)
        picks.append(best[1])
        cur = (cur * (len(picks) - 1) + oof_r[best[1]]) / len(picks)
        a = roc_auc_score(y, cur)
        if a > best_overall[0]:
            best_overall = (a, list(picks))
    auc, plist = best_overall
    weights = {m: plist.count(m) / len(plist) for m in set(plist)}
    return auc, weights


def main():
    train_ids = np.load(C.TRAIN_IDS_NPY, allow_pickle=True)
    test_ids = np.load(C.TEST_IDS_NPY, allow_pickle=True)
    y = pd.read_csv(C.YTRAIN_CSV).set_index(C.ID_COL).reindex(train_ids)[C.TARGET].to_numpy()

    models = list_models()
    if not models:
        print("artifacts/oof/ 에 모델이 없습니다.")
        return

    oof_r, test_r = {}, {}
    for m in models:
        o, t = load_oof(m)
        assert len(o) == len(y) and len(t) == len(test_ids), f"{m}: 길이 불일치"
        oof_r[m], test_r[m] = _rank(o), _rank(t)
        print(f"  {m:26s} OOF AUC = {roc_auc_score(y, o):.5f}")

    avg = np.mean([oof_r[m] for m in models], axis=0)
    print(f"\nrank-average   OOF AUC = {roc_auc_score(y, avg):.5f}  ({len(models)} models)")

    hc_auc, weights = hill_climb(oof_r, y, models)
    print(f"hill-climbing  OOF AUC = {hc_auc:.5f}")
    print("  weights:", {m: round(w, 3) for m, w in sorted(weights.items(), key=lambda x: -x[1])})

    test_blend = np.zeros(len(test_ids))
    for m, w in weights.items():
        test_blend += w * test_r[m]

    sub = pd.DataFrame({C.ID_COL: test_ids, C.TARGET: test_blend})
    out = C.SUB_DIR / "submission_blend.csv"
    sub.to_csv(out, index=False)
    print(f"\nsaved: {out}  (rows={len(sub)})  [hill-climbing weighted]")


if __name__ == "__main__":
    main()
