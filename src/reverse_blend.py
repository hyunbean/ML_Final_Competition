"""Reverse Blend (drop-one) — '좋은 모델 추가'가 아니라 '불필요 모델 제거'로 robust 블렌드.

GPT 진단: 신호 수렴(corr 0.997). 멤버 하나씩 빼서 최종 rank 변화 측정 →
변화 거의 없는 멤버=불필요(과적합 위험만 추가). 제거형 단순 블렌드가 private서 더 강한 경우 많음.
strong 풀(CV>=THRESH) equal rank-avg 기준, drop-one OOF AUC + test rank corr.
실행: python -m src.reverse_blend
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from . import config as C
from .oof_io import list_models, load_oof

THRESH = 0.712     # 이 CV 이상 strong 멤버만 후보
DROP_EPS = 0.0003  # 빼도 OOF 변화 < 이 값 = 불필요


def _rk(a):
    return rankdata(a) / len(a)


def main():
    train_ids = np.load(C.TRAIN_IDS_NPY, allow_pickle=True)
    test_ids = np.load(C.TEST_IDS_NPY, allow_pickle=True)
    y = pd.read_csv(C.YTRAIN_CSV).set_index(C.ID_COL).reindex(train_ids)[C.TARGET].to_numpy()
    allm = list_models()
    oof = {m: _rk(load_oof(m)[0]) for m in allm}
    test = {m: _rk(load_oof(m)[1]) for m in allm}
    cvs = {m: roc_auc_score(y, oof[m]) for m in allm}
    pool = [m for m in allm if cvs[m] >= THRESH]
    print(f"strong 풀 {len(pool)}개 (CV>={THRESH}):")
    for m in sorted(pool, key=lambda x: -cvs[x]):
        print(f"   {cvs[m]:.4f}  {m}")

    full_oof = np.mean([oof[m] for m in pool], axis=0)
    full_auc = roc_auc_score(y, full_oof)
    print(f"\n=== equal rank-avg ({len(pool)}멤버) OOF AUC = {full_auc:.5f} ===")

    print("\n=== drop-one (빼면 OOF 얼마나 변하나) ===")
    rows = []
    for m in pool:
        rest = [k for k in pool if k != m]
        a = roc_auc_score(y, np.mean([oof[k] for k in rest], axis=0))
        rows.append((a - full_auc, m, a))
    for delta, m, a in sorted(rows):
        tag = "필수(빼면↓)" if delta < -DROP_EPS else ("불필요(빼도무관)" if abs(delta) <= DROP_EPS else "해로움(빼면↑)")
        print(f"   drop {m:24s} ΔOOF={delta:+.5f}  ({tag})")

    # 제거형 robust 블렌드: 빼면 오르거나 무관한 멤버 제거 → 필수만
    essential = [m for (d, m, a) in rows if d < -DROP_EPS]
    if not essential:
        essential = [max(pool, key=lambda x: cvs[x])]
    rob_oof = np.mean([oof[m] for m in essential], axis=0)
    rob_test = np.mean([test[m] for m in essential], axis=0)
    rob_auc = roc_auc_score(y, rob_oof)
    print(f"\n=== 제거형 robust 블렌드: 필수 {len(essential)}멤버 OOF AUC = {rob_auc:.5f} ===")
    print(f"   {essential}")
    out = C.SUB_DIR / "submission_reverse.csv"
    pd.DataFrame({C.ID_COL: test_ids, C.TARGET: rob_test}).to_csv(out, index=False)
    print(f"saved: {out}")
    # 비교: full vs robust corr
    print(f"\nfull vs robust test corr = {np.corrcoef(_rk(full_oof), _rk(rob_oof))[0,1]:.4f}")


if __name__ == "__main__":
    main()
