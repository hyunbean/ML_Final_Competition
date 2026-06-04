"""특정 모델(또는 여러 모델 평균)의 test 예측 → Kaggle 제출 CSV.

사용:
  python -m src.make_submission lgbm_full                       # 단일 모델
  python -m src.make_submission lgbm_full catboost_full xgb_full # 단순 평균
출력: artifacts/submissions/submission_<이름>.csv  (custid, gender=positive 확률)

참고: 제출 형식 = custid,gender / index 없음. (앙상블 블렌드는 src.ensemble 사용)
"""
import sys
import numpy as np
import pandas as pd

from . import config as C
from .oof_io import load_oof


def main():
    models = sys.argv[1:]
    if not models:
        print("사용: python -m src.make_submission <model> [model2 ...]")
        return

    test_ids = np.load(C.TEST_IDS_NPY, allow_pickle=True)
    preds = []
    for m in models:
        _, t = load_oof(m)
        if len(t) != len(test_ids):
            raise ValueError(f"{m}: test 길이 {len(t)} != {len(test_ids)} — 정규순서 확인")
        preds.append(t)
    pred = np.mean(preds, axis=0)

    sub = pd.DataFrame({C.ID_COL: test_ids, C.TARGET: pred})
    name = "_".join(models) if len(models) <= 2 else f"{len(models)}models"
    out = C.SUB_DIR / f"submission_{name}.csv"
    sub.to_csv(out, index=False)
    print(f"saved: {out}")
    print(f"  rows={len(sub)}  models={models}")
    print(sub.head())


if __name__ == "__main__":
    main()
