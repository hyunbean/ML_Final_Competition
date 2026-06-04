"""공유된 모든 OOF를 모아 블렌딩 → 제출 파일 생성.

팀원들이 artifacts/oof/ 에 각자 {model}__oof.npy / {model}__test.npy 를 넣어두면,
여기서 전부 읽어 OOF로 성능을 검증하고 test 예측을 합쳐 제출 파일을 만든다.
기본은 rank-average(간단·강건). 이후 Hill-Climbing / 로지스틱 메타로 교체 가능.

실행: python -m src.ensemble
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from . import config as C
from .oof_io import list_models, load_oof


def main():
    train_ids = np.load(C.TRAIN_IDS_NPY, allow_pickle=True)
    test_ids = np.load(C.TEST_IDS_NPY, allow_pickle=True)
    y = pd.read_csv(C.YTRAIN_CSV).set_index(C.ID_COL).reindex(train_ids)[C.TARGET].to_numpy()

    models = list_models()
    if not models:
        print("artifacts/oof/ 에 모델 예측이 없습니다. 먼저 학습(train_*.py)부터.")
        return

    oofs, tests = {}, {}
    for m in models:
        o, t = load_oof(m)
        assert len(o) == len(y) and len(t) == len(test_ids), f"{m}: 길이 불일치 — 정규순서 확인!"
        oofs[m], tests[m] = o, t
        print(f"  {m:24s}  OOF AUC = {roc_auc_score(y, o):.5f}")

    # rank-average 블렌딩
    oof_blend = np.mean([rankdata(oofs[m]) / len(y) for m in models], axis=0)
    test_blend = np.mean([rankdata(tests[m]) / len(test_ids) for m in models], axis=0)
    print(f"\n==== blend OOF AUC = {roc_auc_score(y, oof_blend):.5f}  ({len(models)} models) ====")

    sub = pd.DataFrame({C.ID_COL: test_ids, C.TARGET: test_blend})
    out = C.SUB_DIR / "submission_blend.csv"
    sub.to_csv(out, index=False)
    print(f"saved: {out}  (rows={len(sub)})")


if __name__ == "__main__":
    main()
