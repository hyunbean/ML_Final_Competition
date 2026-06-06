"""DE 블렌드 최적화 — differential evolution으로 OOF AUC 직접 최대화 (음수 가중치 허용).

Caruana(비음수·discrete)가 못 닿는 영역을 연속·음수허용 전역최적화로. rank 정규화 후.
실행: python -m src.blend_de
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from scipy.optimize import differential_evolution
from sklearn.metrics import roc_auc_score

from . import config as C
from .oof_io import list_models, load_oof


def rk(a):
    return rankdata(a) / len(a)


def main():
    train_ids = np.load(C.TRAIN_IDS_NPY, allow_pickle=True)
    test_ids = np.load(C.TEST_IDS_NPY, allow_pickle=True)
    y = pd.read_csv(C.YTRAIN_CSV).set_index(C.ID_COL).reindex(train_ids)[C.TARGET].to_numpy()
    models = list_models()
    O = np.column_stack([rk(load_oof(m)[0]) for m in models])
    T = np.column_stack([rk(load_oof(m)[1]) for m in models])
    # 개별 AUC 상위로 차원 축소(DE 효율) — 너무 약한 건 0 근처라 제외
    aucs = np.array([roc_auc_score(y, O[:, j]) for j in range(O.shape[1])])
    keep = np.argsort(aucs)[::-1][:20]
    Ok, Tk = O[:, keep], T[:, keep]
    print(f"{len(models)} models → DE 대상 top20")

    def neg_auc(w):
        return -roc_auc_score(y, Ok @ w)

    bounds = [(-0.3, 1.0)] * Ok.shape[1]
    res = differential_evolution(neg_auc, bounds, maxiter=80, popsize=20, tol=1e-7,
                                 seed=C.SEED, polish=True, workers=1)
    w = res.x
    auc = roc_auc_score(y, Ok @ w)
    print(f"\n==== DE 블렌드 OOF AUC = {auc:.5f} ====")
    print("  weights:", {models[keep[i]]: round(float(w[i]), 3) for i in np.argsort(np.abs(w))[::-1][:10]})
    test_blend = Tk @ w
    out = C.SUB_DIR / "submission_de.csv"
    pd.DataFrame({C.ID_COL: test_ids, C.TARGET: test_blend}).to_csv(out, index=False)
    print(f"saved: {out.name}")


if __name__ == "__main__":
    main()
