"""스태킹 메타러너 — OOF 위에 L2 로지스틱 회귀로 2단 학습.

- 각 모델 OOF를 rank 정규화 → 메타 입력.
- 기존 folds로 out-of-fold 스태킹(정직한 CV): fold f의 메타예측은 나머지 fold로 학습한 메타가 생성.
- hill-climbing(ensemble.py)과 CV 비교. 순환 방지 위해 oof/ 에는 저장하지 않고 submission만 생성.

실행: python -m src.stack
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from . import config as C
from .oof_io import list_models, load_oof


def _rank(a):
    return rankdata(a) / len(a)


def main():
    train_ids = np.load(C.TRAIN_IDS_NPY, allow_pickle=True)
    test_ids = np.load(C.TEST_IDS_NPY, allow_pickle=True)
    folds = np.load(C.FOLDS_NPY)
    y = pd.read_csv(C.YTRAIN_CSV).set_index(C.ID_COL).reindex(train_ids)[C.TARGET].to_numpy()

    models = list_models()
    Xo = np.column_stack([_rank(load_oof(m)[0]) for m in models])
    Xt = np.column_stack([_rank(load_oof(m)[1]) for m in models])
    print(f"stacking {len(models)} models: {models}")

    for C_reg in (0.1, 0.5, 1.0, 2.0):
        meta_oof = np.full(len(y), np.nan)
        test_acc = np.zeros(len(test_ids))
        coefs = np.zeros(len(models))
        for f in range(C.N_FOLDS):
            tr, va = np.where(folds != f)[0], np.where(folds == f)[0]
            sc = StandardScaler().fit(Xo[tr])
            lr = LogisticRegression(C=C_reg, max_iter=2000).fit(sc.transform(Xo[tr]), y[tr])
            meta_oof[va] = lr.predict_proba(sc.transform(Xo[va]))[:, 1]
            test_acc += lr.predict_proba(sc.transform(Xt))[:, 1]
            coefs += lr.coef_[0]
        cv = roc_auc_score(y, meta_oof)
        print(f"  [C={C_reg:>4}] stack CV AUC = {cv:.5f}")
        if C_reg == 1.0:
            best_cv, best_test, best_coef = cv, test_acc / C.N_FOLDS, coefs / C.N_FOLDS

    print(f"\n==== stack_logreg (C=1.0)  CV AUC = {best_cv:.5f} ====")
    order = np.argsort(-np.abs(best_coef))
    print("  meta coef (|w| desc):")
    for i in order:
        print(f"    {models[i]:26s} {best_coef[i]:+.3f}")

    out = C.SUB_DIR / "submission_stack.csv"
    pd.DataFrame({C.ID_COL: test_ids, C.TARGET: best_test}).to_csv(out, index=False)
    print(f"\nsaved: {out}  (rows={len(test_ids)})")


if __name__ == "__main__":
    main()
