"""김민형 0607 핸드오프 OOF/test → artifacts/oof/ (우리 folds.npy로 생성됨 = 스태킹 안전).

temporal(corr0.52)·textsvd(0.80)·hypothesis(0.83)가 핵심 직교 멤버. bestblend69(0.997)는 중복.
실행: python -m src.import_mh_0607
"""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import config as C
from .oof_io import save_predictions

SRC = os.path.join(os.path.dirname(os.path.dirname(__file__)), "김민형_0607")
MODELS = ["mh_temporal_lgb", "mh_textsvd_lgb", "mh_hypothesis_lgb", "mh_heavy_lgb", "mh_bestblend69"]


def main():
    train_ids = np.load(C.TRAIN_IDS_NPY, allow_pickle=True)
    test_ids = np.load(C.TEST_IDS_NPY, allow_pickle=True)
    y = pd.read_csv(C.YTRAIN_CSV).set_index(C.ID_COL).reindex(train_ids)[C.TARGET].to_numpy()
    for m in MODELS:
        fo = os.path.join(SRC, f"{m}_oof.csv"); ft = os.path.join(SRC, f"{m}_test.csv")
        if not (os.path.exists(fo) and os.path.exists(ft)):
            print(f"  {m}: 파일 없음, 스킵"); continue
        oof = pd.read_csv(fo).set_index("custid")["pred"].reindex(train_ids).to_numpy()
        test = pd.read_csv(ft).set_index("custid")["pred"].reindex(test_ids).to_numpy()
        nan = np.isnan(oof).sum() + np.isnan(test).sum()
        cv = roc_auc_score(y, oof)
        print(f"  {m}: CV={cv:.5f}  (nan={nan})")
        save_predictions(m, oof, test, meta=dict(cv_auc=float(cv), seed=2026, n_folds=C.N_FOLDS,
                         feature_set="김민형 0607 핸드오프", created_by="minhyung",
                         notes=f"{m} (우리 folds.npy 재생성, 직접 스태킹)"))
    print("done -> artifacts/oof/")


if __name__ == "__main__":
    main()
