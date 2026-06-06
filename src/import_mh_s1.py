"""김민형 session1 모델 OOF 14종을 우리 oof_io로 import (custid 정렬 맞춤).

그의 블렌드선 redundant였으나 우리 블렌드는 구성이 달라 기여 가능 (특히 증강딥 0.713~0.715).
custid 정렬: 그의 mega parquet custid 순서 = npy 순서 → 우리 정규순서로 reindex.
실행: python -m src.import_mh_s1
"""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import config as C
from .oof_io import save_predictions

SRC = C.ROOT / "mega피처_김민형"
OOFDIR = SRC / "feature_attempts" / "model_oof_attempts"


def main():
    train_ids = np.load(C.TRAIN_IDS_NPY, allow_pickle=True)
    test_ids = np.load(C.TEST_IDS_NPY, allow_pickle=True)
    htr = pd.read_parquet(SRC / "mega_train.parquet", columns=[C.ID_COL])[C.ID_COL].to_numpy()
    hte = pd.read_parquet(SRC / "mega_test.parquet", columns=[C.ID_COL])[C.ID_COL].to_numpy()
    y = pd.read_csv(C.YTRAIN_CSV).set_index(C.ID_COL).reindex(train_ids)[C.TARGET].to_numpy()
    print(f"his train custid={len(htr)}, test={len(hte)}")

    names = sorted(set(f[4:-4] for f in os.listdir(OOFDIR) if f.startswith("oof_") and f.endswith(".npy")))
    for nm in names:
        oof = np.load(OOFDIR / f"oof_{nm}.npy")
        test = np.load(OOFDIR / f"test_{nm}.npy")
        if len(oof) != len(htr) or len(test) != len(hte):
            print(f"  ! {nm}: 길이 불일치 oof{len(oof)}/test{len(test)} → 스킵")
            continue
        o = pd.Series(oof, index=htr).reindex(train_ids).to_numpy()
        t = pd.Series(test, index=hte).reindex(test_ids).to_numpy()
        if np.isnan(o).any() or np.isnan(t).any():
            print(f"  ! {nm}: custid 정렬 NaN → 스킵")
            continue
        auc = float(roc_auc_score(y, o))
        save_predictions(f"mh_s1_{nm}", o, t, meta=dict(
            cv_auc=auc, seed=42, n_folds=5, feature_set="minhyung session1",
            created_by="minhyung", notes=f"s1 model oof import ({nm})"))
        print(f"  mh_s1_{nm}: AUC={auc:.5f}")


if __name__ == "__main__":
    main()
