"""Feature Fusion + AutoGluon — 우리376 + 김민형 mega572 융합 피처에 AutoGluon.

fusion_lgbm(단일)은 weight 0(mh_22가 메가를 더 잘 뽑음) → 메가엔 AG가 강하니
fused 피처에 AG best_quality. 그의 mega-AG(0.718~0.722)에 우리 피처까지 더해 넘기는 게 목표.
폴드 동일 검증됨(TE 안전). 5-fold OOF.

실행: pip install "autogluon.tabular[all]" pyarrow gensim lightgbm
     python -m src.folds → python -m src.train_fusion_ag [TIME=600] [PRESET=best_quality]
"""
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from autogluon.tabular import TabularPredictor

from . import config as C
from .features import build_features
from .train_fusion import _merge_mega, MEGA_DIR
from .oof_io import save_predictions

MODEL_NAME = "fusion_ag"
TIME = int(sys.argv[1]) if len(sys.argv) > 1 else 600
PRESET = sys.argv[2] if len(sys.argv) > 2 else "best_quality"


def main():
    X, y, Xtest = build_features()
    train_ids = np.load(C.TRAIN_IDS_NPY, allow_pickle=True)
    test_ids = np.load(C.TEST_IDS_NPY, allow_pickle=True)
    folds = np.load(C.FOLDS_NPY)

    Xf = _merge_mega(X, train_ids, f"{MEGA_DIR}/mega_train.parquet").reset_index(drop=True)
    Xtf = _merge_mega(Xtest, test_ids, f"{MEGA_DIR}/mega_test.parquet").reset_index(drop=True)
    Xtf = Xtf.reindex(columns=Xf.columns, fill_value=0.0)
    Xf.columns = [f"c{i}" for i in range(Xf.shape[1])]          # AG용 안전한 컬럼명
    Xtf.columns = Xf.columns
    print(f"fusion+AG X={Xf.shape}  time={TIME}/fold  preset={PRESET}")

    oof = np.full(len(y), np.nan)
    test_sum = np.zeros(len(test_ids))
    for f in range(C.N_FOLDS):
        tr, va = np.where(folds != f)[0], np.where(folds == f)[0]
        trdf = Xf.iloc[tr].copy(); trdf[C.TARGET] = y[tr]
        pred = TabularPredictor(label=C.TARGET, eval_metric="roc_auc",
                                path=str(C.ARTIFACTS / f"ag_fusion_fold{f}")).fit(
            trdf, time_limit=TIME, presets=PRESET, verbosity=1)
        oof[va] = pred.predict_proba(Xf.iloc[va])[1].values
        test_sum += pred.predict_proba(Xtf)[1].values
        print(f"[fold {f}] AUC={roc_auc_score(y[va], oof[va]):.5f}")

    cv = float(roc_auc_score(y, oof))
    print(f"\n==== {MODEL_NAME}  CV AUC = {cv:.5f} ====")
    save_predictions(MODEL_NAME, oof, test_sum / C.N_FOLDS, meta=dict(
        cv_auc=cv, seed=C.SEED, n_folds=C.N_FOLDS, feature_set="our376+mega572 AutoGluon",
        created_by="hyunbean", notes=f"fusion + AutoGluon {PRESET} {TIME}s/fold"))


if __name__ == "__main__":
    main()
