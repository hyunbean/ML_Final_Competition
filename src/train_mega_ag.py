"""mega-only AutoGluon (강한 단일) — 김민형 mega 572피처에 장시간 best_quality AG.

fusion_ag(our+mega)는 우리 피처가 노이즈라 0.713로 떨어짐 → mega만 쓰면 더 강함.
긴 time + GPU NN으로 강한 단일 1개 만들어 블렌드 step-up 노림. 폴드 동일(TE 안전).
실행: pip install "autogluon.tabular[all]" pyarrow
     python -m src.train_mega_ag [TIME=3600] [PRESET=best_quality]
"""
import sys
import shutil
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from autogluon.tabular import TabularPredictor

from . import config as C
from .train_fusion import MEGA_DIR

MODEL_NAME = "mega_ag"
TIME = int(sys.argv[1]) if len(sys.argv) > 1 else 3600
PRESET = sys.argv[2] if len(sys.argv) > 2 else "best_quality"


def main():
    from .oof_io import save_predictions
    train_ids = np.load(C.TRAIN_IDS_NPY, allow_pickle=True)
    test_ids = np.load(C.TEST_IDS_NPY, allow_pickle=True)
    folds = np.load(C.FOLDS_NPY)
    y = pd.read_csv(C.YTRAIN_CSV).set_index(C.ID_COL).reindex(train_ids)[C.TARGET].to_numpy()

    tr = pd.read_parquet(f"{MEGA_DIR}/mega_train.parquet").set_index(C.ID_COL).reindex(train_ids).reset_index(drop=True)
    te = pd.read_parquet(f"{MEGA_DIR}/mega_test.parquet").set_index(C.ID_COL).reindex(test_ids).reset_index(drop=True)
    tr.columns = [f"c{i}" for i in range(tr.shape[1])]; te.columns = tr.columns
    print(f"mega-AG X={tr.shape}  time={TIME}/fold preset={PRESET}")

    oof = np.full(len(y), np.nan); test_sum = np.zeros(len(test_ids))
    for f in range(C.N_FOLDS):
        trn, va = np.where(folds != f)[0], np.where(folds == f)[0]
        df = tr.iloc[trn].copy(); df[C.TARGET] = y[trn]
        path = C.ARTIFACTS / f"ag_mega_fold{f}"
        if path.exists():
            shutil.rmtree(path)              # 이전 run 잔여 제거 (overwrite 경고/충돌 방지)
        pred = TabularPredictor(label=C.TARGET, eval_metric="roc_auc",
                                path=str(path)).fit(
            df, time_limit=TIME, presets=PRESET, verbosity=1,
            dynamic_stacking=False,                                   # DyStack OFF (작업2배·OOM·FileNotFound 원인)
            num_gpus=1, num_cpus=8,
            ag_args_ensemble={"fold_fitting_strategy": "sequential_local"},  # fold 순차학습 → Ray 병렬 OOM 제거
            ag_args_fit={"ag.max_memory_usage_ratio": 0.7})          # 메모리 보수적
        oof[va] = pred.predict_proba(tr.iloc[va])[1].values
        test_sum += pred.predict_proba(te)[1].values
        print(f"[fold {f}] AUC={roc_auc_score(y[va], oof[va]):.5f}")
    cv = float(roc_auc_score(y, oof))
    print(f"\n==== {MODEL_NAME}  CV AUC = {cv:.5f} ====")
    save_predictions(MODEL_NAME, oof, test_sum / C.N_FOLDS, meta=dict(
        cv_auc=cv, seed=C.SEED, n_folds=C.N_FOLDS, feature_set=f"mega572 AutoGluon {PRESET}",
        created_by="hyunbean", notes=f"mega-only AutoGluon {PRESET} {TIME}s/fold"))


if __name__ == "__main__":
    main()
