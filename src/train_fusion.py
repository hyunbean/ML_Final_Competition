"""Feature Fusion — 우리 376피처 + 김민형 mega(572) 합쳐 새 강한 모델.

폴드 검증: 우리 folds.npy == mega_folds (seed42 StratifiedKFold, 일치율 1.0) →
mega TE 36개 포함 전부 누수 없음. 예측 블렌드는 0.729 포화 → 피처레벨 융합으로 돌파 시도.

mega 파일 위치: $MEGA_DIR (기본 <repo>/민형_mega/). mega_train.parquet/mega_test.parquet 필요.
실행: pip install lightgbm gensim pyarrow → python -m src.folds → python -m src.train_fusion
"""
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

from . import config as C
from .features import build_features
from .oof_io import save_predictions

MODEL_NAME = "fusion_lgbm"
MEGA_DIR = os.environ.get("MEGA_DIR", str(C.ROOT / "민형_mega"))
PARAMS = dict(objective="binary", metric="auc", learning_rate=0.02, num_leaves=96,
              max_depth=-1, min_child_samples=120, feature_fraction=0.4,
              bagging_fraction=0.8, bagging_freq=1, lambda_l1=0.1, lambda_l2=2.0,
              n_estimators=4000, n_jobs=-1, verbosity=-1)


def _merge_mega(X, ids, parquet):
    mega = pd.read_parquet(parquet)
    X = X.copy(); X[C.ID_COL] = ids
    out = X.merge(mega, on=C.ID_COL, how="left").set_index(C.ID_COL).reindex(ids)
    return out


def main():
    X, y, Xtest = build_features()
    train_ids = np.load(C.TRAIN_IDS_NPY, allow_pickle=True)
    test_ids = np.load(C.TEST_IDS_NPY, allow_pickle=True)
    folds = np.load(C.FOLDS_NPY)

    Xf = _merge_mega(X, train_ids, f"{MEGA_DIR}/mega_train.parquet")
    Xtf = _merge_mega(Xtest, test_ids, f"{MEGA_DIR}/mega_test.parquet")
    Xtf = Xtf.reindex(columns=Xf.columns, fill_value=0.0)
    Xv = np.nan_to_num(Xf.values.astype(np.float32), posinf=0, neginf=0)
    Xtv = np.nan_to_num(Xtf.values.astype(np.float32), posinf=0, neginf=0)
    print(f"fusion X={Xv.shape} (우리{X.shape[1]} + mega 합침)")

    oof = np.full(len(y), np.nan)
    test_sum = np.zeros(len(Xtv))
    for f in range(C.N_FOLDS):
        tr, va = np.where(folds != f)[0], np.where(folds == f)[0]
        m = lgb.LGBMClassifier(**PARAMS)
        m.fit(Xv[tr], y[tr], eval_set=[(Xv[va], y[va])],
              callbacks=[lgb.early_stopping(150, verbose=False)])
        oof[va] = m.predict_proba(Xv[va])[:, 1]
        test_sum += m.predict_proba(Xtv)[:, 1]
        print(f"[fold {f}] AUC={roc_auc_score(y[va], oof[va]):.5f}")

    cv = float(roc_auc_score(y, oof))
    print(f"\n==== {MODEL_NAME}  CV AUC = {cv:.5f} ====")
    save_predictions(MODEL_NAME, oof, test_sum / C.N_FOLDS, meta=dict(
        cv_auc=cv, seed=C.SEED, n_folds=C.N_FOLDS, feature_set="our376 + mega572 (fold-matched, TE safe)",
        created_by="hyunbean", notes="feature-level fusion: 우리 피처 + 김민형 mega(572) LGBM"))


if __name__ == "__main__":
    main()
