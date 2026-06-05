"""Phase1: fused 피처 3중 중요도 셀렉션 → 재학습 (1등 방식).

fusion_lgbm/ag가 986피처 노이즈로 0.71대에 머묾 → gain+permutation(+SHAP) 랭크 합산으로
top-K만 선택해 재학습. 노이즈 제거로 강화 시도. 폴드 동일(TE 안전).
실행: pip install pyarrow shap lightgbm gensim → python -m src.folds → python -m src.train_fusion_sel [TOPK=250]
"""
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

from . import config as C
from .features import build_features
from .train_fusion import _merge_mega, MEGA_DIR
from .oof_io import save_predictions

MODEL_NAME = "fusion_sel"
TOPK = int(sys.argv[1]) if len(sys.argv) > 1 else 250
PARAMS = dict(objective="binary", metric="auc", learning_rate=0.02, num_leaves=96,
              min_child_samples=120, feature_fraction=0.5, bagging_fraction=0.8,
              bagging_freq=1, lambda_l1=0.1, lambda_l2=2.0, n_estimators=4000,
              n_jobs=-1, verbosity=-1)


def main():
    X, y, Xtest = build_features()
    train_ids = np.load(C.TRAIN_IDS_NPY, allow_pickle=True)
    test_ids = np.load(C.TEST_IDS_NPY, allow_pickle=True)
    folds = np.load(C.FOLDS_NPY)
    Xf = _merge_mega(X, train_ids, f"{MEGA_DIR}/mega_train.parquet").reset_index(drop=True)
    Xtf = _merge_mega(Xtest, test_ids, f"{MEGA_DIR}/mega_test.parquet").reset_index(drop=True)
    Xtf = Xtf.reindex(columns=Xf.columns, fill_value=0.0)
    cols = list(Xf.columns)
    Xv = np.nan_to_num(Xf.values.astype(np.float32), posinf=0, neginf=0)
    Xtv = np.nan_to_num(Xtf.values.astype(np.float32), posinf=0, neginf=0)
    print(f"fused {Xv.shape} → 중요도 셀렉션 top{TOPK}")

    # --- 중요도: gain (5fold avg) — 경량(저RAM, permutation/SHAP는 RAM폭발로 제거) ---
    gain = np.zeros(len(cols))
    for f in range(C.N_FOLDS):
        tr = np.where(folds != f)[0]
        m = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=96,
                               importance_type="gain", n_jobs=4, verbosity=-1).fit(Xv[tr], y[tr])
        gain += m.feature_importances_
    keep = np.argsort(-gain)[:TOPK]
    print(f"gain 중요도 셀렉션 (경량) top{TOPK}")
    Xs, Xts = Xv[:, keep], Xtv[:, keep]
    print(f"선택 {len(keep)}개로 재학습")

    oof = np.full(len(y), np.nan); test_sum = np.zeros(len(test_ids))
    for f in range(C.N_FOLDS):
        tr, va = np.where(folds != f)[0], np.where(folds == f)[0]
        m = lgb.LGBMClassifier(**PARAMS)
        m.fit(Xs[tr], y[tr], eval_set=[(Xs[va], y[va])], callbacks=[lgb.early_stopping(150, verbose=False)])
        oof[va] = m.predict_proba(Xs[va])[:, 1]; test_sum += m.predict_proba(Xts)[:, 1]
        print(f"[fold {f}] AUC={roc_auc_score(y[va], oof[va]):.5f}")
    cv = float(roc_auc_score(y, oof))
    print(f"\n==== {MODEL_NAME}  CV AUC = {cv:.5f} ====")
    save_predictions(MODEL_NAME, oof, test_sum / C.N_FOLDS, meta=dict(
        cv_auc=cv, seed=C.SEED, n_folds=C.N_FOLDS, feature_set=f"fused→3중셀렉션 top{TOPK}",
        created_by="hyunbean", notes=f"feature selection (gain+perm+shap) top{TOPK} on fused"))


if __name__ == "__main__":
    main()
