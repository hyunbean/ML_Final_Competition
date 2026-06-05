"""강한 teacher pseudo-labeling + fused 피처 — 마지막 강+decorrelated 시도.

pl_lgbm은 약한 teacher(lgbm_optuna 0.705)로도 블렌드 weight 0.20 먹었음.
이번엔 ① teacher = 현재 0.729 합동 블렌드 ② 피처 = our376+mega572 fused
→ 더 강한 가짜라벨 + 더 풍부한 피처 → mh_22 흔들기 시도.
폴드 동일(TE 안전). 실행: python -m src.train_pl_fusion
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

from . import config as C
from .features import build_features
from .train_fusion import _merge_mega, MEGA_DIR
from .oof_io import save_predictions

MODEL_NAME = "pl_fusion"
TEACHER = C.SUB_DIR / "submission_blend.csv"   # 현재 best 합동 hill-climb(0.729) test 예측
Q = 0.30                                        # 상/하위 30% 확신 샘플
PARAMS = dict(objective="binary", metric="auc", learning_rate=0.02, num_leaves=96,
              min_child_samples=120, feature_fraction=0.4, bagging_fraction=0.8,
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
    Xv = np.nan_to_num(Xf.values.astype(np.float32), posinf=0, neginf=0)
    Xtv = np.nan_to_num(Xtf.values.astype(np.float32), posinf=0, neginf=0)

    teach = pd.read_csv(TEACHER).set_index(C.ID_COL).reindex(test_ids)[C.TARGET].to_numpy()
    hi, lo = np.quantile(teach, 1 - Q), np.quantile(teach, Q)
    conf = np.where((teach >= hi) | (teach <= lo))[0]
    pl_y = (teach[conf] >= hi).astype(int)
    Xc = Xtv[conf]
    print(f"fused X={Xv.shape}  teacher=0.729블렌드  확신 test={len(conf)} (pos={pl_y.mean():.3f})")

    oof = np.full(len(y), np.nan)
    test_sum = np.zeros(len(test_ids))
    for f in range(C.N_FOLDS):
        tr, va = np.where(folds != f)[0], np.where(folds == f)[0]
        Xa = np.vstack([Xv[tr], Xc]); ya = np.concatenate([y[tr], pl_y])
        m = lgb.LGBMClassifier(**PARAMS)
        m.fit(Xa, ya, eval_set=[(Xv[va], y[va])], callbacks=[lgb.early_stopping(150, verbose=False)])
        oof[va] = m.predict_proba(Xv[va])[:, 1]
        test_sum += m.predict_proba(Xtv)[:, 1]
        print(f"[fold {f}] AUC={roc_auc_score(y[va], oof[va]):.5f}")

    cv = float(roc_auc_score(y, oof))
    print(f"\n==== {MODEL_NAME}  CV AUC = {cv:.5f} (⚠️ teacher 누수로 낙관 가능→LB) ====")
    save_predictions(MODEL_NAME, oof, test_sum / C.N_FOLDS, meta=dict(
        cv_auc=cv, seed=C.SEED, n_folds=C.N_FOLDS, feature_set="fused + pseudo(0.729 teacher)",
        created_by="hyunbean", notes=f"strong-teacher pseudo({len(conf)}) on fused our+mega"))


if __name__ == "__main__":
    main()
