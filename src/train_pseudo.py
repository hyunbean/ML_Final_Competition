"""Pseudo-labeling (자기학습) — test 19995명을 준지도로 활용해 LGBM 재학습.

강한 기존 모델(lgbm_optuna) test 예측에서 '확신 높은' 샘플만 가짜라벨로 붙여
각 fold 학습셋에 추가 → 데이터 늘려 재학습. tabular에서 ~1% 보고된 다른 레버.

⚠️ 가짜라벨 출처가 전체train 학습 모델이라 CV가 약간 낙관적일 수 있음 → LB로 검증.
fold마다 OOF는 진짜 train val로만 측정. 실행: python -m src.folds → python -m src.train_pseudo
"""
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

from . import config as C
from .features import build_features
from .oof_io import save_predictions

MODEL_NAME = "pl_lgbm"
SRC = "lgbm_optuna"          # 가짜라벨 출처(강한 단일)
Q = 0.25                     # 상위/하위 25%만 확신 샘플로
PARAMS = dict(objective="binary", metric="auc", learning_rate=0.02, num_leaves=63,
              feature_fraction=0.6, bagging_fraction=0.8, bagging_freq=1,
              min_child_samples=40, n_estimators=3000, n_jobs=-1, verbosity=-1)


def main():
    X, y, Xtest = build_features()
    folds = np.load(C.FOLDS_NPY)
    src = np.load(C.OOF_DIR / f"{SRC}__test.npy")
    hi, lo = np.quantile(src, 1 - Q), np.quantile(src, Q)
    conf = np.where((src >= hi) | (src <= lo))[0]
    pl_y = (src[conf] >= hi).astype(int)
    Xc = Xtest.iloc[conf]
    print(f"X={X.shape}  확신 test 샘플={len(conf)} (pos={pl_y.mean():.3f}) 추가")

    oof = np.full(len(y), np.nan)
    test_sum = np.zeros(len(Xtest))
    for f in range(C.N_FOLDS):
        tr, va = np.where(folds != f)[0], np.where(folds == f)[0]
        Xa = np.vstack([X.iloc[tr].values, Xc.values])
        ya = np.concatenate([y[tr], pl_y])
        m = lgb.LGBMClassifier(**PARAMS)
        m.fit(Xa, ya, eval_set=[(X.iloc[va].values, y[va])],
              callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[va] = m.predict_proba(X.iloc[va].values)[:, 1]
        test_sum += m.predict_proba(Xtest.values)[:, 1]
        print(f"[fold {f}] AUC={roc_auc_score(y[va], oof[va]):.5f}")

    cv = float(roc_auc_score(y, oof))
    print(f"\n==== {MODEL_NAME}  CV AUC = {cv:.5f} (⚠️ 낙관 가능, LB로 검증) ====")
    save_predictions(MODEL_NAME, oof, test_sum / C.N_FOLDS, meta=dict(
        cv_auc=cv, seed=C.SEED, n_folds=C.N_FOLDS, feature_set=f"full + pseudo({SRC},q{Q})",
        created_by="hyunbean", notes=f"pseudo-labeling: confident {len(conf)} test rows added"))


if __name__ == "__main__":
    main()
