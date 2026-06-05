"""Adversarial Validation — train vs test 분포 차이 진단 + 시프트 피처 탐지.

LGBM으로 '이 행이 train이냐 test냐'를 맞춰본다.
- AUC≈0.5 → 분포 동일 = CV 신뢰 OK
- AUC 높음 → 분포 차이(covariate shift) = CV-LB 갭 원인. 상위 importance 피처가 범인.
진단 전용(OOF 저장 안 함). 실행: python -m src.adversarial_validation
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from . import config as C
from .features import build_features

PARAMS = dict(objective="binary", metric="auc", learning_rate=0.05, num_leaves=63,
              feature_fraction=0.7, n_estimators=500, n_jobs=-1, verbosity=-1)


def main():
    X, _, Xtest = build_features()
    Xall = pd.concat([X, Xtest], ignore_index=True)
    is_test = np.r_[np.zeros(len(X)), np.ones(len(Xtest))]
    print(f"train={len(X)} test={len(Xtest)} feats={X.shape[1]}")

    oof = np.zeros(len(Xall))
    imp = np.zeros(X.shape[1])
    skf = StratifiedKFold(5, shuffle=True, random_state=C.SEED)
    for tr, va in skf.split(Xall, is_test):
        m = lgb.LGBMClassifier(**PARAMS)
        m.fit(Xall.iloc[tr], is_test[tr])
        oof[va] = m.predict_proba(Xall.iloc[va])[:, 1]
        imp += m.feature_importances_
    auc = roc_auc_score(is_test, oof)
    print(f"\n==== Adversarial AUC = {auc:.4f} ====")
    if auc < 0.55:
        print("✅ 분포 거의 동일 → CV 신뢰 가능 (시프트 거의 없음)")
    else:
        print("⚠️ 분포 차이 있음(covariate shift) → 아래 피처가 train/test 구분에 기여")
    top = pd.Series(imp, index=X.columns).sort_values(ascending=False).head(20)
    print("\n[시프트 기여 상위 20 피처]")
    for k, v in top.items():
        print(f"  {k:30s} {v:.0f}")


if __name__ == "__main__":
    main()
