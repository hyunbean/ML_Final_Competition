"""Adversarial validation — train vs test 구분 AUC. ~0.5면 분포동일(sample-weight 무의미), 0.6+면 시프트.
실행: pip install lightgbm gensim → python -m src.adv_validation
"""
import numpy as np, lightgbm as lgb
from sklearn.model_selection import cross_val_score
from . import config as C
from .features import build_features


def main():
    X, y, Xt = build_features()
    import pandas as pd
    Xa = pd.concat([X, Xt], ignore_index=True)
    ya = np.r_[np.zeros(len(X)), np.ones(len(Xt))]
    m = lgb.LGBMClassifier(n_estimators=300, num_leaves=31, random_state=C.SEED, verbose=-1)
    auc = cross_val_score(m, Xa, ya, cv=5, scoring="roc_auc").mean()
    print(f"\n==== Adversarial AUC = {auc:.4f} ====")
    print("~0.5 = train/test 분포동일(sample-weighting 무의미) / 0.6+ = 시프트존재(가중치 가치)")
    if auc > 0.6:
        m.fit(Xa, ya)
        imp = sorted(zip(m.feature_importances_, X.columns), reverse=True)[:15]
        print("시프트 주도 피처 top15:", [c for _, c in imp])


if __name__ == "__main__":
    main()
