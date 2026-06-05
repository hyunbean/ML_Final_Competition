"""P3: OpenFE 자동 피처생성 — 우리 피처에서 고가치 교호항/변환을 자동 발굴.

OpenFE가 후보 피처(곱/나눗셈/groupby 등)를 생성·랭크 → 상위 N개만 원본에 추가해 LGBM.
수동으로 놓친 상호작용을 잡는 게 목적. 5-fold OOF.
실행: pip install openfe lightgbm gensim → python -m src.folds → python -m src.train_openfe [NNEW=60]
"""
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

from . import config as C
from .features import build_features
from .oof_io import save_predictions

MODEL_NAME = "openfe_lgbm"
NNEW = int(sys.argv[1]) if len(sys.argv) > 1 else 60
PARAMS = dict(objective="binary", metric="auc", learning_rate=0.02, num_leaves=64,
              min_child_samples=80, feature_fraction=0.6, bagging_fraction=0.8,
              bagging_freq=1, n_estimators=3000, n_jobs=4, verbosity=-1)


def main():
    from openfe import OpenFE, transform
    X, y, Xtest = build_features()
    X = X.copy(); Xtest = Xtest.copy()
    X.columns = [f"c{i}" for i in range(X.shape[1])]      # OpenFE 안전 컬럼명
    Xtest.columns = X.columns
    ys = pd.Series(y, index=X.index)
    print(f"OpenFE 자동생성 시작 (base {X.shape[1]} → 상위 {NNEW} 추가)")

    ofe = OpenFE()
    feats = ofe.fit(data=X, label=ys, n_jobs=8)
    Xtr2, Xte2 = transform(X, Xtest, feats[:NNEW], n_jobs=8)
    Xv = np.nan_to_num(Xtr2.values.astype(np.float32), posinf=0, neginf=0)
    Xtv = np.nan_to_num(Xte2.values.astype(np.float32), posinf=0, neginf=0)
    print(f"OpenFE 후 X={Xv.shape}")

    folds = np.load(C.FOLDS_NPY)
    oof = np.full(len(y), np.nan); test_sum = np.zeros(len(Xtv))
    for f in range(C.N_FOLDS):
        tr, va = np.where(folds != f)[0], np.where(folds == f)[0]
        m = lgb.LGBMClassifier(**PARAMS)
        m.fit(Xv[tr], y[tr], eval_set=[(Xv[va], y[va])], callbacks=[lgb.early_stopping(120, verbose=False)])
        oof[va] = m.predict_proba(Xv[va])[:, 1]; test_sum += m.predict_proba(Xtv)[:, 1]
        print(f"[fold {f}] AUC={roc_auc_score(y[va], oof[va]):.5f}")
    cv = float(roc_auc_score(y, oof))
    print(f"\n==== {MODEL_NAME}  CV AUC = {cv:.5f} ====")
    save_predictions(MODEL_NAME, oof, test_sum / C.N_FOLDS, meta=dict(
        cv_auc=cv, seed=C.SEED, n_folds=C.N_FOLDS, feature_set=f"our + OpenFE top{NNEW}",
        created_by="hyunbean", notes=f"OpenFE auto feature generation +{NNEW}"))


if __name__ == "__main__":
    main()
