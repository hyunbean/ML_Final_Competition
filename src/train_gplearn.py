"""gplearn 유전프로그래밍 파생피처 ('gp_lgbm') — top 중요피처에서 수식 자동생성 → 다양성.

SymbolicTransformer로 top-K 피처의 비선형 조합 수식 N개 생성 → LGBM. 단독성능보다 직교 다양성 목적.
실행: pip install gplearn lightgbm gensim → python -m src.train_gplearn
"""
import numpy as np
import lightgbm as lgb
from gplearn.genetic import SymbolicTransformer
from sklearn.metrics import roc_auc_score

from . import config as C
from .oof_io import save_predictions
from .features import build_features

MODEL_NAME = "gp_lgbm"
TOPK = 40


def main():
    folds = np.load(C.FOLDS_NPY)
    test_ids = np.load(C.TEST_IDS_NPY, allow_pickle=True)
    X, y, Xt = build_features()
    # 빠른 importance로 top-K 선정 (1회)
    base = lgb.LGBMClassifier(n_estimators=300, num_leaves=63, random_state=C.SEED, verbose=-1).fit(X, y)
    top = np.argsort(base.feature_importances_)[::-1][:TOPK]
    Xs = X.iloc[:, top].fillna(0).to_numpy(np.float32)
    Xts = Xt.iloc[:, top].fillna(0).to_numpy(np.float32)
    print(f"gplearn top{TOPK}피처로 수식 생성")
    oof = np.full(len(y), np.nan); test_sum = np.zeros(len(test_ids))
    params = dict(objective="binary", metric="auc", learning_rate=0.03, num_leaves=31,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, n_jobs=-1,
                  random_state=C.SEED, verbose=-1)
    for f in range(C.N_FOLDS):
        tri, va = np.where(folds != f)[0], np.where(folds == f)[0]
        gp = SymbolicTransformer(generations=15, population_size=1500, n_components=40,
                                 function_set=("add", "sub", "mul", "div", "sqrt", "log", "abs"),
                                 parsimony_coefficient=0.001, random_state=C.SEED, n_jobs=-1)
        gp.fit(Xs[tri], y[tri])
        Ftr = np.hstack([Xs, gp.transform(Xs)]); Fte = np.hstack([Xts, gp.transform(Xts)])
        m = lgb.LGBMClassifier(n_estimators=2000, **params)
        m.fit(Ftr[tri], y[tri], eval_set=[(Ftr[va], y[va])], callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[va] = m.predict_proba(Ftr[va])[:, 1]; test_sum += m.predict_proba(Fte)[:, 1]
        print(f"[fold {f}] AUC={roc_auc_score(y[va], oof[va]):.5f}")
    cv = float(roc_auc_score(y, oof))
    print(f"\n==== {MODEL_NAME}  CV AUC = {cv:.5f} ====")
    save_predictions(MODEL_NAME, oof, test_sum / C.N_FOLDS, meta=dict(
        cv_auc=cv, seed=C.SEED, n_folds=C.N_FOLDS, feature_set="gplearn 수식파생 + top40",
        created_by="hyunbean", notes="genetic programming symbolic features"))


if __name__ == "__main__":
    main()
